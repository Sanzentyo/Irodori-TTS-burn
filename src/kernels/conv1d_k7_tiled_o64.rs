//! Device-limit-gated T64/O64/Cin16 DACVAE k=7 Conv1d kernel.
//!
//! Production selects this tile only for seven measured official-v4 shapes.
//! The prior O32 then O16 route remains the fallback whenever this kernel's
//! exact f32/layout/device/resource contract is unavailable. The guarded O64
//! output tail covers C=96 without out-of-bounds bias, weight, or output access.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::conv1d_k7_tiled::Conv1dK7Dilation;
use super::precision::{KernelFloatPrecision, common_float_precision};

const BATCH: usize = 1;
const KERNEL_SIZE: usize = 7;
const LOCAL_TIME_LANES: usize = 16;
const LOCAL_CHANNEL_LANES: usize = 16;
/// Time positions produced by one O64 workgroup.
pub const TIME_TILE: usize = 64;
/// Output channels covered by one O64 workgroup, including a guarded tail.
pub const OUTPUT_CHANNEL_TILE: usize = 64;
/// Input channels consumed between workgroup barrier pairs.
pub const INPUT_CHANNEL_TILE: usize = 16;
/// Invocations in the fixed 16x16 O64 workgroup.
pub const WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
const REQUIRED_BINDINGS: u32 = 4;

/// Workgroup storage used by T64/O64/Cin16 for a supported dilation.
pub const fn required_shared_memory_bytes(dilation: Conv1dK7Dilation) -> usize {
    let input_span = TIME_TILE + 6 * dilation.value();
    let input_elements = INPUT_CHANNEL_TILE * input_span;
    let weight_elements = OUTPUT_CHANNEL_TILE * INPUT_CHANNEL_TILE * KERNEL_SIZE;
    (input_elements + weight_elements) * core::mem::size_of::<f32>()
}

/// Number of O64 output-channel workgroups, including the guarded tail.
pub fn output_channel_tiles(channels: usize) -> Option<usize> {
    (channels > 0 && channels.is_multiple_of(INPUT_CHANNEL_TILE))
        .then(|| channels.div_ceil(OUTPUT_CHANNEL_TILE))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LaunchGeometry {
    input_span: usize,
    input_tile_size: usize,
    weight_load_size: usize,
    weight_tile_size: usize,
    shared_bytes: usize,
    time_tiles: u32,
    output_channel_tiles: u32,
}

impl LaunchGeometry {
    fn new(channels: usize, length: usize, dilation: Conv1dK7Dilation) -> Option<Self> {
        if length == 0 || !channels.is_multiple_of(INPUT_CHANNEL_TILE) {
            return None;
        }

        let input_span = TIME_TILE.checked_add(6usize.checked_mul(dilation.value())?)?;
        let input_tile_size = INPUT_CHANNEL_TILE.checked_mul(input_span)?;
        let weight_load_size = OUTPUT_CHANNEL_TILE
            .checked_mul(INPUT_CHANNEL_TILE)?
            .checked_mul(KERNEL_SIZE)?;
        let weight_tile_size = weight_load_size;
        let shared_bytes = input_tile_size
            .checked_add(weight_tile_size)?
            .checked_mul(core::mem::size_of::<f32>())?;
        let time_tiles_usize = length.div_ceil(TIME_TILE);
        let output_channel_tiles_usize = output_channel_tiles(channels)?;
        let time_tiles = u32::try_from(time_tiles_usize).ok()?;
        let output_channel_tiles = u32::try_from(output_channel_tiles_usize).ok()?;

        let input_elements = channels.checked_mul(length)?;
        let weight_elements = channels.checked_mul(channels)?.checked_mul(KERNEL_SIZE)?;
        let output_channel_extent = output_channel_tiles_usize.checked_mul(OUTPUT_CHANNEL_TILE)?;
        let final_time_base = time_tiles_usize.checked_sub(1)?.checked_mul(TIME_TILE)?;
        let largest_staged_time = final_time_base.checked_add(input_span - 1)?;

        for value in [
            channels,
            length,
            input_elements,
            weight_elements,
            output_channel_extent,
            input_span,
            input_tile_size,
            weight_load_size,
            weight_tile_size,
        ] {
            u32::try_from(value).ok()?;
        }
        i32::try_from(largest_staged_time).ok()?;
        i32::try_from(3usize.checked_mul(dilation.value())?).ok()?;

        Some(Self {
            input_span,
            input_tile_size,
            weight_load_size,
            weight_tile_size,
            shared_bytes,
            time_tiles,
            output_channel_tiles,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceLimits {
    max_bindings: u32,
    max_shared_memory_size: usize,
    max_cube_count: (u32, u32, u32),
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
}

impl DeviceLimits {
    fn supports(self, geometry: LaunchGeometry) -> bool {
        self.max_bindings >= REQUIRED_BINDINGS
            && self.max_shared_memory_size >= geometry.shared_bytes
            && self.max_units_per_cube >= WORKGROUP_SIZE as u32
            && self.max_cube_dim.0 >= LOCAL_TIME_LANES as u32
            && self.max_cube_dim.1 >= LOCAL_CHANNEL_LANES as u32
            && self.max_cube_dim.2 >= 1
            && self.max_cube_count.0 >= geometry.time_tiles
            && self.max_cube_count.1 >= geometry.output_channel_tiles
            && self.max_cube_count.2 >= BATCH as u32
    }
}

/// Check O64 shape and device limits without allocating or launching work.
pub fn device_supports_conv1d_k7_tiled_o64(
    input: &CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
) -> bool {
    if KernelFloatPrecision::from_dtype(input.dtype).is_none() || input.meta.num_dims() != 3 {
        return false;
    }
    let shape = input.meta.shape();
    if shape[0] != BATCH {
        return false;
    }
    let Some(geometry) = LaunchGeometry::new(shape[1], shape[2], dilation) else {
        return false;
    };
    let hardware = &input.client.properties().hardware;
    DeviceLimits {
        max_bindings: hardware.max_bindings,
        max_shared_memory_size: hardware.max_shared_memory_size,
        max_cube_count: hardware.max_cube_count,
        max_units_per_cube: hardware.max_units_per_cube,
        max_cube_dim: hardware.max_cube_dim,
    }
    .supports(geometry)
}

/// Check the complete non-panicking production O64 launch contract.
pub fn conv1d_k7_tiled_o64_contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
) -> bool {
    if input.meta.num_dims() != 3 || weight.meta.num_dims() != 3 || bias.meta.num_dims() != 1 {
        return false;
    }
    let input_shape = input.meta.shape();
    let channels = input_shape[1];
    let weight_shape = weight.meta.shape();
    common_float_precision([input.dtype, weight.dtype, bias.dtype]).is_some()
        && [input, weight, bias]
            .into_iter()
            .all(|tensor| tensor.device == input.device)
        && input_shape[0] == BATCH
        && input_shape[2] > 0
        && channels > 0
        && channels.is_multiple_of(INPUT_CHANNEL_TILE)
        && [weight_shape[0], weight_shape[1], weight_shape[2]] == [channels, channels, KERNEL_SIZE]
        && bias.meta.shape()[0] == channels
        && input.is_contiguous()
        && weight.is_contiguous()
        && bias.is_contiguous()
        && device_supports_conv1d_k7_tiled_o64(input, dilation)
}

#[derive(Debug)]
struct Conv1dK7TiledO64Kernel {
    precision: KernelFloatPrecision,
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_load_size: u32,
    weight_tile_size: u32,
}

impl KernelSource for Conv1dK7TiledO64Kernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("conv1d_k7_tiled_o64.wgsl"),
                include_str!("conv1d_k7_tiled_o64_f16.wgsl"),
            )
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.to_string())
            .register("padding", self.padding.to_string())
            .register("input_span", self.input_span.to_string())
            .register("input_tile_size", self.input_tile_size.to_string())
            .register("weight_load_size", self.weight_load_size.to_string())
            .register("weight_tile_size", self.weight_tile_size.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.channels,
            self.precision,
            self.length,
            self.dilation,
            self.padding,
            self.input_span,
            self.input_tile_size,
            self.weight_load_size,
            self.weight_tile_size,
        ))
    }
}

/// Compute exact-order f32 same-length Conv1d with the T64/O64/Cin16 tile.
///
/// Call [`conv1d_k7_tiled_o64_contract_is_compatible`] before production use.
/// Required layouts are contiguous input `[1, C, L]`, OIK weight `[C, C, 7]`,
/// and bias `[C]`. Padding is exactly `3 * dilation`.
///
/// # Panics
///
/// Panics when called directly outside the documented launch contract.
pub fn conv1d_k7_same_tiled_o64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
) -> CubeTensor<WgpuRuntime> {
    assert!(
        conv1d_k7_tiled_o64_contract_is_compatible(&input, &weight, &bias, dilation),
        "T64/O64 k=7 Conv1d requires compatible f32 shape/layout/device/resource limits"
    );

    let input_shape = input.meta.shape();
    let precision = KernelFloatPrecision::from_dtype(input.dtype)
        .expect("compatible O64 contract accepted only f32 or f16");
    let batch = input_shape[0];
    let channels = input_shape[1];
    let length = input_shape[2];
    let geometry = LaunchGeometry::new(channels, length, dilation)
        .expect("validated O64 launch geometry must remain representable");
    let input_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("validated input/output element count must not overflow");
    let output_bytes = input_elements
        .checked_mul(precision.element_bytes())
        .expect("output byte count overflow");
    let client = input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, channels, length]),
        client.empty(output_bytes),
        precision.dtype(),
    );

    let dilation_value = dilation.value();
    let kernel = Conv1dK7TiledO64Kernel {
        precision,
        channels: u32::try_from(channels).expect("validated C must fit u32"),
        length: u32::try_from(length).expect("validated L must fit u32"),
        dilation: u32::try_from(dilation_value).expect("validated dilation must fit u32"),
        padding: u32::try_from(3 * dilation_value).expect("validated padding must fit u32"),
        input_span: u32::try_from(geometry.input_span).expect("validated input span must fit u32"),
        input_tile_size: u32::try_from(geometry.input_tile_size)
            .expect("validated input tile size must fit u32"),
        weight_load_size: u32::try_from(geometry.weight_load_size)
            .expect("validated weight load size must fit u32"),
        weight_tile_size: u32::try_from(geometry.weight_tile_size)
            .expect("validated weight tile size must fit u32"),
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(LOCAL_TIME_LANES as u32, LOCAL_CHANNEL_LANES as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(weight.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(
        task,
        CubeCount::new_3d(
            geometry.time_tiles,
            geometry.output_channel_tiles,
            BATCH as u32,
        ),
        bindings,
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_memory_matches_measured_dilations() {
        assert_eq!(
            [
                Conv1dK7Dilation::One,
                Conv1dK7Dilation::Three,
                Conv1dK7Dilation::Nine,
            ]
            .map(required_shared_memory_bytes),
            [33_152, 33_920, 36_224]
        );
    }

    #[test]
    fn official_shapes_include_the_guarded_c96_tail() {
        for (channels, length, expected_output_tiles) in [
            (768, 600, 12),
            (384, 6_000, 6),
            (192, 48_000, 3),
            (96, 96_000, 2),
        ] {
            for dilation in [
                Conv1dK7Dilation::One,
                Conv1dK7Dilation::Three,
                Conv1dK7Dilation::Nine,
            ] {
                let geometry = LaunchGeometry::new(channels, length, dilation)
                    .expect("every official shape must have safe O64 geometry");
                assert_eq!(geometry.output_channel_tiles, expected_output_tiles);
                assert!(geometry.shared_bytes <= 49_152);
            }
        }
    }

    #[test]
    fn launch_limit_check_rejects_each_nonportable_shortfall() {
        let geometry = LaunchGeometry::new(96, 96_000, Conv1dK7Dilation::Nine)
            .expect("official shape must have valid launch geometry");
        let sufficient = DeviceLimits {
            max_bindings: 4,
            max_shared_memory_size: geometry.shared_bytes,
            max_cube_count: (geometry.time_tiles, geometry.output_channel_tiles, 1),
            max_units_per_cube: 256,
            max_cube_dim: (16, 16, 1),
        };
        assert!(sufficient.supports(geometry));
        assert!(
            !DeviceLimits {
                max_shared_memory_size: geometry.shared_bytes - 1,
                ..sufficient
            }
            .supports(geometry)
        );
        assert!(
            !DeviceLimits {
                max_units_per_cube: 255,
                ..sufficient
            }
            .supports(geometry)
        );
        assert!(
            !DeviceLimits {
                max_cube_dim: (16, 15, 1),
                ..sufficient
            }
            .supports(geometry)
        );
        assert!(
            !DeviceLimits {
                max_cube_count: (geometry.time_tiles - 1, geometry.output_channel_tiles, 1),
                ..sufficient
            }
            .supports(geometry)
        );
    }

    fn scalar_fma(
        input: &[f32],
        weight: &[f32],
        bias: f32,
        channels: usize,
        length: usize,
        output_time: usize,
        dilation: usize,
    ) -> f32 {
        let padding = 3 * dilation;
        let mut accumulator = bias;
        for input_channel in 0..channels {
            for kernel_index in 0..KERNEL_SIZE {
                let source_time =
                    output_time as isize + (kernel_index * dilation) as isize - padding as isize;
                let value = if (0..length as isize).contains(&source_time) {
                    input[input_channel * length + source_time as usize]
                } else {
                    0.0
                };
                accumulator = value.mul_add(
                    weight[input_channel * KERNEL_SIZE + kernel_index],
                    accumulator,
                );
            }
        }
        accumulator
    }

    fn tiled_fma(
        input: &[f32],
        weight: &[f32],
        bias: f32,
        channels: usize,
        length: usize,
        output_time: usize,
        dilation: usize,
    ) -> f32 {
        let padding = 3 * dilation;
        let mut accumulator = bias;
        for input_channel_base in (0..channels).step_by(INPUT_CHANNEL_TILE) {
            for tile_input_channel in 0..INPUT_CHANNEL_TILE {
                let input_channel = input_channel_base + tile_input_channel;
                for kernel_index in 0..KERNEL_SIZE {
                    let source_time = output_time as isize + (kernel_index * dilation) as isize
                        - padding as isize;
                    let value = if (0..length as isize).contains(&source_time) {
                        input[input_channel * length + source_time as usize]
                    } else {
                        0.0
                    };
                    accumulator = value.mul_add(
                        weight[input_channel * KERNEL_SIZE + kernel_index],
                        accumulator,
                    );
                }
            }
        }
        accumulator
    }

    #[test]
    fn cpu_tile_sequence_is_bitwise_scalar_fma_order() {
        let channels = 32;
        let length = 19;
        let input = (0..channels * length)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) / 16.0)
            .collect::<Vec<_>>();
        let weight = (0..channels * KERNEL_SIZE)
            .map(|index| ((index * 7 % 31) as f32 - 15.0) / 256.0)
            .collect::<Vec<_>>();
        let bias = -3.0 / 128.0;

        for dilation in [1, 3, 9] {
            for output_time in 0..length {
                assert_eq!(
                    tiled_fma(
                        &input,
                        &weight,
                        bias,
                        channels,
                        length,
                        output_time,
                        dilation,
                    )
                    .to_bits(),
                    scalar_fma(
                        &input,
                        &weight,
                        bias,
                        channels,
                        length,
                        output_time,
                        dilation,
                    )
                    .to_bits(),
                );
            }
        }
    }

    #[test]
    fn source_contract_is_exact_and_fully_templated() {
        let shader = include_str!("conv1d_k7_tiled_o64.wgsl");
        let input_channel_loop = shader
            .find("var tile_input_channel = 0u")
            .expect("shader must iterate tile input channels");
        let kernel_loop = shader[input_channel_loop..]
            .find("var kernel_index = 0u")
            .map(|offset| input_channel_loop + offset)
            .expect("kernel loop must be inside the input-channel loop");
        let first_fma = shader[kernel_loop..]
            .find("= fma(")
            .map(|offset| kernel_loop + offset)
            .expect("shader must use explicit fma");
        let input_tile_advance = shader[first_fma..]
            .find("input_channel_base += INPUT_CHANNEL_TILE")
            .map(|offset| first_fma + offset)
            .expect("shader must advance only after consuming a tile");
        assert!(input_channel_loop < kernel_loop);
        assert!(kernel_loop < first_fma);
        assert!(first_fma < input_tile_advance);
        assert_eq!(shader.matches("= fma(").count(), 16);
        assert!(shader.contains("if output_channel_3 < CHANNELS"));

        let bindings = shader
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("@group(0)") && line.contains("var<storage"))
            .collect::<Vec<_>>();
        assert_eq!(bindings.len(), 4);
        assert!(
            bindings
                .iter()
                .all(|line| line.contains("var<storage, read_write>"))
        );

        let rendered = Conv1dK7TiledO64Kernel {
            precision: KernelFloatPrecision::F32,
            channels: 96,
            length: 96_000,
            dilation: 9,
            padding: 27,
            input_span: 118,
            input_tile_size: 1_888,
            weight_load_size: 7_168,
            weight_tile_size: 7_168,
        }
        .source()
        .complete();
        assert!(!rendered.contains("{{"));
    }
}
