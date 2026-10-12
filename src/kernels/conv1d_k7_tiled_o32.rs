//! Device-limit-gated T64/O32/WG256 DACVAE k=7 Conv1d kernel.
//!
//! The portable [`super::conv1d_k7_tiled`] T64/O16/WG128 kernel remains the
//! fallback. This wider output tile is selected only for measured official-v4
//! shapes and only when a read-only device-limit check proves that its 256
//! invocations and dilation-dependent 18.4--21.4 KiB workgroup allocation fit.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
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
const TIME_TILE: usize = 64;
const OUTPUT_CHANNEL_TILE: usize = 32;
const INPUT_CHANNEL_TILE: usize = 16;
const WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
const REQUIRED_BINDINGS: u32 = 4;

/// Workgroup storage used by T64/O32 for a supported dilation.
pub const fn required_shared_memory_bytes(dilation: Conv1dK7Dilation) -> usize {
    let input_span = TIME_TILE + 6 * dilation.value();
    let input_elements = INPUT_CHANNEL_TILE * input_span;
    let weight_elements = OUTPUT_CHANNEL_TILE * INPUT_CHANNEL_TILE * KERNEL_SIZE;
    (input_elements + weight_elements) * core::mem::size_of::<f32>()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LaunchGeometry {
    input_span: usize,
    input_tile_size: usize,
    weight_tile_size: usize,
    shared_bytes: usize,
    time_tiles: u32,
    output_channel_tiles: u32,
}

impl LaunchGeometry {
    fn new(channels: usize, length: usize, dilation: Conv1dK7Dilation) -> Option<Self> {
        if channels == 0 || length == 0 || !channels.is_multiple_of(OUTPUT_CHANNEL_TILE) {
            return None;
        }
        let dilation = dilation.value();
        let input_span = TIME_TILE.checked_add(6usize.checked_mul(dilation)?)?;
        let input_tile_size = INPUT_CHANNEL_TILE.checked_mul(input_span)?;
        let weight_tile_size = OUTPUT_CHANNEL_TILE
            .checked_mul(INPUT_CHANNEL_TILE)?
            .checked_mul(KERNEL_SIZE)?;
        let shared_bytes = input_tile_size
            .checked_add(weight_tile_size)?
            .checked_mul(core::mem::size_of::<f32>())?;
        let time_tiles = u32::try_from(length.div_ceil(TIME_TILE)).ok()?;
        let output_channel_tiles = u32::try_from(channels / OUTPUT_CHANNEL_TILE).ok()?;
        let input_elements = channels.checked_mul(length)?;
        let weight_elements = channels.checked_mul(channels)?.checked_mul(KERNEL_SIZE)?;
        u32::try_from(channels).ok()?;
        u32::try_from(length).ok()?;
        u32::try_from(input_elements).ok()?;
        u32::try_from(weight_elements).ok()?;
        u32::try_from(input_span).ok()?;
        u32::try_from(input_tile_size).ok()?;
        u32::try_from(weight_tile_size).ok()?;
        let final_time_base = usize::try_from(time_tiles)
            .ok()?
            .checked_sub(1)?
            .checked_mul(TIME_TILE)?;
        let largest_staged_time = final_time_base.checked_add(input_span - 1)?;
        i32::try_from(largest_staged_time).ok()?;

        Some(Self {
            input_span,
            input_tile_size,
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

/// Check the wider tile's shape and device limits without allocating or
/// launching GPU work.
pub fn device_supports_conv1d_k7_tiled_o32(
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

#[derive(Debug)]
struct Conv1dK7TiledO32Kernel {
    precision: KernelFloatPrecision,
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
}

impl KernelSource for Conv1dK7TiledO32Kernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("conv1d_k7_tiled_o32.wgsl"),
                include_str!("conv1d_k7_tiled_o32_f16.wgsl"),
            )
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.to_string())
            .register("padding", self.padding.to_string())
            .register("input_span", self.input_span.to_string())
            .register("input_tile_size", self.input_tile_size.to_string())
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
            self.weight_tile_size,
        ))
    }
}

/// Compute the exact production f32 same-length Conv1d with a T64/O32 tile.
///
/// Required physical layouts are input `[1, C, L]`, contiguous OIK weight
/// `[C, C, 7]`, and contiguous bias `[C]`. `C` must be divisible by 32.
/// Call [`device_supports_conv1d_k7_tiled_o32`] first when a portable fallback
/// is required.
///
/// # Panics
///
/// Panics for an incompatible dtype, rank, shape, device, parameter layout,
/// insufficient device limits, or an unsafe WGPU/WGSL index calculation.
pub fn conv1d_k7_same_tiled_o32_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
) -> CubeTensor<WgpuRuntime> {
    let precision = common_float_precision([input.dtype, weight.dtype, bias.dtype])
        .expect("T64/O32 k=7 Conv1d tensors must share f32 or f16 dtype");
    for (_, tensor) in [("input", &input), ("weight", &weight), ("bias", &bias)] {
        input.assert_is_on_same_device(tensor);
    }
    assert_eq!(input.meta.num_dims(), 3, "input must be rank 3 [1, C, L]");
    assert_eq!(weight.meta.num_dims(), 3, "weight must be rank 3 [C, C, 7]");
    assert_eq!(bias.meta.num_dims(), 1, "bias must be rank 1 [C]");

    let input_shape = input.meta.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let length = input_shape[2];
    assert_eq!(batch, BATCH, "T64/O32 k=7 Conv1d is specialised for B=1");
    let geometry = LaunchGeometry::new(channels, length, dilation)
        .expect("T64/O32 k=7 Conv1d requires non-empty C/L, C divisible by 32, and safe indices");
    let weight_shape = weight.meta.shape();
    assert_eq!(
        [weight_shape[0], weight_shape[1], weight_shape[2]],
        [channels, channels, KERNEL_SIZE],
        "weight must have shape [C, C, 7]"
    );
    assert_eq!(bias.meta.shape()[0], channels, "bias must have shape [C]");
    assert!(weight.is_contiguous(), "weight must be contiguous OIK");
    assert!(bias.is_contiguous(), "bias must be contiguous");
    assert!(
        device_supports_conv1d_k7_tiled_o32(&input, dilation),
        "T64/O32 k=7 Conv1d exceeds the current device's binding, shared-memory, workgroup, or dispatch limits"
    );

    let input = into_contiguous(input);
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
    let kernel = Conv1dK7TiledO32Kernel {
        precision,
        channels: u32::try_from(channels).expect("validated C must fit u32"),
        length: u32::try_from(length).expect("validated L must fit u32"),
        dilation: u32::try_from(dilation_value).expect("validated dilation must fit u32"),
        padding: u32::try_from(3 * dilation_value).expect("validated padding must fit u32"),
        input_span: u32::try_from(geometry.input_span).expect("validated input span must fit u32"),
        input_tile_size: u32::try_from(geometry.input_tile_size)
            .expect("validated input tile size must fit u32"),
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
            [18_816, 19_584, 21_888]
        );
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
}
