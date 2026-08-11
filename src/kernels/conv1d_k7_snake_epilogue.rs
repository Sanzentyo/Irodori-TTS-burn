//! DACVAE k=7 Conv1d + Snake1d output epilogue.
//!
//! One parameterised WGSL body covers the accepted T64/O16/WG128 and
//! T64/O32/WG256 convolution tiles, then applies the exact `snake.wgsl`
//! operation sequence immediately before each output store. Production uses
//! this only for ResidualUnit `act1` after its five-binding and layout contract
//! passes; the standalone two-dispatch path remains the fail-safe fallback.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::conv1d_k7_tiled::Conv1dK7Dilation;

const BATCH: usize = 1;
const KERNEL_SIZE: usize = 7;
const LOCAL_TIME_LANES: usize = 16;
const TIME_TILE: usize = 64;
const INPUT_CHANNEL_TILE: usize = 16;
const REQUIRED_BINDINGS: u32 = 5;
const PORTABLE_WORKGROUP_STORAGE_BYTES: usize = 16 * 1024;

/// Accepted output-channel tile retained by the fused act1 epilogue.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Conv1dK7SnakeTile {
    /// T64/O16/WG128, the portable accepted fallback.
    Output16,
    /// T64/O32/WG256, the measured official-shape preferred route.
    Output32,
}

impl Conv1dK7SnakeTile {
    /// Human-readable tile label used by the benchmark.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Output16 => "T64/O16",
            Self::Output32 => "T64/O32",
        }
    }

    /// Number of output channels computed by one workgroup.
    pub const fn output_channel_tile(self) -> usize {
        match self {
            Self::Output16 => 16,
            Self::Output32 => 32,
        }
    }

    /// Workgroup Y dimension; X is always 16.
    pub const fn local_channel_lanes(self) -> usize {
        self.output_channel_tile() / 2
    }

    /// Number of invocations in one workgroup.
    pub const fn workgroup_size(self) -> usize {
        LOCAL_TIME_LANES * self.local_channel_lanes()
    }

    /// Exact shared-memory byte requirement for this tile and dilation.
    pub const fn shared_memory_bytes(self, dilation: Conv1dK7Dilation) -> usize {
        let input_span = TIME_TILE + 6 * dilation.value();
        let input_elements = INPUT_CHANNEL_TILE * input_span;
        let weight_elements = self.output_channel_tile() * INPUT_CHANNEL_TILE * KERNEL_SIZE;
        (input_elements + weight_elements) * core::mem::size_of::<f32>()
    }
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
    fn new(
        channels: usize,
        length: usize,
        dilation: Conv1dK7Dilation,
        tile: Conv1dK7SnakeTile,
    ) -> Option<Self> {
        let output_channel_tile = tile.output_channel_tile();
        if channels == 0 || length == 0 || !channels.is_multiple_of(output_channel_tile) {
            return None;
        }

        let input_span = TIME_TILE.checked_add(6usize.checked_mul(dilation.value())?)?;
        let input_tile_size = INPUT_CHANNEL_TILE.checked_mul(input_span)?;
        let weight_tile_size = output_channel_tile
            .checked_mul(INPUT_CHANNEL_TILE)?
            .checked_mul(KERNEL_SIZE)?;
        let shared_bytes = input_tile_size
            .checked_add(weight_tile_size)?
            .checked_mul(core::mem::size_of::<f32>())?;
        if tile == Conv1dK7SnakeTile::Output16 && shared_bytes > PORTABLE_WORKGROUP_STORAGE_BYTES {
            return None;
        }

        let time_tiles = u32::try_from(length.div_ceil(TIME_TILE)).ok()?;
        let output_channel_tiles = u32::try_from(channels / output_channel_tile).ok()?;
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
    fn supports(self, geometry: LaunchGeometry, tile: Conv1dK7SnakeTile) -> bool {
        self.max_bindings >= REQUIRED_BINDINGS
            && self.max_shared_memory_size >= geometry.shared_bytes
            && self.max_units_per_cube >= tile.workgroup_size() as u32
            && self.max_cube_dim.0 >= LOCAL_TIME_LANES as u32
            && self.max_cube_dim.1 >= tile.local_channel_lanes() as u32
            && self.max_cube_dim.2 >= 1
            && self.max_cube_count.0 >= geometry.time_tiles
            && self.max_cube_count.1 >= geometry.output_channel_tiles
            && self.max_cube_count.2 >= BATCH as u32
    }
}

/// Check the fused candidate's shape and five-binding device contract.
///
/// A future production selector must retain `accepted k7 -> Snake` when this
/// returns false; four-binding support for the current convolution alone is
/// not sufficient for this candidate.
pub fn device_supports_conv1d_k7_snake_epilogue(
    input: &CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7SnakeTile,
) -> bool {
    if input.meta.num_dims() != 3 {
        return false;
    }
    let shape = input.meta.shape();
    if shape[0] != BATCH {
        return false;
    }
    let Some(geometry) = LaunchGeometry::new(shape[1], shape[2], dilation, tile) else {
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
    .supports(geometry, tile)
}

#[derive(Debug)]
struct Conv1dK7SnakeEpilogueKernel {
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
    output_channel_tile: u32,
    local_channel_lanes: u32,
    workgroup_size: u32,
}

impl KernelSource for Conv1dK7SnakeEpilogueKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_snake_epilogue.wgsl"))
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.to_string())
            .register("padding", self.padding.to_string())
            .register("input_span", self.input_span.to_string())
            .register("input_tile_size", self.input_tile_size.to_string())
            .register("weight_tile_size", self.weight_tile_size.to_string())
            .register("output_channel_tile", self.output_channel_tile.to_string())
            .register("local_channel_lanes", self.local_channel_lanes.to_string())
            .register("workgroup_size", self.workgroup_size.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.channels,
            self.length,
            self.dilation,
            self.padding,
            self.input_span,
            self.input_tile_size,
            self.weight_tile_size,
            self.output_channel_tile,
            self.local_channel_lanes,
            self.workgroup_size,
        ))
    }
}

/// Compute the accepted f32 k=7 convolution and apply Snake1d before storing.
///
/// Required physical layouts are input `[1, C, L]`, contiguous OIK weight
/// `[C, C, 7]`, contiguous bias `[C]`, and alpha `[1, C, 1]`. The candidate
/// adds only the alpha binding and epilogue to the accepted convolution order.
///
/// # Panics
///
/// Panics for an incompatible dtype, rank, shape, device, layout, unsafe
/// index calculation, or insufficient five-binding/device-limit contract.
pub fn conv1d_k7_same_snake_epilogue_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7SnakeTile,
) -> CubeTensor<WgpuRuntime> {
    for (name, tensor) in [
        ("input", &input),
        ("weight", &weight),
        ("bias", &bias),
        ("alpha", &alpha),
    ] {
        assert_eq!(
            tensor.dtype,
            DType::F32,
            "k=7 Conv1d + Snake candidate only supports f32 {name}"
        );
        input.assert_is_on_same_device(tensor);
    }
    assert_eq!(input.meta.num_dims(), 3, "input must be rank 3 [1, C, L]");
    assert_eq!(weight.meta.num_dims(), 3, "weight must be rank 3 [C, C, 7]");
    assert_eq!(bias.meta.num_dims(), 1, "bias must be rank 1 [C]");
    assert_eq!(alpha.meta.num_dims(), 3, "alpha must be rank 3 [1, C, 1]");

    let input_shape = input.meta.shape();
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];
    assert_eq!(batch, BATCH, "k=7 Conv1d + Snake is specialised for B=1");
    let geometry = LaunchGeometry::new(channels, length, dilation, tile).unwrap_or_else(|| {
        panic!(
            "{} k=7 Conv1d + Snake requires non-empty C/L, a compatible C tile, and safe indices",
            tile.label()
        )
    });

    let weight_shape = weight.meta.shape();
    assert_eq!(
        [weight_shape[0], weight_shape[1], weight_shape[2]],
        [channels, channels, KERNEL_SIZE],
        "weight must have shape [C, C, 7]"
    );
    assert_eq!(bias.meta.shape()[0], channels, "bias must have shape [C]");
    let alpha_shape = alpha.meta.shape();
    assert_eq!(
        [alpha_shape[0], alpha_shape[1], alpha_shape[2]],
        [1, channels, 1],
        "Snake alpha must have shape [1, C, 1]"
    );
    assert!(weight.is_contiguous(), "weight must be contiguous OIK");
    assert!(bias.is_contiguous(), "bias must be contiguous");
    assert!(
        device_supports_conv1d_k7_snake_epilogue(&input, dilation, tile),
        "{} k=7 Conv1d + Snake exceeds the five-binding, shared-memory, workgroup, or dispatch limits",
        tile.label()
    );

    let input = into_contiguous(input);
    let alpha = into_contiguous(alpha);
    let input_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("validated input/output element count must not overflow");
    let output_bytes = input_elements
        .checked_mul(core::mem::size_of::<f32>())
        .expect("output byte count overflow");
    let client = input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, channels, length]),
        client.empty(output_bytes),
        DType::F32,
    );

    let dilation_value = dilation.value();
    let kernel = Conv1dK7SnakeEpilogueKernel {
        channels: u32::try_from(channels).expect("validated C must fit u32"),
        length: u32::try_from(length).expect("validated L must fit u32"),
        dilation: u32::try_from(dilation_value).expect("validated dilation must fit u32"),
        padding: u32::try_from(3 * dilation_value).expect("validated padding must fit u32"),
        input_span: u32::try_from(geometry.input_span).expect("validated input span must fit u32"),
        input_tile_size: u32::try_from(geometry.input_tile_size)
            .expect("validated input tile size must fit u32"),
        weight_tile_size: u32::try_from(geometry.weight_tile_size)
            .expect("validated weight tile size must fit u32"),
        output_channel_tile: u32::try_from(tile.output_channel_tile())
            .expect("validated output-channel tile must fit u32"),
        local_channel_lanes: u32::try_from(tile.local_channel_lanes())
            .expect("validated local-channel lane count must fit u32"),
        workgroup_size: u32::try_from(tile.workgroup_size())
            .expect("validated workgroup size must fit u32"),
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(
                LOCAL_TIME_LANES as u32,
                u32::try_from(tile.local_channel_lanes())
                    .expect("validated local-channel lane count must fit u32"),
            ),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(weight.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding())
        .with_buffer(alpha.handle.binding());
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

    const TEST_CHANNELS: usize = 16;
    const TEST_LENGTH: usize = 19;

    fn deterministic_values(
        length: usize,
        modulus: usize,
        multiplier: usize,
        scale: f32,
    ) -> Vec<f32> {
        let centre = (modulus / 2) as f32;
        (0..length)
            .map(|index| (((index % modulus) * multiplier) % modulus) as f32 - centre)
            .map(|value| value * scale)
            .collect()
    }

    fn scalar_convolution(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        dilation: Conv1dK7Dilation,
    ) -> Vec<f32> {
        let dilation = dilation.value();
        let padding = 3 * dilation;
        let mut output = vec![0.0_f32; TEST_CHANNELS * TEST_LENGTH];
        for output_channel in 0..TEST_CHANNELS {
            for output_time in 0..TEST_LENGTH {
                let mut accumulator = bias[output_channel];
                for input_channel in 0..TEST_CHANNELS {
                    for kernel_index in 0..KERNEL_SIZE {
                        let source_time = output_time as isize + (kernel_index * dilation) as isize
                            - padding as isize;
                        if (0..TEST_LENGTH as isize).contains(&source_time) {
                            let input_index = input_channel * TEST_LENGTH + source_time as usize;
                            let weight_index = (output_channel * TEST_CHANNELS + input_channel)
                                * KERNEL_SIZE
                                + kernel_index;
                            accumulator =
                                input[input_index].mul_add(weight[weight_index], accumulator);
                        }
                    }
                }
                output[output_channel * TEST_LENGTH + output_time] = accumulator;
            }
        }
        output
    }

    fn scalar_snake(x: f32, alpha: f32) -> f32 {
        let sine = (alpha * x).sin();
        x + (sine * sine) / (alpha + 1.0e-9)
    }

    fn normalized_main_prefix(shader: &str) -> String {
        let main = shader
            .split_once("fn main(")
            .expect("shader must define main")
            .1;
        let prefix = main
            .split_once("    let output_base_0")
            .expect("shader must compute output_base_0")
            .0;
        prefix
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn shared_memory_matches_both_accepted_tiles() {
        let dilations = [
            Conv1dK7Dilation::One,
            Conv1dK7Dilation::Three,
            Conv1dK7Dilation::Nine,
        ];
        assert_eq!(
            dilations.map(|dilation| { Conv1dK7SnakeTile::Output16.shared_memory_bytes(dilation) }),
            [11_648, 12_416, 14_720]
        );
        assert_eq!(
            dilations.map(|dilation| { Conv1dK7SnakeTile::Output32.shared_memory_bytes(dilation) }),
            [18_816, 19_584, 21_888]
        );
    }

    #[test]
    fn fifth_binding_is_a_hard_candidate_requirement() {
        let tile = Conv1dK7SnakeTile::Output32;
        let geometry = LaunchGeometry::new(96, 96_000, Conv1dK7Dilation::Nine, tile)
            .expect("official shape must have valid launch geometry");
        let sufficient = DeviceLimits {
            max_bindings: REQUIRED_BINDINGS,
            max_shared_memory_size: geometry.shared_bytes,
            max_cube_count: (geometry.time_tiles, geometry.output_channel_tiles, 1),
            max_units_per_cube: 256,
            max_cube_dim: (16, 16, 1),
        };
        assert!(sufficient.supports(geometry, tile));
        assert!(
            !DeviceLimits {
                max_bindings: REQUIRED_BINDINGS - 1,
                ..sufficient
            }
            .supports(geometry, tile)
        );
    }

    #[test]
    fn scalar_materialized_and_epilogue_snake_are_bit_exact() {
        let input = deterministic_values(TEST_CHANNELS * TEST_LENGTH, 29, 11, 1.0 / 16.0);
        let weight = deterministic_values(
            TEST_CHANNELS * TEST_CHANNELS * KERNEL_SIZE,
            31,
            7,
            1.0 / 256.0,
        );
        let bias = deterministic_values(TEST_CHANNELS, 17, 5, 1.0 / 128.0);
        let alpha = deterministic_values(TEST_CHANNELS, 13, 3, 1.0 / 32.0)
            .into_iter()
            .map(|value| value.abs() + 0.25)
            .collect::<Vec<_>>();

        for dilation in [
            Conv1dK7Dilation::One,
            Conv1dK7Dilation::Three,
            Conv1dK7Dilation::Nine,
        ] {
            let materialized = scalar_convolution(&input, &weight, &bias, dilation);
            let baseline = materialized
                .iter()
                .enumerate()
                .map(|(index, &x)| scalar_snake(x, alpha[index / TEST_LENGTH]))
                .collect::<Vec<_>>();
            let epilogue = scalar_convolution(&input, &weight, &bias, dilation)
                .into_iter()
                .enumerate()
                .map(|(index, x)| scalar_snake(x, alpha[index / TEST_LENGTH]))
                .collect::<Vec<_>>();
            assert_eq!(
                baseline
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                epilogue
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "dilation {} changed f32 operation order",
                dilation.value()
            );
        }
    }

    #[test]
    fn candidate_preserves_accepted_convolution_and_snake_source_contracts() {
        let candidate = include_str!("conv1d_k7_snake_epilogue.wgsl");
        let accepted_o16 = include_str!("conv1d_k7_tiled.wgsl");
        let accepted_o32 = include_str!("conv1d_k7_tiled_o32.wgsl");
        assert_eq!(
            normalized_main_prefix(candidate),
            normalized_main_prefix(accepted_o16),
            "candidate convolution body drifted from accepted O16"
        );
        assert_eq!(
            normalized_main_prefix(candidate),
            normalized_main_prefix(accepted_o32),
            "candidate convolution body drifted from accepted O32"
        );

        for line in [
            "let a = alpha_buf[output_channel];",
            "let sine = sin(a * x);",
            "return x + (sine * sine) / (a + 1e-9);",
        ] {
            assert!(candidate.contains(line), "missing exact Snake line: {line}");
        }
        for (accumulator, output_channel) in [
            ("00", "0"),
            ("01", "1"),
            ("10", "0"),
            ("11", "1"),
            ("20", "0"),
            ("21", "1"),
            ("30", "0"),
            ("31", "1"),
        ] {
            let call = format!(
                "snake_epilogue(accumulator_{accumulator}, output_channel_{output_channel})"
            );
            assert!(candidate.contains(&call), "missing epilogue call {call}");
        }
        let storage_bindings = candidate
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("@group(0)") && line.contains("var<storage"))
            .collect::<Vec<_>>();
        assert_eq!(storage_bindings.len(), REQUIRED_BINDINGS as usize);
        assert!(
            storage_bindings
                .iter()
                .all(|line| line.contains("var<storage, read_write>")),
            "SourceKernel storage bindings must use uniform read_write access"
        );
    }
}
