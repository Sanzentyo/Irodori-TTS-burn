//! Production T256/O32 DACVAE k=7 Conv1d + act1 Snake epilogue.
//!
//! The conservative selector promotes nine measured released decoder shapes
//! over the accepted T128+Snake path. Callers must retain that complete T128
//! chain whenever this module's exact five-buffer contract is absent.

use super::conv1d_k7_tiled::Conv1dK7Dilation;
use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const BATCH: usize = 1;
const KERNEL_SIZE: usize = 7;
const REQUIRED_BINDINGS: u32 = 5;
const LOCAL_TIME_LANES: usize = 16;
const LOCAL_CHANNEL_LANES: usize = 16;
/// Time positions produced by one T256 workgroup.
pub const TIME_TILE: usize = 256;
/// Output channels produced by one T256 workgroup.
pub const OUTPUT_CHANNEL_TILE: usize = 32;
/// Invocations in the fixed 16x16 workgroup.
pub const WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
/// Logical f32 accumulators owned by one invocation (eight vec4 values).
pub const ACCUMULATORS_PER_INVOCATION: usize = 32;

/// Input-channel reduction tile selected per measured production shape.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Conv1dK7T256Tile {
    /// T256/O32/Cin16. Half the Cin8 barriers, but 30.4--33.4 KiB shared.
    Cin16,
    /// T256/O32/Cin8. 15.2--16.7 KiB shared, but twice the Cin16 barriers.
    Cin8,
}

impl Conv1dK7T256Tile {
    /// Stable production and benchmark label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cin16 => "t256-o32-c16-v8t",
            Self::Cin8 => "t256-o32-c8-v8t",
        }
    }

    /// Input channels reduced between pairs of workgroup barriers.
    pub const fn input_channel_tile(self) -> usize {
        match self {
            Self::Cin16 => 16,
            Self::Cin8 => 8,
        }
    }

    /// Workgroup storage required for a supported dilation.
    pub const fn shared_memory_bytes(self, dilation: Conv1dK7Dilation) -> usize {
        let input_channels = self.input_channel_tile();
        let input_span = TIME_TILE + 6 * dilation.value();
        let input_elements = input_channels * input_span;
        let weight_elements = OUTPUT_CHANNEL_TILE * input_channels * KERNEL_SIZE;
        (input_elements + weight_elements) * core::mem::size_of::<f32>()
    }
}

/// Select the conservative T256+Snake tile for an exact released shape.
///
/// The final rotating one-shot was bit-exact for both tiles on all 12 shapes.
/// Its fixed T128 baseline summed to 37.432 ms while the per-shape selected
/// minimum summed to 36.386 ms. Three close or losing shapes deliberately stay
/// on the established T128+Snake selector.
pub const fn production_tile_for_shape(
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
) -> Option<Conv1dK7T256Tile> {
    match (channels, length, dilation) {
        (768, 600, Conv1dK7Dilation::One | Conv1dK7Dilation::Nine)
        | (384, 6_000, Conv1dK7Dilation::One)
        | (192, 48_000, Conv1dK7Dilation::One | Conv1dK7Dilation::Three)
        | (96, 96_000, Conv1dK7Dilation::One | Conv1dK7Dilation::Three) => {
            Some(Conv1dK7T256Tile::Cin16)
        }
        (192, 48_000, Conv1dK7Dilation::Nine) | (96, 96_000, Conv1dK7Dilation::Nine) => {
            Some(Conv1dK7T256Tile::Cin8)
        }
        _ => None,
    }
}

/// Checked T256 launch geometry used by production preflight and accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchGeometry {
    /// Input time span staged for each reduction tile.
    pub input_span: usize,
    /// Scalar input elements in workgroup storage.
    pub input_tile_size: usize,
    /// Scalar weight elements in workgroup storage.
    pub weight_tile_size: usize,
    /// Total workgroup storage in bytes.
    pub shared_bytes: usize,
    /// Dispatch count along time.
    pub time_tiles: u32,
    /// Dispatch count along output channels.
    pub output_channel_tiles: u32,
}

impl LaunchGeometry {
    /// Build geometry for a T256 shape, returning `None` on every unsupported
    /// shape or unrepresentable index calculation.
    pub fn new(
        channels: usize,
        length: usize,
        dilation: Conv1dK7Dilation,
        tile: Conv1dK7T256Tile,
    ) -> Option<Self> {
        let input_channel_tile = tile.input_channel_tile();
        if channels == 0
            || length == 0
            || !channels.is_multiple_of(OUTPUT_CHANNEL_TILE)
            || !channels.is_multiple_of(input_channel_tile)
        {
            return None;
        }

        let halo = 6usize.checked_mul(dilation.value())?;
        let input_span = TIME_TILE.checked_add(halo)?;
        let input_tile_size = input_channel_tile.checked_mul(input_span)?;
        let weight_tile_size = OUTPUT_CHANNEL_TILE
            .checked_mul(input_channel_tile)?
            .checked_mul(KERNEL_SIZE)?;
        let shared_bytes = input_tile_size
            .checked_add(weight_tile_size)?
            .checked_mul(core::mem::size_of::<f32>())?;
        let time_tiles_usize = length.div_ceil(TIME_TILE);
        let output_channel_tiles_usize = channels / OUTPUT_CHANNEL_TILE;
        let time_tiles = u32::try_from(time_tiles_usize).ok()?;
        let output_channel_tiles = u32::try_from(output_channel_tiles_usize).ok()?;

        let input_elements = channels.checked_mul(length)?;
        let weight_elements = channels.checked_mul(channels)?.checked_mul(KERNEL_SIZE)?;
        let final_time_base = time_tiles_usize.checked_sub(1)?.checked_mul(TIME_TILE)?;
        let largest_staged_time = final_time_base.checked_add(input_span - 1)?;
        let largest_output_time = final_time_base.checked_add(TIME_TILE - 1)?;
        for value in [
            channels,
            length,
            input_elements,
            weight_elements,
            input_span,
            input_tile_size,
            weight_tile_size,
            largest_output_time,
        ] {
            u32::try_from(value).ok()?;
        }
        i32::try_from(largest_staged_time).ok()?;
        i32::try_from(3usize.checked_mul(dilation.value())?).ok()?;

        Some(Self {
            input_span,
            input_tile_size,
            weight_tile_size,
            shared_bytes,
            time_tiles,
            output_channel_tiles,
        })
    }

    /// Total dispatched workgroups for B=1.
    pub fn workgroups(self) -> Option<usize> {
        usize::try_from(self.time_tiles)
            .ok()?
            .checked_mul(usize::try_from(self.output_channel_tiles).ok()?)
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

#[derive(Debug)]
struct Conv1dK7T256SnakeEpilogueKernel {
    tile: Conv1dK7T256Tile,
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
}

impl KernelSource for Conv1dK7T256SnakeEpilogueKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_t256_snake_epilogue.wgsl"))
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.to_string())
            .register("padding", self.padding.to_string())
            .register(
                "input_channel_tile",
                self.tile.input_channel_tile().to_string(),
            )
            .register("input_span", self.input_span.to_string())
            .register("input_tile_size", self.input_tile_size.to_string())
            .register("weight_tile_size", self.weight_tile_size.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.tile,
            self.channels,
            self.length,
            self.dilation,
            self.padding,
            self.input_span,
            self.input_tile_size,
            self.weight_tile_size,
        ))
    }
}

/// Validate the complete production fused launch without allocating or
/// dispatching work.
pub fn conv1d_k7_t256_snake_epilogue_contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T256Tile,
) -> bool {
    if input.meta.num_dims() != 3
        || weight.meta.num_dims() != 3
        || bias.meta.num_dims() != 1
        || alpha.meta.num_dims() != 3
    {
        return false;
    }

    let input_shape = input.meta.shape();
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];
    let weight_shape = weight.meta.shape();
    let alpha_shape = alpha.meta.shape();
    let Some(geometry) = LaunchGeometry::new(channels, length, dilation, tile) else {
        return false;
    };
    let logical_contract = batch == BATCH
        && [weight_shape[0], weight_shape[1], weight_shape[2]] == [channels, channels, KERNEL_SIZE]
        && bias.meta.shape()[0] == channels
        && [alpha_shape[0], alpha_shape[1], alpha_shape[2]] == [BATCH, channels, 1]
        && [input, weight, bias, alpha]
            .into_iter()
            .all(|tensor| tensor.dtype == DType::F32 && tensor.device == input.device)
        && input.is_contiguous()
        && weight.is_contiguous()
        && bias.is_contiguous()
        && alpha.is_contiguous();
    if !logical_contract {
        return false;
    }

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

/// Apply the production Snake scalar expression in the unchanged T256 output
/// store epilogue.
///
/// Call [`conv1d_k7_t256_snake_epilogue_contract_is_compatible`] before this
/// asserting production launcher and retain the established T128 fallback.
///
/// # Panics
///
/// Panics on any incompatible shape, dtype, layout, device, indexing, shared
/// memory, workgroup, dispatch, or five-binding contract.
pub fn conv1d_k7_same_t256_snake_epilogue_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T256Tile,
) -> CubeTensor<WgpuRuntime> {
    assert!(
        conv1d_k7_t256_snake_epilogue_contract_is_compatible(
            &input, &weight, &bias, &alpha, dilation, tile,
        ),
        "{} + Snake requires compatible B1/f32/contiguous NCL+OIK+alpha shape, device, and five-binding resources",
        tile.label(),
    );

    let input_shape = input.meta.shape();
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];
    let geometry = LaunchGeometry::new(channels, length, dilation, tile)
        .expect("validated T256 + Snake geometry must remain representable");
    let output_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("validated output element count must not overflow");
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .expect("validated output byte count must not overflow");
    let client = input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, channels, length]),
        client.empty(output_bytes),
        DType::F32,
    );

    let dilation_value = dilation.value();
    let kernel = Conv1dK7T256SnakeEpilogueKernel {
        tile,
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

    fn normalized_main_prefix(shader: &str) -> String {
        shader
            .split_once("fn main(")
            .expect("shader must define main")
            .1
            .split_once("    let output_base_0")
            .expect("shader must compute output_base_0")
            .0
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn shader_preserves_the_measured_t256_convolution_order() {
        let fused = include_str!("conv1d_k7_t256_snake_epilogue.wgsl");
        let body = normalized_main_prefix(fused);
        assert_eq!(
            body.lines()
                .filter(|line| line.contains("accumulator_") && line.contains(" = fma("))
                .count(),
            KERNEL_SIZE * 8,
        );
        assert_eq!(fused.matches("// tap ").count(), KERNEL_SIZE);
        assert_eq!(fused.matches("var input_vector").count(), 1);
        assert!(body.contains("accumulator_00 = fma(input_vector, vec4<f32>(weight_0)"));
        assert!(body.contains("accumulator_31 = fma(input_vector, vec4<f32>(weight_1)"));
        assert!(!body.contains("dot("));
        assert!(!body.contains("subgroup"));
    }

    #[test]
    fn fused_shader_has_exact_five_bindings_and_production_snake_source() {
        let fused = include_str!("conv1d_k7_t256_snake_epilogue.wgsl");
        let production_snake = include_str!("snake.wgsl");
        let expected_bindings = [
            "@group(0) @binding(0) var<storage, read_write> input_buf:",
            "@group(0) @binding(1) var<storage, read_write> weight_buf:",
            "@group(0) @binding(2) var<storage, read_write> bias_buf:",
            "@group(0) @binding(3) var<storage, read_write> output_buf:",
            "@group(0) @binding(4) var<storage, read_write> alpha_buf:",
        ];
        assert_eq!(REQUIRED_BINDINGS, expected_bindings.len() as u32);
        for binding in expected_bindings {
            assert!(fused.contains(binding), "missing binding: {binding}");
        }
        assert_eq!(
            fused
                .lines()
                .filter(|line| line.trim().starts_with("@group(0)") && line.contains("var<storage"))
                .count(),
            REQUIRED_BINDINGS as usize
        );
        for line in [
            "let a = alpha_buf[output_channel];",
            "let sine = sin(a * x);",
            "return x + (sine * sine) / (a + 1e-9);",
        ] {
            assert!(fused.contains(line), "missing fused Snake line: {line}");
        }
        for line in [
            "let sine = sin(a * x);",
            "output[index] = x + (sine * sine) / (a + 1e-9);",
        ] {
            assert!(
                production_snake.contains(line),
                "production Snake source drifted: {line}"
            );
        }
        for component in ['x', 'y', 'z', 'w'] {
            assert!(fused.contains(&format!(
                "snake_epilogue(value.{component}, output_channel)"
            )));
        }
        assert_eq!(fused.matches("store_output_vec4(output_base_").count(), 8);
    }

    #[test]
    fn exact_official_geometry_and_shared_bytes_are_fixed() {
        let dilations = [
            Conv1dK7Dilation::One,
            Conv1dK7Dilation::Three,
            Conv1dK7Dilation::Nine,
        ];
        let expected_workgroups = [
            (768, 600, 72),
            (384, 6_000, 288),
            (192, 48_000, 1_128),
            (96, 96_000, 1_125),
        ];
        for tile in [Conv1dK7T256Tile::Cin16, Conv1dK7T256Tile::Cin8] {
            for (channels, length, workgroups) in expected_workgroups {
                for dilation in dilations {
                    let geometry = LaunchGeometry::new(channels, length, dilation, tile)
                        .expect("released T256 geometry must remain valid");
                    assert_eq!(geometry.workgroups(), Some(workgroups));
                    assert_eq!(geometry.shared_bytes, tile.shared_memory_bytes(dilation));
                }
            }
        }
        assert_eq!(
            dilations.map(|dilation| Conv1dK7T256Tile::Cin16.shared_memory_bytes(dilation)),
            [31_104, 31_872, 34_176],
        );
        assert_eq!(
            dilations.map(|dilation| Conv1dK7T256Tile::Cin8.shared_memory_bytes(dilation)),
            [15_552, 15_936, 17_088],
        );
        assert!(Conv1dK7T256Tile::Cin16.shared_memory_bytes(Conv1dK7Dilation::Nine) <= 49_152);
        assert_eq!(
            (TIME_TILE, OUTPUT_CHANNEL_TILE, WORKGROUP_SIZE),
            (256, 32, 256)
        );
        assert_eq!(ACCUMULATORS_PER_INVOCATION, 32);
    }

    #[test]
    fn conservative_production_selector_covers_exactly_nine_shapes() {
        let cases = [
            (
                768,
                600,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (768, 600, Conv1dK7Dilation::Three, None),
            (
                768,
                600,
                Conv1dK7Dilation::Nine,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                384,
                6_000,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (384, 6_000, Conv1dK7Dilation::Three, None),
            (384, 6_000, Conv1dK7Dilation::Nine, None),
            (
                192,
                48_000,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                192,
                48_000,
                Conv1dK7Dilation::Three,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                192,
                48_000,
                Conv1dK7Dilation::Nine,
                Some(Conv1dK7T256Tile::Cin8),
            ),
            (
                96,
                96_000,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                96,
                96_000,
                Conv1dK7Dilation::Three,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                96,
                96_000,
                Conv1dK7Dilation::Nine,
                Some(Conv1dK7T256Tile::Cin8),
            ),
        ];
        for (channels, length, dilation, expected) in cases {
            assert_eq!(
                production_tile_for_shape(channels, length, dilation),
                expected
            );
        }
        assert_eq!(
            cases
                .into_iter()
                .filter(|(channels, length, dilation, _)| {
                    production_tile_for_shape(*channels, *length, *dilation).is_some()
                })
                .count(),
            9,
        );
        assert_eq!(
            production_tile_for_shape(96, 95_999, Conv1dK7Dilation::Nine),
            None,
        );
    }

    #[test]
    fn geometry_and_device_limits_fail_closed() {
        assert!(
            LaunchGeometry::new(96, 96_000, Conv1dK7Dilation::One, Conv1dK7T256Tile::Cin8,)
                .is_some(),
        );
        for (channels, length) in [(0, 1), (31, 1), (96, 0)] {
            assert!(
                LaunchGeometry::new(
                    channels,
                    length,
                    Conv1dK7Dilation::One,
                    Conv1dK7T256Tile::Cin8,
                )
                .is_none(),
            );
        }

        let geometry =
            LaunchGeometry::new(96, 96_000, Conv1dK7Dilation::Nine, Conv1dK7T256Tile::Cin16)
                .expect("released shape must have valid T256 geometry");
        let sufficient = DeviceLimits {
            max_bindings: REQUIRED_BINDINGS,
            max_shared_memory_size: geometry.shared_bytes,
            max_cube_count: (geometry.time_tiles, geometry.output_channel_tiles, 1),
            max_units_per_cube: WORKGROUP_SIZE as u32,
            max_cube_dim: (LOCAL_TIME_LANES as u32, LOCAL_CHANNEL_LANES as u32, 1),
        };
        assert!(sufficient.supports(geometry));
        let insufficient = [
            DeviceLimits {
                max_bindings: REQUIRED_BINDINGS - 1,
                ..sufficient
            },
            DeviceLimits {
                max_shared_memory_size: geometry.shared_bytes - 1,
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (geometry.time_tiles - 1, geometry.output_channel_tiles, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (geometry.time_tiles, geometry.output_channel_tiles - 1, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (geometry.time_tiles, geometry.output_channel_tiles, 0),
                ..sufficient
            },
            DeviceLimits {
                max_units_per_cube: WORKGROUP_SIZE as u32 - 1,
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (LOCAL_TIME_LANES as u32 - 1, LOCAL_CHANNEL_LANES as u32, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (LOCAL_TIME_LANES as u32, LOCAL_CHANNEL_LANES as u32 - 1, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (LOCAL_TIME_LANES as u32, LOCAL_CHANNEL_LANES as u32, 0),
                ..sufficient
            },
        ];
        assert!(
            insufficient
                .into_iter()
                .all(|limits| !limits.supports(geometry)),
        );
    }

    #[test]
    fn source_template_is_complete_for_both_tiles() {
        for tile in [Conv1dK7T256Tile::Cin8, Conv1dK7T256Tile::Cin16] {
            let geometry = LaunchGeometry::new(96, 96_000, Conv1dK7Dilation::Nine, tile)
                .expect("released geometry");
            let source = Conv1dK7T256SnakeEpilogueKernel {
                tile,
                channels: 96,
                length: 96_000,
                dilation: 9,
                padding: 27,
                input_span: geometry.input_span as u32,
                input_tile_size: geometry.input_tile_size as u32,
                weight_tile_size: geometry.weight_tile_size as u32,
            }
            .source()
            .complete();
            assert!(!source.contains("{{"));
            assert!(source.contains(&format!(
                "const INPUT_CHANNEL_TILE: u32 = {}u;",
                tile.input_channel_tile(),
            )));
        }
    }
}
