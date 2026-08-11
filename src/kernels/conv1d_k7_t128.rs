//! Production T128/O32 DACVAE k=7 Conv1d tiles.
//!
//! The measured selector uses Cin16 or Cin8 per released decoder shape. Both
//! tiles retain checkpoint-native contiguous NCL/OIK layouts and the accepted
//! f32 accumulation order. Callers must preserve the prior O64/O32/O16 route
//! as a fail-closed fallback whenever this module's exact contract is absent.

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
const REQUIRED_BINDINGS: u32 = 4;
const LOCAL_TIME_LANES: usize = 16;
const LOCAL_CHANNEL_LANES: usize = 16;
/// Time positions produced by one T128 workgroup.
pub const TIME_TILE: usize = 128;
/// Output channels produced by one T128 workgroup.
pub const OUTPUT_CHANNEL_TILE: usize = 32;
/// Invocations in the fixed 16x16 workgroup.
pub const WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
/// Logical f32 accumulators owned by one invocation (four vec4 values).
pub const ACCUMULATORS_PER_INVOCATION: usize = 16;

/// Input-channel reduction tile selected per measured production shape.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Conv1dK7T128Tile {
    /// T128/O32/Cin16. Fewer barrier encounters, but 22.4--25.4 KiB shared.
    Cin16,
    /// T128/O32/Cin8. Portable shared allocation, but twice as many barriers.
    Cin8,
}

impl Conv1dK7T128Tile {
    /// Stable benchmark label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cin16 => "t128-o32-c16-v4t",
            Self::Cin8 => "t128-o32-c8-v4t",
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

/// Select the measured T128 tile for an exact released decoder shape.
///
/// The same RTX 3060 Ti run measured current/Cin16/Cin8 medians (µs):
/// C768 d1 2202.362/1064.579/1111.188, d3 2198.749/1613.195/1630.933,
/// d9 2212.349/2085.852/1971.204; C384 d1 3494.116/1994.153/2011.507,
/// d3 3478.036/2864.728/2859.340, d9 4905.554/3660.924/3450.642;
/// C192 d1 6771.664/4057.195/4007.558, d3 6767.053/5695.509/5677.795,
/// d9 9764.613/7351.391/6766.262; C96 d1 4376.802/2045.700/2051.904,
/// d3 4349.462/2861.484/2893.990, d9 4956.975/3686.087/3435.230.
/// C384/d3 retains Cin16 because its 5.4 µs difference is a tie. This
/// conservative selection totals 37.753 ms versus 55.478 ms (1.470x).
pub const fn production_tile_for_shape(
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
) -> Option<Conv1dK7T128Tile> {
    match (channels, length, dilation) {
        (768, 600, Conv1dK7Dilation::One | Conv1dK7Dilation::Three)
        | (384, 6_000, Conv1dK7Dilation::One | Conv1dK7Dilation::Three)
        | (96, 96_000, Conv1dK7Dilation::One | Conv1dK7Dilation::Three) => {
            Some(Conv1dK7T128Tile::Cin16)
        }
        (768, 600, Conv1dK7Dilation::Nine)
        | (384, 6_000, Conv1dK7Dilation::Nine)
        | (192, 48_000, Conv1dK7Dilation::One | Conv1dK7Dilation::Three | Conv1dK7Dilation::Nine)
        | (96, 96_000, Conv1dK7Dilation::Nine) => Some(Conv1dK7T128Tile::Cin8),
        _ => None,
    }
}

/// Checked launch geometry used by the benchmark's static accounting.
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
    /// Build geometry for an exact T128 shape, returning `None` on every
    /// unsupported shape or unrepresentable index calculation.
    pub fn new(
        channels: usize,
        length: usize,
        dilation: Conv1dK7Dilation,
        tile: Conv1dK7T128Tile,
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
struct Conv1dK7T128Kernel {
    tile: Conv1dK7T128Tile,
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
}

impl KernelSource for Conv1dK7T128Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_t128.wgsl"))
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

/// Validate the complete production launch contract without allocating or
/// dispatching work.
pub fn conv1d_k7_t128_contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T128Tile,
) -> bool {
    if input.meta.num_dims() != 3 || weight.meta.num_dims() != 3 || bias.meta.num_dims() != 1 {
        return false;
    }
    let input_shape = input.meta.shape();
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];
    let weight_shape = weight.meta.shape();
    let Some(geometry) = LaunchGeometry::new(channels, length, dilation, tile) else {
        return false;
    };
    let logical_contract = batch == BATCH
        && [weight_shape[0], weight_shape[1], weight_shape[2]] == [channels, channels, KERNEL_SIZE]
        && bias.meta.shape()[0] == channels
        && [input, weight, bias]
            .into_iter()
            .all(|tensor| tensor.dtype == DType::F32 && tensor.device == input.device)
        && input.is_contiguous()
        && weight.is_contiguous()
        && bias.is_contiguous();
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

/// Compute exact-order f32 same-length Conv1d with a T128/O32 tile.
///
/// Required physical layouts are contiguous input `[1, C, L]`, contiguous OIK
/// weight `[C, C, 7]`, and contiguous bias `[C]`. Production callers must call
/// [`conv1d_k7_t128_contract_is_compatible`] and retain a portable fallback.
///
/// # Panics
///
/// Panics for any dtype, rank, shape, device, layout, indexing, binding,
/// dispatch, workgroup, or shared-memory contract mismatch.
pub fn conv1d_k7_same_t128_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T128Tile,
) -> CubeTensor<WgpuRuntime> {
    assert!(
        conv1d_k7_t128_contract_is_compatible(&input, &weight, &bias, dilation, tile),
        "{} requires compatible B1/F32/contiguous NCL+OIK shape, device, and resources",
        tile.label(),
    );

    let input_shape = input.meta.shape();
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];
    let geometry = LaunchGeometry::new(channels, length, dilation, tile)
        .expect("validated T128 geometry must remain representable");
    let input_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("validated output element count must not overflow");
    let output_bytes = input_elements
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
    let kernel = Conv1dK7T128Kernel {
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

    const DILATIONS: [Conv1dK7Dilation; 3] = [
        Conv1dK7Dilation::One,
        Conv1dK7Dilation::Three,
        Conv1dK7Dilation::Nine,
    ];

    #[test]
    fn exact_official_geometry_and_shared_bytes() {
        let expected_workgroups = [
            (768, 600, 120),
            (384, 6_000, 564),
            (192, 48_000, 2_250),
            (96, 96_000, 2_250),
        ];
        for tile in [Conv1dK7T128Tile::Cin16, Conv1dK7T128Tile::Cin8] {
            for (channels, length, workgroups) in expected_workgroups {
                for dilation in DILATIONS {
                    let geometry = LaunchGeometry::new(channels, length, dilation, tile)
                        .expect("official shape must be representable");
                    assert_eq!(geometry.workgroups(), Some(workgroups));
                    assert_eq!(geometry.shared_bytes, tile.shared_memory_bytes(dilation));
                }
            }
        }
        assert_eq!(
            DILATIONS.map(|dilation| Conv1dK7T128Tile::Cin16.shared_memory_bytes(dilation)),
            [22_912, 23_680, 25_984]
        );
        assert_eq!(
            DILATIONS.map(|dilation| Conv1dK7T128Tile::Cin8.shared_memory_bytes(dilation)),
            [11_456, 11_840, 12_992]
        );
    }

    #[test]
    fn released_shape_selector_matches_the_measured_tie_policy() {
        let cases = [
            (768, 600, Conv1dK7Dilation::One, Conv1dK7T128Tile::Cin16),
            (768, 600, Conv1dK7Dilation::Three, Conv1dK7T128Tile::Cin16),
            (768, 600, Conv1dK7Dilation::Nine, Conv1dK7T128Tile::Cin8),
            (384, 6_000, Conv1dK7Dilation::One, Conv1dK7T128Tile::Cin16),
            (384, 6_000, Conv1dK7Dilation::Three, Conv1dK7T128Tile::Cin16),
            (384, 6_000, Conv1dK7Dilation::Nine, Conv1dK7T128Tile::Cin8),
            (192, 48_000, Conv1dK7Dilation::One, Conv1dK7T128Tile::Cin8),
            (192, 48_000, Conv1dK7Dilation::Three, Conv1dK7T128Tile::Cin8),
            (192, 48_000, Conv1dK7Dilation::Nine, Conv1dK7T128Tile::Cin8),
            (96, 96_000, Conv1dK7Dilation::One, Conv1dK7T128Tile::Cin16),
            (96, 96_000, Conv1dK7Dilation::Three, Conv1dK7T128Tile::Cin16),
            (96, 96_000, Conv1dK7Dilation::Nine, Conv1dK7T128Tile::Cin8),
        ];
        for (channels, length, dilation, expected) in cases {
            assert_eq!(
                production_tile_for_shape(channels, length, dilation),
                Some(expected)
            );
        }
        assert_eq!(
            production_tile_for_shape(32, 73, Conv1dK7Dilation::One),
            None
        );
        assert_eq!(
            production_tile_for_shape(96, 95_999, Conv1dK7Dilation::Nine),
            None
        );
    }

    #[test]
    fn geometry_rejects_incompatible_shapes() {
        assert!(
            LaunchGeometry::new(96, 96_000, Conv1dK7Dilation::One, Conv1dK7T128Tile::Cin8,)
                .is_some()
        );
        for (channels, length) in [(0, 1), (31, 1), (96, 0)] {
            assert!(
                LaunchGeometry::new(
                    channels,
                    length,
                    Conv1dK7Dilation::One,
                    Conv1dK7T128Tile::Cin8,
                )
                .is_none()
            );
        }
    }

    #[test]
    fn device_limits_fail_closed_for_every_launch_resource() {
        let geometry =
            LaunchGeometry::new(96, 96_000, Conv1dK7Dilation::Nine, Conv1dK7T128Tile::Cin16)
                .expect("official shape must have valid T128 geometry");
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
                .all(|limits| !limits.supports(geometry))
        );
    }

    #[test]
    fn shader_contract_is_vector_componentwise_and_uniform_read_write() {
        let shader = include_str!("conv1d_k7_t128.wgsl");
        for binding in 0..4 {
            assert!(shader.contains(&format!(
                "@group(0) @binding({binding}) var<storage, read_write>"
            )));
        }
        assert_eq!(shader.matches("// tap ").count(), KERNEL_SIZE);
        assert!(shader.contains("accumulator_00 = fma(input_0, vec4<f32>(weight_0)"));
        assert!(!shader.contains("dot("));
        assert!(!shader.contains("subgroup"));
    }
}
