//! Production residue-class kernel for the two long dilated DACVAE k=7 calls.
//!
//! The accepted rotating A/B was bit-exact over 18,432,000 outputs and reduced
//! the exact d3+d9 median sum from 11.693 ms to 7.914 ms (3.779 ms, 1.477x).
//! It maps exact `[1, 192, 48_000]` NCL input into compact
//! `[residue][channel][q]` storage, then evaluates each residue as an ordinary
//! dilation-one, same-padding k=7 convolution. The core retains the production
//! input-channel-then-tap FMA order, bias initialization, and scalar Snake
//! expression before scattering directly to `t = residue + q * d`.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::conv1d_k7_tiled::Conv1dK7Dilation;

const BATCH: usize = 1;
const CHANNELS: usize = 192;
const LENGTH: usize = 48_000;
const KERNEL_SIZE: usize = 7;
const PADDING_D1: usize = 3;
const INPUT_CHANNEL_TILE: usize = 16;
const OUTPUT_CHANNEL_TILE: usize = 32;
const TIME_TILE: usize = 256;
const LOCAL_TIME_LANES: usize = 16;
const LOCAL_CHANNEL_LANES: usize = 16;
const CORE_WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
const PACK_WORKGROUP_SIZE: usize = 256;
const PACKED_ELEMENTS: usize = BATCH * CHANNELS * LENGTH;
const F32_BYTES: usize = size_of::<f32>();
const PACKED_BYTES: usize = PACKED_ELEMENTS * F32_BYTES;
const INPUT_SPAN_D1: usize = TIME_TILE + 2 * PADDING_D1;
const INPUT_TILE_SIZE: usize = INPUT_CHANNEL_TILE * INPUT_SPAN_D1;
const WEIGHT_TILE_SIZE: usize = OUTPUT_CHANNEL_TILE * INPUT_CHANNEL_TILE * KERNEL_SIZE;
const SHARED_BYTES: usize = (INPUT_TILE_SIZE + WEIGHT_TILE_SIZE) * F32_BYTES;
const PACK_BINDINGS: u32 = 2;
const CORE_BINDINGS: u32 = 5;

/// The only two dilations admitted by this production kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum ResidueDilation {
    /// Decoder block 2, residual unit 1.
    Three = 3,
    /// Decoder block 2, residual unit 2.
    Nine = 9,
}

impl ResidueDilation {
    /// Numeric dilation used by WGSL indexing.
    pub const fn value(self) -> usize {
        self as usize
    }

    /// Stable benchmark label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Three => "residue-d1-d3",
            Self::Nine => "residue-d1-d9",
        }
    }

    /// Floor of the original length divided by the residue count.
    pub const fn base_length(self) -> usize {
        LENGTH / self.value()
    }

    /// Number of leading residues that contain one extra element.
    pub const fn remainder(self) -> usize {
        LENGTH % self.value()
    }

    /// Longest residue sequence, used for the rectangular core dispatch.
    pub const fn max_residue_length(self) -> usize {
        self.base_length() + if self.remainder() == 0 { 0 } else { 1 }
    }

    /// Exact logical length of one compact residue sequence.
    pub const fn residue_length(self, residue: usize) -> Option<usize> {
        if residue >= self.value() {
            return None;
        }
        Some(self.base_length() + if residue < self.remainder() { 1 } else { 0 })
    }

    /// Number of time positions stored before one residue block.
    pub const fn residue_prefix_q(self, residue: usize) -> Option<usize> {
        if residue >= self.value() {
            return None;
        }
        let extra = if residue < self.remainder() {
            residue
        } else {
            self.remainder()
        };
        Some(residue * self.base_length() + extra)
    }

    /// Compact `[residue][channel][q]` index for an original NCL element.
    pub const fn packed_index(self, channel: usize, time: usize) -> Option<usize> {
        if channel >= CHANNELS || time >= LENGTH {
            return None;
        }
        let residue = time % self.value();
        let q = time / self.value();
        let residue_length = match self.residue_length(residue) {
            Some(value) => value,
            None => return None,
        };
        let residue_prefix_q = match self.residue_prefix_q(residue) {
            Some(value) => value,
            None => return None,
        };
        Some(residue_prefix_q * CHANNELS + channel * residue_length + q)
    }
}

/// Exact static launch and temporary-storage accounting for one production call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidueLaunchGeometry {
    /// Selected exact decoder dilation.
    pub dilation: ResidueDilation,
    /// Compact pack elements; equal to the source NCL element count.
    pub packed_elements: usize,
    /// Compact temporary bytes retained between the two dispatches.
    pub temporary_bytes: usize,
    /// Logical source read plus compact destination write bytes for the pack.
    pub pack_read_write_bytes: usize,
    /// Workgroups in the one-dimensional pack dispatch.
    pub pack_workgroups: u32,
    /// Time tiles along the longest residue.
    pub core_time_tiles: u32,
    /// Output-channel tiles in the core dispatch.
    pub core_output_channel_tiles: u32,
    /// Residue count in the core dispatch Z dimension.
    pub core_residues: u32,
    /// Total workgroups in the core dispatch.
    pub core_workgroups: usize,
    /// Workgroup barriers in the complete core dispatch.
    pub core_barriers: usize,
    /// Fixed d1/Cin16 workgroup storage.
    pub core_shared_bytes: usize,
    /// Pack plus core dispatches.
    pub dispatches: usize,
}

impl ResidueLaunchGeometry {
    /// Construct the infallible geometry for one admitted exact shape.
    pub const fn new(dilation: ResidueDilation) -> Self {
        let max_residue_length = dilation.max_residue_length();
        let core_time_tiles = max_residue_length.div_ceil(TIME_TILE);
        let core_output_channel_tiles = CHANNELS / OUTPUT_CHANNEL_TILE;
        let core_workgroups = core_time_tiles * core_output_channel_tiles * dilation.value();
        let barriers_per_workgroup = 2 * (CHANNELS / INPUT_CHANNEL_TILE);
        Self {
            dilation,
            packed_elements: PACKED_ELEMENTS,
            temporary_bytes: PACKED_BYTES,
            pack_read_write_bytes: 2 * PACKED_BYTES,
            pack_workgroups: PACKED_ELEMENTS.div_ceil(PACK_WORKGROUP_SIZE) as u32,
            core_time_tiles: core_time_tiles as u32,
            core_output_channel_tiles: core_output_channel_tiles as u32,
            core_residues: dilation.value() as u32,
            core_workgroups,
            core_barriers: core_workgroups * barriers_per_workgroup,
            core_shared_bytes: SHARED_BYTES,
            dispatches: 2,
        }
    }
}

#[derive(Debug)]
struct ResiduePackKernel {
    dilation: ResidueDilation,
}

impl KernelSource for ResiduePackKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_residue_pack.wgsl"))
            .register("channels", CHANNELS.to_string())
            .register("length", LENGTH.to_string())
            .register("elements", PACKED_ELEMENTS.to_string())
            .register("dilation", self.dilation.value().to_string())
            .register("base_length", self.dilation.base_length().to_string())
            .register("remainder", self.dilation.remainder().to_string())
            .register("workgroup_size", PACK_WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.dilation)
    }
}

#[derive(Debug)]
struct ResidueD1SnakeCoreKernel {
    dilation: ResidueDilation,
}

impl KernelSource for ResidueD1SnakeCoreKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_residue_d1_snake.wgsl"))
            .register("channels", CHANNELS.to_string())
            .register("length", LENGTH.to_string())
            .register("dilation", self.dilation.value().to_string())
            .register("base_length", self.dilation.base_length().to_string())
            .register("remainder", self.dilation.remainder().to_string())
            .register("input_span", INPUT_SPAN_D1.to_string())
            .register("input_tile_size", INPUT_TILE_SIZE.to_string())
            .register("weight_tile_size", WEIGHT_TILE_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.dilation)
    }
}

fn exact_shape<const D: usize>(tensor: &CubeTensor<WgpuRuntime>, expected: [usize; D]) -> bool {
    tensor.meta.num_dims() == D && tensor.meta.shape().dims::<D>() == expected
}

fn exact_input_contract(input: &CubeTensor<WgpuRuntime>) -> bool {
    exact_shape(input, [BATCH, CHANNELS, LENGTH])
        && input.dtype == DType::F32
        && input.is_contiguous()
}

fn device_supports_geometry(
    input: &CubeTensor<WgpuRuntime>,
    geometry: ResidueLaunchGeometry,
) -> bool {
    let properties = input.client.properties();
    let hardware = &properties.hardware;
    let Ok(temporary_bytes) = u64::try_from(geometry.temporary_bytes) else {
        return false;
    };
    temporary_bytes <= properties.memory.max_page_size
        && hardware.max_bindings >= PACK_BINDINGS.max(CORE_BINDINGS)
        && hardware.max_shared_memory_size >= geometry.core_shared_bytes
        && hardware.max_units_per_cube >= CORE_WORKGROUP_SIZE.max(PACK_WORKGROUP_SIZE) as u32
        && hardware.max_cube_dim.0 >= PACK_WORKGROUP_SIZE as u32
        && hardware.max_cube_dim.1 >= LOCAL_CHANNEL_LANES as u32
        && hardware.max_cube_dim.2 >= 1
        && hardware.max_cube_count.0 >= geometry.pack_workgroups.max(geometry.core_time_tiles)
        && hardware.max_cube_count.1 >= geometry.core_output_channel_tiles
        && hardware.max_cube_count.2 >= geometry.core_residues
}

fn packed_contract_is_compatible(
    packed: &CubeTensor<WgpuRuntime>,
    reference: &CubeTensor<WgpuRuntime>,
) -> bool {
    exact_shape(packed, [PACKED_ELEMENTS])
        && packed.dtype == DType::F32
        && packed.device == reference.device
        && packed.is_contiguous()
}

/// Select only the two accepted production shapes.
pub const fn production_dilation_for_shape(
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
) -> Option<ResidueDilation> {
    match (channels, length, dilation) {
        (CHANNELS, LENGTH, Conv1dK7Dilation::Three) => Some(ResidueDilation::Three),
        (CHANNELS, LENGTH, Conv1dK7Dilation::Nine) => Some(ResidueDilation::Nine),
        _ => None,
    }
}

/// Validate all exact logical, physical, device, and resource requirements.
pub fn conv1d_k7_residue_d1_snake_contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
    dilation: ResidueDilation,
) -> bool {
    let geometry = ResidueLaunchGeometry::new(dilation);
    exact_input_contract(input)
        && exact_shape(weight, [CHANNELS, CHANNELS, KERNEL_SIZE])
        && exact_shape(bias, [CHANNELS])
        && exact_shape(alpha, [BATCH, CHANNELS, 1])
        && [weight, bias, alpha].into_iter().all(|tensor| {
            tensor.dtype == DType::F32 && tensor.device == input.device && tensor.is_contiguous()
        })
        && device_supports_geometry(input, geometry)
}

/// Launch only the compact residue pack used by the diagnostic timing split.
///
/// The returned one-dimensional tensor has the exact physical order
/// `[residue][channel][q]`, with variable residue lengths and no padding.
pub fn try_pack_conv1d_k7_residue_input_wgsl(
    input: CubeTensor<WgpuRuntime>,
    dilation: ResidueDilation,
) -> Option<CubeTensor<WgpuRuntime>> {
    let geometry = ResidueLaunchGeometry::new(dilation);
    if !exact_input_contract(&input) || !device_supports_geometry(&input, geometry) {
        return None;
    }

    let client = input.client.clone();
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([PACKED_ELEMENTS]),
        client.empty(PACKED_BYTES),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResiduePackKernel { dilation },
            CubeDim::new_1d(PACK_WORKGROUP_SIZE as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(packed.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(geometry.pack_workgroups), bindings);
    Some(packed)
}

/// Launch only the residue-d1 convolution/Snake core from a validated pack.
fn conv1d_k7_residue_d1_snake_from_packed_wgsl(
    packed: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: ResidueDilation,
) -> Option<CubeTensor<WgpuRuntime>> {
    let geometry = ResidueLaunchGeometry::new(dilation);
    if !packed_contract_is_compatible(&packed, &weight)
        || !exact_shape(&weight, [CHANNELS, CHANNELS, KERNEL_SIZE])
        || !exact_shape(&bias, [CHANNELS])
        || !exact_shape(&alpha, [BATCH, CHANNELS, 1])
        || [&weight, &bias, &alpha].into_iter().any(|tensor| {
            tensor.dtype != DType::F32 || tensor.device != packed.device || !tensor.is_contiguous()
        })
        || !device_supports_geometry(&packed, geometry)
    {
        return None;
    }

    let client = packed.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        packed.device.clone(),
        Shape::from([BATCH, CHANNELS, LENGTH]),
        client.empty(PACKED_BYTES),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResidueD1SnakeCoreKernel { dilation },
            CubeDim::new_2d(LOCAL_TIME_LANES as u32, LOCAL_CHANNEL_LANES as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(packed.handle.binding())
        .with_buffer(weight.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding())
        .with_buffer(alpha.handle.binding());
    client.launch(
        task,
        CubeCount::new_3d(
            geometry.core_time_tiles,
            geometry.core_output_channel_tiles,
            geometry.core_residues,
        ),
        bindings,
    );
    Some(output)
}

/// Try the accepted production path: one compact pack plus one fused core.
///
/// Returns `None` without dispatch when the exact two-shape logical, physical,
/// device, allocation, or resource contract is absent. Production callers can
/// then retain the established T256, T128, and legacy fallback chain.
pub fn try_conv1d_k7_same_residue_d1_snake_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: ResidueDilation,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !conv1d_k7_residue_d1_snake_contract_is_compatible(&input, &weight, &bias, &alpha, dilation)
    {
        return None;
    }
    let packed = try_pack_conv1d_k7_residue_input_wgsl(input, dilation)?;
    conv1d_k7_residue_d1_snake_from_packed_wgsl(packed, weight, bias, alpha, dilation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_selector_admits_exactly_c192_l48000_d3_and_d9() {
        assert_eq!(
            production_dilation_for_shape(192, 48_000, Conv1dK7Dilation::Three),
            Some(ResidueDilation::Three),
        );
        assert_eq!(
            production_dilation_for_shape(192, 48_000, Conv1dK7Dilation::Nine),
            Some(ResidueDilation::Nine),
        );
        for (channels, length, dilation) in [
            (192, 48_000, Conv1dK7Dilation::One),
            (192, 47_999, Conv1dK7Dilation::Three),
            (191, 48_000, Conv1dK7Dilation::Three),
            (96, 96_000, Conv1dK7Dilation::Nine),
        ] {
            assert_eq!(
                production_dilation_for_shape(channels, length, dilation),
                None
            );
        }
    }

    #[test]
    fn exact_geometry_and_accounting_are_fixed() {
        let d3 = ResidueLaunchGeometry::new(ResidueDilation::Three);
        let d9 = ResidueLaunchGeometry::new(ResidueDilation::Nine);
        for geometry in [d3, d9] {
            assert_eq!(geometry.packed_elements, 9_216_000);
            assert_eq!(geometry.temporary_bytes, 36_864_000);
            assert_eq!(geometry.pack_read_write_bytes, 73_728_000);
            assert_eq!(geometry.pack_workgroups, 36_000);
            assert_eq!(geometry.core_output_channel_tiles, 6);
            assert_eq!(geometry.core_workgroups, 1_134);
            assert_eq!(geometry.core_barriers, 27_216);
            assert_eq!(geometry.core_shared_bytes, 31_104);
            assert_eq!(geometry.dispatches, 2);
        }
        assert_eq!((d3.core_time_tiles, d3.core_residues), (63, 3));
        assert_eq!((d9.core_time_tiles, d9.core_residues), (21, 9));
    }

    #[test]
    fn compact_residue_lengths_partition_the_exact_input() {
        for dilation in [ResidueDilation::Three, ResidueDilation::Nine] {
            let mut prefix = 0;
            for residue in 0..dilation.value() {
                assert_eq!(dilation.residue_prefix_q(residue), Some(prefix));
                prefix += dilation
                    .residue_length(residue)
                    .expect("in-range residue has a length");
            }
            assert_eq!(prefix, LENGTH);
            assert_eq!(dilation.residue_length(dilation.value()), None);
            assert_eq!(dilation.residue_prefix_q(dilation.value()), None);
        }
        assert_eq!(
            (0..3)
                .map(|residue| ResidueDilation::Three.residue_length(residue).unwrap())
                .collect::<Vec<_>>(),
            vec![16_000; 3],
        );
        assert_eq!(
            (0..9)
                .map(|residue| ResidueDilation::Nine.residue_length(residue).unwrap())
                .collect::<Vec<_>>(),
            vec![
                5_334, 5_334, 5_334, 5_333, 5_333, 5_333, 5_333, 5_333, 5_333
            ],
        );
    }

    #[test]
    fn compact_index_blocks_are_exact_and_non_overlapping() {
        for dilation in [ResidueDilation::Three, ResidueDilation::Nine] {
            let first = dilation.packed_index(0, 0).unwrap();
            let last = dilation.packed_index(CHANNELS - 1, LENGTH - 1).unwrap();
            assert_eq!(first, 0);
            assert!(last < PACKED_ELEMENTS);
            for residue in 0..dilation.value() {
                let length = dilation.residue_length(residue).unwrap();
                let first_time = residue;
                let last_time = residue + (length - 1) * dilation.value();
                let block_start = dilation.residue_prefix_q(residue).unwrap() * CHANNELS;
                let block_end = block_start + CHANNELS * length - 1;
                assert_eq!(dilation.packed_index(0, first_time), Some(block_start));
                assert_eq!(
                    dilation.packed_index(CHANNELS - 1, last_time),
                    Some(block_end),
                );
            }
        }
    }

    #[test]
    fn residue_d1_boundary_map_matches_original_dilation() {
        for dilation in [ResidueDilation::Three, ResidueDilation::Nine] {
            let d = dilation.value() as isize;
            for time in [0usize, 1, 2, 26, 27, 28, LENGTH - 3, LENGTH - 2, LENGTH - 1] {
                let residue = time % dilation.value();
                let q = (time / dilation.value()) as isize;
                for tap in 0..KERNEL_SIZE {
                    let original = time as isize - 3 * d + tap as isize * d;
                    let residue_source_q = q - 3 + tap as isize;
                    if (0..LENGTH as isize).contains(&original) {
                        assert_eq!(original % d, residue as isize);
                        assert_eq!(original, residue as isize + residue_source_q * d);
                    } else {
                        let residue_length = dilation.residue_length(residue).unwrap() as isize;
                        assert!(residue_source_q < 0 || residue_source_q >= residue_length);
                    }
                }
            }
        }
    }

    #[test]
    fn pack_shader_encodes_the_compact_mapping() {
        let shader = include_str!("conv1d_k7_residue_pack.wgsl");
        assert_eq!(shader.matches("@group(0) @binding(").count(), 2);
        assert!(shader.contains("let residue = time % DILATION;"));
        assert!(shader.contains("let q = time / DILATION;"));
        assert!(
            shader.contains(
                "let residue_prefix_q = residue * BASE_LENGTH + min(residue, REMAINDER);"
            )
        );
        assert!(shader.contains("packed_buf[packed_index] = input_buf[input_index];"));
    }

    #[test]
    fn core_shader_preserves_fma_order_and_direct_scatter() {
        let shader = include_str!("conv1d_k7_residue_d1_snake.wgsl");
        let current = include_str!("conv1d_k7_t256_snake_vec4_store.wgsl");
        assert_eq!(shader.matches("@group(0) @binding(").count(), 5);
        assert_eq!(shader.matches("workgroupBarrier();").count(), 2);
        assert_eq!(shader.matches(" = fma(").count(), 56);
        let fma_assignments = |source: &'static str| {
            source
                .lines()
                .map(str::trim)
                .filter(|line| line.contains(" = fma("))
                .collect::<Vec<_>>()
        };
        assert_eq!(fma_assignments(shader), fma_assignments(current));
        let tap_offsets = (0..KERNEL_SIZE)
            .map(|tap| shader.find(&format!("// tap {tap}")).expect("tap marker"))
            .collect::<Vec<_>>();
        assert!(tap_offsets.windows(2).all(|window| window[0] < window[1]));
        assert!(shader.contains("let source_q = i32(q_base + tile_q) - PADDING;"));
        assert!(shader.contains("let output_time = residue + q * DILATION;"));
        assert!(shader.contains("return x + (sine * sine) / (a + 1e-9);"));
        assert!(!shader.contains("array<vec4<f32>>"));
    }
}
