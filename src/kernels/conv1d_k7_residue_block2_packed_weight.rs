//! Isolated Block2 C192/L48000 residue-core candidate.
//!
//! This module is deliberately disconnected from the production decoder.  It
//! keeps the accepted compact residue input and exact input-channel-then-tap
//! FMA order, but changes the core's output mapping and weight cache:
//!
//! - a one-time OIK -> `[Cin, K7, Cout]` weight pack makes four adjacent output
//!   channels one `vec4<f32>`;
//! - every 32-lane subgroup owns one such output vector and 32 adjacent residue
//!   positions, so input shared-memory reads are contiguous across the subgroup;
//! - weights are subgroup-uniform packed-buffer reads instead of a second
//!   14,336-byte workgroup tile.
//!
//! The timed path remains two dispatches (the accepted input pack plus this
//! core).  The weight pack is an inference-preparation cache and must be timed
//! and reported separately by an A/B harness.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

const BATCH: usize = 1;
const CHANNELS: usize = 192;
const LENGTH: usize = 48_000;
const KERNEL_SIZE: usize = 7;
const VECTOR_WIDTH: usize = 4;
const OUTPUT_VECTORS: usize = CHANNELS / VECTOR_WIDTH;
const TIME_TILE: usize = 256;
const OUTPUT_CHANNEL_TILE: usize = 32;
const INPUT_CHANNEL_TILE: usize = 16;
const INPUT_SPAN: usize = TIME_TILE + KERNEL_SIZE - 1;
const INPUT_TILE_ELEMENTS: usize = INPUT_CHANNEL_TILE * INPUT_SPAN;
const CORE_SHARED_BYTES: usize = INPUT_TILE_ELEMENTS * size_of::<f32>();
const LOCAL_TIME_LANES: usize = 32;
const LOCAL_OUTPUT_VECTOR_LANES: usize = 8;
const CORE_WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_OUTPUT_VECTOR_LANES;
const WEIGHT_PACK_WORKGROUP_SIZE: usize = 256;
const WEIGHT_ELEMENTS: usize = CHANNELS * CHANNELS * KERNEL_SIZE;
const WEIGHT_BYTES: usize = WEIGHT_ELEMENTS * size_of::<f32>();
const VEC4_BYTES: u64 = 16;
const PACKED_ELEMENTS: usize = BATCH * CHANNELS * LENGTH;
const OUTPUT_BYTES: usize = PACKED_ELEMENTS * size_of::<f32>();
const CORE_BINDINGS: u32 = 5;
const WEIGHT_PACK_BINDINGS: u32 = 2;

/// The two production residue decompositions evaluated by this candidate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum Block2ResidueDilation {
    Three = 3,
    Nine = 9,
}

impl Block2ResidueDilation {
    pub const fn value(self) -> usize {
        self as usize
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Three => "block2-d3",
            Self::Nine => "block2-d9",
        }
    }

    pub const fn base_length(self) -> usize {
        LENGTH / self.value()
    }

    pub const fn remainder(self) -> usize {
        LENGTH % self.value()
    }

    pub const fn max_residue_length(self) -> usize {
        self.base_length() + if self.remainder() == 0 { 0 } else { 1 }
    }
}

/// Exact static accounting for one candidate core launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Block2PackedWeightGeometry {
    pub dilation: Block2ResidueDilation,
    pub time_tile: usize,
    pub output_channel_tile: usize,
    pub input_channel_tile: usize,
    pub workgroup: [u32; 3],
    pub core_time_tiles: u32,
    pub core_output_channel_tiles: u32,
    pub core_residues: u32,
    pub core_workgroups: usize,
    pub barriers_per_workgroup: usize,
    pub core_barriers: usize,
    pub core_shared_bytes: usize,
    pub cached_weight_bytes: usize,
    pub cached_weight_pack_workgroups: u32,
}

impl Block2PackedWeightGeometry {
    pub const fn new(dilation: Block2ResidueDilation) -> Self {
        let core_time_tiles = dilation.max_residue_length().div_ceil(TIME_TILE);
        let core_output_channel_tiles = CHANNELS / OUTPUT_CHANNEL_TILE;
        let core_workgroups = core_time_tiles * core_output_channel_tiles * dilation.value();
        let barriers_per_workgroup = 2 * (CHANNELS / INPUT_CHANNEL_TILE);
        Self {
            dilation,
            time_tile: TIME_TILE,
            output_channel_tile: OUTPUT_CHANNEL_TILE,
            input_channel_tile: INPUT_CHANNEL_TILE,
            workgroup: [LOCAL_TIME_LANES as u32, LOCAL_OUTPUT_VECTOR_LANES as u32, 1],
            core_time_tiles: core_time_tiles as u32,
            core_output_channel_tiles: core_output_channel_tiles as u32,
            core_residues: dilation.value() as u32,
            core_workgroups,
            barriers_per_workgroup,
            core_barriers: core_workgroups * barriers_per_workgroup,
            core_shared_bytes: CORE_SHARED_BYTES,
            cached_weight_bytes: WEIGHT_BYTES,
            cached_weight_pack_workgroups: (WEIGHT_ELEMENTS / VECTOR_WIDTH)
                .div_ceil(WEIGHT_PACK_WORKGROUP_SIZE)
                as u32,
        }
    }
}

#[derive(Debug)]
struct Block2WeightPackKernel;

impl KernelSource for Block2WeightPackKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_residue_block2_weight_pack.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[derive(Debug)]
struct Block2PackedWeightCoreKernel {
    dilation: Block2ResidueDilation,
}

impl KernelSource for Block2PackedWeightCoreKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_residue_block2_packed_weight.wgsl"))
            .register("dilation", self.dilation.value().to_string())
            .register("base_length", self.dilation.base_length().to_string())
            .register("remainder", self.dilation.remainder().to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.dilation)
    }
}

fn exact_shape<const D: usize>(tensor: &CubeTensor<WgpuRuntime>, expected: [usize; D]) -> bool {
    tensor.meta.num_dims() == D && tensor.meta.shape().dims::<D>() == expected
}

fn exact_contiguous_f32<const D: usize>(
    tensor: &CubeTensor<WgpuRuntime>,
    expected: [usize; D],
) -> bool {
    exact_shape(tensor, expected) && tensor.dtype == DType::F32 && tensor.is_contiguous()
}

fn device_supports_weight_pack(weight: &CubeTensor<WgpuRuntime>) -> bool {
    let properties = weight.client.properties();
    let hardware = &properties.hardware;
    let Ok(weight_bytes) = u64::try_from(WEIGHT_BYTES) else {
        return false;
    };
    weight_bytes <= properties.memory.max_page_size
        && properties.memory.alignment >= VEC4_BYTES
        && properties.memory.alignment.is_multiple_of(VEC4_BYTES)
        && hardware.max_bindings >= WEIGHT_PACK_BINDINGS
        && hardware.max_units_per_cube >= WEIGHT_PACK_WORKGROUP_SIZE as u32
        && hardware.max_cube_dim.0 >= WEIGHT_PACK_WORKGROUP_SIZE as u32
        && hardware.max_cube_count.0
            >= Block2PackedWeightGeometry::new(Block2ResidueDilation::Three)
                .cached_weight_pack_workgroups
}

fn vec4_binding_is_compatible(tensor: &CubeTensor<WgpuRuntime>) -> bool {
    let binding = tensor.handle.clone().binding();
    binding.size_in_used().is_multiple_of(VEC4_BYTES)
        && tensor.client.properties().memory.alignment >= VEC4_BYTES
        && tensor
            .client
            .properties()
            .memory
            .alignment
            .is_multiple_of(VEC4_BYTES)
        && binding.offset_start.unwrap_or(0).is_multiple_of(VEC4_BYTES)
}

fn device_supports_core(
    reference: &CubeTensor<WgpuRuntime>,
    geometry: Block2PackedWeightGeometry,
) -> bool {
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    let Ok(output_bytes) = u64::try_from(OUTPUT_BYTES) else {
        return false;
    };
    output_bytes <= properties.memory.max_page_size
        && hardware.max_bindings >= CORE_BINDINGS
        && hardware.max_shared_memory_size >= geometry.core_shared_bytes
        && hardware.max_units_per_cube >= CORE_WORKGROUP_SIZE as u32
        && hardware.max_cube_dim.0 >= geometry.workgroup[0]
        && hardware.max_cube_dim.1 >= geometry.workgroup[1]
        && hardware.max_cube_dim.2 >= geometry.workgroup[2]
        && hardware.max_cube_count.0 >= geometry.core_time_tiles
        && hardware.max_cube_count.1 >= geometry.core_output_channel_tiles
        && hardware.max_cube_count.2 >= geometry.core_residues
}

/// Validate an exact checkpoint-native OIK weight before the one-time pack.
pub fn block2_weight_pack_contract_is_compatible(weight: &CubeTensor<WgpuRuntime>) -> bool {
    exact_contiguous_f32(weight, [CHANNELS, CHANNELS, KERNEL_SIZE])
        && device_supports_weight_pack(weight)
}

/// Pack OIK `[Cout, Cin, K7]` into vector-addressable `[Cin, K7, Cout]`.
///
/// This is a one-time inference cache.  It is not part of steady-state decode
/// timing and is never invoked by production code in this isolated module.
pub fn try_pack_block2_weight_kto_wgsl(
    weight_oik: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !block2_weight_pack_contract_is_compatible(&weight_oik) {
        return None;
    }
    let geometry = Block2PackedWeightGeometry::new(Block2ResidueDilation::Three);
    let client = weight_oik.client.clone();
    let packed_handle = client.empty(WEIGHT_BYTES);
    if !packed_handle.size_in_used().is_multiple_of(VEC4_BYTES)
        || !packed_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(VEC4_BYTES)
    {
        return None;
    }
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        weight_oik.device.clone(),
        Shape::from([CHANNELS, KERNEL_SIZE, CHANNELS]),
        packed_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            Block2WeightPackKernel,
            CubeDim::new_1d(WEIGHT_PACK_WORKGROUP_SIZE as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(weight_oik.handle.binding())
        .with_buffer(packed.handle.clone().binding());
    client.launch(
        task,
        CubeCount::new_1d(geometry.cached_weight_pack_workgroups),
        bindings,
    );
    Some(packed)
}

/// Validate the complete packed-input/packed-weight core contract.
pub fn block2_packed_weight_core_contract_is_compatible(
    packed_input: &CubeTensor<WgpuRuntime>,
    packed_weight_kto: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
    dilation: Block2ResidueDilation,
) -> bool {
    let geometry = Block2PackedWeightGeometry::new(dilation);
    exact_contiguous_f32(packed_input, [PACKED_ELEMENTS])
        && exact_contiguous_f32(packed_weight_kto, [CHANNELS, KERNEL_SIZE, CHANNELS])
        && exact_contiguous_f32(bias, [CHANNELS])
        && exact_contiguous_f32(alpha, [BATCH, CHANNELS, 1])
        && vec4_binding_is_compatible(packed_weight_kto)
        && [packed_weight_kto, bias, alpha]
            .into_iter()
            .all(|tensor| tensor.device == packed_input.device)
        && device_supports_core(packed_input, geometry)
}

/// Launch only the isolated packed-weight core after the accepted input pack.
pub fn try_block2_residue_packed_weight_core_wgsl(
    packed_input: CubeTensor<WgpuRuntime>,
    packed_weight_kto: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: Block2ResidueDilation,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !block2_packed_weight_core_contract_is_compatible(
        &packed_input,
        &packed_weight_kto,
        &bias,
        &alpha,
        dilation,
    ) {
        return None;
    }
    let geometry = Block2PackedWeightGeometry::new(dilation);
    let client = packed_input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        packed_input.device.clone(),
        Shape::from([BATCH, CHANNELS, LENGTH]),
        client.empty(OUTPUT_BYTES),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            Block2PackedWeightCoreKernel { dilation },
            CubeDim::new_2d(LOCAL_TIME_LANES as u32, LOCAL_OUTPUT_VECTOR_LANES as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(packed_input.handle.binding())
        .with_buffer(packed_weight_kto.handle.binding())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_preserves_work_and_removes_the_weight_shared_tile() {
        for dilation in [Block2ResidueDilation::Three, Block2ResidueDilation::Nine] {
            let geometry = Block2PackedWeightGeometry::new(dilation);
            assert_eq!(geometry.time_tile, 256);
            assert_eq!(geometry.output_channel_tile, 32);
            assert_eq!(geometry.input_channel_tile, 16);
            assert_eq!(geometry.workgroup, [32, 8, 1]);
            assert_eq!(geometry.core_workgroups, 1_134);
            assert_eq!(geometry.barriers_per_workgroup, 24);
            assert_eq!(geometry.core_barriers, 27_216);
            assert_eq!(geometry.core_shared_bytes, 16_768);
            assert_eq!(geometry.cached_weight_bytes, 1_032_192);
            assert_eq!(geometry.cached_weight_pack_workgroups, 252);
        }
    }

    #[test]
    fn residue_geometry_is_exact() {
        let d3 = Block2PackedWeightGeometry::new(Block2ResidueDilation::Three);
        let d9 = Block2PackedWeightGeometry::new(Block2ResidueDilation::Nine);
        assert_eq!((d3.core_time_tiles, d3.core_residues), (63, 3));
        assert_eq!((d9.core_time_tiles, d9.core_residues), (21, 9));
        assert_eq!(Block2ResidueDilation::Three.base_length(), 16_000);
        assert_eq!(Block2ResidueDilation::Nine.base_length(), 5_333);
        assert_eq!(Block2ResidueDilation::Nine.remainder(), 3);
    }

    #[test]
    fn shader_contract_keeps_exact_fma_and_snake_order() {
        let shader = include_str!("conv1d_k7_residue_block2_packed_weight.wgsl");
        assert_eq!(shader.matches("@group(0) @binding(").count(), 5);
        assert_eq!(shader.matches("workgroupBarrier();").count(), 2);
        assert_eq!(shader.matches("// tap ").count(), KERNEL_SIZE);
        assert_eq!(shader.matches(" = fma(").count(), KERNEL_SIZE * 8);
        assert!(shader.contains("let input_channel = input_channel_base + tile_input_channel;"));
        assert!(shader.contains("return x + (sine * sine) / (a + 1e-9);"));
        assert!(shader.contains("let output_time = residue + q * DILATION;"));
        assert!(!shader.contains("var<workgroup> weight"));
    }

    #[test]
    fn weight_pack_is_oik_to_cin_tap_cout_vec4() {
        let shader = include_str!("conv1d_k7_residue_block2_weight_pack.wgsl");
        assert_eq!(shader.matches("@group(0) @binding(").count(), 2);
        assert!(shader.contains("(output_channel * CHANNELS + input_channel) * KERNEL_SIZE + tap"));
        assert!(shader.contains("packed_weight[packed_vector_index] = vec4<f32>("));
    }
}
