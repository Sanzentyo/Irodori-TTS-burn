//! Isolated C96/L96000 residue-class candidate for dilated DACVAE k=7 + Snake.
//!
//! This file is intentionally disconnected from `src/kernels.rs` and the codec
//! selector.  It reuses the accepted production pack and core WGSL sources
//! byte-for-byte, but supplies a shape descriptor for the two remaining
//! C96/L96000 dilations.  Promotion requires a separate rotating A/B against
//! the current production launcher and then full-decoder validation.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

const BATCH: usize = 1;
const KERNEL_SIZE: usize = 7;
const PADDING_D1: usize = 3;
const INPUT_CHANNEL_TILE: usize = 16;
const OUTPUT_CHANNEL_TILE: usize = 32;
const TIME_TILE: usize = 256;
const LOCAL_TIME_LANES: usize = 16;
const LOCAL_CHANNEL_LANES: usize = 16;
const CORE_WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
const PACK_WORKGROUP_SIZE: usize = 256;
const INPUT_SPAN_D1: usize = TIME_TILE + 2 * PADDING_D1;
const INPUT_TILE_SIZE: usize = INPUT_CHANNEL_TILE * INPUT_SPAN_D1;
const WEIGHT_TILE_SIZE: usize = OUTPUT_CHANNEL_TILE * INPUT_CHANNEL_TILE * KERNEL_SIZE;
const SHARED_BYTES: usize = (INPUT_TILE_SIZE + WEIGHT_TILE_SIZE) * size_of::<f32>();
const PACK_BINDINGS: u32 = 2;
const CORE_BINDINGS: u32 = 5;

/// The exact candidate shape.  Fields remain part of the kernel identity so a
/// later isolated shape experiment cannot alias this compiled pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ResidueCandidateShape {
    channels: usize,
    length: usize,
}

impl ResidueCandidateShape {
    /// Released decoder block-3 residual shape.
    pub(crate) const C96_L96000: Self = Self {
        channels: 96,
        length: 96_000,
    };

    pub(crate) const fn channels(self) -> usize {
        self.channels
    }

    pub(crate) const fn length(self) -> usize {
        self.length
    }

    pub(crate) const fn elements(self) -> usize {
        BATCH * self.channels * self.length
    }

    pub(crate) const fn bytes(self) -> usize {
        self.elements() * size_of::<f32>()
    }

    const fn core_is_compatible(self) -> bool {
        self.channels == Self::C96_L96000.channels
            && self.length == Self::C96_L96000.length
            && self.channels.is_multiple_of(INPUT_CHANNEL_TILE)
            && self.channels.is_multiple_of(OUTPUT_CHANNEL_TILE)
    }
}

/// The two remaining released dilations screened by this candidate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub(crate) enum ResidueCandidateDilation {
    Three = 3,
    Nine = 9,
}

impl ResidueCandidateDilation {
    pub(crate) const fn value(self) -> usize {
        self as usize
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Three => "c96-l96000-residue-d3",
            Self::Nine => "c96-l96000-residue-d9",
        }
    }
}

/// Shape, dilation, and fixed tile identity for one candidate pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ResidueCandidateDescriptor {
    shape: ResidueCandidateShape,
    dilation: ResidueCandidateDilation,
    input_channel_tile: usize,
    output_channel_tile: usize,
    time_tile: usize,
}

impl ResidueCandidateDescriptor {
    pub(crate) const fn c96_l96000(dilation: ResidueCandidateDilation) -> Self {
        Self {
            shape: ResidueCandidateShape::C96_L96000,
            dilation,
            input_channel_tile: INPUT_CHANNEL_TILE,
            output_channel_tile: OUTPUT_CHANNEL_TILE,
            time_tile: TIME_TILE,
        }
    }

    pub(crate) const fn shape(self) -> ResidueCandidateShape {
        self.shape
    }

    pub(crate) const fn dilation(self) -> ResidueCandidateDilation {
        self.dilation
    }

    pub(crate) const fn base_length(self) -> usize {
        self.shape.length / self.dilation.value()
    }

    pub(crate) const fn remainder(self) -> usize {
        self.shape.length % self.dilation.value()
    }

    pub(crate) const fn max_residue_length(self) -> usize {
        self.base_length() + if self.remainder() != 0 { 1 } else { 0 }
    }

    pub(crate) const fn residue_length(self, residue: usize) -> Option<usize> {
        if residue >= self.dilation.value() {
            return None;
        }
        Some(self.base_length() + if residue < self.remainder() { 1 } else { 0 })
    }

    pub(crate) const fn residue_prefix_q(self, residue: usize) -> Option<usize> {
        if residue >= self.dilation.value() {
            return None;
        }
        Some(residue * self.base_length() + residue.min(self.remainder()))
    }

    pub(crate) const fn packed_index(self, channel: usize, time: usize) -> Option<usize> {
        if channel >= self.shape.channels || time >= self.shape.length {
            return None;
        }
        let residue = time % self.dilation.value();
        let q = time / self.dilation.value();
        let residue_length = match self.residue_length(residue) {
            Some(value) => value,
            None => return None,
        };
        let residue_prefix_q = match self.residue_prefix_q(residue) {
            Some(value) => value,
            None => return None,
        };
        Some(residue_prefix_q * self.shape.channels + channel * residue_length + q)
    }

    const fn source_identity(self) -> (Self, usize, usize, usize) {
        (
            self,
            self.input_channel_tile,
            self.output_channel_tile,
            self.time_tile,
        )
    }

    const fn is_exact_candidate(self) -> bool {
        self.shape.core_is_compatible()
            && self.input_channel_tile == INPUT_CHANNEL_TILE
            && self.output_channel_tile == OUTPUT_CHANNEL_TILE
            && self.time_tile == TIME_TILE
    }
}

/// Static launch geometry for the two-dispatch candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidueCandidateGeometry {
    pub(crate) descriptor: ResidueCandidateDescriptor,
    pub(crate) packed_elements: usize,
    pub(crate) temporary_bytes: usize,
    pub(crate) pack_workgroups: u32,
    pub(crate) core_time_tiles: u32,
    pub(crate) core_output_channel_tiles: u32,
    pub(crate) core_residues: u32,
    pub(crate) core_workgroups: usize,
    pub(crate) core_barriers: usize,
    pub(crate) core_shared_bytes: usize,
    pub(crate) dispatches: usize,
    pub(crate) persistent_bytes: usize,
    pub(crate) output_unpack_dispatches: usize,
}

impl ResidueCandidateGeometry {
    pub(crate) const fn new(descriptor: ResidueCandidateDescriptor) -> Self {
        let shape = descriptor.shape;
        let core_time_tiles = descriptor.max_residue_length().div_ceil(TIME_TILE);
        let core_output_channel_tiles = shape.channels / OUTPUT_CHANNEL_TILE;
        let core_workgroups =
            core_time_tiles * core_output_channel_tiles * descriptor.dilation.value();
        let barriers_per_workgroup = 2 * (shape.channels / INPUT_CHANNEL_TILE);
        Self {
            descriptor,
            packed_elements: shape.elements(),
            temporary_bytes: shape.bytes(),
            pack_workgroups: shape.elements().div_ceil(PACK_WORKGROUP_SIZE) as u32,
            core_time_tiles: core_time_tiles as u32,
            core_output_channel_tiles: core_output_channel_tiles as u32,
            core_residues: descriptor.dilation.value() as u32,
            core_workgroups,
            core_barriers: core_workgroups * barriers_per_workgroup,
            core_shared_bytes: SHARED_BYTES,
            dispatches: 2,
            persistent_bytes: 0,
            output_unpack_dispatches: 0,
        }
    }
}

/// Exact semantic buffer traffic performed by the current WGSL source.
///
/// Read-write storage declarations are required by the sliced allocator, but
/// these counts follow the actual source operations rather than binding access
/// modes.  They exclude the common output allocation and host readback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidueCandidateTraffic {
    pub(crate) pack_read_bytes: usize,
    pub(crate) pack_write_bytes: usize,
    pub(crate) core_input_read_bytes: usize,
    pub(crate) core_weight_read_bytes: usize,
    pub(crate) core_bias_read_bytes: usize,
    pub(crate) core_alpha_read_bytes: usize,
    pub(crate) core_output_write_bytes: usize,
    pub(crate) total_bytes: usize,
}

impl ResidueCandidateTraffic {
    pub(crate) fn new(descriptor: ResidueCandidateDescriptor) -> Self {
        let geometry = ResidueCandidateGeometry::new(descriptor);
        let shape = descriptor.shape;
        let scalar_bytes = size_of::<f32>();
        let core_input_read_elements = core_input_read_elements(descriptor);
        let core_weight_read_elements =
            geometry.core_workgroups * (shape.channels / INPUT_CHANNEL_TILE) * WEIGHT_TILE_SIZE;
        let core_bias_read_elements = geometry.core_workgroups * CORE_WORKGROUP_SIZE * 2;
        let output_elements = shape.elements();
        let pack_read_bytes = output_elements * scalar_bytes;
        let pack_write_bytes = output_elements * scalar_bytes;
        let core_input_read_bytes = core_input_read_elements * scalar_bytes;
        let core_weight_read_bytes = core_weight_read_elements * scalar_bytes;
        let core_bias_read_bytes = core_bias_read_elements * scalar_bytes;
        let core_alpha_read_bytes = output_elements * scalar_bytes;
        let core_output_write_bytes = output_elements * scalar_bytes;
        let total_bytes = pack_read_bytes
            + pack_write_bytes
            + core_input_read_bytes
            + core_weight_read_bytes
            + core_bias_read_bytes
            + core_alpha_read_bytes
            + core_output_write_bytes;
        Self {
            pack_read_bytes,
            pack_write_bytes,
            core_input_read_bytes,
            core_weight_read_bytes,
            core_bias_read_bytes,
            core_alpha_read_bytes,
            core_output_write_bytes,
            total_bytes,
        }
    }
}

fn core_input_read_elements(descriptor: ResidueCandidateDescriptor) -> usize {
    let shape = descriptor.shape;
    let output_channel_tiles = shape.channels / OUTPUT_CHANNEL_TILE;
    let mut total_q = 0_usize;
    for residue in 0..descriptor.dilation.value() {
        let q_length = descriptor
            .residue_length(residue)
            .expect("loop residue is in range");
        for tile in 0..q_length.div_ceil(TIME_TILE) {
            let q_base = tile * TIME_TILE;
            let first = q_base.saturating_sub(PADDING_D1);
            let end = (q_base + TIME_TILE + PADDING_D1).min(q_length);
            total_q += end.saturating_sub(first);
        }
    }
    total_q * shape.channels * output_channel_tiles
}

#[derive(Debug)]
struct ResidueCandidatePackKernel {
    descriptor: ResidueCandidateDescriptor,
}

impl KernelSource for ResidueCandidatePackKernel {
    fn source(&self) -> SourceTemplate {
        let descriptor = self.descriptor;
        let shape = descriptor.shape;
        SourceTemplate::new(include_str!("conv1d_k7_residue_pack.wgsl"))
            .register("channels", shape.channels.to_string())
            .register("length", shape.length.to_string())
            .register("elements", shape.elements().to_string())
            .register("dilation", descriptor.dilation.value().to_string())
            .register("base_length", descriptor.base_length().to_string())
            .register("remainder", descriptor.remainder().to_string())
            .register("workgroup_size", PACK_WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.descriptor.source_identity())
    }
}

#[derive(Debug)]
struct ResidueCandidateCoreKernel {
    descriptor: ResidueCandidateDescriptor,
}

impl KernelSource for ResidueCandidateCoreKernel {
    fn source(&self) -> SourceTemplate {
        let descriptor = self.descriptor;
        let shape = descriptor.shape;
        SourceTemplate::new(include_str!("conv1d_k7_residue_d1_snake.wgsl"))
            .register("channels", shape.channels.to_string())
            .register("length", shape.length.to_string())
            .register("dilation", descriptor.dilation.value().to_string())
            .register("base_length", descriptor.base_length().to_string())
            .register("remainder", descriptor.remainder().to_string())
            .register("input_span", INPUT_SPAN_D1.to_string())
            .register("input_tile_size", INPUT_TILE_SIZE.to_string())
            .register("weight_tile_size", WEIGHT_TILE_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.descriptor.source_identity())
    }
}

/// Owned candidate inputs, grouped so logical and physical validation cannot
/// accidentally omit one of the five bindings.
#[derive(Debug)]
pub(crate) struct ResidueCandidateInputs {
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
}

impl ResidueCandidateInputs {
    pub(crate) fn new(
        input: CubeTensor<WgpuRuntime>,
        weight: CubeTensor<WgpuRuntime>,
        bias: CubeTensor<WgpuRuntime>,
        alpha: CubeTensor<WgpuRuntime>,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            alpha,
        }
    }
}

fn exact_shape<const D: usize>(tensor: &CubeTensor<WgpuRuntime>, expected: [usize; D]) -> bool {
    tensor.meta.num_dims() == D && tensor.meta.shape().dims::<D>() == expected
}

fn device_supports_geometry(
    reference: &CubeTensor<WgpuRuntime>,
    geometry: ResidueCandidateGeometry,
) -> bool {
    let properties = reference.client.properties();
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

/// Full no-allocation/no-dispatch preflight for the exact candidate.
pub(crate) fn residue_candidate_contract_is_compatible(
    inputs: &ResidueCandidateInputs,
    descriptor: ResidueCandidateDescriptor,
) -> bool {
    if !descriptor.is_exact_candidate() {
        return false;
    }
    let shape = descriptor.shape;
    let geometry = ResidueCandidateGeometry::new(descriptor);
    exact_shape(&inputs.input, [BATCH, shape.channels, shape.length])
        && exact_shape(
            &inputs.weight,
            [shape.channels, shape.channels, KERNEL_SIZE],
        )
        && exact_shape(&inputs.bias, [shape.channels])
        && exact_shape(&inputs.alpha, [BATCH, shape.channels, 1])
        && [&inputs.input, &inputs.weight, &inputs.bias, &inputs.alpha]
            .into_iter()
            .all(|tensor| {
                tensor.dtype == DType::F32
                    && tensor.device == inputs.input.device
                    && tensor.is_contiguous()
            })
        && device_supports_geometry(&inputs.input, geometry)
}

fn packed_contract_is_compatible(
    packed: &CubeTensor<WgpuRuntime>,
    reference: &CubeTensor<WgpuRuntime>,
    descriptor: ResidueCandidateDescriptor,
) -> bool {
    let shape = descriptor.shape;
    exact_shape(packed, [shape.elements()])
        && packed.dtype == DType::F32
        && packed.device == reference.device
        && packed.is_contiguous()
}

/// Launch the isolated compact pack for diagnostic timing.
pub(crate) fn try_pack_residue_candidate(
    input: CubeTensor<WgpuRuntime>,
    descriptor: ResidueCandidateDescriptor,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !descriptor.is_exact_candidate() {
        return None;
    }
    let shape = descriptor.shape;
    let geometry = ResidueCandidateGeometry::new(descriptor);
    if !exact_shape(&input, [BATCH, shape.channels, shape.length])
        || input.dtype != DType::F32
        || !input.is_contiguous()
        || !device_supports_geometry(&input, geometry)
    {
        return None;
    }

    let client = input.client.clone();
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([shape.elements()]),
        client.empty(shape.bytes()),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResidueCandidatePackKernel { descriptor },
            CubeDim::new_1d(PACK_WORKGROUP_SIZE as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(packed.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(geometry.pack_workgroups), bindings);
    Some(packed)
}

fn try_residue_candidate_from_packed(
    packed: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    descriptor: ResidueCandidateDescriptor,
) -> Option<CubeTensor<WgpuRuntime>> {
    let shape = descriptor.shape;
    let geometry = ResidueCandidateGeometry::new(descriptor);
    if !packed_contract_is_compatible(&packed, &weight, descriptor)
        || !exact_shape(&weight, [shape.channels, shape.channels, KERNEL_SIZE])
        || !exact_shape(&bias, [shape.channels])
        || !exact_shape(&alpha, [BATCH, shape.channels, 1])
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
        Shape::from([BATCH, shape.channels, shape.length]),
        client.empty(shape.bytes()),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResidueCandidateCoreKernel { descriptor },
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

/// Launch one compact pack plus the byte-identical production d1 core.
pub(crate) fn try_conv1d_k7_residue_c96_candidate(
    inputs: ResidueCandidateInputs,
    descriptor: ResidueCandidateDescriptor,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !residue_candidate_contract_is_compatible(&inputs, descriptor) {
        return None;
    }
    let packed = try_pack_residue_candidate(inputs.input, descriptor)?;
    try_residue_candidate_from_packed(packed, inputs.weight, inputs.bias, inputs.alpha, descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;

    const D3: ResidueCandidateDescriptor =
        ResidueCandidateDescriptor::c96_l96000(ResidueCandidateDilation::Three);
    const D9: ResidueCandidateDescriptor =
        ResidueCandidateDescriptor::c96_l96000(ResidueCandidateDilation::Nine);

    #[test]
    fn descriptors_are_exactly_c96_l96000_d3_and_d9() {
        for descriptor in [D3, D9] {
            assert!(descriptor.is_exact_candidate());
            assert_eq!(descriptor.shape.channels(), 96);
            assert_eq!(descriptor.shape.length(), 96_000);
            assert_eq!(descriptor.shape.elements(), 9_216_000);
            assert_eq!(descriptor.shape.bytes(), 36_864_000);
            assert_eq!(descriptor.input_channel_tile, 16);
            assert_eq!(descriptor.output_channel_tile, 32);
            assert_eq!(descriptor.time_tile, 256);
        }
    }

    #[test]
    fn exact_geometry_matches_the_byte_identical_core() {
        let d3 = ResidueCandidateGeometry::new(D3);
        assert_eq!((d3.core_time_tiles, d3.core_residues), (125, 3));
        assert_eq!(d3.core_workgroups, 1_125);
        assert_eq!(d3.core_barriers, 13_500);

        let d9 = ResidueCandidateGeometry::new(D9);
        assert_eq!((d9.core_time_tiles, d9.core_residues), (42, 9));
        assert_eq!(d9.core_workgroups, 1_134);
        assert_eq!(d9.core_barriers, 13_608);

        for geometry in [d3, d9] {
            assert_eq!(geometry.pack_workgroups, 36_000);
            assert_eq!(geometry.core_output_channel_tiles, 3);
            assert_eq!(geometry.core_shared_bytes, 31_104);
            assert_eq!(geometry.dispatches, 2);
            assert_eq!(geometry.persistent_bytes, 0);
            assert_eq!(geometry.output_unpack_dispatches, 0);
        }
    }

    #[test]
    fn compact_residue_blocks_partition_the_input() {
        for descriptor in [D3, D9] {
            let mut prefix = 0;
            for residue in 0..descriptor.dilation.value() {
                assert_eq!(descriptor.residue_prefix_q(residue), Some(prefix));
                prefix += descriptor.residue_length(residue).unwrap();
            }
            assert_eq!(prefix, descriptor.shape.length);
            assert_eq!(descriptor.residue_length(descriptor.dilation.value()), None);
            assert!(descriptor.packed_index(95, 95_999).unwrap() < descriptor.shape.elements());
        }
        assert_eq!(
            (0..3)
                .map(|residue| D3.residue_length(residue).unwrap())
                .collect::<Vec<_>>(),
            vec![32_000; 3],
        );
        assert_eq!(
            (0..9)
                .map(|residue| D9.residue_length(residue).unwrap())
                .collect::<Vec<_>>(),
            vec![
                10_667, 10_667, 10_667, 10_667, 10_667, 10_667, 10_666, 10_666, 10_666
            ],
        );
    }

    #[test]
    fn traffic_is_exact_for_source_operations() {
        let d3 = ResidueCandidateTraffic::new(D3);
        assert_eq!(d3.pack_read_bytes, 36_864_000);
        assert_eq!(d3.pack_write_bytes, 36_864_000);
        assert_eq!(d3.core_input_read_bytes, 113_163_264);
        assert_eq!(d3.core_weight_read_bytes, 96_768_000);
        assert_eq!(d3.core_bias_read_bytes, 2_304_000);
        assert_eq!(d3.core_alpha_read_bytes, 36_864_000);
        assert_eq!(d3.core_output_write_bytes, 36_864_000);
        assert_eq!(d3.total_bytes, 359_691_264);

        let d9 = ResidueCandidateTraffic::new(D9);
        assert_eq!(d9.pack_read_bytes, 36_864_000);
        assert_eq!(d9.pack_write_bytes, 36_864_000);
        assert_eq!(d9.core_input_read_bytes, 113_142_528);
        assert_eq!(d9.core_weight_read_bytes, 97_542_144);
        assert_eq!(d9.core_bias_read_bytes, 2_322_432);
        assert_eq!(d9.core_alpha_read_bytes, 36_864_000);
        assert_eq!(d9.core_output_write_bytes, 36_864_000);
        assert_eq!(d9.total_bytes, 360_463_104);
    }

    #[test]
    fn included_wgsl_retains_the_accepted_fma_and_snake_order() {
        let core = include_str!("conv1d_k7_residue_d1_snake.wgsl");
        let pack = include_str!("conv1d_k7_residue_pack.wgsl");
        assert_eq!(core.matches(" = fma(").count(), 56);
        assert_eq!(core.matches("workgroupBarrier();").count(), 2);
        assert!(core.contains("let output_time = residue + q * DILATION;"));
        assert!(core.contains("return x + (sine * sine) / (a + 1e-9);"));
        assert!(pack.contains("let residue = time % DILATION;"));
        assert!(pack.contains("packed_buf[packed_index] = input_buf[input_index];"));
    }
}
