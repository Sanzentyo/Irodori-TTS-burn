//! Production residue-class kernel for long dilated DACVAE k=7 calls.
//!
//! The accepted rotating A/B was bit-exact over 18,432,000 outputs and reduced
//! the exact d3+d9 median sum from 11.693 ms to 7.914 ms (3.779 ms, 1.477x).
//! It maps decoder-family `[1, 192, L]` NCL input into compact
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
const C96: usize = 96;
const C192: usize = 192;
const C384: usize = 384;
const KERNEL_SIZE: usize = 7;
const PADDING_D1: usize = 3;
const INPUT_CHANNEL_TILE: usize = 8;
const OUTPUT_CHANNEL_TILE: usize = 32;
const TIME_TILE: usize = 256;
const LOCAL_TIME_LANES: usize = 32;
const LOCAL_CHANNEL_LANES: usize = 8;
const CORE_WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
const PACK_WORKGROUP_SIZE: usize = 256;
const F32_BYTES: usize = size_of::<f32>();
const INPUT_SPAN_D1: usize = TIME_TILE + 2 * PADDING_D1;
const INPUT_TILE_SIZE: usize = INPUT_CHANNEL_TILE * INPUT_SPAN_D1;
const WEIGHT_VECTOR_TILE_SIZE: usize = (OUTPUT_CHANNEL_TILE / 4) * INPUT_CHANNEL_TILE * KERNEL_SIZE;
const SHARED_BYTES: usize = (INPUT_TILE_SIZE + 4 * WEIGHT_VECTOR_TILE_SIZE) * F32_BYTES;
const PACK_BINDINGS: u32 = 2;
const WEIGHT_VECTOR_PACK_BINDINGS: u32 = 2;
const CORE_BINDINGS: u32 = 5;
const WEIGHT_VECTOR_PACK_WORKGROUP_SIZE: usize = 256;

const fn decoder_stage_length_is_compatible(channels: usize, length: usize) -> bool {
    length > 0
        && match channels {
            C96 => length.is_multiple_of(1_920),
            C192 => length.is_multiple_of(960),
            C384 => length.is_multiple_of(120),
            _ => false,
        }
}

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
    pub const fn base_length(self, length: usize) -> usize {
        length / self.value()
    }

    /// Number of leading residues that contain one extra element.
    pub const fn remainder(self, length: usize) -> usize {
        length % self.value()
    }

    /// Longest residue sequence, used for the rectangular core dispatch.
    pub const fn max_residue_length(self, length: usize) -> usize {
        self.base_length(length) + if self.remainder(length) == 0 { 0 } else { 1 }
    }

    /// Exact logical length of one compact residue sequence.
    pub const fn residue_length(self, length: usize, residue: usize) -> Option<usize> {
        if residue >= self.value() {
            return None;
        }
        Some(
            self.base_length(length)
                + if residue < self.remainder(length) {
                    1
                } else {
                    0
                },
        )
    }

    /// Number of time positions stored before one residue block.
    pub const fn residue_prefix_q(self, length: usize, residue: usize) -> Option<usize> {
        if residue >= self.value() {
            return None;
        }
        let extra = if residue < self.remainder(length) {
            residue
        } else {
            self.remainder(length)
        };
        Some(residue * self.base_length(length) + extra)
    }

    /// Compact `[residue][channel][q]` index for an original NCL element.
    pub const fn packed_index(
        self,
        channels: usize,
        length: usize,
        channel: usize,
        time: usize,
    ) -> Option<usize> {
        if channel >= channels || time >= length {
            return None;
        }
        let residue = time % self.value();
        let q = time / self.value();
        let residue_length = match self.residue_length(length, residue) {
            Some(value) => value,
            None => return None,
        };
        let residue_prefix_q = match self.residue_prefix_q(length, residue) {
            Some(value) => value,
            None => return None,
        };
        Some(residue_prefix_q * channels + channel * residue_length + q)
    }
}

/// Exact static launch and temporary-storage accounting for one production call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidueLaunchGeometry {
    /// Selected exact decoder dilation.
    pub dilation: ResidueDilation,
    /// Exact dynamic decoder-stage channel count.
    pub channels: usize,
    /// Exact dynamic decoder-stage length.
    pub length: usize,
    /// Compact pack elements; equal to the source NCL element count.
    pub packed_elements: usize,
    /// Compact temporary bytes retained between the two dispatches.
    pub temporary_bytes: usize,
    /// Logical source read plus compact destination write bytes for the pack.
    pub pack_read_write_bytes: usize,
    /// Logical workgroups in the pack dispatch. The launcher maps these onto
    /// one or two dispatch dimensions as required by the device limits.
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
    /// Fixed d1/Cin8 workgroup storage.
    pub core_shared_bytes: usize,
    /// Pack plus core dispatches.
    pub dispatches: usize,
}

impl ResidueLaunchGeometry {
    /// Construct checked geometry for one admitted decoder-family shape.
    pub fn new(dilation: ResidueDilation, channels: usize, length: usize) -> Option<Self> {
        if !matches!(channels, C96 | C192 | C384)
            || !channels.is_multiple_of(INPUT_CHANNEL_TILE)
            || !channels.is_multiple_of(OUTPUT_CHANNEL_TILE)
            || !decoder_stage_length_is_compatible(channels, length)
        {
            return None;
        }
        let packed_elements = BATCH.checked_mul(channels)?.checked_mul(length)?;
        let packed_bytes = packed_elements.checked_mul(F32_BYTES)?;
        let max_residue_length = dilation.max_residue_length(length);
        let core_time_tiles = max_residue_length.div_ceil(TIME_TILE);
        let core_output_channel_tiles = channels / OUTPUT_CHANNEL_TILE;
        let core_workgroups = core_time_tiles
            .checked_mul(core_output_channel_tiles)
            .and_then(|value| value.checked_mul(dilation.value()))?;
        let barriers_per_workgroup = 2 * (channels / INPUT_CHANNEL_TILE);
        Some(Self {
            dilation,
            channels,
            length,
            packed_elements,
            temporary_bytes: packed_bytes,
            pack_read_write_bytes: packed_bytes.checked_mul(2)?,
            pack_workgroups: match u32::try_from(packed_elements.div_ceil(PACK_WORKGROUP_SIZE)) {
                Ok(value) => value,
                Err(_) => return None,
            },
            core_time_tiles: core_time_tiles as u32,
            core_output_channel_tiles: core_output_channel_tiles as u32,
            core_residues: dilation.value() as u32,
            core_workgroups,
            core_barriers: core_workgroups * barriers_per_workgroup,
            core_shared_bytes: SHARED_BYTES,
            dispatches: 2,
        })
    }
}

#[derive(Debug)]
struct ResiduePackKernel {
    dilation: ResidueDilation,
    channels: usize,
    length: usize,
    dispatch_x: u32,
}

impl KernelSource for ResiduePackKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_residue_pack.wgsl"))
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register(
                "elements",
                (BATCH * self.channels * self.length).to_string(),
            )
            .register("dilation", self.dilation.value().to_string())
            .register(
                "base_length",
                self.dilation.base_length(self.length).to_string(),
            )
            .register(
                "remainder",
                self.dilation.remainder(self.length).to_string(),
            )
            .register("workgroup_size", PACK_WORKGROUP_SIZE.to_string())
            .register("dispatch_x", self.dispatch_x.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.dilation, self.channels, self.length, self.dispatch_x))
    }
}

#[derive(Debug)]
struct ResidueD1SnakeCoreKernel {
    dilation: ResidueDilation,
    channels: usize,
    length: usize,
}

#[derive(Debug)]
struct ResidueWeightVectorPackKernel {
    channels: usize,
    vector_elements: usize,
}

impl KernelSource for ResidueWeightVectorPackKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_residue_weight_vector_pack.wgsl"))
            .register("channels", self.channels.to_string())
            .register("vector_elements", self.vector_elements.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.channels, self.vector_elements))
    }
}

impl KernelSource for ResidueD1SnakeCoreKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_residue_d1_snake.wgsl"))
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.value().to_string())
            .register(
                "base_length",
                self.dilation.base_length(self.length).to_string(),
            )
            .register(
                "remainder",
                self.dilation.remainder(self.length).to_string(),
            )
            .register("input_channel_tile", INPUT_CHANNEL_TILE.to_string())
            .register("input_span", INPUT_SPAN_D1.to_string())
            .register("input_tile_size", INPUT_TILE_SIZE.to_string())
            .register(
                "weight_vector_tile_size",
                WEIGHT_VECTOR_TILE_SIZE.to_string(),
            )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.dilation,
            self.channels,
            self.length,
            INPUT_CHANNEL_TILE,
            LOCAL_TIME_LANES,
            LOCAL_CHANNEL_LANES,
        ))
    }
}

fn exact_shape<const D: usize>(tensor: &CubeTensor<WgpuRuntime>, expected: [usize; D]) -> bool {
    tensor.meta.num_dims() == D && tensor.meta.shape().dims::<D>() == expected
}

fn exact_input_contract(input: &CubeTensor<WgpuRuntime>, channels: usize, length: usize) -> bool {
    exact_shape(input, [BATCH, channels, length])
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
    let Some(pack_dispatch) = pack_dispatch_2d(geometry.pack_workgroups, hardware.max_cube_count)
    else {
        return false;
    };
    temporary_bytes <= properties.memory.max_page_size
        && hardware.max_bindings >= PACK_BINDINGS.max(CORE_BINDINGS)
        && hardware.max_shared_memory_size >= geometry.core_shared_bytes
        && hardware.max_units_per_cube >= CORE_WORKGROUP_SIZE.max(PACK_WORKGROUP_SIZE) as u32
        && hardware.max_cube_dim.0 >= PACK_WORKGROUP_SIZE as u32
        && hardware.max_cube_dim.1 >= LOCAL_CHANNEL_LANES as u32
        && hardware.max_cube_dim.2 >= 1
        && hardware.max_cube_count.0 >= pack_dispatch.x.max(geometry.core_time_tiles)
        && hardware.max_cube_count.1 >= pack_dispatch.y.max(geometry.core_output_channel_tiles)
        && hardware.max_cube_count.2 >= geometry.core_residues
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackDispatch2d {
    x: u32,
    y: u32,
}

fn pack_dispatch_2d(workgroups: u32, max_cube_count: (u32, u32, u32)) -> Option<PackDispatch2d> {
    if workgroups == 0 || max_cube_count.0 == 0 || max_cube_count.1 == 0 {
        return None;
    }
    let x = workgroups.min(max_cube_count.0);
    let y = workgroups.div_ceil(x);
    (y <= max_cube_count.1).then_some(PackDispatch2d { x, y })
}

fn packed_contract_is_compatible(
    packed: &CubeTensor<WgpuRuntime>,
    reference: &CubeTensor<WgpuRuntime>,
    geometry: ResidueLaunchGeometry,
) -> bool {
    exact_shape(packed, [geometry.packed_elements])
        && packed.dtype == DType::F32
        && packed.device == reference.device
        && packed.is_contiguous()
}

fn packed_weight_vector_contract_is_compatible(
    packed: &CubeTensor<WgpuRuntime>,
    reference: &CubeTensor<WgpuRuntime>,
    channels: usize,
) -> bool {
    exact_shape(packed, [channels, KERNEL_SIZE, channels])
        && packed.dtype == DType::F32
        && packed.device == reference.device
        && packed.is_contiguous()
        && packed
            .handle
            .clone()
            .binding()
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(4 * F32_BYTES as u64)
}

/// Select the two measured dilations for exact decoder-family C96/C192/C384
/// lengths.
pub const fn production_dilation_for_shape(
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
) -> Option<ResidueDilation> {
    if !matches!(channels, C96 | C192 | C384)
        || !decoder_stage_length_is_compatible(channels, length)
    {
        return None;
    }
    match dilation {
        Conv1dK7Dilation::Three => Some(ResidueDilation::Three),
        Conv1dK7Dilation::Nine => Some(ResidueDilation::Nine),
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
    let [batch, channels, length] = input.meta.shape().dims::<3>();
    let Some(geometry) = ResidueLaunchGeometry::new(dilation, channels, length) else {
        return false;
    };
    batch == BATCH
        && exact_input_contract(input, channels, length)
        && packed_weight_vector_contract_is_compatible(weight, input, channels)
        && exact_shape(bias, [channels])
        && exact_shape(alpha, [BATCH, channels, 1])
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
    let [_, channels, length] = input.meta.shape().dims::<3>();
    let geometry = ResidueLaunchGeometry::new(dilation, channels, length)?;
    if !exact_input_contract(&input, channels, length)
        || !device_supports_geometry(&input, geometry)
    {
        return None;
    }

    let client = input.client.clone();
    let dispatch = pack_dispatch_2d(
        geometry.pack_workgroups,
        client.properties().hardware.max_cube_count,
    )?;
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([geometry.packed_elements]),
        client.empty(geometry.temporary_bytes),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResiduePackKernel {
                dilation,
                channels,
                length,
                dispatch_x: dispatch.x,
            },
            CubeDim::new_1d(PACK_WORKGROUP_SIZE as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(packed.handle.clone().binding());
    client.launch(task, CubeCount::new_2d(dispatch.x, dispatch.y), bindings);
    Some(packed)
}

/// Pack checkpoint-native OIK weights into invocation-owned output vectors.
///
/// The returned logical shape remains `[Cin, K7, Cout]`, while its physical
/// scalar order is `[w(o), w(o+8), w(o+16), w(o+24)]` for each production
/// output lane. This is an inference-preparation cache and never belongs to
/// decode timing.
pub fn try_pack_conv1d_k7_residue_weight_vectors_wgsl(
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let [output_channels, input_channels, kernel_size] = weight.meta.shape().dims::<3>();
    if output_channels != input_channels
        || !matches!(output_channels, C96 | C192 | C384)
        || kernel_size != KERNEL_SIZE
        || weight.dtype != DType::F32
        || !weight.is_contiguous()
    {
        return None;
    }
    let channels = output_channels;
    let scalar_elements = channels.checked_mul(channels)?.checked_mul(KERNEL_SIZE)?;
    let vector_elements = scalar_elements / 4;
    let output_bytes = scalar_elements.checked_mul(F32_BYTES)?;
    let vector_workgroups = vector_elements.div_ceil(WEIGHT_VECTOR_PACK_WORKGROUP_SIZE);
    let vector_workgroups = u32::try_from(vector_workgroups).ok()?;
    let properties = weight.client.properties();
    let hardware = &properties.hardware;
    if hardware.max_bindings < WEIGHT_VECTOR_PACK_BINDINGS
        || hardware.max_units_per_cube < WEIGHT_VECTOR_PACK_WORKGROUP_SIZE as u32
        || hardware.max_cube_dim.0 < WEIGHT_VECTOR_PACK_WORKGROUP_SIZE as u32
        || hardware.max_cube_count.0 < vector_workgroups
        || u64::try_from(output_bytes).ok()? > properties.memory.max_page_size
    {
        return None;
    }

    let client = weight.client.clone();
    let handle = client.empty(output_bytes);
    if !handle
        .offset_start
        .unwrap_or(0)
        .is_multiple_of(4 * F32_BYTES as u64)
    {
        return None;
    }
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        weight.device.clone(),
        Shape::from([channels, KERNEL_SIZE, channels]),
        handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResidueWeightVectorPackKernel {
                channels,
                vector_elements,
            },
            CubeDim::new_1d(WEIGHT_VECTOR_PACK_WORKGROUP_SIZE as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(weight.handle.clone().binding())
        .with_buffer(packed.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(vector_workgroups), bindings);
    packed_weight_vector_contract_is_compatible(&packed, &weight, channels).then_some(packed)
}

/// Launch only the residue-d1 convolution/Snake core from a validated pack.
fn conv1d_k7_residue_d1_snake_from_packed_wgsl(
    packed: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: ResidueDilation,
    channels: usize,
    length: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    let geometry = ResidueLaunchGeometry::new(dilation, channels, length)?;
    if !packed_contract_is_compatible(&packed, &weight, geometry)
        || !packed_weight_vector_contract_is_compatible(&weight, &packed, channels)
        || !exact_shape(&bias, [channels])
        || !exact_shape(&alpha, [BATCH, channels, 1])
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
        Shape::from([BATCH, channels, length]),
        client.empty(geometry.temporary_bytes),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResidueD1SnakeCoreKernel {
                dilation,
                channels,
                length,
            },
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
/// Returns `None` without dispatch when the decoder-family logical, physical,
/// device, allocation, or resource contract is absent. Production callers can
/// then retain the established T256, T128, and legacy fallback chain.
pub fn try_conv1d_k7_same_residue_d1_snake_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: ResidueDilation,
) -> Option<CubeTensor<WgpuRuntime>> {
    let [_, channels, length] = input.meta.shape().dims::<3>();
    if !packed_weight_vector_contract_is_compatible(&weight, &input, channels)
        || !conv1d_k7_residue_d1_snake_contract_is_compatible(
            &input, &weight, &bias, &alpha, dilation,
        )
    {
        return None;
    }
    let packed = try_pack_conv1d_k7_residue_input_wgsl(input, dilation)?;
    conv1d_k7_residue_d1_snake_from_packed_wgsl(
        packed, weight, bias, alpha, dilation, channels, length,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE_LENGTH: usize = 48_000;

    #[test]
    fn production_selector_admits_decoder_family_lengths_for_d3_and_d9() {
        for length in [12_480, 24_000, 48_000, 96_000, 192_000] {
            assert_eq!(
                production_dilation_for_shape(192, length, Conv1dK7Dilation::Three),
                Some(ResidueDilation::Three),
            );
            assert_eq!(
                production_dilation_for_shape(192, length, Conv1dK7Dilation::Nine),
                Some(ResidueDilation::Nine),
            );
        }
        for (channels, length, dilation) in [
            (192, 48_000, Conv1dK7Dilation::One),
            (192, 47_999, Conv1dK7Dilation::Three),
            (191, 48_000, Conv1dK7Dilation::Three),
            (64, 96_000, Conv1dK7Dilation::Nine),
        ] {
            assert_eq!(
                production_dilation_for_shape(channels, length, dilation),
                None
            );
        }
    }

    #[test]
    fn exact_geometry_and_accounting_are_fixed() {
        let d3 =
            ResidueLaunchGeometry::new(ResidueDilation::Three, C192, REFERENCE_LENGTH).unwrap();
        let d9 = ResidueLaunchGeometry::new(ResidueDilation::Nine, C192, REFERENCE_LENGTH).unwrap();
        for geometry in [d3, d9] {
            assert_eq!(geometry.packed_elements, 9_216_000);
            assert_eq!(geometry.temporary_bytes, 36_864_000);
            assert_eq!(geometry.pack_read_write_bytes, 73_728_000);
            assert_eq!(geometry.pack_workgroups, 36_000);
            assert_eq!(geometry.core_output_channel_tiles, 6);
            assert_eq!(geometry.core_workgroups, 1_134);
            assert_eq!(geometry.core_barriers, 54_432);
            assert_eq!(geometry.core_shared_bytes, 15_552);
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
                assert_eq!(
                    dilation.residue_prefix_q(REFERENCE_LENGTH, residue),
                    Some(prefix)
                );
                prefix += dilation
                    .residue_length(REFERENCE_LENGTH, residue)
                    .expect("in-range residue has a length");
            }
            assert_eq!(prefix, REFERENCE_LENGTH);
            assert_eq!(
                dilation.residue_length(REFERENCE_LENGTH, dilation.value()),
                None
            );
            assert_eq!(
                dilation.residue_prefix_q(REFERENCE_LENGTH, dilation.value()),
                None
            );
        }
        assert_eq!(
            (0..3)
                .map(|residue| {
                    ResidueDilation::Three
                        .residue_length(REFERENCE_LENGTH, residue)
                        .unwrap()
                })
                .collect::<Vec<_>>(),
            vec![16_000; 3],
        );
        assert_eq!(
            (0..9)
                .map(|residue| {
                    ResidueDilation::Nine
                        .residue_length(REFERENCE_LENGTH, residue)
                        .unwrap()
                })
                .collect::<Vec<_>>(),
            vec![
                5_334, 5_334, 5_334, 5_333, 5_333, 5_333, 5_333, 5_333, 5_333
            ],
        );
    }

    #[test]
    fn compact_index_blocks_are_exact_and_non_overlapping() {
        for dilation in [ResidueDilation::Three, ResidueDilation::Nine] {
            let first = dilation.packed_index(C192, REFERENCE_LENGTH, 0, 0).unwrap();
            let last = dilation
                .packed_index(C192, REFERENCE_LENGTH, C192 - 1, REFERENCE_LENGTH - 1)
                .unwrap();
            assert_eq!(first, 0);
            assert!(last < BATCH * C192 * REFERENCE_LENGTH);
            for residue in 0..dilation.value() {
                let length = dilation.residue_length(REFERENCE_LENGTH, residue).unwrap();
                let first_time = residue;
                let last_time = residue + (length - 1) * dilation.value();
                let block_start = dilation
                    .residue_prefix_q(REFERENCE_LENGTH, residue)
                    .unwrap()
                    * C192;
                let block_end = block_start + C192 * length - 1;
                assert_eq!(
                    dilation.packed_index(C192, REFERENCE_LENGTH, 0, first_time),
                    Some(block_start)
                );
                assert_eq!(
                    dilation.packed_index(C192, REFERENCE_LENGTH, C192 - 1, last_time),
                    Some(block_end),
                );
            }
        }
    }

    #[test]
    fn c96_geometry_and_production_selector_are_enabled() {
        let geometry = ResidueLaunchGeometry::new(ResidueDilation::Three, C96, 192_000).unwrap();
        assert_eq!(geometry.channels, C96);
        assert_eq!(geometry.packed_elements, 18_432_000);
        assert_eq!(geometry.temporary_bytes, 73_728_000);
        assert_eq!(geometry.core_output_channel_tiles, 3);
        assert_eq!(
            production_dilation_for_shape(C96, 192_000, Conv1dK7Dilation::Three),
            Some(ResidueDilation::Three),
        );
    }

    #[test]
    fn c384_geometry_and_production_selector_are_enabled() {
        let geometry = ResidueLaunchGeometry::new(ResidueDilation::Nine, C384, 24_000).unwrap();
        assert_eq!(geometry.channels, C384);
        assert_eq!(geometry.packed_elements, 9_216_000);
        assert_eq!(geometry.temporary_bytes, 36_864_000);
        assert_eq!(geometry.core_output_channel_tiles, 12);
        assert_eq!(
            production_dilation_for_shape(C384, 24_000, Conv1dK7Dilation::Nine),
            Some(ResidueDilation::Nine),
        );
    }

    #[test]
    fn residue_d1_boundary_map_matches_original_dilation() {
        for dilation in [ResidueDilation::Three, ResidueDilation::Nine] {
            let d = dilation.value() as isize;
            for time in [
                0usize,
                1,
                2,
                26,
                27,
                28,
                REFERENCE_LENGTH - 3,
                REFERENCE_LENGTH - 2,
                REFERENCE_LENGTH - 1,
            ] {
                let residue = time % dilation.value();
                let q = (time / dilation.value()) as isize;
                for tap in 0..KERNEL_SIZE {
                    let original = time as isize - 3 * d + tap as isize * d;
                    let residue_source_q = q - 3 + tap as isize;
                    if (0..REFERENCE_LENGTH as isize).contains(&original) {
                        assert_eq!(original % d, residue as isize);
                        assert_eq!(original, residue as isize + residue_source_q * d);
                    } else {
                        let residue_length =
                            dilation.residue_length(REFERENCE_LENGTH, residue).unwrap() as isize;
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
        assert!(shader.contains("let linear_group = group_id.y * DISPATCH_X + group_id.x;"));
        assert!(shader.contains("let q = time / DILATION;"));
        assert!(
            shader.contains(
                "let residue_prefix_q = residue * BASE_LENGTH + min(residue, REMAINDER);"
            )
        );
        assert!(shader.contains("packed_buf[packed_index] = input_buf[input_index];"));
    }

    #[test]
    fn long_pack_dispatches_split_across_the_second_dimension() {
        let limits = (65_535, 65_535, 65_535);
        assert_eq!(
            pack_dispatch_2d(36_000, limits),
            Some(PackDispatch2d { x: 36_000, y: 1 })
        );
        assert_eq!(
            pack_dispatch_2d(72_000, limits),
            Some(PackDispatch2d { x: 65_535, y: 2 })
        );
        assert_eq!(
            pack_dispatch_2d(144_000, limits),
            Some(PackDispatch2d { x: 65_535, y: 3 })
        );
        assert_eq!(pack_dispatch_2d(1, (0, 65_535, 65_535)), None);
    }

    #[test]
    fn core_shader_preserves_fma_order_and_direct_scatter() {
        let shader = include_str!("conv1d_k7_residue_d1_snake.wgsl");
        assert_eq!(shader.matches("@group(0) @binding(").count(), 5);
        assert_eq!(shader.matches("workgroupBarrier();").count(), 2);
        assert_eq!(shader.matches(" = fma(").count(), 56);
        for accumulator in [
            "accumulator_00",
            "accumulator_01",
            "accumulator_02",
            "accumulator_03",
            "accumulator_10",
            "accumulator_11",
            "accumulator_12",
            "accumulator_13",
        ] {
            assert_eq!(
                shader.matches(&format!("{accumulator} = fma(")).count(),
                KERNEL_SIZE,
            );
        }
        let tap_offsets = (0..KERNEL_SIZE)
            .map(|tap| shader.find(&format!("// tap {tap}")).expect("tap marker"))
            .collect::<Vec<_>>();
        assert!(tap_offsets.windows(2).all(|window| window[0] < window[1]));
        assert!(shader.contains("let source_q = i32(q_base + tile_q) - PADDING;"));
        assert_eq!(shader.matches("+ TIME_TILE / 2u").count(), 2);
        assert!(!shader.contains("input_base_2"));
        assert!(!shader.contains("input_base_3"));
        assert!(shader.contains("let output_time = residue + q * DILATION;"));
        assert!(shader.contains("return x + (sine * sine) / (a + 1e-9);"));
        assert!(!shader.contains("packed_input: array<vec4<f32>>"));
        assert!(!shader.contains("output_buf:   array<vec4<f32>>"));
    }

    #[test]
    fn core_source_template_is_complete_and_balanced() {
        let source = ResidueD1SnakeCoreKernel {
            dilation: ResidueDilation::Three,
            channels: C192,
            length: REFERENCE_LENGTH,
        }
        .source()
        .complete();
        assert!(!source.contains("{{"));
        let mut brace_depth = 0_i32;
        for byte in source.bytes() {
            match byte {
                b'{' => brace_depth += 1,
                b'}' => {
                    brace_depth -= 1;
                    assert!(brace_depth >= 0, "closing brace without an opening brace");
                }
                _ => {}
            }
        }
        assert_eq!(brace_depth, 0);
        assert!(source.contains("@compute @workgroup_size(32, 8, 1)"));
    }

    #[test]
    fn packed_weight_vectors_are_a_bijection_over_checkpoint_oik() {
        for channels in [C96, C192, C384] {
            let vectors = channels / 4;
            let mut seen = vec![false; channels * channels * KERNEL_SIZE];
            for input_channel in 0..channels {
                for tap in 0..KERNEL_SIZE {
                    for vector in 0..vectors {
                        let output_tile = vector / (OUTPUT_CHANNEL_TILE / 4);
                        let output_lane = vector % (OUTPUT_CHANNEL_TILE / 4);
                        for output_channel in [
                            output_tile * OUTPUT_CHANNEL_TILE + output_lane,
                            output_tile * OUTPUT_CHANNEL_TILE
                                + output_lane
                                + OUTPUT_CHANNEL_TILE / 4,
                            output_tile * OUTPUT_CHANNEL_TILE
                                + output_lane
                                + OUTPUT_CHANNEL_TILE / 2,
                            output_tile * OUTPUT_CHANNEL_TILE
                                + output_lane
                                + 3 * OUTPUT_CHANNEL_TILE / 4,
                        ] {
                            let source_index =
                                (output_channel * channels + input_channel) * KERNEL_SIZE + tap;
                            assert!(!seen[source_index]);
                            seen[source_index] = true;
                        }
                    }
                }
            }
            assert!(seen.into_iter().all(|visited| visited));
        }
    }

    #[test]
    fn packed_weight_shader_and_core_keep_invocation_owned_output_lanes() {
        let pack = include_str!("conv1d_k7_residue_weight_vector_pack.wgsl");
        assert!(pack.contains("output_channel_1 = output_channel_0 + 8u"));
        assert!(pack.contains("output_channel_3 = output_channel_0 + 24u"));
        assert!(pack.contains("packed_vectors[packed_index] = vec4<f32>"));

        let core = include_str!("conv1d_k7_residue_d1_snake.wgsl");
        assert!(core.contains("weight_buf:   array<vec4<f32>>"));
        assert!(core.contains("weight_vector = weight_tile[weight_base + 6u]"));
        assert!(core.contains("weight_0 = weight_vector.x"));
        assert!(core.contains("weight_3 = weight_vector.w"));
    }
}
