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
use burn::tensor::Shape;
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::conv1d_k7_tiled::Conv1dK7Dilation;
use super::precision::{KernelFloatPrecision, common_float_precision};

const BATCH: usize = 1;
const C96: usize = 96;
const C192: usize = 192;
const C384: usize = 384;
const KERNEL_SIZE: usize = 7;
const PADDING_D1: usize = 3;
const F16_INPUT_CHANNEL_TILE: usize = 4;
const SHORT_INPUT_CHANNEL_TILE: usize = 8;
const LONG_INPUT_CHANNEL_TILE: usize = 16;
const OUTPUT_CHANNEL_TILE: usize = 32;
const TIME_TILE: usize = 256;
const LOCAL_TIME_LANES: usize = 32;
const LOCAL_CHANNEL_LANES: usize = 8;
const CORE_WORKGROUP_SIZE: usize = LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES;
const PACK_WORKGROUP_SIZE: usize = 256;
const F32_BYTES: usize = size_of::<f32>();
const INPUT_SPAN_D1: usize = TIME_TILE + 2 * PADDING_D1;
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

const fn reference_stage_length(channels: usize) -> Option<usize> {
    match channels {
        C96 => Some(96_000),
        C192 => Some(48_000),
        C384 => Some(6_000),
        _ => None,
    }
}

const fn production_input_channel_tile(
    channels: usize,
    length: usize,
    precision: KernelFloatPrecision,
    dilation: ResidueDilation,
) -> Option<usize> {
    match reference_stage_length(channels) {
        Some(_)
            if matches!(precision, KernelFloatPrecision::F16)
                && matches!(dilation, ResidueDilation::One) =>
        {
            Some(LONG_INPUT_CHANNEL_TILE)
        }
        Some(_) if matches!(precision, KernelFloatPrecision::F16) => Some(F16_INPUT_CHANNEL_TILE),
        Some(reference) if length > reference => Some(LONG_INPUT_CHANNEL_TILE),
        Some(_) => Some(SHORT_INPUT_CHANNEL_TILE),
        None => None,
    }
}

/// Dilations admitted by this production kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum ResidueDilation {
    /// Ordinary dilation-one input is already in compact residue order.
    One = 1,
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
            Self::One => "residue-d1-d1",
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
    precision: KernelFloatPrecision,
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
    /// Length-selected d1 workgroup storage.
    pub core_shared_bytes: usize,
    /// Length/precision-aware input-channel tile. F16 d3/d9 keeps Cin4, F16 d1
    /// uses Cin16, and F32 selects Cin8/Cin16 by reference length.
    pub input_channel_tile: usize,
    /// Pack plus core dispatches.
    pub dispatches: usize,
}

impl ResidueLaunchGeometry {
    /// Construct checked geometry for one admitted decoder-family shape.
    pub fn new(dilation: ResidueDilation, channels: usize, length: usize) -> Option<Self> {
        Self::new_with_precision(dilation, channels, length, KernelFloatPrecision::F32)
    }

    fn new_with_precision(
        dilation: ResidueDilation,
        channels: usize,
        length: usize,
        precision: KernelFloatPrecision,
    ) -> Option<Self> {
        let input_channel_tile =
            production_input_channel_tile(channels, length, precision, dilation)?;
        if !matches!(channels, C96 | C192 | C384)
            || !channels.is_multiple_of(input_channel_tile)
            || !channels.is_multiple_of(OUTPUT_CHANNEL_TILE)
            || !decoder_stage_length_is_compatible(channels, length)
        {
            return None;
        }
        let packed_elements = BATCH.checked_mul(channels)?.checked_mul(length)?;
        let packed_bytes = packed_elements.checked_mul(precision.element_bytes())?;
        let max_residue_length = dilation.max_residue_length(length);
        let core_time_tiles = max_residue_length.div_ceil(TIME_TILE);
        let core_output_channel_tiles = channels / OUTPUT_CHANNEL_TILE;
        let core_workgroups = core_time_tiles
            .checked_mul(core_output_channel_tiles)
            .and_then(|value| value.checked_mul(dilation.value()))?;
        let input_tile_size = input_channel_tile.checked_mul(INPUT_SPAN_D1)?;
        let weight_vector_tile_size = (OUTPUT_CHANNEL_TILE / 4)
            .checked_mul(input_channel_tile)?
            .checked_mul(KERNEL_SIZE)?;
        let shared_bytes = input_tile_size
            .checked_add(4 * weight_vector_tile_size)?
            .checked_mul(F32_BYTES)?;
        let barriers_per_workgroup = 2 * (channels / input_channel_tile);
        Some(Self {
            dilation,
            precision,
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
            core_shared_bytes: shared_bytes,
            input_channel_tile,
            dispatches: 2,
        })
    }
}

#[derive(Debug)]
struct ResiduePackKernel {
    precision: KernelFloatPrecision,
    dilation: ResidueDilation,
    channels: usize,
    length: usize,
    dispatch_x: u32,
}

impl KernelSource for ResiduePackKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("conv1d_k7_residue_pack.wgsl"),
                include_str!("conv1d_k7_residue_pack_f16.wgsl"),
            )
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
        KernelId::new::<Self>().info((
            self.precision,
            self.dilation,
            self.channels,
            self.length,
            self.dispatch_x,
        ))
    }
}

#[derive(Debug)]
struct ResidueD1SnakeCoreKernel {
    precision: KernelFloatPrecision,
    dilation: ResidueDilation,
    channels: usize,
    length: usize,
}

#[derive(Debug)]
struct ResidueWeightVectorPackKernel {
    precision: KernelFloatPrecision,
    channels: usize,
    vector_elements: usize,
}

impl KernelSource for ResidueWeightVectorPackKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("conv1d_k7_residue_weight_vector_pack.wgsl"),
                include_str!("conv1d_k7_residue_weight_vector_pack_f16.wgsl"),
            )
            .register("channels", self.channels.to_string())
            .register("vector_elements", self.vector_elements.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.channels, self.vector_elements))
    }
}

impl KernelSource for ResidueD1SnakeCoreKernel {
    fn source(&self) -> SourceTemplate {
        let geometry = ResidueLaunchGeometry::new_with_precision(
            self.dilation,
            self.channels,
            self.length,
            self.precision,
        )
        .expect("kernel construction requires admitted residue geometry");
        let input_tile_size = geometry.input_channel_tile * INPUT_SPAN_D1;
        let weight_vector_tile_size =
            (OUTPUT_CHANNEL_TILE / 4) * geometry.input_channel_tile * KERNEL_SIZE;
        let source = match (self.precision, self.dilation) {
            (KernelFloatPrecision::F16, ResidueDilation::One) => {
                SourceTemplate::new(include_str!("conv1d_k7_residue_d1_snake_d1_f16.wgsl"))
            }
            _ => self.precision.source(
                include_str!("conv1d_k7_residue_d1_snake.wgsl"),
                include_str!("conv1d_k7_residue_d1_snake_f16.wgsl"),
            ),
        };
        source
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
            .register(
                "input_channel_tile",
                geometry.input_channel_tile.to_string(),
            )
            .register("input_span", INPUT_SPAN_D1.to_string())
            .register("input_tile_size", input_tile_size.to_string())
            .register(
                "weight_vector_tile_size",
                weight_vector_tile_size.to_string(),
            )
    }

    fn id(&self) -> KernelId {
        let input_channel_tile = production_input_channel_tile(
            self.channels,
            self.length,
            self.precision,
            self.dilation,
        )
        .expect("kernel identity requires admitted decoder channels");
        KernelId::new::<Self>().info((
            self.precision,
            self.dilation,
            self.channels,
            self.length,
            input_channel_tile,
            LOCAL_TIME_LANES,
            LOCAL_CHANNEL_LANES,
        ))
    }
}

fn exact_shape<const D: usize>(tensor: &CubeTensor<WgpuRuntime>, expected: [usize; D]) -> bool {
    tensor.meta.num_dims() == D && tensor.meta.shape().dims::<D>() == expected
}

fn binding_is_compatible(
    tensor: &CubeTensor<WgpuRuntime>,
    required_elements: usize,
    precision: KernelFloatPrecision,
    alignment: u64,
) -> bool {
    let Some(required_bytes) = required_elements
        .checked_mul(precision.element_bytes())
        .and_then(|bytes| u64::try_from(bytes).ok())
    else {
        return false;
    };
    let binding = tensor.handle.clone().binding();
    tensor.client.properties().memory.alignment >= alignment
        && tensor
            .client
            .properties()
            .memory
            .alignment
            .is_multiple_of(alignment)
        && binding.size_in_used() >= required_bytes
        && binding.offset_start.unwrap_or(0).is_multiple_of(alignment)
}

fn exact_input_contract(
    input: &CubeTensor<WgpuRuntime>,
    channels: usize,
    length: usize,
    precision: KernelFloatPrecision,
) -> bool {
    let Some(required_elements) = channels.checked_mul(length) else {
        return false;
    };
    exact_shape(input, [BATCH, channels, length])
        && input.is_contiguous()
        && binding_is_compatible(
            input,
            required_elements,
            precision,
            precision.element_bytes() as u64,
        )
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
        && packed.dtype == geometry.precision.dtype()
        && packed.device == reference.device
        && packed.is_contiguous()
        && binding_is_compatible(
            packed,
            geometry.packed_elements,
            geometry.precision,
            geometry.precision.element_bytes() as u64,
        )
}

fn packed_weight_vector_contract_is_compatible(
    packed: &CubeTensor<WgpuRuntime>,
    reference: &CubeTensor<WgpuRuntime>,
    channels: usize,
    precision: KernelFloatPrecision,
) -> bool {
    let Some(required_elements) = channels
        .checked_mul(KERNEL_SIZE)
        .and_then(|elements| elements.checked_mul(channels))
    else {
        return false;
    };
    exact_shape(packed, [channels, KERNEL_SIZE, channels])
        && packed.dtype == precision.dtype()
        && packed.device == reference.device
        && packed.is_contiguous()
        && binding_is_compatible(
            packed,
            required_elements,
            precision,
            (4 * precision.element_bytes()) as u64,
        )
}

/// Select measured dilations for exact decoder-family C96/C192/C384 lengths.
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
        Conv1dK7Dilation::One => Some(ResidueDilation::One),
        Conv1dK7Dilation::Three => Some(ResidueDilation::Three),
        Conv1dK7Dilation::Nine => Some(ResidueDilation::Nine),
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
    let Some(precision) =
        common_float_precision([input.dtype, weight.dtype, bias.dtype, alpha.dtype])
    else {
        return false;
    };
    let Some(geometry) =
        ResidueLaunchGeometry::new_with_precision(dilation, channels, length, precision)
    else {
        return false;
    };
    batch == BATCH
        && (dilation != ResidueDilation::One || precision == KernelFloatPrecision::F16)
        && (dilation != ResidueDilation::One || length.is_multiple_of(4))
        && exact_input_contract(input, channels, length, precision)
        && packed_weight_vector_contract_is_compatible(weight, input, channels, precision)
        && exact_shape(bias, [channels])
        && exact_shape(alpha, [BATCH, channels, 1])
        && [weight, bias, alpha]
            .into_iter()
            .all(|tensor| tensor.device == input.device && tensor.is_contiguous())
        && binding_is_compatible(bias, channels, precision, precision.element_bytes() as u64)
        && binding_is_compatible(alpha, channels, precision, precision.element_bytes() as u64)
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
    let precision = common_float_precision([input.dtype])?;
    let geometry =
        ResidueLaunchGeometry::new_with_precision(dilation, channels, length, precision)?;
    if !exact_input_contract(&input, channels, length, precision)
        || !device_supports_geometry(&input, geometry)
    {
        return None;
    }

    let client = input.client.clone();
    let dispatch = pack_dispatch_2d(
        geometry.pack_workgroups,
        client.properties().hardware.max_cube_count,
    )?;
    let packed_handle = client.empty(geometry.temporary_bytes);
    if packed_handle.size_in_used() < u64::try_from(geometry.temporary_bytes).ok()?
        || !packed_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(precision.element_bytes() as u64)
    {
        return None;
    }
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([geometry.packed_elements]),
        packed_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResiduePackKernel {
                precision,
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
    let precision = common_float_precision([weight.dtype])?;
    if output_channels != input_channels
        || !matches!(output_channels, C96 | C192 | C384)
        || kernel_size != KERNEL_SIZE
        || !weight.is_contiguous()
    {
        return None;
    }
    let channels = output_channels;
    let scalar_elements = channels.checked_mul(channels)?.checked_mul(KERNEL_SIZE)?;
    let vector_elements = scalar_elements / 4;
    let output_bytes = scalar_elements.checked_mul(precision.element_bytes())?;
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

    if !binding_is_compatible(
        &weight,
        scalar_elements,
        precision,
        precision.element_bytes() as u64,
    ) {
        return None;
    }

    let client = weight.client.clone();
    let handle = client.empty(output_bytes);
    if handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of((4 * precision.element_bytes()) as u64)
    {
        return None;
    }
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        weight.device.clone(),
        Shape::from([channels, KERNEL_SIZE, channels]),
        handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResidueWeightVectorPackKernel {
                precision,
                channels,
                vector_elements,
            },
            CubeDim::new_1d(WEIGHT_VECTOR_PACK_WORKGROUP_SIZE as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(weight.handle.clone().binding())
        .with_buffer(packed.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(vector_workgroups), bindings);
    packed_weight_vector_contract_is_compatible(&packed, &weight, channels, precision)
        .then_some(packed)
}

/// Launch only the residue-d1 convolution/Snake core from a validated pack.
pub fn conv1d_k7_residue_d1_snake_from_packed_wgsl(
    packed: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: ResidueDilation,
    channels: usize,
    length: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    let precision = common_float_precision([packed.dtype, weight.dtype, bias.dtype, alpha.dtype])?;
    let geometry =
        ResidueLaunchGeometry::new_with_precision(dilation, channels, length, precision)?;
    if !packed_contract_is_compatible(&packed, &weight, geometry)
        || !packed_weight_vector_contract_is_compatible(&weight, &packed, channels, precision)
        || !exact_shape(&bias, [channels])
        || !exact_shape(&alpha, [BATCH, channels, 1])
        || [&weight, &bias, &alpha]
            .into_iter()
            .any(|tensor| tensor.device != packed.device || !tensor.is_contiguous())
        || !binding_is_compatible(&bias, channels, precision, precision.element_bytes() as u64)
        || !binding_is_compatible(
            &alpha,
            channels,
            precision,
            precision.element_bytes() as u64,
        )
        || !device_supports_geometry(&packed, geometry)
    {
        return None;
    }

    let client = packed.client.clone();
    let output_handle = client.empty(geometry.temporary_bytes);
    if output_handle.size_in_used() < u64::try_from(geometry.temporary_bytes).ok()?
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(precision.element_bytes() as u64)
    {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        packed.device.clone(),
        Shape::from([BATCH, channels, length]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ResidueD1SnakeCoreKernel {
                precision,
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
    let precision = common_float_precision([input.dtype, weight.dtype, bias.dtype, alpha.dtype])?;
    if !packed_weight_vector_contract_is_compatible(&weight, &input, channels, precision)
        || !conv1d_k7_residue_d1_snake_contract_is_compatible(
            &input, &weight, &bias, &alpha, dilation,
        )
    {
        return None;
    }
    let packed = if dilation == ResidueDilation::One {
        CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::from([channels.checked_mul(length)?]),
            input.handle.clone(),
            input.dtype,
        )
    } else {
        try_pack_conv1d_k7_residue_input_wgsl(input, dilation)?
    };
    conv1d_k7_residue_d1_snake_from_packed_wgsl(
        packed, weight, bias, alpha, dilation, channels, length,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE_LENGTH: usize = 48_000;

    #[test]
    fn production_selector_admits_decoder_family_lengths_for_d1_d3_and_d9() {
        for length in [12_480, 24_000, 48_000, 96_000, 192_000] {
            assert_eq!(
                production_dilation_for_shape(192, length, Conv1dK7Dilation::One),
                Some(ResidueDilation::One),
            );
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
    fn f16_dilation_one_is_a_zero_copy_ncl_view_with_cin16() {
        let geometry = ResidueLaunchGeometry::new_with_precision(
            ResidueDilation::One,
            C192,
            REFERENCE_LENGTH,
            KernelFloatPrecision::F16,
        )
        .expect("released F16 d1 geometry");
        assert_eq!(geometry.input_channel_tile, LONG_INPUT_CHANNEL_TILE);
        assert_eq!(geometry.core_residues, 1);
        for (channel, time) in [(0, 0), (17, 12_345), (191, REFERENCE_LENGTH - 1)] {
            assert_eq!(
                ResidueDilation::One.packed_index(C192, REFERENCE_LENGTH, channel, time,),
                Some(channel * REFERENCE_LENGTH + time),
            );
        }
        let source = include_str!("conv1d_k7_residue_d1_snake_d1_f16.wgsl");
        assert!(source.contains("output_buf: array<vec4<f16>>"));
        assert_eq!(source.matches("output_buf[output_index / 4u] =").count(), 1);
        assert!(!source.contains("output_buf[output_base + output_time]"));
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
            assert_eq!(geometry.input_channel_tile, SHORT_INPUT_CHANNEL_TILE);
            assert_eq!(geometry.dispatches, 2);
        }
        assert_eq!((d3.core_time_tiles, d3.core_residues), (63, 3));
        assert_eq!((d9.core_time_tiles, d9.core_residues), (21, 9));
    }

    #[test]
    fn f16_cin4_halves_storage_and_shared_memory_with_explicit_barrier_tradeoff() {
        let f16 = ResidueLaunchGeometry::new_with_precision(
            ResidueDilation::Three,
            C192,
            REFERENCE_LENGTH,
            KernelFloatPrecision::F16,
        )
        .expect("released F16 geometry");
        assert_eq!(f16.temporary_bytes, 18_432_000);
        assert_eq!(f16.pack_read_write_bytes, 36_864_000);
        assert_eq!(f16.input_channel_tile, F16_INPUT_CHANNEL_TILE);
        assert_eq!(f16.core_shared_bytes, 7_776);
        assert_eq!(f16.core_barriers, 108_864);
    }

    #[test]
    fn long_shapes_select_cin16_and_halved_per_workgroup_barriers() {
        for (channels, reference, long) in [
            (C384, 6_000, 6_120),
            (C192, 48_000, 48_960),
            (C96, 96_000, 97_920),
        ] {
            let short = ResidueLaunchGeometry::new(ResidueDilation::Three, channels, reference)
                .expect("reference decoder shape");
            let long = ResidueLaunchGeometry::new(ResidueDilation::Three, channels, long)
                .expect("long decoder shape");
            assert_eq!(short.input_channel_tile, SHORT_INPUT_CHANNEL_TILE);
            assert_eq!(short.core_shared_bytes, 15_552);
            assert_eq!(long.input_channel_tile, LONG_INPUT_CHANNEL_TILE);
            assert_eq!(long.core_shared_bytes, 31_104);
            assert_eq!(
                2 * (channels / long.input_channel_tile),
                channels / SHORT_INPUT_CHANNEL_TILE,
            );
        }
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
    }

    #[test]
    fn core_source_template_is_complete_and_balanced() {
        let source = ResidueD1SnakeCoreKernel {
            precision: KernelFloatPrecision::F32,
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
}
