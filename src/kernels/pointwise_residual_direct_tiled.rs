//! Production direct f32 pointwise-residual kernels for the codec tail.
//!
//! The fixed T64/O96/K32 winner consumes a one-time contiguous `[1,K,O]`
//! weight directly, evaluates the released C192 and C96 projections at every
//! decoder length in NCL order, and fuses the bias, shortcut, and (where needed)
//! next-unit Snake boundary into one dispatch. A guarded final time tile extends
//! the measured C192/C96 path to the released C384 decoder stage. Every unsupported shape,
//! physical layout, dtype, device, or resource limit is rejected before any
//! allocation or dispatch so the caller can retain the established finalizer
//! and generic fallbacks.

use core::fmt;

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::precision::{KernelFloatPrecision, common_float_precision};

pub const TIME_TILE_T64: usize = 64;
pub const OUTPUT_TILE_O96: usize = 96;
pub const REDUCTION_TILE_K32: usize = 32;
pub const WORKGROUP_SIZE: u32 = 256;
pub const RELEASED_SHAPES: [(usize, usize); 2] = [(192, 48_000), (96, 96_000)];

const BATCH: usize = 1;
const RAW_BINDINGS: u32 = 5;
const ACTIVATED_BINDINGS: u32 = 6;
const PAIR_BINDINGS: u32 = 7;
const F32_BYTES: usize = size_of::<f32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PointwiseOutputContract {
    Raw,
    Pair,
    ActivatedOnly,
}

impl PointwiseOutputContract {
    const fn required_bindings(self) -> u32 {
        match self {
            Self::Raw => RAW_BINDINGS,
            Self::Pair => PAIR_BINDINGS,
            Self::ActivatedOnly => ACTIVATED_BINDINGS,
        }
    }

    const fn writes_raw(self) -> bool {
        !matches!(self, Self::ActivatedOnly)
    }

    const fn writes_activated(self) -> bool {
        !matches!(self, Self::Raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PointwiseShapeSet {
    Released,
    #[cfg(feature = "profile")]
    IncludeC768BlockBoundary,
}

impl PointwiseShapeSet {
    const fn supports_channels(self, channels: usize) -> bool {
        match self {
            Self::Released => matches!(channels, 384 | 192 | 96),
            #[cfg(feature = "profile")]
            Self::IncludeC768BlockBoundary => matches!(channels, 768 | 384 | 192 | 96),
        }
    }
}

fn supported_decoder_shape(
    channels: usize,
    length: usize,
    tile: PointwiseKTile,
    shape_set: PointwiseShapeSet,
) -> bool {
    shape_set.supports_channels(channels)
        && length > 0
        && channels.is_multiple_of(tile.reduction())
        && channels.is_multiple_of(tile.output_tile())
}

/// The only production tile accepted by the isolated production-weight screen.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PointwiseKTile {
    T64O96K32Vec4,
}

impl PointwiseKTile {
    pub const PRODUCTION: Self = Self::T64O96K32Vec4;

    pub const fn time_tile(self) -> usize {
        TIME_TILE_T64
    }

    pub const fn reduction(self) -> usize {
        REDUCTION_TILE_K32
    }

    pub const fn output_tile(self) -> usize {
        OUTPUT_TILE_O96
    }

    pub const fn workgroup_x(self) -> u32 {
        32
    }

    pub const fn workgroup_y(self) -> u32 {
        8
    }

    pub const fn local_time_outputs(self) -> usize {
        self.time_tile() / self.workgroup_x() as usize
    }

    pub const fn local_output_channels(self) -> usize {
        self.output_tile() / self.workgroup_y() as usize
    }

    pub const fn vector_width(self) -> usize {
        4
    }

    pub const fn outputs_per_thread(self) -> usize {
        self.local_time_outputs() * self.local_output_channels()
    }

    pub const fn fma_statements_per_reduction_step(self) -> usize {
        self.outputs_per_thread() / self.vector_width()
    }

    /// One padding element per time row avoids shared-bank aliasing.
    pub const fn input_stride(self) -> usize {
        self.reduction() + 1
    }

    pub const fn input_tile_elements(self) -> usize {
        self.time_tile() * self.reduction()
    }

    pub const fn input_tile_storage_elements(self) -> usize {
        self.time_tile() * self.input_stride()
    }

    pub const fn weight_tile_elements(self) -> usize {
        self.reduction() * self.output_tile()
    }

    pub const fn shared_memory_bytes(self) -> usize {
        (self.input_tile_storage_elements() + self.weight_tile_elements()) * F32_BYTES
    }

    pub const fn label(self) -> &'static str {
        "t64_o96_k32_wg32x8_vec4"
    }
}

/// Failure before an exact-shape candidate can allocate or dispatch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PointwiseDirectError {
    message: String,
}

impl PointwiseDirectError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PointwiseDirectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PointwiseDirectError {}

/// Owned tensors common to the raw and prepared-pair launchers.
#[derive(Clone, Debug)]
pub struct PointwiseResidualDirectInputs {
    input_ncl: CubeTensor<WgpuRuntime>,
    input_layout: PointwiseInputLayout,
    packed_weight_kco: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    residual_ncl: CubeTensor<WgpuRuntime>,
    residual_layout: PointwiseInputLayout,
}

impl PointwiseResidualDirectInputs {
    pub fn new(
        input_ncl: CubeTensor<WgpuRuntime>,
        packed_weight_kco: CubeTensor<WgpuRuntime>,
        bias: CubeTensor<WgpuRuntime>,
        residual_ncl: CubeTensor<WgpuRuntime>,
    ) -> Self {
        Self {
            input_ncl,
            input_layout: PointwiseInputLayout::Ncl,
            packed_weight_kco,
            bias,
            residual_ncl,
            residual_layout: PointwiseInputLayout::Ncl,
        }
    }

    /// Construct inputs whose projection activation is physically contiguous
    /// NHWC (`[1, length, channels]`), while the residual remains NCL.
    pub fn new_nhwc(
        input_nhwc: CubeTensor<WgpuRuntime>,
        packed_weight_kco: CubeTensor<WgpuRuntime>,
        bias: CubeTensor<WgpuRuntime>,
        residual_ncl: CubeTensor<WgpuRuntime>,
    ) -> Self {
        Self {
            input_ncl: input_nhwc,
            input_layout: PointwiseInputLayout::Nhwc,
            packed_weight_kco,
            bias,
            residual_ncl,
            residual_layout: PointwiseInputLayout::Ncl,
        }
    }

    /// Construct an all-NHWC residual-state boundary. This is profile-only
    /// until the complete decoder block layout transition is accepted.
    #[cfg(feature = "profile")]
    pub fn new_nhwc_state(
        input_nhwc: CubeTensor<WgpuRuntime>,
        packed_weight_kco: CubeTensor<WgpuRuntime>,
        bias: CubeTensor<WgpuRuntime>,
        residual_nhwc: CubeTensor<WgpuRuntime>,
    ) -> Self {
        Self {
            input_ncl: input_nhwc,
            input_layout: PointwiseInputLayout::Nhwc,
            packed_weight_kco,
            bias,
            residual_ncl: residual_nhwc,
            residual_layout: PointwiseInputLayout::Nhwc,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PointwiseInputLayout {
    Ncl,
    Nhwc,
}

impl PointwiseInputLayout {
    const fn tile_input_channel(self) -> &'static str {
        match self {
            Self::Ncl => "load_index / TIME_TILE",
            Self::Nhwc => "load_index % K_TILE",
        }
    }

    const fn tile_time(self) -> &'static str {
        match self {
            Self::Ncl => "load_index - tile_input_channel * TIME_TILE",
            Self::Nhwc => "load_index / K_TILE",
        }
    }

    const fn input_index(self) -> &'static str {
        match self {
            Self::Ncl => "input_channel * LENGTH + time",
            Self::Nhwc => "time * CHANNELS + input_channel",
        }
    }

    const fn output_index(self) -> &'static str {
        match self {
            Self::Ncl => "output_channel * LENGTH + time",
            Self::Nhwc => "time * CHANNELS + output_channel",
        }
    }
}

/// Raw residual plus its next-unit Snake activation from one dispatch.
#[derive(Debug)]
pub struct PointwiseResidualDirectPair {
    raw_ncl: CubeTensor<WgpuRuntime>,
    activated_ncl: CubeTensor<WgpuRuntime>,
}

/// Raw residual plus its next-unit Snake activation in compact residue order.
#[derive(Debug)]
pub struct PointwiseResidualDirectResiduePair {
    raw_ncl: CubeTensor<WgpuRuntime>,
    activated_residue: CubeTensor<WgpuRuntime>,
}

impl PointwiseResidualDirectResiduePair {
    pub fn into_tensors(self) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
        (self.raw_ncl, self.activated_residue)
    }
}

impl PointwiseResidualDirectPair {
    pub fn into_tensors(self) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
        (self.raw_ncl, self.activated_ncl)
    }
}

#[derive(Clone, Copy, Debug)]
struct LaunchGeometry {
    precision: KernelFloatPrecision,
    channels: usize,
    length: usize,
    output_bytes: usize,
    time_workgroups: u32,
    output_workgroups: u32,
}

#[derive(Debug)]
struct PointwiseDirectRawKernel {
    geometry: LaunchGeometry,
    tile: PointwiseKTile,
    input_layout: PointwiseInputLayout,
    residual_layout: PointwiseInputLayout,
    output_layout: PointwiseInputLayout,
}

#[derive(Debug)]
struct PointwiseDirectPairKernel {
    geometry: LaunchGeometry,
    tile: PointwiseKTile,
    raw_layout: PointwiseInputLayout,
    activated_layout: PointwiseInputLayout,
    input_layout: PointwiseInputLayout,
    residual_layout: PointwiseInputLayout,
}

#[derive(Debug)]
struct PointwiseDirectActivatedKernel {
    geometry: LaunchGeometry,
    tile: PointwiseKTile,
    input_layout: PointwiseInputLayout,
    residual_layout: PointwiseInputLayout,
}

#[derive(Debug)]
struct PointwiseDirectResiduePairKernel {
    geometry: LaunchGeometry,
    tile: PointwiseKTile,
    dilation: crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation,
}

fn source_template(
    source: &'static str,
    geometry: LaunchGeometry,
    tile: PointwiseKTile,
) -> SourceTemplate {
    SourceTemplate::new(source)
        .register("channels", geometry.channels.to_string())
        .register("length", geometry.length.to_string())
        .register("k_tile", tile.reduction().to_string())
        .register("input_stride", tile.input_stride().to_string())
        .register(
            "input_tile_elements",
            tile.input_tile_elements().to_string(),
        )
        .register(
            "input_tile_storage_elements",
            tile.input_tile_storage_elements().to_string(),
        )
        .register(
            "weight_tile_elements",
            tile.weight_tile_elements().to_string(),
        )
        .register(
            "weight_vector_tile_elements",
            (tile.weight_tile_elements() / 4).to_string(),
        )
}

impl KernelSource for PointwiseDirectRawKernel {
    fn source(&self) -> SourceTemplate {
        let source = match self.geometry.precision {
            KernelFloatPrecision::F32 => {
                include_str!("pointwise_residual_direct_t64_o96_vec4_raw.wgsl")
            }
            KernelFloatPrecision::F16 => {
                include_str!("pointwise_residual_direct_t64_o96_vec4_raw_f16.wgsl")
            }
        };
        source_template(source, self.geometry, self.tile)
            .register("tile_input_channel", self.input_layout.tile_input_channel())
            .register("tile_time", self.input_layout.tile_time())
            .register("input_index", self.input_layout.input_index())
            .register("residual_index", self.residual_layout.output_index())
            .register("raw_index", self.output_layout.output_index())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.geometry.channels,
            self.geometry.precision,
            self.geometry.length,
            self.tile.time_tile(),
            self.tile.reduction(),
            self.tile.output_tile(),
            self.tile.workgroup_x(),
            self.tile.workgroup_y(),
            self.input_layout,
            self.residual_layout,
            self.output_layout,
        ))
    }
}

impl KernelSource for PointwiseDirectPairKernel {
    fn source(&self) -> SourceTemplate {
        let source = match self.geometry.precision {
            KernelFloatPrecision::F32 => {
                include_str!("pointwise_residual_direct_t64_o96_vec4_pair.wgsl")
            }
            KernelFloatPrecision::F16 => {
                include_str!("pointwise_residual_direct_t64_o96_vec4_pair_f16.wgsl")
            }
        };
        source_template(source, self.geometry, self.tile)
            .register("raw_index", self.raw_layout.output_index())
            .register("activated_index", self.activated_layout.output_index())
            .register("tile_input_channel", self.input_layout.tile_input_channel())
            .register("tile_time", self.input_layout.tile_time())
            .register("input_index", self.input_layout.input_index())
            .register("residual_index", self.residual_layout.output_index())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.geometry.channels,
            self.geometry.precision,
            self.geometry.length,
            self.tile.time_tile(),
            self.tile.reduction(),
            self.tile.output_tile(),
            self.tile.workgroup_x(),
            self.tile.workgroup_y(),
            self.raw_layout,
            self.activated_layout,
            self.input_layout,
            self.residual_layout,
        ))
    }
}

impl KernelSource for PointwiseDirectActivatedKernel {
    fn source(&self) -> SourceTemplate {
        let source = match self.geometry.precision {
            KernelFloatPrecision::F32 => {
                include_str!("pointwise_residual_direct_t64_o96_vec4_activated.wgsl")
            }
            KernelFloatPrecision::F16 => {
                include_str!("pointwise_residual_direct_t64_o96_vec4_activated_f16.wgsl")
            }
        };
        source_template(source, self.geometry, self.tile)
            .register("tile_input_channel", self.input_layout.tile_input_channel())
            .register("tile_time", self.input_layout.tile_time())
            .register("input_index", self.input_layout.input_index())
            .register("residual_index", self.residual_layout.output_index())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.geometry.channels,
            self.geometry.precision,
            self.geometry.length,
            self.tile.time_tile(),
            self.tile.reduction(),
            self.tile.output_tile(),
            self.tile.workgroup_x(),
            self.tile.workgroup_y(),
            self.input_layout,
            self.residual_layout,
        ))
    }
}

impl KernelSource for PointwiseDirectResiduePairKernel {
    fn source(&self) -> SourceTemplate {
        source_template(
            include_str!("pointwise_residual_direct_t64_o96_vec4_pair_residue_f16.wgsl"),
            self.geometry,
            self.tile,
        )
        .register("dilation", self.dilation.value().to_string())
        .register(
            "base_length",
            self.dilation.base_length(self.geometry.length).to_string(),
        )
        .register(
            "remainder",
            self.dilation.remainder(self.geometry.length).to_string(),
        )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.geometry.channels,
            self.geometry.precision,
            self.geometry.length,
            self.dilation,
            self.tile.time_tile(),
            self.tile.reduction(),
            self.tile.output_tile(),
            self.tile.workgroup_x(),
            self.tile.workgroup_y(),
        ))
    }
}

fn exact_strides<const D: usize>(tensor: &CubeTensor<WgpuRuntime>) -> [usize; D] {
    let strides = tensor.meta.strides();
    std::array::from_fn(|index| strides[index])
}

fn validate_tensor(
    name: &str,
    tensor: &CubeTensor<WgpuRuntime>,
    rank: usize,
    reference: &CubeTensor<WgpuRuntime>,
) -> Result<(), PointwiseDirectError> {
    if KernelFloatPrecision::from_dtype(tensor.dtype).is_none() {
        return Err(PointwiseDirectError::new(format!(
            "{name} must be f32 or f16, got {:?}",
            tensor.dtype
        )));
    }
    if tensor.meta.num_dims() != rank {
        return Err(PointwiseDirectError::new(format!(
            "{name} must have rank {rank}, got {}",
            tensor.meta.num_dims()
        )));
    }
    if tensor.device != reference.device {
        return Err(PointwiseDirectError::new(format!(
            "{name} is on a different WGPU device"
        )));
    }
    if !tensor.is_contiguous() {
        return Err(PointwiseDirectError::new(format!(
            "{name} must be contiguous, got strides {:?}",
            tensor.meta.strides()
        )));
    }
    Ok(())
}

fn checked_bytes(
    precision: KernelFloatPrecision,
    elements: usize,
    label: &str,
) -> Result<usize, PointwiseDirectError> {
    elements
        .checked_mul(precision.element_bytes())
        .ok_or_else(|| PointwiseDirectError::new(format!("{label} byte count overflows usize")))
}

fn tensor_contract_diagnostic(
    name: &str,
    tensor: &CubeTensor<WgpuRuntime>,
    reference: &CubeTensor<WgpuRuntime>,
) -> String {
    format!(
        "{name}[shape={:?},strides={:?},dtype={:?},contiguous={},same_device={}]",
        tensor.meta.shape(),
        tensor.meta.strides(),
        tensor.dtype,
        tensor.is_contiguous(),
        tensor.device == reference.device,
    )
}

fn contract_diagnostic(
    inputs: &PointwiseResidualDirectInputs,
    alpha: Option<&CubeTensor<WgpuRuntime>>,
    tile: PointwiseKTile,
    output_contract: PointwiseOutputContract,
) -> String {
    let reference = &inputs.input_ncl;
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    let alpha = alpha.map_or_else(
        || "alpha[absent]".to_owned(),
        |alpha| tensor_contract_diagnostic("alpha", alpha, reference),
    );
    format!(
        "contract_snapshot tile={} required_bindings={} required_shared={}B required_workgroup=[{},{},1] {}; {}; {}; {}; {}; device[bindings={},shared={}B,units={},dim={:?},count={:?},page={}B]",
        tile.label(),
        output_contract.required_bindings(),
        tile.shared_memory_bytes(),
        tile.workgroup_x(),
        tile.workgroup_y(),
        tensor_contract_diagnostic("input_ncl", &inputs.input_ncl, reference),
        tensor_contract_diagnostic("packed_weight_kco", &inputs.packed_weight_kco, reference,),
        tensor_contract_diagnostic("bias", &inputs.bias, reference),
        tensor_contract_diagnostic("residual_ncl", &inputs.residual_ncl, reference),
        alpha,
        hardware.max_bindings,
        hardware.max_shared_memory_size,
        hardware.max_units_per_cube,
        hardware.max_cube_dim,
        hardware.max_cube_count,
        properties.memory.max_page_size,
    )
}

fn validate_contract_inner(
    inputs: &PointwiseResidualDirectInputs,
    alpha: Option<&CubeTensor<WgpuRuntime>>,
    tile: PointwiseKTile,
    output_contract: PointwiseOutputContract,
    shape_set: PointwiseShapeSet,
) -> Result<LaunchGeometry, PointwiseDirectError> {
    if output_contract.writes_activated() != alpha.is_some() {
        return Err(PointwiseDirectError::new(
            "activated output requires exactly one alpha binding",
        ));
    }
    let reference = &inputs.input_ncl;
    validate_tensor("input_ncl", &inputs.input_ncl, 3, reference)?;
    validate_tensor("packed_weight_kco", &inputs.packed_weight_kco, 3, reference)?;
    validate_tensor("bias", &inputs.bias, 1, reference)?;
    validate_tensor("residual_ncl", &inputs.residual_ncl, 3, reference)?;
    if let Some(alpha) = alpha {
        validate_tensor("alpha", alpha, 3, reference)?;
    }
    let precision = common_float_precision(
        [
            inputs.input_ncl.dtype,
            inputs.packed_weight_kco.dtype,
            inputs.bias.dtype,
            inputs.residual_ncl.dtype,
        ]
        .into_iter()
        .chain(alpha.map(|tensor| tensor.dtype)),
    )
    .ok_or_else(|| PointwiseDirectError::new("all bindings must share f32 or f16 dtype"))?;

    let input_shape = inputs.input_ncl.meta.shape().dims::<3>();
    let [batch, channels, length] = match inputs.input_layout {
        PointwiseInputLayout::Ncl => input_shape,
        PointwiseInputLayout::Nhwc => [input_shape[0], input_shape[2], input_shape[1]],
    };
    if batch != BATCH || !supported_decoder_shape(channels, length, tile, shape_set) {
        return Err(PointwiseDirectError::new(format!(
            "unsupported input shape {input_shape:?}; expected B=1, C in the {:?} shape set, and positive L with guarded T{} tails",
            shape_set,
            tile.time_tile(),
        )));
    }
    let elements = channels
        .checked_mul(length)
        .ok_or_else(|| PointwiseDirectError::new("C*L overflows usize"))?;
    let weight_elements = channels
        .checked_mul(channels)
        .ok_or_else(|| PointwiseDirectError::new("C*C overflows usize"))?;

    let exact_ncl_strides = [elements, length, 1];
    let expected_input_strides = match inputs.input_layout {
        PointwiseInputLayout::Ncl => exact_ncl_strides,
        PointwiseInputLayout::Nhwc => [elements, channels, 1],
    };
    if exact_strides::<3>(&inputs.input_ncl) != expected_input_strides {
        return Err(PointwiseDirectError::new(format!(
            "input must have exact {:?} strides {expected_input_strides:?}, got {:?}",
            inputs.input_layout,
            exact_strides::<3>(&inputs.input_ncl)
        )));
    }
    if inputs.packed_weight_kco.meta.shape().dims::<3>() != [1, channels, channels]
        || exact_strides::<3>(&inputs.packed_weight_kco) != [weight_elements, channels, 1]
    {
        return Err(PointwiseDirectError::new(format!(
            "packed_weight_kco must be exact contiguous [1,{channels},{channels}] with physical k*C+o rows"
        )));
    }
    if inputs.bias.meta.shape().dims::<1>() != [channels] || exact_strides::<1>(&inputs.bias) != [1]
    {
        return Err(PointwiseDirectError::new(format!(
            "bias must be exact contiguous [{channels}]"
        )));
    }
    let expected_residual_shape = match inputs.residual_layout {
        PointwiseInputLayout::Ncl => [BATCH, channels, length],
        PointwiseInputLayout::Nhwc => [BATCH, length, channels],
    };
    let expected_residual_strides = match inputs.residual_layout {
        PointwiseInputLayout::Ncl => exact_ncl_strides,
        PointwiseInputLayout::Nhwc => [elements, channels, 1],
    };
    if inputs.residual_ncl.meta.shape().dims::<3>() != expected_residual_shape
        || exact_strides::<3>(&inputs.residual_ncl) != expected_residual_strides
    {
        return Err(PointwiseDirectError::new(format!(
            "residual must be exact contiguous {:?} shape {expected_residual_shape:?} with strides {expected_residual_strides:?}",
            inputs.residual_layout,
        )));
    }
    if let Some(alpha) = alpha
        && (alpha.meta.shape().dims::<3>() != [1, channels, 1]
            || exact_strides::<3>(alpha) != [channels, 1, 1])
    {
        return Err(PointwiseDirectError::new(format!(
            "alpha must be exact contiguous [1,{channels},1]"
        )));
    }

    let output_bytes = checked_bytes(precision, elements, "output")?;
    let time_workgroups = u32::try_from(length.div_ceil(tile.time_tile()))
        .map_err(|_| PointwiseDirectError::new("time workgroup count exceeds u32"))?;
    let output_workgroups = u32::try_from(channels.div_ceil(tile.output_tile()))
        .map_err(|_| PointwiseDirectError::new("output workgroup count exceeds u32"))?;
    let properties = reference.client.properties();
    let page_limit = properties.memory.max_page_size;
    let mut buffers = vec![
        ("input_ncl", elements),
        ("packed_weight_kco", weight_elements),
        ("bias", channels),
        ("residual_ncl", elements),
    ];
    if output_contract.writes_raw() {
        buffers.push(("raw_ncl", elements));
    }
    if output_contract.writes_activated() {
        buffers.push(("alpha", channels));
        buffers.push(("activated_ncl", elements));
    }
    for (name, buffer_elements) in buffers {
        let bytes = checked_bytes(precision, buffer_elements, name)?;
        let bytes = u64::try_from(bytes)
            .map_err(|_| PointwiseDirectError::new(format!("{name} byte count exceeds u64")))?;
        if bytes > page_limit {
            return Err(PointwiseDirectError::new(format!(
                "{name} requires {bytes} bytes, device page limit is {page_limit}"
            )));
        }
    }

    let hardware = &properties.hardware;
    let required_bindings = output_contract.required_bindings();
    if hardware.max_bindings < required_bindings
        || hardware.max_shared_memory_size < tile.shared_memory_bytes()
        || hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < tile.workgroup_x()
        || hardware.max_cube_dim.1 < tile.workgroup_y()
        || hardware.max_cube_dim.2 < 1
        || hardware.max_cube_count.0 < time_workgroups
        || hardware.max_cube_count.1 < output_workgroups
        || hardware.max_cube_count.2 < 1
    {
        return Err(PointwiseDirectError::new(format!(
            "{} requires bindings={required_bindings}, shared={}B, workgroup=[{},{},1], dispatch=[{time_workgroups},{output_workgroups},1]; device exposes bindings={}, shared={}B, units={}, dim={:?}, count={:?}",
            tile.label(),
            tile.shared_memory_bytes(),
            tile.workgroup_x(),
            tile.workgroup_y(),
            hardware.max_bindings,
            hardware.max_shared_memory_size,
            hardware.max_units_per_cube,
            hardware.max_cube_dim,
            hardware.max_cube_count,
        )));
    }

    Ok(LaunchGeometry {
        precision,
        channels,
        length,
        output_bytes,
        time_workgroups,
        output_workgroups,
    })
}

fn validate_contract(
    inputs: &PointwiseResidualDirectInputs,
    alpha: Option<&CubeTensor<WgpuRuntime>>,
    tile: PointwiseKTile,
    output_contract: PointwiseOutputContract,
) -> Result<LaunchGeometry, PointwiseDirectError> {
    validate_contract_for_shape_set(
        inputs,
        alpha,
        tile,
        output_contract,
        PointwiseShapeSet::Released,
    )
}

fn validate_contract_for_shape_set(
    inputs: &PointwiseResidualDirectInputs,
    alpha: Option<&CubeTensor<WgpuRuntime>>,
    tile: PointwiseKTile,
    output_contract: PointwiseOutputContract,
    shape_set: PointwiseShapeSet,
) -> Result<LaunchGeometry, PointwiseDirectError> {
    validate_contract_inner(inputs, alpha, tile, output_contract, shape_set).map_err(|error| {
        PointwiseDirectError::new(format!(
            "{error}; {}",
            contract_diagnostic(inputs, alpha, tile, output_contract)
        ))
    })
}

/// Complete physical/device preflight without allocation or dispatch.
pub fn pointwise_residual_direct_contract_is_compatible(
    inputs: &PointwiseResidualDirectInputs,
    alpha: Option<&CubeTensor<WgpuRuntime>>,
    tile: PointwiseKTile,
) -> bool {
    let output_contract = if alpha.is_some() {
        PointwiseOutputContract::Pair
    } else {
        PointwiseOutputContract::Raw
    };
    validate_contract(inputs, alpha, tile, output_contract).is_ok()
}

fn allocate_output(
    reference: &CubeTensor<WgpuRuntime>,
    geometry: LaunchGeometry,
    layout: PointwiseInputLayout,
) -> CubeTensor<WgpuRuntime> {
    let shape = match layout {
        PointwiseInputLayout::Ncl => [BATCH, geometry.channels, geometry.length],
        PointwiseInputLayout::Nhwc => [BATCH, geometry.length, geometry.channels],
    };
    CubeTensor::new_contiguous(
        reference.client.clone(),
        reference.device.clone(),
        Shape::from(shape),
        reference.client.empty(geometry.output_bytes),
        geometry.precision.dtype(),
    )
}

/// Direct pointwise projection followed by ordered bias and residual adds.
pub fn pointwise_residual_direct_raw_wgsl(
    inputs: PointwiseResidualDirectInputs,
    tile: PointwiseKTile,
) -> Result<CubeTensor<WgpuRuntime>, PointwiseDirectError> {
    let geometry = validate_contract(&inputs, None, tile, PointwiseOutputContract::Raw)?;
    let output = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Ncl);
    let kernel = PointwiseDirectRawKernel {
        geometry,
        tile,
        input_layout: inputs.input_layout,
        residual_layout: inputs.residual_layout,
        output_layout: PointwiseInputLayout::Ncl,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(tile.workgroup_x(), tile.workgroup_y()),
        ));
    inputs.input_ncl.client.launch(
        task,
        CubeCount::new_3d(geometry.time_workgroups, geometry.output_workgroups, 1),
        KernelArguments::new()
            .with_buffer(inputs.input_ncl.handle.binding())
            .with_buffer(inputs.packed_weight_kco.handle.binding())
            .with_buffer(inputs.bias.handle.binding())
            .with_buffer(inputs.residual_ncl.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Ok(output)
}

/// Direct pointwise projection with raw residual and next-Snake outputs.
pub fn pointwise_residual_direct_snake_pair_wgsl(
    inputs: PointwiseResidualDirectInputs,
    alpha: CubeTensor<WgpuRuntime>,
    tile: PointwiseKTile,
) -> Result<PointwiseResidualDirectPair, PointwiseDirectError> {
    let geometry = validate_contract(&inputs, Some(&alpha), tile, PointwiseOutputContract::Pair)?;
    let raw_ncl = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Ncl);
    let activated_ncl = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Ncl);
    let kernel = PointwiseDirectPairKernel {
        geometry,
        tile,
        raw_layout: PointwiseInputLayout::Ncl,
        activated_layout: PointwiseInputLayout::Ncl,
        input_layout: inputs.input_layout,
        residual_layout: inputs.residual_layout,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(tile.workgroup_x(), tile.workgroup_y()),
        ));
    inputs.input_ncl.client.launch(
        task,
        CubeCount::new_3d(geometry.time_workgroups, geometry.output_workgroups, 1),
        KernelArguments::new()
            .with_buffer(inputs.input_ncl.handle.binding())
            .with_buffer(inputs.packed_weight_kco.handle.binding())
            .with_buffer(inputs.bias.handle.binding())
            .with_buffer(inputs.residual_ncl.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(raw_ncl.handle.clone().binding())
            .with_buffer(activated_ncl.handle.clone().binding()),
    );
    Ok(PointwiseResidualDirectPair {
        raw_ncl,
        activated_ncl,
    })
}

/// Direct pointwise residual followed by a post-storage-cast Snake, returning
/// only the activation consumed by the following decoder block.
///
/// F16 explicitly rounds the residual before evaluating Snake, matching the
/// former `pointwise -> storage -> standalone Snake` numerical boundary while
/// removing the intermediate allocation, write/read, and consumer dispatch.
pub fn pointwise_residual_direct_snake_activated_wgsl(
    inputs: PointwiseResidualDirectInputs,
    alpha: CubeTensor<WgpuRuntime>,
    tile: PointwiseKTile,
) -> Result<CubeTensor<WgpuRuntime>, PointwiseDirectError> {
    let geometry = validate_contract(
        &inputs,
        Some(&alpha),
        tile,
        PointwiseOutputContract::ActivatedOnly,
    )?;
    let activated_ncl = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Ncl);
    let kernel = PointwiseDirectActivatedKernel {
        geometry,
        tile,
        input_layout: inputs.input_layout,
        residual_layout: inputs.residual_layout,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(tile.workgroup_x(), tile.workgroup_y()),
        ));
    inputs.input_ncl.client.launch(
        task,
        CubeCount::new_3d(geometry.time_workgroups, geometry.output_workgroups, 1),
        KernelArguments::new()
            .with_buffer(inputs.input_ncl.handle.binding())
            .with_buffer(inputs.packed_weight_kco.handle.binding())
            .with_buffer(inputs.bias.handle.binding())
            .with_buffer(inputs.residual_ncl.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(activated_ncl.handle.clone().binding()),
    );
    Ok(activated_ncl)
}

/// Profile-only extension of the activated-only block-boundary store to C768.
///
/// The released production shape set remains unchanged. This entry point is
/// deliberately narrow: C768 is accepted only for the one-output post-cast
/// Snake contract, so screening it cannot silently replace the other C768
/// pointwise operations in the decoder.
#[cfg(feature = "profile")]
pub fn pointwise_residual_direct_snake_activated_c768_profile_wgsl(
    inputs: PointwiseResidualDirectInputs,
    alpha: CubeTensor<WgpuRuntime>,
    tile: PointwiseKTile,
) -> Result<CubeTensor<WgpuRuntime>, PointwiseDirectError> {
    let geometry = validate_contract_for_shape_set(
        &inputs,
        Some(&alpha),
        tile,
        PointwiseOutputContract::ActivatedOnly,
        PointwiseShapeSet::IncludeC768BlockBoundary,
    )?;
    if geometry.channels != 768 {
        return Err(PointwiseDirectError::new(format!(
            "profile C768 block-boundary route requires exactly 768 channels, got {}",
            geometry.channels
        )));
    }
    let activated_ncl = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Ncl);
    let kernel = PointwiseDirectActivatedKernel {
        geometry,
        tile,
        input_layout: inputs.input_layout,
        residual_layout: inputs.residual_layout,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(tile.workgroup_x(), tile.workgroup_y()),
        ));
    inputs.input_ncl.client.launch(
        task,
        CubeCount::new_3d(geometry.time_workgroups, geometry.output_workgroups, 1),
        KernelArguments::new()
            .with_buffer(inputs.input_ncl.handle.binding())
            .with_buffer(inputs.packed_weight_kco.handle.binding())
            .with_buffer(inputs.bias.handle.binding())
            .with_buffer(inputs.residual_ncl.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(activated_ncl.handle.clone().binding()),
    );
    Ok(activated_ncl)
}

/// Direct pointwise residual whose prepared Snake result is contiguous NHWC.
///
/// The raw shortcut remains contiguous NCL. This layout is consumed directly
/// by the following implicit-GEMM convolution and removes its input transpose.
pub fn pointwise_residual_direct_snake_nhwc_pair_wgsl(
    inputs: PointwiseResidualDirectInputs,
    alpha: CubeTensor<WgpuRuntime>,
    tile: PointwiseKTile,
) -> Result<PointwiseResidualDirectPair, PointwiseDirectError> {
    let geometry = validate_contract(&inputs, Some(&alpha), tile, PointwiseOutputContract::Pair)?;
    let raw_ncl = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Ncl);
    let activated_nhwc = CubeTensor::new_contiguous(
        inputs.input_ncl.client.clone(),
        inputs.input_ncl.device.clone(),
        Shape::from([BATCH, geometry.length, geometry.channels]),
        inputs.input_ncl.client.empty(geometry.output_bytes),
        geometry.precision.dtype(),
    );
    let kernel = PointwiseDirectPairKernel {
        geometry,
        tile,
        raw_layout: PointwiseInputLayout::Ncl,
        activated_layout: PointwiseInputLayout::Nhwc,
        input_layout: inputs.input_layout,
        residual_layout: inputs.residual_layout,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(tile.workgroup_x(), tile.workgroup_y()),
        ));
    inputs.input_ncl.client.launch(
        task,
        CubeCount::new_3d(geometry.time_workgroups, geometry.output_workgroups, 1),
        KernelArguments::new()
            .with_buffer(inputs.input_ncl.handle.binding())
            .with_buffer(inputs.packed_weight_kco.handle.binding())
            .with_buffer(inputs.bias.handle.binding())
            .with_buffer(inputs.residual_ncl.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(raw_ncl.handle.clone().binding())
            .with_buffer(activated_nhwc.handle.clone().binding()),
    );
    Ok(PointwiseResidualDirectPair {
        raw_ncl,
        activated_ncl: activated_nhwc,
    })
}

/// Profile-only transition whose raw and following-Snake outputs both remain
/// contiguous NHWC. The shortcut may be NCL at a block entrance or NHWC for
/// an already-prepared residual state; its layout is encoded by `inputs`.
#[cfg(feature = "profile")]
pub fn pointwise_residual_direct_snake_nhwc_outputs_pair_wgsl(
    inputs: PointwiseResidualDirectInputs,
    alpha: CubeTensor<WgpuRuntime>,
    tile: PointwiseKTile,
) -> Result<PointwiseResidualDirectPair, PointwiseDirectError> {
    if inputs.input_layout != PointwiseInputLayout::Nhwc {
        return Err(PointwiseDirectError::new(
            "NHWC residual outputs require an NHWC projection input",
        ));
    }
    let geometry = validate_contract(&inputs, Some(&alpha), tile, PointwiseOutputContract::Pair)?;
    let raw_nhwc = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Nhwc);
    let activated_nhwc = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Nhwc);
    let kernel = PointwiseDirectPairKernel {
        geometry,
        tile,
        raw_layout: PointwiseInputLayout::Nhwc,
        activated_layout: PointwiseInputLayout::Nhwc,
        input_layout: inputs.input_layout,
        residual_layout: inputs.residual_layout,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(tile.workgroup_x(), tile.workgroup_y()),
        ));
    inputs.input_ncl.client.launch(
        task,
        CubeCount::new_3d(geometry.time_workgroups, geometry.output_workgroups, 1),
        KernelArguments::new()
            .with_buffer(inputs.input_ncl.handle.binding())
            .with_buffer(inputs.packed_weight_kco.handle.binding())
            .with_buffer(inputs.bias.handle.binding())
            .with_buffer(inputs.residual_ncl.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(raw_nhwc.handle.clone().binding())
            .with_buffer(activated_nhwc.handle.clone().binding()),
    );
    Ok(PointwiseResidualDirectPair {
        raw_ncl: raw_nhwc,
        activated_ncl: activated_nhwc,
    })
}

/// F16 direct pointwise projection whose Snake output is already in the exact
/// compact layout consumed by the following residue-d1 core.
pub fn pointwise_residual_direct_snake_residue_pair_wgsl(
    inputs: PointwiseResidualDirectInputs,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation,
    tile: PointwiseKTile,
) -> Result<PointwiseResidualDirectResiduePair, PointwiseDirectError> {
    let geometry = validate_contract(&inputs, Some(&alpha), tile, PointwiseOutputContract::Pair)?;
    if geometry.precision != KernelFloatPrecision::F16 {
        return Err(PointwiseDirectError::new(
            "direct residue pair is an F16-only measured route",
        ));
    }
    let raw_ncl = allocate_output(&inputs.input_ncl, geometry, PointwiseInputLayout::Ncl);
    let activated_residue = CubeTensor::new_contiguous(
        inputs.input_ncl.client.clone(),
        inputs.input_ncl.device.clone(),
        Shape::from([geometry.channels * geometry.length]),
        inputs.input_ncl.client.empty(geometry.output_bytes),
        geometry.precision.dtype(),
    );
    let kernel = PointwiseDirectResiduePairKernel {
        geometry,
        tile,
        dilation,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(tile.workgroup_x(), tile.workgroup_y()),
        ));
    inputs.input_ncl.client.launch(
        task,
        CubeCount::new_3d(geometry.time_workgroups, geometry.output_workgroups, 1),
        KernelArguments::new()
            .with_buffer(inputs.input_ncl.handle.binding())
            .with_buffer(inputs.packed_weight_kco.handle.binding())
            .with_buffer(inputs.bias.handle.binding())
            .with_buffer(inputs.residual_ncl.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(raw_ncl.handle.clone().binding())
            .with_buffer(activated_residue.handle.clone().binding()),
    );
    Ok(PointwiseResidualDirectResiduePair {
        raw_ncl,
        activated_residue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_contracts_account_for_exact_bindings_and_outputs() {
        assert_eq!(PointwiseOutputContract::Raw.required_bindings(), 5);
        assert_eq!(PointwiseOutputContract::Pair.required_bindings(), 7);
        assert_eq!(
            PointwiseOutputContract::ActivatedOnly.required_bindings(),
            6
        );
        assert!(PointwiseOutputContract::Raw.writes_raw());
        assert!(!PointwiseOutputContract::Raw.writes_activated());
        assert!(PointwiseOutputContract::Pair.writes_raw());
        assert!(PointwiseOutputContract::Pair.writes_activated());
        assert!(!PointwiseOutputContract::ActivatedOnly.writes_raw());
        assert!(PointwiseOutputContract::ActivatedOnly.writes_activated());
    }

    #[test]
    fn released_shapes_exactly_tile_and_have_identical_element_counts() {
        let tile = PointwiseKTile::PRODUCTION;
        for (channels, length) in RELEASED_SHAPES {
            assert_eq!(channels * length, 9_216_000);
            assert!(channels.is_multiple_of(tile.reduction()));
            assert!(length.is_multiple_of(tile.time_tile()));
        }
        assert_eq!(
            tile.outputs_per_thread() * WORKGROUP_SIZE as usize,
            tile.time_tile() * tile.output_tile()
        );
        assert_eq!(tile.workgroup_x() * tile.workgroup_y(), WORKGROUP_SIZE);
    }

    #[test]
    fn all_sweep_lengths_are_supported_for_direct_decoder_channels() {
        let tile = PointwiseKTile::PRODUCTION;
        for latent_steps in [13, 25, 50, 100, 200] {
            assert!(supported_decoder_shape(
                384,
                latent_steps * 120,
                tile,
                PointwiseShapeSet::Released
            ));
            assert!(supported_decoder_shape(
                192,
                latent_steps * 960,
                tile,
                PointwiseShapeSet::Released
            ));
            assert!(supported_decoder_shape(
                96,
                latent_steps * 1_920,
                tile,
                PointwiseShapeSet::Released
            ));
        }
        assert!(!supported_decoder_shape(
            192,
            0,
            tile,
            PointwiseShapeSet::Released
        ));
        assert!(supported_decoder_shape(
            96,
            95_999,
            tile,
            PointwiseShapeSet::Released
        ));
        assert!(!supported_decoder_shape(
            768,
            95_999,
            tile,
            PointwiseShapeSet::Released
        ));
        #[cfg(feature = "profile")]
        assert!(supported_decoder_shape(
            768,
            6_000,
            tile,
            PointwiseShapeSet::IncludeC768BlockBoundary
        ));
    }

    #[test]
    fn padded_shared_memory_accounting_is_exact() {
        let tile = PointwiseKTile::PRODUCTION;
        assert_eq!(tile.input_tile_elements(), 2_048);
        assert_eq!(tile.input_tile_storage_elements(), 2_112);
        assert_eq!(tile.weight_tile_elements(), 3_072);
        assert_eq!(tile.shared_memory_bytes(), 20_736);
        assert!(tile.shared_memory_bytes() <= 49_152);
    }

    #[test]
    fn released_six_call_workgroup_accounting_is_exact() {
        fn suite_workgroups(tile: PointwiseKTile) -> usize {
            RELEASED_SHAPES
                .into_iter()
                .map(|(channels, length)| {
                    3 * (length / tile.time_tile()) * channels.div_ceil(tile.output_tile())
                })
                .sum()
        }

        let tile = PointwiseKTile::PRODUCTION;
        assert_eq!(suite_workgroups(tile), 9_000);
        assert_eq!(tile.local_time_outputs(), 2);
        assert_eq!(tile.local_output_channels(), 12);
        assert_eq!(tile.outputs_per_thread(), 24);
        assert_eq!(tile.fma_statements_per_reduction_step(), 6);
    }

    #[test]
    fn kernel_cache_ids_include_complete_execution_tile() {
        let tile = PointwiseKTile::PRODUCTION;
        let c192 = LaunchGeometry {
            precision: KernelFloatPrecision::F32,
            channels: 192,
            length: 48_000,
            output_bytes: 192 * 48_000 * F32_BYTES,
            time_workgroups: 750,
            output_workgroups: 2,
        };
        let c96 = LaunchGeometry {
            precision: KernelFloatPrecision::F32,
            channels: 96,
            length: 96_000,
            output_bytes: 96 * 96_000 * F32_BYTES,
            time_workgroups: 1_500,
            output_workgroups: 1,
        };
        let raw_c192 = PointwiseDirectRawKernel {
            geometry: c192,
            tile,
            input_layout: PointwiseInputLayout::Ncl,
            residual_layout: PointwiseInputLayout::Ncl,
            output_layout: PointwiseInputLayout::Ncl,
        }
        .id()
        .stable_format();
        let raw_c96 = PointwiseDirectRawKernel {
            geometry: c96,
            tile,
            input_layout: PointwiseInputLayout::Ncl,
            residual_layout: PointwiseInputLayout::Ncl,
            output_layout: PointwiseInputLayout::Ncl,
        }
        .id()
        .stable_format();
        let pair_c192 = PointwiseDirectPairKernel {
            geometry: c192,
            tile,
            raw_layout: PointwiseInputLayout::Ncl,
            activated_layout: PointwiseInputLayout::Ncl,
            input_layout: PointwiseInputLayout::Ncl,
            residual_layout: PointwiseInputLayout::Ncl,
        }
        .id()
        .stable_format();
        let f16_c192 = LaunchGeometry {
            precision: KernelFloatPrecision::F16,
            output_bytes: c192.output_bytes / 2,
            ..c192
        };
        let residue_d3 = PointwiseDirectResiduePairKernel {
            geometry: f16_c192,
            tile,
            dilation: crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation::Three,
        }
        .id()
        .stable_format();
        let residue_d9 = PointwiseDirectResiduePairKernel {
            geometry: f16_c192,
            tile,
            dilation: crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation::Nine,
        }
        .id()
        .stable_format();
        assert_ne!(raw_c192, raw_c96);
        assert_ne!(raw_c192, pair_c192);
        assert_ne!(pair_c192, residue_d3);
        assert_ne!(residue_d3, residue_d9);
        for component in ["48000", "64", "32", "96", "8"] {
            assert!(
                raw_c192.contains(component),
                "missing id component {component}"
            );
        }
    }

    #[test]
    fn shaders_keep_uniform_bindings_and_ordered_epilogues() {
        let raw = include_str!("pointwise_residual_direct_t64_o96_vec4_raw.wgsl");
        let pair = include_str!("pointwise_residual_direct_t64_o96_vec4_pair.wgsl");
        let tile = PointwiseKTile::PRODUCTION;

        assert_eq!(
            raw.matches("var<storage, read_write>").count(),
            RAW_BINDINGS as usize
        );
        assert_eq!(
            pair.matches("var<storage, read_write>").count(),
            PAIR_BINDINGS as usize
        );
        for shader in [raw, pair] {
            let biased = shader
                .find("let biased = accumulator + bias[output_channel]")
                .unwrap();
            let residual = shader.find("biased + residual_ncl").unwrap();
            assert!(biased < residual);
            assert_eq!(
                shader.matches(" = fma(").count(),
                tile.fma_statements_per_reduction_step()
            );
        }
        let raw_store = pair.find("raw_ncl[{{ raw_index }}] = raw").unwrap();
        let snake_store = pair
            .find("activated_output[{{ activated_index }}]")
            .unwrap();
        assert!(raw_store < snake_store);
        assert!(pair.contains("raw + (sine * sine) / (a + 1e-9)"));
    }

    #[test]
    fn vector_output_tail_guards_cover_each_released_channel_exactly_once() {
        let tile = PointwiseKTile::PRODUCTION;
        for channels in [96_usize, 192, 384] {
            let mut covered = Vec::new();
            let mut guarded = 0_usize;
            for group in 0..channels.div_ceil(tile.output_tile()) {
                for local_y in 0..tile.workgroup_y() as usize {
                    let output_base =
                        group * tile.output_tile() + local_y * tile.local_output_channels();
                    for local_output in 0..tile.local_output_channels() {
                        let output_channel = output_base + local_output;
                        if output_channel < channels {
                            covered.push(output_channel);
                        } else {
                            guarded += 1;
                        }
                    }
                }
            }
            assert_eq!(
                covered,
                (0..channels).collect::<Vec<_>>(),
                "{} C{channels}",
                tile.label()
            );
            let expected_guarded =
                channels.div_ceil(tile.output_tile()) * tile.output_tile() - channels;
            assert_eq!(
                expected_guarded, 0,
                "released shapes exactly fill O96 groups"
            );
            assert_eq!(guarded, expected_guarded, "{} C{channels}", tile.label());
        }
    }

    #[test]
    fn guarded_time_dispatch_covers_non_multiple_lengths_exactly_once() {
        let tile = PointwiseKTile::PRODUCTION;
        for length in [1_usize, 63, 64, 65, 39_960, 82_200] {
            let workgroups = length.div_ceil(tile.time_tile());
            let mut covered = Vec::new();
            let mut guarded = 0_usize;
            for group in 0..workgroups {
                for local_time in 0..tile.time_tile() {
                    let time = group * tile.time_tile() + local_time;
                    if time < length {
                        covered.push(time);
                    } else {
                        guarded += 1;
                    }
                }
            }
            assert_eq!(covered, (0..length).collect::<Vec<_>>());
            assert_eq!(guarded, workgroups * tile.time_tile() - length);
        }
    }
}
