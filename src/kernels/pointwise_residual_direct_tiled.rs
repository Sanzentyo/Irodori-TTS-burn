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
const PAIR_BINDINGS: u32 = 7;
const F32_BYTES: usize = size_of::<f32>();

fn supported_decoder_shape(channels: usize, length: usize, tile: PointwiseKTile) -> bool {
    matches!(channels, 384 | 192 | 96)
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
    packed_weight_kco: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    residual_ncl: CubeTensor<WgpuRuntime>,
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
            packed_weight_kco,
            bias,
            residual_ncl,
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
}

#[derive(Debug)]
struct PointwiseDirectPairKernel {
    geometry: LaunchGeometry,
    tile: PointwiseKTile,
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
) -> String {
    let reference = &inputs.input_ncl;
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    let is_pair = alpha.is_some();
    let alpha = alpha.map_or_else(
        || "alpha[absent]".to_owned(),
        |alpha| tensor_contract_diagnostic("alpha", alpha, reference),
    );
    format!(
        "contract_snapshot tile={} required_bindings={} required_shared={}B required_workgroup=[{},{},1] {}; {}; {}; {}; {}; device[bindings={},shared={}B,units={},dim={:?},count={:?},page={}B]",
        tile.label(),
        if is_pair { PAIR_BINDINGS } else { RAW_BINDINGS },
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
) -> Result<LaunchGeometry, PointwiseDirectError> {
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
    let [batch, channels, length] = input_shape;
    if batch != BATCH || !supported_decoder_shape(channels, length, tile) {
        return Err(PointwiseDirectError::new(format!(
            "unsupported input shape {input_shape:?}; expected B=1, C in [384,192,96], and positive L with guarded T{} tails",
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
    if exact_strides::<3>(&inputs.input_ncl) != exact_ncl_strides {
        return Err(PointwiseDirectError::new(format!(
            "input_ncl must have exact contiguous strides {exact_ncl_strides:?}, got {:?}",
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
    if inputs.residual_ncl.meta.shape().dims::<3>() != [BATCH, channels, length]
        || exact_strides::<3>(&inputs.residual_ncl) != exact_ncl_strides
    {
        return Err(PointwiseDirectError::new(format!(
            "residual_ncl must be exact contiguous [1,{channels},{length}]"
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
        ("raw_ncl", elements),
    ];
    if alpha.is_some() {
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
    let required_bindings = if alpha.is_some() {
        PAIR_BINDINGS
    } else {
        RAW_BINDINGS
    };
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
) -> Result<LaunchGeometry, PointwiseDirectError> {
    validate_contract_inner(inputs, alpha, tile).map_err(|error| {
        PointwiseDirectError::new(format!(
            "{error}; {}",
            contract_diagnostic(inputs, alpha, tile)
        ))
    })
}

/// Complete physical/device preflight without allocation or dispatch.
pub fn pointwise_residual_direct_contract_is_compatible(
    inputs: &PointwiseResidualDirectInputs,
    alpha: Option<&CubeTensor<WgpuRuntime>>,
    tile: PointwiseKTile,
) -> bool {
    validate_contract(inputs, alpha, tile).is_ok()
}

fn allocate_output(
    reference: &CubeTensor<WgpuRuntime>,
    geometry: LaunchGeometry,
) -> CubeTensor<WgpuRuntime> {
    CubeTensor::new_contiguous(
        reference.client.clone(),
        reference.device.clone(),
        Shape::from([BATCH, geometry.channels, geometry.length]),
        reference.client.empty(geometry.output_bytes),
        geometry.precision.dtype(),
    )
}

/// Direct pointwise projection followed by ordered bias and residual adds.
pub fn pointwise_residual_direct_raw_wgsl(
    inputs: PointwiseResidualDirectInputs,
    tile: PointwiseKTile,
) -> Result<CubeTensor<WgpuRuntime>, PointwiseDirectError> {
    let geometry = validate_contract(&inputs, None, tile)?;
    let output = allocate_output(&inputs.input_ncl, geometry);
    let kernel = PointwiseDirectRawKernel { geometry, tile };
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
    let geometry = validate_contract(&inputs, Some(&alpha), tile)?;
    let raw_ncl = allocate_output(&inputs.input_ncl, geometry);
    let activated_ncl = allocate_output(&inputs.input_ncl, geometry);
    let kernel = PointwiseDirectPairKernel { geometry, tile };
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

/// F16 direct pointwise projection whose Snake output is already in the exact
/// compact layout consumed by the following residue-d1 core.
pub fn pointwise_residual_direct_snake_residue_pair_wgsl(
    inputs: PointwiseResidualDirectInputs,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation,
    tile: PointwiseKTile,
) -> Result<PointwiseResidualDirectResiduePair, PointwiseDirectError> {
    let geometry = validate_contract(&inputs, Some(&alpha), tile)?;
    if geometry.precision != KernelFloatPrecision::F16 {
        return Err(PointwiseDirectError::new(
            "direct residue pair is an F16-only measured route",
        ));
    }
    let raw_ncl = allocate_output(&inputs.input_ncl, geometry);
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
            assert!(supported_decoder_shape(384, latent_steps * 120, tile));
            assert!(supported_decoder_shape(192, latent_steps * 960, tile));
            assert!(supported_decoder_shape(96, latent_steps * 1_920, tile));
        }
        assert!(!supported_decoder_shape(192, 0, tile));
        assert!(supported_decoder_shape(96, 95_999, tile));
        assert!(!supported_decoder_shape(768, 95_999, tile));
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
        }
        .id()
        .stable_format();
        let raw_c96 = PointwiseDirectRawKernel {
            geometry: c96,
            tile,
        }
        .id()
        .stable_format();
        let pair_c192 = PointwiseDirectPairKernel {
            geometry: c192,
            tile,
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
        let raw_store = pair.find("raw_ncl[output_index] = raw").unwrap();
        let snake_store = pair.find("activated_ncl[output_index]").unwrap();
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
