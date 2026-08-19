//! Decoder-shape cached-column col2im path for DACVAE ConvTranspose1d.
//!
//! Released decoder cases 1--3 already store checkpoint weights contiguously as
//! `[Cin, Cout * kernel]`. Reinterpreting that allocation as the exact
//! column-major logical matrix `[Cout * kernel, Cin]` is metadata-only. This
//! production path passes that view to CubeCL's tuned matrix multiplication and
//! finalizes the columns with a 1D-specialized kernel matching Burn's `col2im`
//! evaluation order. No persistent cache or checkpoint copy is required.

use core::fmt;

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use burn_cubecl::{
    kernel::{
        into_contiguous_aligned,
        matmul::{MatmulStrategy, matmul},
    },
    ops::{permute, reshape},
};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

const WORKGROUP_SIZE: u32 = 256;
#[cfg(feature = "profile")]
const SNAKE_PAIR_TILE: u32 = 16;
const STANDARD_BINDINGS: u32 = 3;
#[cfg(feature = "profile")]
const SNAKE_PAIR_BINDINGS: u32 = 5;

/// Released decoder ConvTranspose1d channel/stride geometries.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CachedCol2ImCase {
    /// `768 -> 384`, `stride=10`, `kernel=20`.
    Case1,
    /// `384 -> 192`, `stride=8`, `kernel=16`.
    Case2,
    /// `192 -> 96`, `stride=2`, `kernel=4`.
    Case3,
}

impl CachedCol2ImCase {
    /// Exact `(Cin, Cout, stride)` dimensions.
    pub const fn dimensions(self) -> (usize, usize, usize) {
        match self {
            Self::Case1 => (768, 384, 10),
            Self::Case2 => (384, 192, 8),
            Self::Case3 => (192, 96, 2),
        }
    }

    /// Input channels.
    pub const fn input_channels(self) -> usize {
        self.dimensions().0
    }

    /// Output channels.
    pub const fn output_channels(self) -> usize {
        self.dimensions().1
    }

    /// Reference two-second input length used by the isolated benchmark.
    pub const fn input_length(self) -> usize {
        match self {
            Self::Case1 => 600,
            Self::Case2 => 6_000,
            Self::Case3 => 48_000,
        }
    }

    /// Upsampling stride.
    pub const fn stride(self) -> usize {
        self.dimensions().2
    }

    /// Released kernels are exactly twice the stride.
    pub const fn kernel_size(self) -> usize {
        2 * self.stride()
    }

    /// Released padding is exactly half the stride.
    pub const fn padding(self) -> usize {
        self.stride() / 2
    }

    /// Exact output length for `padding_out=0` and `dilation=1`.
    pub const fn output_length_for_input(self, input_length: usize) -> Option<usize> {
        input_length.checked_mul(self.stride())
    }

    /// Reference two-second output length used by the isolated benchmark.
    pub const fn output_length(self) -> usize {
        self.input_length() * self.stride()
    }

    /// GEMM rows (`Cout * kernel`).
    pub const fn columns_rows(self) -> usize {
        self.output_channels() * self.kernel_size()
    }

    /// Number of f32 values in the GEMM columns allocation.
    pub const fn columns_elements_for_input(self, input_length: usize) -> Option<usize> {
        self.columns_rows().checked_mul(input_length)
    }

    /// Reference two-second columns element count.
    pub const fn columns_elements(self) -> usize {
        self.columns_rows() * self.input_length()
    }

    /// Number of f32 output values for `B=1`.
    pub const fn output_elements_for_input(self, input_length: usize) -> Option<usize> {
        let Some(output_length) = self.output_length_for_input(input_length) else {
            return None;
        };
        self.output_channels().checked_mul(output_length)
    }

    /// Reference two-second output element count.
    pub const fn output_elements(self) -> usize {
        self.output_channels() * self.output_length()
    }

    /// Admit only lengths produced by the corresponding released decoder stage.
    pub const fn supports_input_length(self, input_length: usize) -> bool {
        let divisor = match self {
            Self::Case1 => 12,
            Self::Case2 => 120,
            Self::Case3 => 960,
        };
        input_length >= 25 * divisor && input_length.is_multiple_of(divisor)
    }
}

/// Validation failure that prevents launching the cached-col2im path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CachedCol2ImError {
    message: String,
}

impl CachedCol2ImError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CachedCol2ImError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CachedCol2ImError {}

#[derive(Debug)]
struct CachedCol2ImFinalizeKernel {
    precision: KernelFloatPrecision,
    output_channels: u32,
    input_length: u32,
    output_length: u32,
    stride: u32,
    kernel_size: u32,
    padding: u32,
    output_elements: u32,
    dispatch_x: u32,
}

#[derive(Debug)]
#[cfg(feature = "profile")]
struct CachedCol2ImFinalizeSnakePairKernel {
    precision: KernelFloatPrecision,
    output_channels: u32,
    input_length: u32,
    output_length: u32,
    stride: u32,
    kernel_size: u32,
    padding: u32,
}

/// Exact outputs of a col2im finalizer that also prepares the following
/// residual unit's Snake activation.
#[derive(Debug)]
#[cfg(feature = "profile")]
pub struct CachedCol2ImSnakePair {
    pub raw_ncl: CubeTensor<WgpuRuntime>,
    pub activated_nhwc: CubeTensor<WgpuRuntime>,
}

impl KernelSource for CachedCol2ImFinalizeKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("conv_transpose1d_cached_col2im.wgsl"),
                include_str!("conv_transpose1d_cached_col2im_f16.wgsl"),
            )
            .register("output_channels", self.output_channels.to_string())
            .register("input_length", self.input_length.to_string())
            .register("output_length", self.output_length.to_string())
            .register("stride", self.stride.to_string())
            .register("kernel_size", self.kernel_size.to_string())
            .register("padding", self.padding.to_string())
            .register("output_elements", self.output_elements.to_string())
            .register("dispatch_x", self.dispatch_x.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.output_channels,
            self.precision,
            self.input_length,
            self.output_length,
            self.stride,
            self.kernel_size,
            self.padding,
            self.output_elements,
            self.dispatch_x,
        ))
    }
}

#[cfg(feature = "profile")]
impl KernelSource for CachedCol2ImFinalizeSnakePairKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("conv_transpose1d_cached_col2im_snake_pair.wgsl"),
                include_str!("conv_transpose1d_cached_col2im_snake_pair_f16.wgsl"),
            )
            .register("output_channels", self.output_channels.to_string())
            .register("input_length", self.input_length.to_string())
            .register("output_length", self.output_length.to_string())
            .register("stride", self.stride.to_string())
            .register("kernel_size", self.kernel_size.to_string())
            .register("padding", self.padding.to_string())
            .register("tile", SNAKE_PAIR_TILE.to_string())
            .register("tile_stride", (SNAKE_PAIR_TILE + 1).to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.output_channels,
            self.input_length,
            self.output_length,
            self.stride,
            self.kernel_size,
            self.padding,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FinalizeDispatch2d {
    x: u32,
    y: u32,
}

fn finalize_dispatch_2d(
    workgroups: u32,
    max_cube_count: (u32, u32, u32),
) -> Result<FinalizeDispatch2d, CachedCol2ImError> {
    if workgroups == 0 || max_cube_count.0 == 0 || max_cube_count.1 == 0 {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im requires non-zero finalizer workgroups and device limits, got workgroups={workgroups}, limits={max_cube_count:?}"
        )));
    }
    let x = workgroups.min(max_cube_count.0);
    let y = workgroups.div_ceil(x);
    if y > max_cube_count.1 {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im finalizer requires ({x}, {y}) workgroups, exceeding device limits {max_cube_count:?}"
        )));
    }
    Ok(FinalizeDispatch2d { x, y })
}

fn checked_u32(value: usize, label: &str) -> Result<u32, CachedCol2ImError> {
    u32::try_from(value).map_err(|_| {
        CachedCol2ImError::new(format!(
            "cached col2im {label}={value} exceeds WGSL u32 indexing"
        ))
    })
}

fn tensor_bytes(tensor: &CubeTensor<WgpuRuntime>, label: &str) -> Result<usize, CachedCol2ImError> {
    tensor
        .meta
        .num_elements()
        .checked_mul(
            KernelFloatPrecision::from_dtype(tensor.dtype)
                .ok_or_else(|| CachedCol2ImError::new(format!("{label} must be f32 or f16")))?
                .element_bytes(),
        )
        .ok_or_else(|| CachedCol2ImError::new(format!("{label} byte count overflow")))
}

fn validate_rank_and_layout(
    tensor: &CubeTensor<WgpuRuntime>,
    rank: usize,
    label: &str,
) -> Result<(), CachedCol2ImError> {
    if KernelFloatPrecision::from_dtype(tensor.dtype).is_none() {
        return Err(CachedCol2ImError::new(format!(
            "{label} must be f32 or f16, got {}",
            tensor.dtype.name()
        )));
    }
    if tensor.meta.num_dims() != rank {
        return Err(CachedCol2ImError::new(format!(
            "{label} must have rank {rank}, got {}",
            tensor.meta.num_dims()
        )));
    }
    if !tensor.is_contiguous() {
        return Err(CachedCol2ImError::new(format!(
            "{label} must be contiguous, strides={:?}",
            tensor.meta.strides()
        )));
    }
    Ok(())
}

fn validate_resources(
    reference: &CubeTensor<WgpuRuntime>,
    buffers: &[(&str, usize)],
    workgroups: u32,
    required_bindings: u32,
) -> Result<FinalizeDispatch2d, CachedCol2ImError> {
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    if hardware.max_bindings < required_bindings {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im requires {required_bindings} storage bindings, device supports {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE || hardware.max_cube_dim.0 < WORKGROUP_SIZE {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im requires workgroup_size({WORKGROUP_SIZE},1,1), device supports units={} dims={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim
        )));
    }
    let dispatch = finalize_dispatch_2d(workgroups, hardware.max_cube_count)?;

    let page_limit = properties.memory.max_page_size;
    for &(label, bytes) in buffers {
        let bytes_u64 = u64::try_from(bytes).map_err(|_| {
            CachedCol2ImError::new(format!(
                "cached col2im {label} buffer byte count {bytes} exceeds u64"
            ))
        })?;
        if bytes_u64 > page_limit {
            return Err(CachedCol2ImError::new(format!(
                "cached col2im {label} buffer requires {bytes} bytes, device page limit is {page_limit}"
            )));
        }
    }
    Ok(dispatch)
}

#[cfg(feature = "profile")]
fn validate_snake_pair_resources(
    reference: &CubeTensor<WgpuRuntime>,
    buffers: &[(&str, usize)],
    time_tiles: u32,
    channel_tiles: u32,
) -> Result<(), CachedCol2ImError> {
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    if hardware.max_bindings < SNAKE_PAIR_BINDINGS {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im Snake pair requires {SNAKE_PAIR_BINDINGS} storage bindings, device supports {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < SNAKE_PAIR_TILE
        || hardware.max_cube_dim.1 < SNAKE_PAIR_TILE
    {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im Snake pair requires workgroup_size({SNAKE_PAIR_TILE},{SNAKE_PAIR_TILE},1), device supports units={} dims={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim
        )));
    }
    if time_tiles == 0
        || channel_tiles == 0
        || time_tiles > hardware.max_cube_count.0
        || channel_tiles > hardware.max_cube_count.1
    {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im Snake pair dispatch ({time_tiles},{channel_tiles},1) exceeds device limits {:?}",
            hardware.max_cube_count
        )));
    }
    let page_limit = properties.memory.max_page_size;
    for &(label, bytes) in buffers {
        let bytes_u64 = u64::try_from(bytes).map_err(|_| {
            CachedCol2ImError::new(format!(
                "cached col2im {label} buffer byte count {bytes} exceeds u64"
            ))
        })?;
        if bytes_u64 > page_limit {
            return Err(CachedCol2ImError::new(format!(
                "cached col2im {label} buffer requires {bytes} bytes, device page limit is {page_limit}"
            )));
        }
    }
    Ok(())
}

fn validate_cached_col2im_inputs(
    input: &CubeTensor<WgpuRuntime>,
    source_weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    case: CachedCol2ImCase,
) -> Result<usize, CachedCol2ImError> {
    validate_rank_and_layout(input, 3, "input")?;
    validate_rank_and_layout(source_weight, 3, "source weight")?;
    validate_rank_and_layout(bias, 1, "bias")?;
    let precision = common_float_precision([input.dtype, source_weight.dtype, bias.dtype])
        .ok_or_else(|| {
            CachedCol2ImError::new("input, source weight, and bias must share f32 or f16 dtype")
        })?;
    if input.device != source_weight.device || input.device != bias.device {
        return Err(CachedCol2ImError::new(format!(
            "input, source weight, and bias must be on one device, got {:?}, {:?}, and {:?}",
            input.device, source_weight.device, bias.device
        )));
    }

    let actual_input = input.meta.shape().dims::<3>();
    let [batch, input_channels, input_length] = actual_input;
    if batch != 1
        || input_channels != case.input_channels()
        || !case.supports_input_length(input_length)
    {
        return Err(CachedCol2ImError::new(format!(
            "input shape mismatch: expected B=1, C={}, and a positive released decoder-stage length; got {actual_input:?}",
            case.input_channels(),
        )));
    }
    let expected_weight = [
        case.input_channels(),
        case.output_channels(),
        case.kernel_size(),
    ];
    let actual_weight = source_weight.meta.shape().dims::<3>();
    if actual_weight != expected_weight {
        return Err(CachedCol2ImError::new(format!(
            "source weight shape mismatch: expected {expected_weight:?}, got {actual_weight:?}"
        )));
    }
    let expected_bias = [case.output_channels()];
    let actual_bias = bias.meta.shape().dims::<1>();
    if actual_bias != expected_bias {
        return Err(CachedCol2ImError::new(format!(
            "bias shape mismatch: expected {expected_bias:?}, got {actual_bias:?}"
        )));
    }

    let columns_elements = case
        .columns_elements_for_input(input_length)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im columns element count overflow"))?;
    let output_elements = case
        .output_elements_for_input(input_length)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output element count overflow"))?;
    for (label, elements) in [
        ("input", input.meta.num_elements()),
        ("source weight", source_weight.meta.num_elements()),
        ("bias", bias.meta.num_elements()),
        ("columns", columns_elements),
        ("output", output_elements),
    ] {
        checked_u32(elements, &format!("{label} elements"))?;
    }
    let output_elements_u32 = checked_u32(output_elements, "output elements")?;
    let workgroups = output_elements_u32.div_ceil(WORKGROUP_SIZE);
    let buffers = [
        ("input", tensor_bytes(input, "input")?),
        (
            "source weight",
            tensor_bytes(source_weight, "source weight")?,
        ),
        ("bias", tensor_bytes(bias, "bias")?),
        (
            "columns",
            columns_elements
                .checked_mul(precision.element_bytes())
                .ok_or_else(|| CachedCol2ImError::new("columns byte count overflow"))?,
        ),
        (
            "output",
            output_elements
                .checked_mul(precision.element_bytes())
                .ok_or_else(|| CachedCol2ImError::new("output byte count overflow"))?,
        ),
    ];
    validate_resources(input, &buffers, workgroups, STANDARD_BINDINGS)?;
    Ok(input_length)
}

/// Execute the released B=1 cached-column ConvTranspose1d path.
///
/// `source_weight` remains checkpoint-native contiguous
/// `[Cin, Cout, kernel]`. Reshape plus permutation creates a zero-copy logical
/// `[1, Cout * kernel, Cin]` column-major view. CubeCL's default tuned GEMM
/// multiplies it by contiguous input `[1, Cin, Lin]`; the exact 1D finalizer
/// then produces `[1, Cout, Lin * stride]`. There is no persistent allocation
/// beyond the existing source parameter.
///
/// # Errors
///
/// Returns an error before the production launch for unsupported case, dtype,
/// shape, layout, device, integer range, binding size, or device resources.
/// A matmul setup/launch error is also returned so callers can fall back to
/// Burn's ordinary ConvTranspose1d implementation.
pub fn conv_transpose1d_cached_col2im_wgsl(
    input: CubeTensor<WgpuRuntime>,
    source_weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    case: CachedCol2ImCase,
) -> Result<CubeTensor<WgpuRuntime>, CachedCol2ImError> {
    let columns = matmul_cached_col2im_columns_wgsl(input, source_weight, &bias, case)?;
    finalize_cached_col2im_wgsl(columns, bias, case)
}

/// Execute cached-column ConvTranspose1d and fuse the following residual
/// unit's Snake/layout preparation into the finalizer dispatch.
#[cfg(feature = "profile")]
pub fn conv_transpose1d_cached_col2im_snake_pair_wgsl(
    input: CubeTensor<WgpuRuntime>,
    source_weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    case: CachedCol2ImCase,
) -> Result<CachedCol2ImSnakePair, CachedCol2ImError> {
    let columns = matmul_cached_col2im_columns_wgsl(input, source_weight, &bias, case)?;
    finalize_cached_col2im_snake_pair_wgsl(columns, bias, alpha, case)
}

/// Execute only the tuned GEMM half of the cached-column path.
///
/// This public split is used by the profile-only decoder path to put a device
/// synchronization boundary between GEMM and the exact col2im finalizer. The
/// ordinary production entry point above calls the same function without that
/// boundary, so normal decode dispatch and synchronization behavior is
/// unchanged.
pub fn matmul_cached_col2im_columns_wgsl(
    input: CubeTensor<WgpuRuntime>,
    source_weight: CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    case: CachedCol2ImCase,
) -> Result<CubeTensor<WgpuRuntime>, CachedCol2ImError> {
    let input_length = validate_cached_col2im_inputs(&input, &source_weight, bias, case)?;

    let weight = reshape(
        source_weight,
        Shape::new([1, case.input_channels(), case.columns_rows()]),
    );
    let weight = permute(weight, &[0, 2, 1]);
    let expected_column_strides = [
        case.input_channels() * case.columns_rows(),
        1,
        case.columns_rows(),
    ];
    if &weight.meta.strides()[..] != expected_column_strides.as_slice() {
        return Err(CachedCol2ImError::new(format!(
            "zero-copy column weight stride mismatch: expected {expected_column_strides:?}, got {:?}",
            weight.meta.strides()
        )));
    }

    let output_dtype = input.dtype;
    let columns = matmul(weight, input, None, MatmulStrategy::default(), output_dtype)
        .map_err(|error| CachedCol2ImError::new(format!("cached col2im matmul failed: {error}")))?;
    let columns = reshape(columns, Shape::new([case.columns_rows(), input_length]));
    Ok(into_contiguous_aligned(columns))
}

/// Finalize contiguous `[Cout * kernel, Lin]` GEMM columns into
/// `[1, Cout, Lin * stride]`.
///
/// The only accepted geometries are released cases 1--3 with `B=1`, `k=2s`,
/// `padding=s/2`, `padding_out=0`, `dilation=1`, and `groups=1`. For each
/// output, this evaluates Burn's col2im order exactly: initialize `value` to
/// zero, add the first and possible second column in ascending input-time
/// order, then add bias. All validation happens before allocation or launch.
///
/// # Errors
///
/// Returns an error without launching for dtype, rank, shape, device,
/// contiguity, integer-range, binding-size, workgroup, or dispatch mismatches.
pub fn finalize_cached_col2im_wgsl(
    columns: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    case: CachedCol2ImCase,
) -> Result<CubeTensor<WgpuRuntime>, CachedCol2ImError> {
    validate_rank_and_layout(&columns, 2, "columns")?;
    validate_rank_and_layout(&bias, 1, "bias")?;
    let precision = common_float_precision([columns.dtype, bias.dtype])
        .ok_or_else(|| CachedCol2ImError::new("columns and bias must share f32 or f16 dtype"))?;
    if columns.device != bias.device {
        return Err(CachedCol2ImError::new(format!(
            "columns and bias must be on one device, got {:?} and {:?}",
            columns.device, bias.device
        )));
    }

    let actual_columns = columns.meta.shape().dims::<2>();
    let [column_rows, input_length] = actual_columns;
    if column_rows != case.columns_rows() || !case.supports_input_length(input_length) {
        return Err(CachedCol2ImError::new(format!(
            "columns shape mismatch: expected rows={} and a positive released decoder-stage length; got {actual_columns:?}",
            case.columns_rows(),
        )));
    }
    let expected_bias = [case.output_channels()];
    let actual_bias = bias.meta.shape().dims::<1>();
    if actual_bias != expected_bias {
        return Err(CachedCol2ImError::new(format!(
            "bias shape mismatch: expected {expected_bias:?}, got {actual_bias:?}"
        )));
    }

    let output_length = case
        .output_length_for_input(input_length)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output length overflow"))?;
    let output_elements = case
        .output_elements_for_input(input_length)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output element count overflow"))?;
    let columns_elements = case
        .columns_elements_for_input(input_length)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im columns element count overflow"))?;
    let output_bytes = output_elements
        .checked_mul(precision.element_bytes())
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output byte count overflow"))?;
    checked_u32(columns_elements, "columns elements")?;
    let output_elements_u32 = checked_u32(output_elements, "output elements")?;
    let workgroups = output_elements_u32.div_ceil(WORKGROUP_SIZE);
    let buffers = [
        ("columns", tensor_bytes(&columns, "columns")?),
        ("bias", tensor_bytes(&bias, "bias")?),
        ("output", output_bytes),
    ];
    let dispatch = validate_resources(&columns, &buffers, workgroups, STANDARD_BINDINGS)?;

    let client = columns.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        columns.device.clone(),
        Shape::from([1, case.output_channels(), output_length]),
        client.empty(output_bytes),
        precision.dtype(),
    );
    let kernel = CachedCol2ImFinalizeKernel {
        precision,
        output_channels: checked_u32(case.output_channels(), "output channels")?,
        input_length: checked_u32(input_length, "input length")?,
        output_length: checked_u32(output_length, "output length")?,
        stride: checked_u32(case.stride(), "stride")?,
        kernel_size: checked_u32(case.kernel_size(), "kernel size")?,
        padding: checked_u32(case.padding(), "padding")?,
        output_elements: output_elements_u32,
        dispatch_x: dispatch.x,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    let bindings = KernelArguments::new()
        .with_buffer(columns.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_2d(dispatch.x, dispatch.y), bindings);
    Ok(output)
}

/// Finalize col2im into the raw NCL residual and the following residual unit's
/// post-storage-cast Snake activation in NHWC layout.
///
/// The raw value is first rounded to the selected storage dtype. Snake then
/// reloads that rounded value into F32, preserving the former
/// `finalizer -> storage -> Snake` numerical boundary while avoiding the raw
/// read and a standalone dispatch.
#[cfg(feature = "profile")]
pub fn finalize_cached_col2im_snake_pair_wgsl(
    columns: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    case: CachedCol2ImCase,
) -> Result<CachedCol2ImSnakePair, CachedCol2ImError> {
    validate_rank_and_layout(&columns, 2, "columns")?;
    validate_rank_and_layout(&bias, 1, "bias")?;
    validate_rank_and_layout(&alpha, 3, "Snake alpha")?;
    let precision =
        common_float_precision([columns.dtype, bias.dtype, alpha.dtype]).ok_or_else(|| {
            CachedCol2ImError::new("columns, bias, and Snake alpha must share f32 or f16 dtype")
        })?;
    if columns.device != bias.device || columns.device != alpha.device {
        return Err(CachedCol2ImError::new(format!(
            "columns, bias, and Snake alpha must be on one device, got {:?}, {:?}, and {:?}",
            columns.device, bias.device, alpha.device
        )));
    }

    let actual_columns = columns.meta.shape().dims::<2>();
    let [column_rows, input_length] = actual_columns;
    if column_rows != case.columns_rows() || !case.supports_input_length(input_length) {
        return Err(CachedCol2ImError::new(format!(
            "columns shape mismatch: expected rows={} and a released decoder-stage length; got {actual_columns:?}",
            case.columns_rows(),
        )));
    }
    let expected_bias = [case.output_channels()];
    if bias.meta.shape().dims::<1>() != expected_bias {
        return Err(CachedCol2ImError::new(format!(
            "bias shape mismatch: expected {expected_bias:?}, got {:?}",
            bias.meta.shape().dims::<1>()
        )));
    }
    let expected_alpha = [1, case.output_channels(), 1];
    if alpha.meta.shape().dims::<3>() != expected_alpha {
        return Err(CachedCol2ImError::new(format!(
            "Snake alpha shape mismatch: expected {expected_alpha:?}, got {:?}",
            alpha.meta.shape().dims::<3>()
        )));
    }

    let output_length = case
        .output_length_for_input(input_length)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output length overflow"))?;
    let output_elements = case
        .output_elements_for_input(input_length)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output element count overflow"))?;
    let output_bytes = output_elements
        .checked_mul(precision.element_bytes())
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output byte count overflow"))?;
    checked_u32(columns.meta.num_elements(), "columns elements")?;
    checked_u32(output_elements, "output elements")?;
    let time_tiles = checked_u32(output_length, "output length")?.div_ceil(SNAKE_PAIR_TILE);
    let channel_tiles =
        checked_u32(case.output_channels(), "output channels")?.div_ceil(SNAKE_PAIR_TILE);
    let buffers = [
        ("columns", tensor_bytes(&columns, "columns")?),
        ("bias", tensor_bytes(&bias, "bias")?),
        ("Snake alpha", tensor_bytes(&alpha, "Snake alpha")?),
        ("raw output", output_bytes),
        ("activated output", output_bytes),
    ];
    validate_snake_pair_resources(&columns, &buffers, time_tiles, channel_tiles)?;

    let client = columns.client.clone();
    let raw_ncl = CubeTensor::new_contiguous(
        client.clone(),
        columns.device.clone(),
        Shape::from([1, case.output_channels(), output_length]),
        client.empty(output_bytes),
        precision.dtype(),
    );
    let activated_nhwc = CubeTensor::new_contiguous(
        client.clone(),
        columns.device.clone(),
        Shape::from([1, output_length, case.output_channels()]),
        client.empty(output_bytes),
        precision.dtype(),
    );
    let kernel = CachedCol2ImFinalizeSnakePairKernel {
        precision,
        output_channels: checked_u32(case.output_channels(), "output channels")?,
        input_length: checked_u32(input_length, "input length")?,
        output_length: checked_u32(output_length, "output length")?,
        stride: checked_u32(case.stride(), "stride")?,
        kernel_size: checked_u32(case.kernel_size(), "kernel size")?,
        padding: checked_u32(case.padding(), "padding")?,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> = Box::new(
        SourceKernel::new(kernel, CubeDim::new_2d(SNAKE_PAIR_TILE, SNAKE_PAIR_TILE)),
    );
    let bindings = KernelArguments::new()
        .with_buffer(columns.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(alpha.handle.binding())
        .with_buffer(raw_ncl.handle.clone().binding())
        .with_buffer(activated_nhwc.handle.clone().binding());
    client.launch(task, CubeCount::new_2d(time_tiles, channel_tiles), bindings);
    Ok(CachedCol2ImSnakePair {
        raw_ncl,
        activated_nhwc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_input_length(case: CachedCol2ImCase) -> usize {
        match case {
            CachedCol2ImCase::Case1 => 600,
            CachedCol2ImCase::Case2 => 6_000,
            CachedCol2ImCase::Case3 => 48_000,
        }
    }

    #[test]
    fn released_shapes_match_exact_gemm_and_output_sizes() {
        let expected = [
            (CachedCol2ImCase::Case1, [7_680, 600], [384, 6_000]),
            (CachedCol2ImCase::Case2, [3_072, 6_000], [192, 48_000]),
            (CachedCol2ImCase::Case3, [384, 48_000], [96, 96_000]),
        ];
        for (case, columns, output) in expected {
            let input_length = reference_input_length(case);
            assert_eq!([case.columns_rows(), input_length], columns);
            assert_eq!(
                [
                    case.output_channels(),
                    case.output_length_for_input(input_length).unwrap()
                ],
                output,
            );
            assert_eq!(case.kernel_size(), 2 * case.stride());
            assert_eq!(case.padding(), case.stride() / 2);
        }
    }

    #[test]
    fn every_sweep_length_is_admitted_by_its_decoder_stage() {
        for latent_steps in [25, 50, 100, 200] {
            assert!(CachedCol2ImCase::Case1.supports_input_length(latent_steps * 12));
            assert!(CachedCol2ImCase::Case2.supports_input_length(latent_steps * 120));
            assert!(CachedCol2ImCase::Case3.supports_input_length(latent_steps * 960));
        }
        assert!(!CachedCol2ImCase::Case1.supports_input_length(13 * 12));
        assert!(!CachedCol2ImCase::Case3.supports_input_length(47_999));
    }

    #[test]
    fn every_released_output_has_at_most_two_ordered_contributors() {
        for case in [
            CachedCol2ImCase::Case1,
            CachedCol2ImCase::Case2,
            CachedCol2ImCase::Case3,
        ] {
            let input_length = reference_input_length(case);
            let output_length = case.output_length_for_input(input_length).unwrap();
            for output_time in 0..output_length {
                let padded_time = output_time + case.padding();
                let start = if padded_time >= case.kernel_size() {
                    (padded_time - case.kernel_size()) / case.stride() + 1
                } else {
                    0
                };
                let end = (padded_time / case.stride() + 1).min(input_length);
                assert!(end >= start);
                assert!(end - start <= 2);
                for input_time in start..end {
                    let kernel_index = padded_time - input_time * case.stride();
                    assert!(kernel_index < case.kernel_size());
                }
            }
        }
    }

    fn burn_contributors(
        case: CachedCol2ImCase,
        input_length: usize,
        output_time: usize,
    ) -> Vec<(usize, usize)> {
        let padded_time = output_time + case.padding();
        let start = if padded_time >= case.kernel_size() {
            (padded_time - case.kernel_size()) / case.stride() + 1
        } else {
            0
        };
        let end = (padded_time / case.stride() + 1).min(input_length);
        (start..end)
            .map(|input_time| (padded_time - input_time * case.stride(), input_time))
            .collect()
    }

    fn specialized_contributors(
        case: CachedCol2ImCase,
        input_length: usize,
        output_time: usize,
    ) -> Vec<(usize, usize)> {
        let padded_time = output_time + case.padding();
        let input_start = if padded_time >= case.kernel_size() {
            (padded_time - case.kernel_size()) / case.stride() + 1
        } else {
            0
        };
        let input_end = (padded_time / case.stride() + 1).min(input_length);
        [input_start, input_start + 1]
            .into_iter()
            .filter(|&input_time| input_time < input_end)
            .map(|input_time| (padded_time - input_time * case.stride(), input_time))
            .collect()
    }

    #[test]
    fn specialized_contributor_order_equals_burn_for_every_output_time() {
        for case in [
            CachedCol2ImCase::Case1,
            CachedCol2ImCase::Case2,
            CachedCol2ImCase::Case3,
        ] {
            let input_length = reference_input_length(case);
            let output_length = case.output_length_for_input(input_length).unwrap();
            for output_time in 0..output_length {
                assert_eq!(
                    specialized_contributors(case, input_length, output_time),
                    burn_contributors(case, input_length, output_time),
                    "case={case:?}, output_time={output_time}"
                );
            }
        }
    }

    #[test]
    fn shader_storage_and_evaluation_order_match_contract() {
        let shader = include_str!("conv_transpose1d_cached_col2im.wgsl");
        assert_eq!(shader.matches("var<storage, read_write>").count(), 3);
        assert!(!shader.contains("var<storage, read>"));

        let zero = shader
            .find("var value = 0.0;")
            .expect("zero initialization");
        let first = shader
            .find("value = value + columns_buf[first_column_index];")
            .expect("first ordered addition");
        let second = shader
            .find("value = value + columns_buf[second_column_index];")
            .expect("second ordered addition");
        let bias = shader
            .find("output_buf[output_index] = value + bias_buf[output_channel];")
            .expect("bias addition");
        assert!(zero < first && first < second && second < bias);
        assert!(shader.contains("group_id.y * DISPATCH_X + group_id.x"));
    }

    #[test]
    fn finalizer_dispatch_spills_into_y_without_changing_linear_coverage() {
        let max = (65_535, 65_535, 65_535);
        assert_eq!(
            finalize_dispatch_2d(65_535, max).unwrap(),
            FinalizeDispatch2d { x: 65_535, y: 1 }
        );
        let dispatch = finalize_dispatch_2d(239_760, max).unwrap();
        assert_eq!(dispatch, FinalizeDispatch2d { x: 65_535, y: 4 });
        assert!(dispatch.x * (dispatch.y - 1) < 239_760);
        assert!(dispatch.x * dispatch.y >= 239_760);
        assert!(finalize_dispatch_2d(65_536, (1, 65_535, 1)).is_err());
    }
}
