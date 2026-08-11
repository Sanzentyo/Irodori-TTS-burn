//! Exact-shape cached-column col2im path for DACVAE ConvTranspose1d.
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
use burn::tensor::{DType, Shape};
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

const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 3;
const F32_BYTES: usize = size_of::<f32>();

/// Released decoder ConvTranspose1d shapes that currently use Burn col2im.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CachedCol2ImCase {
    /// `768 -> 384`, `Lin=600`, `stride=10`, `kernel=20`.
    Case1,
    /// `384 -> 192`, `Lin=6000`, `stride=8`, `kernel=16`.
    Case2,
    /// `192 -> 96`, `Lin=48000`, `stride=2`, `kernel=4`.
    Case3,
}

impl CachedCol2ImCase {
    /// Exact `(Cin, Cout, Lin, stride)` dimensions.
    pub const fn dimensions(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Case1 => (768, 384, 600, 10),
            Self::Case2 => (384, 192, 6_000, 8),
            Self::Case3 => (192, 96, 48_000, 2),
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

    /// Input length, which is also the GEMM `N` dimension.
    pub const fn input_length(self) -> usize {
        self.dimensions().2
    }

    /// Upsampling stride.
    pub const fn stride(self) -> usize {
        self.dimensions().3
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
    pub const fn output_length(self) -> usize {
        self.input_length() * self.stride()
    }

    /// GEMM rows (`Cout * kernel`).
    pub const fn columns_rows(self) -> usize {
        self.output_channels() * self.kernel_size()
    }

    /// Number of f32 values in the GEMM columns allocation.
    pub const fn columns_elements(self) -> usize {
        self.columns_rows() * self.input_length()
    }

    /// Number of f32 output values for `B=1`.
    pub const fn output_elements(self) -> usize {
        self.output_channels() * self.output_length()
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
    output_channels: u32,
    input_length: u32,
    output_length: u32,
    stride: u32,
    kernel_size: u32,
    padding: u32,
    output_elements: u32,
}

impl KernelSource for CachedCol2ImFinalizeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv_transpose1d_cached_col2im.wgsl"))
            .register("output_channels", self.output_channels.to_string())
            .register("input_length", self.input_length.to_string())
            .register("output_length", self.output_length.to_string())
            .register("stride", self.stride.to_string())
            .register("kernel_size", self.kernel_size.to_string())
            .register("padding", self.padding.to_string())
            .register("output_elements", self.output_elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.output_channels,
            self.input_length,
            self.output_length,
            self.stride,
            self.kernel_size,
            self.padding,
            self.output_elements,
        ))
    }
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
        .checked_mul(F32_BYTES)
        .ok_or_else(|| CachedCol2ImError::new(format!("{label} byte count overflow")))
}

fn validate_rank_and_layout(
    tensor: &CubeTensor<WgpuRuntime>,
    rank: usize,
    label: &str,
) -> Result<(), CachedCol2ImError> {
    if tensor.dtype != DType::F32 {
        return Err(CachedCol2ImError::new(format!(
            "{label} must be f32, got {}",
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
) -> Result<(), CachedCol2ImError> {
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im requires {REQUIRED_BINDINGS} storage bindings, device supports {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE || hardware.max_cube_dim.0 < WORKGROUP_SIZE {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im requires workgroup_size({WORKGROUP_SIZE},1,1), device supports units={} dims={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim
        )));
    }
    if hardware.max_cube_count.0 < workgroups {
        return Err(CachedCol2ImError::new(format!(
            "cached col2im dispatch x={workgroups} exceeds device limit {:?}",
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
) -> Result<(), CachedCol2ImError> {
    validate_rank_and_layout(input, 3, "input")?;
    validate_rank_and_layout(source_weight, 3, "source weight")?;
    validate_rank_and_layout(bias, 1, "bias")?;
    if input.device != source_weight.device || input.device != bias.device {
        return Err(CachedCol2ImError::new(format!(
            "input, source weight, and bias must be on one device, got {:?}, {:?}, and {:?}",
            input.device, source_weight.device, bias.device
        )));
    }

    let expected_input = [1, case.input_channels(), case.input_length()];
    let actual_input = input.meta.shape().dims::<3>();
    if actual_input != expected_input {
        return Err(CachedCol2ImError::new(format!(
            "input shape mismatch: expected {expected_input:?}, got {actual_input:?}"
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

    for (label, elements) in [
        ("input", input.meta.num_elements()),
        ("source weight", source_weight.meta.num_elements()),
        ("bias", bias.meta.num_elements()),
        ("columns", case.columns_elements()),
        ("output", case.output_elements()),
    ] {
        checked_u32(elements, &format!("{label} elements"))?;
    }
    let output_elements_u32 = checked_u32(case.output_elements(), "output elements")?;
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
            case.columns_elements()
                .checked_mul(F32_BYTES)
                .ok_or_else(|| CachedCol2ImError::new("columns byte count overflow"))?,
        ),
        (
            "output",
            case.output_elements()
                .checked_mul(F32_BYTES)
                .ok_or_else(|| CachedCol2ImError::new("output byte count overflow"))?,
        ),
    ];
    validate_resources(input, &buffers, workgroups)
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
    validate_cached_col2im_inputs(&input, &source_weight, &bias, case)?;

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

    let columns = matmul(weight, input, None, MatmulStrategy::default(), DType::F32)
        .map_err(|error| CachedCol2ImError::new(format!("cached col2im matmul failed: {error}")))?;
    let columns = reshape(
        columns,
        Shape::new([case.columns_rows(), case.input_length()]),
    );
    let columns = into_contiguous_aligned(columns);
    finalize_cached_col2im_wgsl(columns, bias, case)
}

/// Finalize contiguous `[Cout * kernel, Lin]` GEMM columns into
/// `[1, Cout, Lin * stride]`.
///
/// The only accepted shapes are released cases 1--3 with `B=1`, `k=2s`,
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
    if columns.device != bias.device {
        return Err(CachedCol2ImError::new(format!(
            "columns and bias must be on one device, got {:?} and {:?}",
            columns.device, bias.device
        )));
    }

    let expected_columns = [case.columns_rows(), case.input_length()];
    let actual_columns = columns.meta.shape().dims::<2>();
    if actual_columns != expected_columns {
        return Err(CachedCol2ImError::new(format!(
            "columns shape mismatch: expected {expected_columns:?}, got {actual_columns:?}"
        )));
    }
    let expected_bias = [case.output_channels()];
    let actual_bias = bias.meta.shape().dims::<1>();
    if actual_bias != expected_bias {
        return Err(CachedCol2ImError::new(format!(
            "bias shape mismatch: expected {expected_bias:?}, got {actual_bias:?}"
        )));
    }

    let output_elements = case.output_elements();
    let output_bytes = output_elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| CachedCol2ImError::new("cached col2im output byte count overflow"))?;
    checked_u32(case.columns_elements(), "columns elements")?;
    let output_elements_u32 = checked_u32(output_elements, "output elements")?;
    let workgroups = output_elements_u32.div_ceil(WORKGROUP_SIZE);
    let buffers = [
        ("columns", tensor_bytes(&columns, "columns")?),
        ("bias", tensor_bytes(&bias, "bias")?),
        ("output", output_bytes),
    ];
    validate_resources(&columns, &buffers, workgroups)?;

    let client = columns.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        columns.device.clone(),
        Shape::from([1, case.output_channels(), case.output_length()]),
        client.empty(output_bytes),
        DType::F32,
    );
    let kernel = CachedCol2ImFinalizeKernel {
        output_channels: checked_u32(case.output_channels(), "output channels")?,
        input_length: checked_u32(case.input_length(), "input length")?,
        output_length: checked_u32(case.output_length(), "output length")?,
        stride: checked_u32(case.stride(), "stride")?,
        kernel_size: checked_u32(case.kernel_size(), "kernel size")?,
        padding: checked_u32(case.padding(), "padding")?,
        output_elements: output_elements_u32,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    let bindings = KernelArguments::new()
        .with_buffer(columns.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups), bindings);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_shapes_match_exact_gemm_and_output_sizes() {
        let expected = [
            (CachedCol2ImCase::Case1, [7_680, 600], [384, 6_000]),
            (CachedCol2ImCase::Case2, [3_072, 6_000], [192, 48_000]),
            (CachedCol2ImCase::Case3, [384, 48_000], [96, 96_000]),
        ];
        for (case, columns, output) in expected {
            assert_eq!([case.columns_rows(), case.input_length()], columns);
            assert_eq!([case.output_channels(), case.output_length()], output);
            assert_eq!(case.kernel_size(), 2 * case.stride());
            assert_eq!(case.padding(), case.stride() / 2);
        }
    }

    #[test]
    fn every_released_output_has_at_most_two_ordered_contributors() {
        for case in [
            CachedCol2ImCase::Case1,
            CachedCol2ImCase::Case2,
            CachedCol2ImCase::Case3,
        ] {
            for output_time in 0..case.output_length() {
                let padded_time = output_time + case.padding();
                let start = if padded_time >= case.kernel_size() {
                    (padded_time - case.kernel_size()) / case.stride() + 1
                } else {
                    0
                };
                let end = (padded_time / case.stride() + 1).min(case.input_length());
                assert!(end >= start);
                assert!(end - start <= 2);
                for input_time in start..end {
                    let kernel_index = padded_time - input_time * case.stride();
                    assert!(kernel_index < case.kernel_size());
                }
            }
        }
    }

    fn burn_contributors(case: CachedCol2ImCase, output_time: usize) -> Vec<(usize, usize)> {
        let padded_time = output_time + case.padding();
        let start = if padded_time >= case.kernel_size() {
            (padded_time - case.kernel_size()) / case.stride() + 1
        } else {
            0
        };
        let end = (padded_time / case.stride() + 1).min(case.input_length());
        (start..end)
            .map(|input_time| (padded_time - input_time * case.stride(), input_time))
            .collect()
    }

    fn specialized_contributors(case: CachedCol2ImCase, output_time: usize) -> Vec<(usize, usize)> {
        let padded_time = output_time + case.padding();
        let input_start = if padded_time >= case.kernel_size() {
            (padded_time - case.kernel_size()) / case.stride() + 1
        } else {
            0
        };
        let input_end = (padded_time / case.stride() + 1).min(case.input_length());
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
            for output_time in 0..case.output_length() {
                assert_eq!(
                    specialized_contributors(case, output_time),
                    burn_contributors(case, output_time),
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
    }
}
