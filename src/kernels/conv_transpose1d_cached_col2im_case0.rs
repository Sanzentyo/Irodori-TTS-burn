//! Exact-shape cached-column production kernel for decoder ConvTranspose case 0.
//!
//! This extends the already accepted cases 1--3 algorithm to the released first
//! upsampler. The decoder keeps the accepted polyphase kernel and Burn operation
//! as fail-safe fallbacks if this exact-shape path rejects the runtime contract:
//!
//! - input `[1, 1536, 50]`;
//! - checkpoint-native contiguous weight `[1536, 768, 24]`;
//! - bias `[768]`;
//! - stride 12, padding 6, output `[1, 768, 600]`.
//!
//! Reshape plus permutation produces a zero-copy logical
//! `[1, 768 * 24, 1536]` column-major view. CubeCL's tuned f32 GEMM writes the
//! compact `[768 * 24, 50]` columns tensor, and the accepted exact col2im
//! shader adds its one or two contributors in ascending input-time order before
//! adding bias. No persistent weight cache is created by this candidate.

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
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

pub const BATCH: usize = 1;
pub const INPUT_CHANNELS: usize = 1_536;
pub const OUTPUT_CHANNELS: usize = 768;
pub const INPUT_LENGTH: usize = 50;
pub const STRIDE: usize = 12;
pub const KERNEL_SIZE: usize = 24;
pub const PADDING: usize = 6;
pub const OUTPUT_LENGTH: usize = 600;
pub const COLUMNS_ROWS: usize = OUTPUT_CHANNELS * KERNEL_SIZE;
pub const INPUT_ELEMENTS: usize = BATCH * INPUT_CHANNELS * INPUT_LENGTH;
pub const WEIGHT_ELEMENTS: usize = INPUT_CHANNELS * OUTPUT_CHANNELS * KERNEL_SIZE;
pub const BIAS_ELEMENTS: usize = OUTPUT_CHANNELS;
pub const COLUMNS_ELEMENTS: usize = COLUMNS_ROWS * INPUT_LENGTH;
pub const OUTPUT_ELEMENTS: usize = BATCH * OUTPUT_CHANNELS * OUTPUT_LENGTH;
pub const MODEL_MAC: usize = COLUMNS_ELEMENTS * INPUT_CHANNELS;

const F32_BYTES: usize = size_of::<f32>();
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 3;
const FINALIZER_WORKGROUPS: u32 = OUTPUT_ELEMENTS.div_ceil(WORKGROUP_SIZE as usize) as u32;

const _: () = assert!(KERNEL_SIZE == 2 * STRIDE);
const _: () = assert!(PADDING == STRIDE / 2);
const _: () = assert!(OUTPUT_LENGTH == INPUT_LENGTH * STRIDE);
const _: () = assert!(WEIGHT_ELEMENTS == 28_311_552);
const _: () = assert!(MODEL_MAC == 1_415_577_600);

/// Exact steady-state memory, traffic, and launch accounting for the screen.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Case0Accounting {
    pub input_bytes: usize,
    pub native_weight_bytes: usize,
    pub bias_bytes: usize,
    pub columns_bytes: usize,
    pub output_bytes: usize,
    pub candidate_persistent_cache_bytes: usize,
    pub current_polyphase_persistent_cache_bytes: usize,
    pub candidate_transient_peak_bytes: usize,
    pub finalizer_column_read_bytes: usize,
    pub candidate_minimum_intermediate_traffic_bytes: usize,
    pub current_polyphase_workgroups: usize,
    pub current_polyphase_barriers: usize,
    pub current_polyphase_staged_bytes: usize,
    pub candidate_logical_dispatches: usize,
    pub candidate_finalizer_workgroups: usize,
    pub model_mac: usize,
}

impl Case0Accounting {
    pub const fn exact() -> Self {
        let input_bytes = INPUT_ELEMENTS * F32_BYTES;
        let native_weight_bytes = WEIGHT_ELEMENTS * F32_BYTES;
        let bias_bytes = BIAS_ELEMENTS * F32_BYTES;
        let columns_bytes = COLUMNS_ELEMENTS * F32_BYTES;
        let output_bytes = OUTPUT_ELEMENTS * F32_BYTES;

        // Each of the 48 interior input positions contributes all 24 taps.
        // The first and last positions contribute 18 taps after output crop.
        let contributors_per_output_channel = 18 + (INPUT_LENGTH - 2) * KERNEL_SIZE + 18;
        let finalizer_column_reads = OUTPUT_CHANNELS * contributors_per_output_channel;
        let finalizer_column_read_bytes = finalizer_column_reads * F32_BYTES;

        // Accepted case-0 Cin32 polyphase geometry: T64/O16/Cin32.
        let current_polyphase_time_tiles = INPUT_LENGTH.div_ceil(64);
        let current_polyphase_workgroups =
            current_polyphase_time_tiles * (OUTPUT_CHANNELS / 16) * STRIDE;
        let current_polyphase_cin_rounds = INPUT_CHANNELS / 32;
        let current_polyphase_barriers =
            current_polyphase_workgroups * current_polyphase_cin_rounds * 2;
        let current_polyphase_shared_bytes = 12_416;
        let current_polyphase_staged_bytes = current_polyphase_workgroups
            * current_polyphase_cin_rounds
            * current_polyphase_shared_bytes;

        Self {
            input_bytes,
            native_weight_bytes,
            bias_bytes,
            columns_bytes,
            output_bytes,
            candidate_persistent_cache_bytes: 0,
            current_polyphase_persistent_cache_bytes: native_weight_bytes,
            candidate_transient_peak_bytes: columns_bytes + output_bytes,
            finalizer_column_read_bytes,
            candidate_minimum_intermediate_traffic_bytes: columns_bytes
                + finalizer_column_read_bytes
                + output_bytes,
            current_polyphase_workgroups,
            current_polyphase_barriers,
            current_polyphase_staged_bytes,
            candidate_logical_dispatches: 2,
            candidate_finalizer_workgroups: FINALIZER_WORKGROUPS as usize,
            model_mac: MODEL_MAC,
        }
    }
}

/// Runtime proof that the logical GEMM weight remains on the source allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZeroCopyWeightProof {
    pub same_managed_memory: bool,
    pub source_binding: String,
    pub view_binding: String,
    pub source_bytes: u64,
    pub view_bytes: u64,
    pub source_offset_start: Option<u64>,
    pub view_offset_start: Option<u64>,
    pub source_offset_end: Option<u64>,
    pub view_offset_end: Option<u64>,
    pub view_shape: [usize; 3],
    pub view_strides: [usize; 3],
    pub persistent_cache_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Case0CachedCol2ImError {
    message: String,
}

impl Case0CachedCol2ImError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Case0CachedCol2ImError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Case0CachedCol2ImError {}

#[derive(Debug)]
struct Case0FinalizeKernel;

impl KernelSource for Case0FinalizeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv_transpose1d_cached_col2im.wgsl"))
            .register("output_channels", OUTPUT_CHANNELS.to_string())
            .register("input_length", INPUT_LENGTH.to_string())
            .register("output_length", OUTPUT_LENGTH.to_string())
            .register("stride", STRIDE.to_string())
            .register("kernel_size", KERNEL_SIZE.to_string())
            .register("padding", PADDING.to_string())
            .register("output_elements", OUTPUT_ELEMENTS.to_string())
            .register("dispatch_x", "1")
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

fn checked_u32(value: usize, label: &str) -> Result<u32, Case0CachedCol2ImError> {
    u32::try_from(value).map_err(|_| {
        Case0CachedCol2ImError::new(format!(
            "case-0 cached col2im {label}={value} exceeds WGSL u32 indexing"
        ))
    })
}

fn validate_exact_tensor<const D: usize>(
    tensor: &CubeTensor<WgpuRuntime>,
    expected: [usize; D],
    label: &str,
) -> Result<(), Case0CachedCol2ImError> {
    if tensor.dtype != DType::F32 {
        return Err(Case0CachedCol2ImError::new(format!(
            "{label} must be f32, got {}",
            tensor.dtype.name()
        )));
    }
    if tensor.meta.num_dims() != D {
        return Err(Case0CachedCol2ImError::new(format!(
            "{label} must have rank {D}, got {}",
            tensor.meta.num_dims()
        )));
    }
    let actual = tensor.meta.shape().dims::<D>();
    if actual != expected {
        return Err(Case0CachedCol2ImError::new(format!(
            "{label} shape mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    if !tensor.is_contiguous() {
        return Err(Case0CachedCol2ImError::new(format!(
            "{label} must be contiguous, strides={:?}",
            tensor.meta.strides()
        )));
    }
    Ok(())
}

fn validate_resources(reference: &CubeTensor<WgpuRuntime>) -> Result<(), Case0CachedCol2ImError> {
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS {
        return Err(Case0CachedCol2ImError::new(format!(
            "case-0 cached col2im requires {REQUIRED_BINDINGS} bindings, device supports {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
        || hardware.max_cube_count.0 < FINALIZER_WORKGROUPS
    {
        return Err(Case0CachedCol2ImError::new(format!(
            "case-0 finalizer geometry unsupported: units={} dims={:?} counts={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim, hardware.max_cube_count
        )));
    }

    let accounting = Case0Accounting::exact();
    let page_limit = properties.memory.max_page_size;
    for (label, bytes) in [
        ("input", accounting.input_bytes),
        ("native weight", accounting.native_weight_bytes),
        ("bias", accounting.bias_bytes),
        ("columns", accounting.columns_bytes),
        ("output", accounting.output_bytes),
    ] {
        let bytes = u64::try_from(bytes)
            .map_err(|_| Case0CachedCol2ImError::new(format!("{label} bytes exceed u64")))?;
        if bytes > page_limit {
            return Err(Case0CachedCol2ImError::new(format!(
                "{label} requires {bytes} bytes, device page limit is {page_limit}"
            )));
        }
    }
    Ok(())
}

fn validate_inputs(
    input: &CubeTensor<WgpuRuntime>,
    source_weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
) -> Result<(), Case0CachedCol2ImError> {
    validate_exact_tensor(input, [BATCH, INPUT_CHANNELS, INPUT_LENGTH], "input")?;
    validate_exact_tensor(
        source_weight,
        [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE],
        "checkpoint-native weight",
    )?;
    validate_exact_tensor(bias, [OUTPUT_CHANNELS], "bias")?;
    if input.device != source_weight.device || input.device != bias.device {
        return Err(Case0CachedCol2ImError::new(format!(
            "input, weight, and bias must share one device, got {:?}, {:?}, {:?}",
            input.device, source_weight.device, bias.device
        )));
    }
    for (label, elements) in [
        ("input", INPUT_ELEMENTS),
        ("weight", WEIGHT_ELEMENTS),
        ("bias", BIAS_ELEMENTS),
        ("columns", COLUMNS_ELEMENTS),
        ("output", OUTPUT_ELEMENTS),
    ] {
        checked_u32(elements, label)?;
    }
    validate_resources(input)
}

/// Build the metadata-only checkpoint-native GEMM weight view and prove it
/// retains the same managed-memory allocation.
pub fn zero_copy_case0_weight_view(
    source_weight: CubeTensor<WgpuRuntime>,
) -> Result<(CubeTensor<WgpuRuntime>, ZeroCopyWeightProof), Case0CachedCol2ImError> {
    // ZERO_COPY_VIEW_BEGIN: this section must stay allocation-free.
    validate_exact_tensor(
        &source_weight,
        [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE],
        "checkpoint-native weight",
    )?;
    let source_binding = source_weight.handle.clone().binding();
    let source_memory = format!("{:?}", source_binding.memory);
    let weight = reshape(
        source_weight,
        Shape::new([BATCH, INPUT_CHANNELS, COLUMNS_ROWS]),
    );
    let weight = permute(weight, &[0, 2, 1]);
    let expected_shape = [BATCH, COLUMNS_ROWS, INPUT_CHANNELS];
    let expected_strides = [INPUT_CHANNELS * COLUMNS_ROWS, 1, COLUMNS_ROWS];
    if weight.meta.shape().dims::<3>() != expected_shape
        || &weight.meta.strides()[..] != expected_strides.as_slice()
    {
        return Err(Case0CachedCol2ImError::new(format!(
            "zero-copy weight metadata mismatch: shape={:?} strides={:?}",
            weight.meta.shape(),
            weight.meta.strides()
        )));
    }
    let view_binding = weight.handle.clone().binding();
    let view_memory = format!("{:?}", view_binding.memory);
    let proof = ZeroCopyWeightProof {
        same_managed_memory: source_memory == view_memory,
        source_binding: source_memory,
        view_binding: view_memory,
        source_bytes: source_binding.size_in_used(),
        view_bytes: view_binding.size_in_used(),
        source_offset_start: source_binding.offset_start,
        view_offset_start: view_binding.offset_start,
        source_offset_end: source_binding.offset_end,
        view_offset_end: view_binding.offset_end,
        view_shape: expected_shape,
        view_strides: expected_strides,
        persistent_cache_bytes: Case0Accounting::exact().candidate_persistent_cache_bytes,
    };
    // ZERO_COPY_VIEW_END
    if !proof.same_managed_memory
        || proof.source_bytes != proof.view_bytes
        || proof.source_offset_start != proof.view_offset_start
        || proof.source_offset_end != proof.view_offset_end
        || proof.persistent_cache_bytes != 0
    {
        return Err(Case0CachedCol2ImError::new(format!(
            "checkpoint-native view allocated or changed its binding: {proof:?}"
        )));
    }
    Ok((weight, proof))
}

/// Execute tuned f32 GEMM plus the exact accepted case-0 col2im finalizer.
pub fn conv_transpose1d_case0_cached_col2im_wgsl(
    input: CubeTensor<WgpuRuntime>,
    source_weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> Result<CubeTensor<WgpuRuntime>, Case0CachedCol2ImError> {
    validate_inputs(&input, &source_weight, &bias)?;
    let (weight, proof) = zero_copy_case0_weight_view(source_weight)?;
    if !proof.same_managed_memory || proof.persistent_cache_bytes != 0 {
        return Err(Case0CachedCol2ImError::new(
            "case-0 candidate requires a zero-copy checkpoint-native weight view",
        ));
    }

    let columns = matmul(weight, input, None, MatmulStrategy::default(), DType::F32)
        .map_err(|error| Case0CachedCol2ImError::new(format!("case-0 GEMM failed: {error}")))?;
    let columns = reshape(columns, Shape::new([COLUMNS_ROWS, INPUT_LENGTH]));
    let columns = into_contiguous_aligned(columns);
    finalize_case0_cached_col2im_wgsl(columns, bias)
}

fn finalize_case0_cached_col2im_wgsl(
    columns: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> Result<CubeTensor<WgpuRuntime>, Case0CachedCol2ImError> {
    validate_exact_tensor(&columns, [COLUMNS_ROWS, INPUT_LENGTH], "contiguous columns")?;
    validate_exact_tensor(&bias, [OUTPUT_CHANNELS], "bias")?;
    if columns.device != bias.device {
        return Err(Case0CachedCol2ImError::new(
            "columns and bias must share one device",
        ));
    }
    validate_resources(&columns)?;

    let output_bytes = OUTPUT_ELEMENTS
        .checked_mul(F32_BYTES)
        .ok_or_else(|| Case0CachedCol2ImError::new("output byte count overflow"))?;
    let client = columns.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        columns.device.clone(),
        Shape::from([BATCH, OUTPUT_CHANNELS, OUTPUT_LENGTH]),
        client.empty(output_bytes),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> = Box::new(
        SourceKernel::new(Case0FinalizeKernel, CubeDim::new_1d(WORKGROUP_SIZE)),
    );
    let bindings = KernelArguments::new()
        .with_buffer(columns.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(
        task,
        CubeCount::new_1d(checked_u32(FINALIZER_WORKGROUPS as usize, "workgroups")?),
        bindings,
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_case0_accounting_matches_audited_geometry() {
        let accounting = Case0Accounting::exact();
        assert_eq!(accounting.input_bytes, 307_200);
        assert_eq!(accounting.native_weight_bytes, 113_246_208);
        assert_eq!(accounting.bias_bytes, 3_072);
        assert_eq!(accounting.columns_bytes, 3_686_400);
        assert_eq!(accounting.output_bytes, 1_843_200);
        assert_eq!(accounting.candidate_persistent_cache_bytes, 0);
        assert_eq!(
            accounting.current_polyphase_persistent_cache_bytes,
            accounting.native_weight_bytes
        );
        assert_eq!(accounting.candidate_transient_peak_bytes, 5_529_600);
        assert_eq!(accounting.finalizer_column_read_bytes, 3_649_536);
        assert_eq!(
            accounting.candidate_minimum_intermediate_traffic_bytes,
            9_179_136
        );
        assert_eq!(accounting.current_polyphase_workgroups, 576);
        assert_eq!(accounting.current_polyphase_barriers, 55_296);
        assert_eq!(accounting.current_polyphase_staged_bytes, 343_277_568);
        assert_eq!(accounting.candidate_logical_dispatches, 2);
        assert_eq!(accounting.candidate_finalizer_workgroups, 1_800);
        assert_eq!(accounting.model_mac, 1_415_577_600);
    }

    #[test]
    fn every_output_has_at_most_two_ordered_contributors() {
        let mut total = 0usize;
        for output_time in 0..OUTPUT_LENGTH {
            let padded_time = output_time + PADDING;
            let start = if padded_time >= KERNEL_SIZE {
                (padded_time - KERNEL_SIZE) / STRIDE + 1
            } else {
                0
            };
            let end = (padded_time / STRIDE + 1).min(INPUT_LENGTH);
            assert!(end >= start && end - start <= 2);
            let contributors = (start..end)
                .map(|input_time| (padded_time - input_time * STRIDE, input_time))
                .collect::<Vec<_>>();
            assert!(contributors.windows(2).all(|pair| pair[0].1 < pair[1].1));
            total += contributors.len();
        }
        assert_eq!(total, 1_188);
    }

    #[test]
    fn accepted_finalizer_preserves_zero_first_second_bias_order() {
        let shader = include_str!("conv_transpose1d_cached_col2im.wgsl");
        let zero = shader
            .find("var value = 0.0;")
            .expect("zero initialization");
        let first = shader
            .find("value = value + columns_buf[first_column_index];")
            .expect("first contributor");
        let second = shader
            .find("value = value + columns_buf[second_column_index];")
            .expect("second contributor");
        let bias = shader
            .find("output_buf[output_index] = value + bias_buf[output_channel];")
            .expect("bias-last output");
        assert!(zero < first && first < second && second < bias);
    }
}
