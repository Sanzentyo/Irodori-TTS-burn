//! Shape-checked materialization kernels for the v4-Small JointAttention tail.
//!
//! The production WGPU path selects these measured kernels only when every
//! dtype, shape, stride, device, and hardware-limit contract is satisfied. The
//! direct-K/V kernel preserves the accepted QKV+gate shader's f32 reduction and
//! elementwise order while writing `[self | context]` K/V directly. The
//! post-SDPA kernel preserves the tuned CubeCL attention kernel and combines
//! only its mandatory output-layout copy with the existing gate multiplication.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

pub const CONTEXT_LEN: usize = 3;
pub const REFERENCE_SEQ_LEN: usize = 50;
pub const NUM_HEADS: usize = 20;
pub const HEAD_DIM: usize = 64;
pub const MODEL_DIM: usize = NUM_HEADS * HEAD_DIM;
pub const COMBINED_DIM: usize = 4 * MODEL_DIM;

const HALF_HEAD_DIM: usize = HEAD_DIM / 2;
const DIRECT_WORKGROUP_SIZE: u32 = 32;
const POST_WORKGROUP_SIZE: u32 = 256;
const DIRECT_BINDINGS: u32 = 8;
const POST_BINDINGS: u32 = 3;
const DIRECT_SHARED_BYTES: usize = 2 * DIRECT_WORKGROUP_SIZE as usize * size_of::<f32>();

/// Q plus directly packed `[self | context]` K/V and the in-place gate buffer.
#[derive(Debug)]
pub struct DirectPackedKvOutput {
    /// Contiguous `[B,H,S,Dh]` query tensor.
    pub q: CubeTensor<WgpuRuntime>,
    /// Contiguous `[B,H,S+3,Dh]` key tensor.
    pub k_all: CubeTensor<WgpuRuntime>,
    /// Contiguous `[B,H,S+3,Dh]` value tensor.
    pub v_all: CubeTensor<WgpuRuntime>,
    pub combined: CubeTensor<WgpuRuntime>,
}

#[derive(Debug)]
struct DirectPackedKvKernel {
    precision: KernelFloatPrecision,
    batch: u32,
    sequence: u32,
    total_sequence: u32,
    eps: f64,
}

impl KernelSource for DirectPackedKvKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("joint_attention_direct_kv.wgsl"),
                include_str!("joint_attention_direct_kv_f16.wgsl"),
            )
            .register("batch", self.batch.to_string())
            .register("sequence", self.sequence.to_string())
            .register("total_sequence", self.total_sequence.to_string())
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.batch,
            self.precision,
            self.sequence,
            self.total_sequence,
            self.eps.to_bits(),
        ))
    }
}

#[derive(Debug)]
struct PostSdpaLayoutGateKernel {
    precision: KernelFloatPrecision,
    elements: u32,
    sequence: u32,
}

impl KernelSource for PostSdpaLayoutGateKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("joint_attention_post_sdpa.wgsl"),
                include_str!("joint_attention_post_sdpa_f16.wgsl"),
            )
            .register("elements", self.elements.to_string())
            .register("sequence", self.sequence.to_string())
            .register("workgroup_size", POST_WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.elements, self.sequence))
    }
}

fn assert_layout<const D: usize>(
    tensor: &CubeTensor<WgpuRuntime>,
    precision: KernelFloatPrecision,
    expected_shape: [usize; D],
    expected_strides: [usize; D],
    name: &str,
) {
    assert_eq!(tensor.dtype, precision.dtype(), "{name} dtype mismatch");
    assert_eq!(tensor.meta.num_dims(), D, "{name} rank mismatch");
    assert_eq!(
        tensor.meta.shape().dims::<D>(),
        expected_shape,
        "{name} shape mismatch"
    );
    assert!(tensor.is_contiguous(), "{name} must be contiguous");
    assert_eq!(
        &tensor.meta.strides()[..],
        &expected_strides,
        "{name} stride mismatch"
    );
}

fn has_layout<const D: usize>(
    tensor: &CubeTensor<WgpuRuntime>,
    expected_shape: [usize; D],
    expected_strides: [usize; D],
) -> bool {
    tensor.meta.num_dims() == D
        && tensor.meta.shape().dims::<D>() == expected_shape
        && tensor.is_contiguous()
        && &tensor.meta.strides()[..] == expected_strides.as_slice()
}

/// Return whether the direct packed-K/V launch is safe for these exact inputs.
///
/// This predicate is deliberately fail-closed and performs no allocation or
/// dispatch. Production uses it before consuming any tensor, so a rejected
/// launch can continue through the existing QKV-postprocess and K/V-cat path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn supports_direct_packed_kv(
    combined: &CubeTensor<WgpuRuntime>,
    qk_weight: &CubeTensor<WgpuRuntime>,
    rope_cos: &CubeTensor<WgpuRuntime>,
    rope_sin: &CubeTensor<WgpuRuntime>,
    ctx_kv: &CubeTensor<WgpuRuntime>,
    eps: f64,
) -> bool {
    if combined.meta.num_dims() != 3 {
        return false;
    }
    if common_float_precision([
        combined.dtype,
        qk_weight.dtype,
        rope_cos.dtype,
        rope_sin.dtype,
        ctx_kv.dtype,
    ])
    .is_none()
    {
        return false;
    }
    let batch = combined.meta.shape()[0];
    let sequence = combined.meta.shape()[1];
    if !matches!(batch, 1 | 2) || sequence < CONTEXT_LEN {
        return false;
    }
    let eps_f32 = eps as f32;
    if !eps.is_finite() || eps <= 0.0 || !eps_f32.is_finite() || eps_f32 <= 0.0 {
        return false;
    }
    if !has_layout(
        combined,
        [batch, sequence, COMBINED_DIM],
        [sequence * COMBINED_DIM, COMBINED_DIM, 1],
    ) {
        return false;
    }
    if !has_layout(
        qk_weight,
        [2, NUM_HEADS, HEAD_DIM],
        [MODEL_DIM, HEAD_DIM, 1],
    ) {
        return false;
    }
    if !has_layout(rope_cos, [sequence, HALF_HEAD_DIM], [HALF_HEAD_DIM, 1])
        || !has_layout(rope_sin, [sequence, HALF_HEAD_DIM], [HALF_HEAD_DIM, 1])
    {
        return false;
    }
    if !has_layout(
        ctx_kv,
        [2, batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM],
        [
            batch * CONTEXT_LEN * MODEL_DIM,
            CONTEXT_LEN * MODEL_DIM,
            MODEL_DIM,
            HEAD_DIM,
            1,
        ],
    ) {
        return false;
    }
    if [qk_weight, rope_cos, rope_sin, ctx_kv]
        .iter()
        .any(|tensor| tensor.device != combined.device)
    {
        return false;
    }

    let Some(workgroups) = batch
        .checked_mul(sequence)
        .and_then(|value| value.checked_mul(NUM_HEADS))
        .and_then(|value| u32::try_from(value).ok())
    else {
        return false;
    };
    let elements_fit_u32 = sequence
        .checked_add(CONTEXT_LEN)
        .and_then(|total| batch.checked_mul(total))
        .and_then(|value| value.checked_mul(MODEL_DIM))
        .is_some_and(|value| u32::try_from(value).is_ok());
    let hardware = &combined.client.properties().hardware;
    elements_fit_u32
        && hardware.max_bindings >= DIRECT_BINDINGS
        && hardware.max_shared_memory_size >= DIRECT_SHARED_BYTES
        && hardware.max_units_per_cube >= DIRECT_WORKGROUP_SIZE
        && hardware.max_cube_dim.0 >= DIRECT_WORKGROUP_SIZE
        && hardware.max_cube_count.0 >= workgroups
}

/// Return whether the post-SDPA layout+gate launch is safe for these inputs.
///
/// Like [`supports_direct_packed_kv`], this is allocation-free and fail-closed.
pub(crate) fn supports_post_sdpa_layout_gate(
    attention: &CubeTensor<WgpuRuntime>,
    combined: &CubeTensor<WgpuRuntime>,
) -> bool {
    if attention.meta.num_dims() != 4
        || common_float_precision([attention.dtype, combined.dtype]).is_none()
    {
        return false;
    }
    let batch = attention.meta.shape()[0];
    let sequence = attention.meta.shape()[2];
    if !matches!(batch, 1 | 2)
        || sequence == 0
        || !has_layout(
            attention,
            [batch, NUM_HEADS, sequence, HEAD_DIM],
            [
                NUM_HEADS * sequence * HEAD_DIM,
                sequence * HEAD_DIM,
                HEAD_DIM,
                1,
            ],
        )
        || !has_layout(
            combined,
            [batch, sequence, COMBINED_DIM],
            [sequence * COMBINED_DIM, COMBINED_DIM, 1],
        )
        || attention.device != combined.device
    {
        return false;
    }

    let Some(elements) = batch
        .checked_mul(sequence)
        .and_then(|value| value.checked_mul(MODEL_DIM))
        .and_then(|value| u32::try_from(value).ok())
    else {
        return false;
    };
    let workgroups = elements.div_ceil(POST_WORKGROUP_SIZE);
    let hardware = &attention.client.properties().hardware;
    hardware.max_bindings >= POST_BINDINGS
        && hardware.max_units_per_cube >= POST_WORKGROUP_SIZE
        && hardware.max_cube_dim.0 >= POST_WORKGROUP_SIZE
        && hardware.max_cube_count.0 >= workgroups
}

fn checked_u32(value: usize, name: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{name}={value} exceeds WGSL u32 indexing"))
}

fn assert_batch(batch: usize) {
    assert!(
        matches!(batch, 1 | 2),
        "exact JointAttention materialization requires B=1 or B=2"
    );
}

/// Run the exact-shape direct packed-K/V QKV+gate post-process kernel.
///
/// `qk_weight` is contiguous `[2,H,Dh]` in Q-then-K order and `ctx_kv` is
/// contiguous `[2,B,3,H,Dh]` in K-then-V order. All tensors must already
/// satisfy the production layout; no implicit materialization is permitted.
///
/// # Panics
///
/// Panics on any dtype/device/shape/stride mismatch, an unsupported batch,
/// invalid epsilon, integer overflow, or insufficient device limits.
#[allow(clippy::too_many_arguments)]
pub fn direct_packed_kv_wgsl(
    combined: CubeTensor<WgpuRuntime>,
    qk_weight: CubeTensor<WgpuRuntime>,
    rope_cos: CubeTensor<WgpuRuntime>,
    rope_sin: CubeTensor<WgpuRuntime>,
    ctx_kv: CubeTensor<WgpuRuntime>,
    eps: f64,
) -> DirectPackedKvOutput {
    assert_eq!(combined.meta.num_dims(), 3, "combined must be rank 3");
    let batch = combined.meta.shape()[0];
    let sequence = combined.meta.shape()[1];
    assert_batch(batch);
    assert!(
        sequence >= CONTEXT_LEN,
        "direct K/V sequence must cover the {CONTEXT_LEN}-token context copy"
    );
    let total_sequence = sequence
        .checked_add(CONTEXT_LEN)
        .expect("packed K/V sequence length overflow");
    assert!(
        eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0,
        "epsilon must be finite, positive, and representable as f32"
    );
    let precision = common_float_precision([
        combined.dtype,
        qk_weight.dtype,
        rope_cos.dtype,
        rope_sin.dtype,
        ctx_kv.dtype,
    ])
    .expect("direct K/V tensors must share f32 or f16 dtype");

    assert_layout(
        &combined,
        precision,
        [batch, sequence, COMBINED_DIM],
        [sequence * COMBINED_DIM, COMBINED_DIM, 1],
        "combined",
    );
    assert_layout(
        &qk_weight,
        precision,
        [2, NUM_HEADS, HEAD_DIM],
        [MODEL_DIM, HEAD_DIM, 1],
        "qk_weight",
    );
    assert_layout(
        &rope_cos,
        precision,
        [sequence, HALF_HEAD_DIM],
        [HALF_HEAD_DIM, 1],
        "rope_cos",
    );
    assert_layout(
        &rope_sin,
        precision,
        [sequence, HALF_HEAD_DIM],
        [HALF_HEAD_DIM, 1],
        "rope_sin",
    );
    assert_layout(
        &ctx_kv,
        precision,
        [2, batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM],
        [
            batch * CONTEXT_LEN * MODEL_DIM,
            CONTEXT_LEN * MODEL_DIM,
            MODEL_DIM,
            HEAD_DIM,
            1,
        ],
        "ctx_kv",
    );
    for tensor in [&qk_weight, &rope_cos, &rope_sin, &ctx_kv] {
        combined.assert_is_on_same_device(tensor);
    }

    let client = combined.client.clone();
    let hardware = &client.properties().hardware;
    assert!(
        hardware.max_bindings >= DIRECT_BINDINGS,
        "direct K/V kernel requires {DIRECT_BINDINGS} bindings, device supports {}",
        hardware.max_bindings
    );
    assert!(
        hardware.max_shared_memory_size >= DIRECT_SHARED_BYTES,
        "direct K/V kernel requires {DIRECT_SHARED_BYTES} shared bytes, device supports {}",
        hardware.max_shared_memory_size
    );
    assert!(
        hardware.max_units_per_cube >= DIRECT_WORKGROUP_SIZE,
        "direct K/V kernel requires {DIRECT_WORKGROUP_SIZE} invocations, device supports {}",
        hardware.max_units_per_cube
    );
    assert!(
        hardware.max_cube_dim.0 >= DIRECT_WORKGROUP_SIZE,
        "direct K/V kernel requires workgroup x={DIRECT_WORKGROUP_SIZE}, device supports {:?}",
        hardware.max_cube_dim
    );

    let workgroups = batch
        .checked_mul(sequence)
        .and_then(|value| value.checked_mul(NUM_HEADS))
        .expect("direct K/V workgroup count overflow");
    let workgroups_u32 = checked_u32(workgroups, "direct K/V workgroups");
    assert!(
        hardware.max_cube_count.0 >= workgroups_u32,
        "direct K/V dispatch exceeds device x limit {:?}",
        hardware.max_cube_count
    );

    let q_elements = batch
        .checked_mul(sequence)
        .and_then(|value| value.checked_mul(MODEL_DIM))
        .expect("Q element count overflow");
    let kv_elements = batch
        .checked_mul(total_sequence)
        .and_then(|value| value.checked_mul(MODEL_DIM))
        .expect("packed K/V element count overflow");
    checked_u32(q_elements, "Q elements");
    checked_u32(kv_elements, "packed K/V elements");
    let q_bytes = q_elements
        .checked_mul(precision.element_bytes())
        .expect("Q byte size overflow");
    let kv_bytes = kv_elements
        .checked_mul(precision.element_bytes())
        .expect("packed K/V byte size overflow");
    let device = combined.device.clone();
    let q = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([batch, NUM_HEADS, sequence, HEAD_DIM]),
        client.empty(q_bytes),
        precision.dtype(),
    );
    let make_kv = || {
        CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            Shape::from([batch, NUM_HEADS, total_sequence, HEAD_DIM]),
            client.empty(kv_bytes),
            precision.dtype(),
        )
    };
    let k_all = make_kv();
    let v_all = make_kv();

    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DirectPackedKvKernel {
                precision,
                batch: checked_u32(batch, "batch"),
                sequence: checked_u32(sequence, "sequence"),
                total_sequence: checked_u32(total_sequence, "total sequence"),
                eps,
            },
            CubeDim::new_1d(DIRECT_WORKGROUP_SIZE),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(combined.handle.clone().binding())
        .with_buffer(qk_weight.handle.binding())
        .with_buffer(rope_cos.handle.binding())
        .with_buffer(rope_sin.handle.binding())
        .with_buffer(ctx_kv.handle.binding())
        .with_buffer(q.handle.clone().binding())
        .with_buffer(k_all.handle.clone().binding())
        .with_buffer(v_all.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups_u32), bindings);

    DirectPackedKvOutput {
        q,
        k_all,
        v_all,
        combined,
    }
}

/// Fuse the mandatory SDPA output layout copy with the existing gate multiply.
///
/// `attention` must be contiguous `[B,H,S,64]`; `combined` must be the
/// contiguous accepted QKV+gate buffer `[B,S,5120]` after in-place sigmoid.
///
/// # Panics
///
/// Panics on any dtype/device/shape/stride mismatch, unsupported batch,
/// integer overflow, or insufficient device limits.
pub fn post_sdpa_layout_gate_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    combined: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    assert_eq!(attention.meta.num_dims(), 4, "attention must be rank 4");
    let batch = attention.meta.shape()[0];
    let sequence = attention.meta.shape()[2];
    assert_batch(batch);
    assert!(sequence > 0, "post-SDPA sequence must be nonzero");
    let precision = common_float_precision([attention.dtype, combined.dtype])
        .expect("post-SDPA tensors must share f32 or f16 dtype");
    assert_layout(
        &attention,
        precision,
        [batch, NUM_HEADS, sequence, HEAD_DIM],
        [
            NUM_HEADS * sequence * HEAD_DIM,
            sequence * HEAD_DIM,
            HEAD_DIM,
            1,
        ],
        "attention",
    );
    assert_layout(
        &combined,
        precision,
        [batch, sequence, COMBINED_DIM],
        [sequence * COMBINED_DIM, COMBINED_DIM, 1],
        "combined",
    );
    attention.assert_is_on_same_device(&combined);

    let elements = batch
        .checked_mul(sequence)
        .and_then(|value| value.checked_mul(MODEL_DIM))
        .expect("post-SDPA element count overflow");
    let elements_u32 = checked_u32(elements, "post-SDPA elements");
    let workgroups_u32 = elements_u32.div_ceil(POST_WORKGROUP_SIZE);
    let client = attention.client.clone();
    let hardware = &client.properties().hardware;
    assert!(
        hardware.max_bindings >= POST_BINDINGS,
        "post-SDPA kernel requires {POST_BINDINGS} bindings, device supports {}",
        hardware.max_bindings
    );
    assert!(
        hardware.max_units_per_cube >= POST_WORKGROUP_SIZE,
        "post-SDPA kernel requires {POST_WORKGROUP_SIZE} invocations, device supports {}",
        hardware.max_units_per_cube
    );
    assert!(
        hardware.max_cube_dim.0 >= POST_WORKGROUP_SIZE,
        "post-SDPA kernel requires workgroup x={POST_WORKGROUP_SIZE}, device supports {:?}",
        hardware.max_cube_dim
    );
    assert!(
        hardware.max_cube_count.0 >= workgroups_u32,
        "post-SDPA dispatch exceeds device x limit {:?}",
        hardware.max_cube_count
    );

    let output_bytes = elements
        .checked_mul(precision.element_bytes())
        .expect("post-SDPA output byte size overflow");
    let output = CubeTensor::new_contiguous(
        client.clone(),
        attention.device.clone(),
        Shape::from([batch, sequence, MODEL_DIM]),
        client.empty(output_bytes),
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            PostSdpaLayoutGateKernel {
                precision,
                elements: elements_u32,
                sequence: checked_u32(sequence, "sequence"),
            },
            CubeDim::new_1d(POST_WORKGROUP_SIZE),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(attention.handle.binding())
        .with_buffer(combined.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups_u32), bindings);
    output
}

/// Current logical read+write bytes for two 2-input K/V concatenations.
pub const fn current_kv_cat_logical_bytes(batch: usize, sequence: usize) -> usize {
    let self_bytes = batch * sequence * MODEL_DIM * size_of::<f32>();
    let ctx_bytes = batch * CONTEXT_LEN * MODEL_DIM * size_of::<f32>();
    let all_bytes = batch * (sequence + CONTEXT_LEN) * MODEL_DIM * size_of::<f32>();
    2 * (self_bytes + ctx_bytes + all_bytes)
}

/// Bytes removed by avoiding the intermediate self K/V write and later read.
pub const fn direct_kv_saved_logical_bytes(batch: usize, sequence: usize) -> usize {
    4 * batch * sequence * MODEL_DIM * size_of::<f32>()
}

/// Bytes removed when the post-SDPA contiguous copy is folded into gating.
pub const fn post_sdpa_saved_logical_bytes(batch: usize, sequence: usize) -> usize {
    2 * batch * sequence * MODEL_DIM * size_of::<f32>()
}

pub const fn direct_shared_bytes() -> usize {
    DIRECT_SHARED_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_shape_resources_and_traffic_are_bounded() {
        let sequence = REFERENCE_SEQ_LEN;
        assert_eq!(MODEL_DIM, 1_280);
        assert_eq!(COMBINED_DIM, 5_120);
        assert_eq!(sequence + CONTEXT_LEN, 53);
        assert_eq!(direct_shared_bytes(), 256);
        assert_eq!(current_kv_cat_logical_bytes(1, sequence), 1_085_440);
        assert_eq!(current_kv_cat_logical_bytes(2, sequence), 2_170_880);
        assert_eq!(direct_kv_saved_logical_bytes(1, sequence), 1_024_000);
        assert_eq!(direct_kv_saved_logical_bytes(2, sequence), 2_048_000);
        assert_eq!(post_sdpa_saved_logical_bytes(1, sequence), 512_000);
        assert_eq!(post_sdpa_saved_logical_bytes(2, sequence), 1_024_000);
    }

    #[test]
    fn post_sdpa_index_mapping_covers_each_source_once_for_b1_b2() {
        for sequence in [13, 25, 50, 100, 200] {
            for batch_count in [1, 2] {
                let elements = batch_count * sequence * MODEL_DIM;
                let mut seen = vec![false; elements];
                for output_index in 0..elements {
                    let token = output_index / MODEL_DIM;
                    let dim = output_index % MODEL_DIM;
                    let batch = token / sequence;
                    let seq = token % sequence;
                    let head = dim / HEAD_DIM;
                    let component = dim % HEAD_DIM;
                    let attention_index =
                        ((batch * NUM_HEADS + head) * sequence + seq) * HEAD_DIM + component;
                    assert!(attention_index < elements);
                    assert!(!seen[attention_index], "duplicate attention source index");
                    seen[attention_index] = true;
                }
                assert!(seen.into_iter().all(|value| value));
            }
        }
    }

    #[test]
    fn direct_kv_prefix_and_context_tail_cover_packed_output_once() {
        for sequence in [13, 25, 50, 100, 200] {
            let total_sequence = sequence + CONTEXT_LEN;
            for batch_count in [1, 2] {
                let elements = batch_count * total_sequence * MODEL_DIM;
                let mut seen = vec![false; elements];

                for row in 0..batch_count * sequence * NUM_HEADS {
                    let head = row % NUM_HEADS;
                    let token = row / NUM_HEADS;
                    let batch = token / sequence;
                    let seq = token % sequence;
                    let base = ((batch * NUM_HEADS + head) * total_sequence + seq) * HEAD_DIM;
                    for component in 0..HEAD_DIM {
                        assert!(!seen[base + component], "duplicate self K/V index");
                        seen[base + component] = true;
                    }
                }

                for row in 0..batch_count * CONTEXT_LEN * NUM_HEADS {
                    let head = row % NUM_HEADS;
                    let token = row / NUM_HEADS;
                    let batch = token / CONTEXT_LEN;
                    let seq = token % CONTEXT_LEN;
                    let base =
                        ((batch * NUM_HEADS + head) * total_sequence + sequence + seq) * HEAD_DIM;
                    for component in 0..HEAD_DIM {
                        assert!(!seen[base + component], "duplicate context K/V index");
                        seen[base + component] = true;
                    }
                }
                assert!(seen.into_iter().all(|value| value));
            }
        }
    }

    #[test]
    fn post_sdpa_mapping_preserves_gate_multiply_order() {
        for sequence in [13, 25, 50, 100, 200] {
            for batch_count in [1, 2] {
                for batch in 0..batch_count {
                    for seq in 0..sequence {
                        for dim in 0..MODEL_DIM {
                            let token = batch * sequence + seq;
                            let head = dim / HEAD_DIM;
                            let component = dim % HEAD_DIM;
                            let attention_index = ((batch * NUM_HEADS + head) * sequence + seq)
                                * HEAD_DIM
                                + component;
                            let output_index = token * MODEL_DIM + dim;
                            let gate_index = token * COMBINED_DIM + 3 * MODEL_DIM + dim;
                            let attention = (attention_index % 257) as f32 * (1.0 / 256.0) - 0.5;
                            let gate = (gate_index % 251) as f32 * (1.0 / 256.0) + 0.25;
                            let current = gate * attention;

                            let fused_token = output_index / MODEL_DIM;
                            let fused_dim = output_index % MODEL_DIM;
                            let fused_batch = fused_token / sequence;
                            let fused_seq = fused_token % sequence;
                            let fused_head = fused_dim / HEAD_DIM;
                            let fused_component = fused_dim % HEAD_DIM;
                            let fused_attention_index =
                                ((fused_batch * NUM_HEADS + fused_head) * sequence + fused_seq)
                                    * HEAD_DIM
                                    + fused_component;
                            let fused_gate_index =
                                fused_token * COMBINED_DIM + 3 * MODEL_DIM + fused_dim;
                            let fused_attention =
                                (fused_attention_index % 257) as f32 * (1.0 / 256.0) - 0.5;
                            let fused_gate = (fused_gate_index % 251) as f32 * (1.0 / 256.0) + 0.25;
                            let fused = fused_gate * fused_attention;
                            assert_eq!(fused.to_bits(), current.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn materialization_shaders_keep_uniform_storage_and_exact_templates() {
        let shaders = [
            (
                "direct",
                include_str!("joint_attention_direct_kv.wgsl"),
                8,
                &["batch", "sequence", "total_sequence", "eps"][..],
            ),
            (
                "post",
                include_str!("joint_attention_post_sdpa.wgsl"),
                3,
                &["elements", "sequence", "workgroup_size"][..],
            ),
        ];
        for (name, shader, binding_count, placeholders) in shaders {
            let bindings = shader
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with("@group(0)") && line.contains("var<storage"))
                .collect::<Vec<_>>();
            assert_eq!(bindings.len(), binding_count, "{name} binding count");
            assert!(
                bindings
                    .iter()
                    .all(|line| line.contains("var<storage, read_write>")),
                "{name} mixes storage access: {bindings:?}"
            );
            for placeholder in placeholders {
                assert!(
                    shader.contains(&format!("{{{{ {placeholder} }}}}")),
                    "{name} omits {placeholder}"
                );
            }
        }
    }
}
