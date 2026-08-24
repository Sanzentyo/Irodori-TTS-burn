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
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::CubeCount;
use cubecl::prelude::*;
use cubecl::server::{Handle, KernelArguments};
use cubecl::std::tensor::{
    ViewMut,
    launch::ViewArg,
    layout::{
        Coords1d,
        simple::{SimpleLayout, SimpleLayoutLaunch},
    },
};
use cubek_matmul::{
    components::global::AccumulatorGlobalScatter,
    definition::{MatmulElems, MatmulGlobalElems},
    routines::{BlueprintStrategy, TileSizeSelection, batch::simple_unit::SimpleUnitSelectionArgs},
};
use cubek_std::InputBinding;

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
const PROJECTION_DIRECT_BINDINGS: u32 = 9;
const POST_BINDINGS: u32 = 3;
const DIRECT_SHARED_BYTES: usize = 2 * DIRECT_WORKGROUP_SIZE as usize * size_of::<f32>();
const PROJECTION_DIRECT_SHARED_BYTES: usize = (64 * 32 + 32 * 32 * 4 + 64 * 32) * size_of::<f32>();
const CUBEK_SCATTER_BINDINGS: u32 = 7;
const DIRECT_NORM_BINDINGS: u32 = 5;

/// Q plus directly packed `[self | context]` K/V and a compact gate view.
#[derive(Debug)]
pub struct DirectPackedKvOutput {
    /// Contiguous `[B,H,S,Dh]` query tensor.
    pub q: CubeTensor<WgpuRuntime>,
    /// Contiguous `[B,H,S+3,Dh]` key tensor.
    pub k_all: CubeTensor<WgpuRuntime>,
    /// Contiguous `[B,H,S+3,Dh]` value tensor.
    pub v_all: CubeTensor<WgpuRuntime>,
    /// Contiguous `[B,S,D]` gate view sharing one allocation with `q`.
    pub gate: CubeTensor<WgpuRuntime>,
}

#[derive(Debug)]
struct DirectPackedKvKernel {
    precision: KernelFloatPrecision,
    rope_f32: bool,
    batch: u32,
    sequence: u32,
    context: u32,
    total_sequence: u32,
    eps: f64,
}

#[derive(Debug)]
struct ProjectionDirectPackedKvKernel {
    batch: u32,
    sequence: u32,
    context: u32,
    eps: f64,
    subgroup: bool,
}

#[derive(Debug)]
struct DirectNormRopeKernel {
    batch: u32,
    sequence: u32,
    context: u32,
    eps: f64,
}

impl KernelSource for DirectNormRopeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("qkv_direct_norm_rope.wgsl"))
            .register("batch", self.batch.to_string())
            .register("sequence", self.sequence.to_string())
            .register("context", self.context.to_string())
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.batch, self.sequence, self.context, self.eps.to_bits()))
    }
}

#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
struct QkvScatterRuntimeArgs {
    q_gate: ViewMut<'static, f32, Coords1d>,
    k_all: ViewMut<'static, f32, Coords1d>,
    v_all: ViewMut<'static, f32, Coords1d>,
    ctx_kv: ViewMut<'static, f32, Coords1d>,
    batch: u32,
    sequence: u32,
    context: u32,
    total_sequence: u32,
}

struct QkvProjectionScatter;

#[cube]
impl AccumulatorGlobalScatter<QkvScatterRuntimeArgs> for QkvProjectionScatter {
    fn store<ES: Numeric>(value: ES, coordinate: (u32, u32), runtime: &mut QkvScatterRuntimeArgs) {
        let row = coordinate.0;
        let column = coordinate.1;
        let batch = row / runtime.sequence;
        let sequence = row - batch * runtime.sequence;
        let head = (column % MODEL_DIM as u32) / HEAD_DIM as u32;
        let dim = column % HEAD_DIM as u32;
        let component = column / MODEL_DIM as u32;
        let projected = f32::cast_from(value);
        let q_offset = ((batch * NUM_HEADS as u32 + head) * runtime.sequence + sequence)
            * HEAD_DIM as u32
            + dim;
        let kv_offset = ((batch * NUM_HEADS as u32 + head) * runtime.total_sequence + sequence)
            * HEAD_DIM as u32
            + dim;
        if component == 0 {
            runtime.q_gate.write(q_offset as usize, projected);
        } else if component == 1 {
            runtime.k_all.write(kv_offset as usize, projected);
        } else if component == 2 {
            runtime.v_all.write(kv_offset as usize, projected);
        } else {
            let q_elements = runtime.batch * runtime.sequence * MODEL_DIM as u32;
            let gate_offset = q_elements + row * MODEL_DIM as u32 + column - 3 * MODEL_DIM as u32;
            let gate = 1.0 / (1.0 + (-projected).exp());
            runtime.q_gate.write(gate_offset as usize, gate);
        }

        // Reuse a disjoint subset of projection stores to copy the prepared
        // context tail. Every context scalar has exactly one owner.
        let linear = row * COMBINED_DIM as u32 + column;
        let context_plane = runtime.batch * runtime.context * MODEL_DIM as u32;
        if linear < 2 * context_plane {
            let context_component = linear / context_plane;
            let within = linear - context_component * context_plane;
            let context_dim = within % HEAD_DIM as u32;
            let context_head = (within / HEAD_DIM as u32) % NUM_HEADS as u32;
            let context_token = within / MODEL_DIM as u32;
            let context_batch = context_token / runtime.context;
            let context_sequence = context_token - context_batch * runtime.context;
            let target = ((context_batch * NUM_HEADS as u32 + context_head)
                * runtime.total_sequence
                + runtime.sequence
                + context_sequence)
                * HEAD_DIM as u32
                + context_dim;
            let cached = runtime.ctx_kv.read(linear as usize);
            if context_component == 0 {
                runtime.k_all.write(target as usize, cached);
            } else {
                runtime.v_all.write(target as usize, cached);
            }
        }
    }
}

impl KernelSource for DirectPackedKvKernel {
    fn source(&self) -> SourceTemplate {
        let source = match (self.precision, self.rope_f32) {
            (KernelFloatPrecision::F32, false) => {
                include_str!("joint_attention_direct_kv.wgsl").to_owned()
            }
            (KernelFloatPrecision::F16, false) => {
                include_str!("joint_attention_direct_kv_f16.wgsl").to_owned()
            }
            (KernelFloatPrecision::F16, true) => include_str!("joint_attention_direct_kv_f16.wgsl")
                .replace("rope_cos: array<f16>", "rope_cos: array<f32>")
                .replace("rope_sin: array<f16>", "rope_sin: array<f32>"),
            (KernelFloatPrecision::F32, true) => {
                unreachable!("f32 projection storage cannot require a mixed-f32 RoPE shader")
            }
        };
        SourceTemplate::new(source)
            .register("batch", self.batch.to_string())
            .register("sequence", self.sequence.to_string())
            .register("context", self.context.to_string())
            .register("total_sequence", self.total_sequence.to_string())
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.batch,
            self.precision,
            self.rope_f32,
            self.sequence,
            self.context,
            self.total_sequence,
            self.eps.to_bits(),
        ))
    }
}

impl KernelSource for ProjectionDirectPackedKvKernel {
    fn source(&self) -> SourceTemplate {
        let (subgroup_enable, norm_storage, norm_prepare) = if self.subgroup {
            (
                "enable subgroups;",
                "",
                r#"
    if (component < 2u) {
        for (var i = 0u; i < 8u; i = i + 1u) {
            var sum = dot(values[i], values[i]);
            sum = sum + subgroupShuffleXor(sum, 1u);
            sum = sum + subgroupShuffleXor(sum, 2u);
            sum = sum + subgroupShuffleXor(sum, 4u);
            sum = sum + subgroupShuffleXor(sum, 8u);
            norm_sums[i] = sum;
        }
    }"#,
            )
        } else {
            (
                "",
                "var<workgroup> norm_partial: array<f32, 2048>;",
                r#"
    if (component < 2u) {
        for (var i = 0u; i < 8u; i = i + 1u) {
            norm_partial[local_rows[i] * 32u + local_id.x] = dot(values[i], values[i]);
        }
        workgroupBarrier();
        var stride = 8u;
        while (stride > 0u) {
            if ((local_id.x % 16u) < stride) {
                for (var i = 0u; i < 8u; i = i + 1u) {
                    let partial = local_rows[i] * 32u + local_id.x;
                    norm_partial[partial] = norm_partial[partial] + norm_partial[partial + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let norm_lane = (local_id.x / 16u) * 16u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            norm_sums[i] = norm_partial[local_rows[i] * 32u + norm_lane];
        }
    }"#,
            )
        };
        SourceTemplate::new(include_str!("dit_projection_direct_packed_kv.wgsl"))
            .register("subgroup_enable", subgroup_enable)
            .register("norm_storage", norm_storage)
            .register("norm_prepare", norm_prepare)
            .register("batch", self.batch.to_string())
            .register("sequence", self.sequence.to_string())
            .register("context", self.context.to_string())
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.batch,
            self.sequence,
            self.context,
            self.eps.to_bits(),
            self.subgroup,
        ))
    }
}

fn direct_binding_precision(
    combined: burn::tensor::DType,
    qk_weight: burn::tensor::DType,
    rope_cos: burn::tensor::DType,
    rope_sin: burn::tensor::DType,
    ctx_kv: burn::tensor::DType,
) -> Option<(KernelFloatPrecision, bool)> {
    let precision = common_float_precision([combined, qk_weight, ctx_kv])?;
    let rope_f32 = match (precision, rope_cos, rope_sin) {
        (KernelFloatPrecision::F16, burn::tensor::DType::F32, burn::tensor::DType::F32) => true,
        (_, cos, sin) if cos == precision.dtype() && sin == precision.dtype() => false,
        _ => return None,
    };
    Some((precision, rope_f32))
}

#[derive(Debug)]
struct PostSdpaLayoutGateKernel {
    precision: KernelFloatPrecision,
    elements: u32,
    sequence: u32,
    gate_stride: u32,
    gate_offset: u32,
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
            .register("gate_stride", self.gate_stride.to_string())
            .register("gate_offset", self.gate_offset.to_string())
            .register("workgroup_size", POST_WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.elements,
            self.sequence,
            self.gate_stride,
            self.gate_offset,
        ))
    }
}

fn gate_layout(
    gate_source: &CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<(usize, usize)> {
    if has_layout(
        gate_source,
        [batch, sequence, MODEL_DIM],
        [sequence * MODEL_DIM, MODEL_DIM, 1],
    ) {
        Some((MODEL_DIM, 0))
    } else if has_layout(
        gate_source,
        [batch, sequence, COMBINED_DIM],
        [sequence * COMBINED_DIM, COMBINED_DIM, 1],
    ) {
        Some((COMBINED_DIM, 3 * MODEL_DIM))
    } else {
        None
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
    if direct_binding_precision(
        combined.dtype,
        qk_weight.dtype,
        rope_cos.dtype,
        rope_sin.dtype,
        ctx_kv.dtype,
    )
    .is_none()
    {
        return false;
    }
    let batch = combined.meta.shape()[0];
    let sequence = combined.meta.shape()[1];
    if ctx_kv.meta.num_dims() != 5 {
        return false;
    }
    let context = ctx_kv.meta.shape()[2];
    if !matches!(batch, 1..=3) || context == 0 || sequence < context {
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
        [2, batch, context, NUM_HEADS, HEAD_DIM],
        [
            batch * context * MODEL_DIM,
            context * MODEL_DIM,
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
        .checked_add(context)
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

/// Return whether projection and direct packed-K/V materialization can share
/// one strict-f32 dispatch.
///
/// The candidate deliberately requires nine storage bindings. Adapters that
/// expose only WebGPU's portable minimum of eight keep the ordinary two-stage
/// route; no implicit packing, cast, or allocation is performed here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn supports_projection_direct_packed_kv(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    qk_weight: &CubeTensor<WgpuRuntime>,
    rope_cos: &CubeTensor<WgpuRuntime>,
    rope_sin: &CubeTensor<WgpuRuntime>,
    ctx_kv: &CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    eps: f64,
    subgroup: bool,
) -> bool {
    let rows = match batch.checked_mul(sequence) {
        Some(rows) => rows,
        None => return false,
    };
    if ctx_kv.meta.num_dims() != 5 {
        return false;
    }
    let context = ctx_kv.meta.shape()[2];
    if !matches!(batch, 1..=3)
        || context == 0
        || sequence < context
        || input.dtype != burn::tensor::DType::F32
        || [weight, qk_weight, rope_cos, rope_sin, ctx_kv]
            .iter()
            .any(|tensor| tensor.dtype != burn::tensor::DType::F32)
        || !eps.is_finite()
        || eps <= 0.0
        || !(eps as f32).is_finite()
        || (eps as f32) <= 0.0
        || !has_layout(input, [rows, MODEL_DIM], [MODEL_DIM, 1])
        || !has_layout(weight, [MODEL_DIM, COMBINED_DIM], [COMBINED_DIM, 1])
        || !has_layout(
            qk_weight,
            [2, NUM_HEADS, HEAD_DIM],
            [MODEL_DIM, HEAD_DIM, 1],
        )
        || !has_layout(rope_cos, [sequence, HALF_HEAD_DIM], [HALF_HEAD_DIM, 1])
        || !has_layout(rope_sin, [sequence, HALF_HEAD_DIM], [HALF_HEAD_DIM, 1])
        || !has_layout(
            ctx_kv,
            [2, batch, context, NUM_HEADS, HEAD_DIM],
            [
                batch * context * MODEL_DIM,
                context * MODEL_DIM,
                MODEL_DIM,
                HEAD_DIM,
                1,
            ],
        )
        || [weight, qk_weight, rope_cos, rope_sin, ctx_kv]
            .iter()
            .any(|tensor| tensor.device != input.device)
    {
        return false;
    }
    let Some(cube_y) = rows.div_ceil(64).try_into().ok() else {
        return false;
    };
    let hardware = &input.client.properties().hardware;
    let shared_bytes = if subgroup {
        PROJECTION_DIRECT_SHARED_BYTES - 2_048 * size_of::<f32>()
    } else {
        PROJECTION_DIRECT_SHARED_BYTES
    };
    // Raw SourceKernel WGSL is parsed by Naga, whose WGSL frontend does not
    // yet implement `enable subgroups;` (wgpu issue #5555). CubeCL-generated
    // kernels can select another compiler path, but this source kernel cannot
    // infer that path from hardware plane support alone. Keep the candidate
    // fail-closed until the compiler capability is represented explicitly.
    let subgroup_supported = !subgroup;
    subgroup_supported
        && hardware.max_bindings >= PROJECTION_DIRECT_BINDINGS
        && hardware.max_shared_memory_size >= shared_bytes
        && hardware.max_units_per_cube >= 32 * 8
        && hardware.max_cube_dim.0 >= 32
        && hardware.max_cube_dim.1 >= 8
        && hardware.max_cube_count.0 >= 40
        && hardware.max_cube_count.1 >= cube_y
}

/// Return whether CubeK can project directly into packed Q/K/V/gate storage.
///
/// Unlike [`supports_projection_direct_packed_kv`], this candidate keeps the
/// regular CubeK matmul and requires its prepared column-major RHS. Q/K
/// normalization and RoPE run in one following in-place dispatch; the full
/// `[B,S,4D]` projection is never allocated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn supports_cubek_projection_direct_packed_kv(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    qk_weight: &CubeTensor<WgpuRuntime>,
    rope_cos: &CubeTensor<WgpuRuntime>,
    rope_sin: &CubeTensor<WgpuRuntime>,
    ctx_kv: &CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    eps: f64,
) -> bool {
    let Some(rows) = batch.checked_mul(sequence) else {
        return false;
    };
    if ctx_kv.meta.num_dims() != 5 {
        return false;
    }
    let context = ctx_kv.meta.shape()[2];
    if !matches!(batch, 1..=3)
        || context == 0
        || sequence < context
        || input.dtype != burn::tensor::DType::F32
        || [weight, qk_weight, rope_cos, rope_sin, ctx_kv]
            .iter()
            .any(|tensor| tensor.dtype != burn::tensor::DType::F32)
        || !eps.is_finite()
        || eps <= 0.0
        || !(eps as f32).is_finite()
        || (eps as f32) <= 0.0
        || !has_layout(input, [rows, MODEL_DIM], [MODEL_DIM, 1])
        || weight.meta.num_dims() != 2
        || weight.meta.shape().dims::<2>() != [MODEL_DIM, COMBINED_DIM]
        || weight.meta.strides()[..] != [1, MODEL_DIM]
        || weight.is_contiguous()
        || !has_layout(
            qk_weight,
            [2, NUM_HEADS, HEAD_DIM],
            [MODEL_DIM, HEAD_DIM, 1],
        )
        || !has_layout(rope_cos, [sequence, HALF_HEAD_DIM], [HALF_HEAD_DIM, 1])
        || !has_layout(rope_sin, [sequence, HALF_HEAD_DIM], [HALF_HEAD_DIM, 1])
        || !has_layout(
            ctx_kv,
            [2, batch, context, NUM_HEADS, HEAD_DIM],
            [
                batch * context * MODEL_DIM,
                context * MODEL_DIM,
                MODEL_DIM,
                HEAD_DIM,
                1,
            ],
        )
        || [weight, qk_weight, rope_cos, rope_sin, ctx_kv]
            .iter()
            .any(|tensor| {
                tensor.device != input.device
                    || !core::ptr::eq(tensor.client.info(), input.client.info())
            })
    {
        return false;
    }
    let Some(norm_workgroups) = rows
        .checked_mul(NUM_HEADS)
        .and_then(|value| u32::try_from(value).ok())
    else {
        return false;
    };
    let Some(max_index) = sequence
        .checked_add(context)
        .and_then(|value| batch.checked_mul(value))
        .and_then(|value| value.checked_mul(MODEL_DIM))
    else {
        return false;
    };
    let hardware = &input.client.properties().hardware;
    u32::try_from(max_index).is_ok()
        && hardware.max_bindings >= CUBEK_SCATTER_BINDINGS.max(DIRECT_NORM_BINDINGS)
        && hardware.max_shared_memory_size >= DIRECT_SHARED_BYTES
        && hardware.max_units_per_cube >= DIRECT_WORKGROUP_SIZE
        && hardware.max_cube_dim.0 >= DIRECT_WORKGROUP_SIZE
        && hardware.max_cube_count.0 >= norm_workgroups
}

/// Return whether the post-SDPA layout+gate launch is safe for these inputs.
///
/// Like [`supports_direct_packed_kv`], this is allocation-free and fail-closed.
pub(crate) fn supports_post_sdpa_layout_gate(
    attention: &CubeTensor<WgpuRuntime>,
    gate_source: &CubeTensor<WgpuRuntime>,
) -> bool {
    if attention.meta.num_dims() != 4
        || common_float_precision([attention.dtype, gate_source.dtype]).is_none()
    {
        return false;
    }
    let batch = attention.meta.shape()[0];
    let sequence = attention.meta.shape()[2];
    if !matches!(batch, 1..=3)
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
        || gate_layout(gate_source, batch, sequence).is_none()
        || attention.device != gate_source.device
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

/// Split the used range of one allocator handle into two adjacent views.
///
/// CubeCL expresses `offset_end` as bytes excluded from the end of the
/// underlying allocation, not as an absolute end address. Keeping that
/// convention here makes the packed Q/gate view correct for both whole-buffer
/// and sub-allocated handles.
fn split_leading_views(
    handle: &Handle,
    first_bytes: usize,
    second_bytes: usize,
) -> (Handle, Handle) {
    let first_bytes = u64::try_from(first_bytes).expect("first view bytes fit u64");
    let second_bytes = u64::try_from(second_bytes).expect("second view bytes fit u64");
    let required = first_bytes
        .checked_add(second_bytes)
        .expect("packed view byte size overflow");
    assert!(
        handle.size_in_used() >= required,
        "packed allocation is smaller than its two requested views"
    );

    let allocation_start = handle.offset_start.unwrap_or(0);
    let split = allocation_start
        .checked_add(first_bytes)
        .expect("packed view split offset overflow");
    let first_end_suffix = handle
        .size()
        .checked_sub(split)
        .expect("packed view split exceeds the underlying allocation");

    let mut first = handle.clone();
    first.offset_end = Some(first_end_suffix);
    let mut second = handle.clone();
    second.offset_start = Some(split);
    assert_eq!(first.size_in_used(), first_bytes);
    assert!(second.size_in_used() >= second_bytes);
    (first, second)
}

fn assert_batch(batch: usize) {
    assert!(
        matches!(batch, 1..=3),
        "exact JointAttention materialization requires B=1, B=2, or B=3"
    );
}

/// Run the exact-shape direct packed-K/V QKV+gate post-process kernel.
///
/// `qk_weight` is contiguous `[2,H,Dh]` in Q-then-K order and `ctx_kv` is
/// contiguous `[2,B,CTX,H,Dh]` in K-then-V order. All tensors must already
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
    assert_eq!(ctx_kv.meta.num_dims(), 5, "context K/V must be rank 5");
    let context = ctx_kv.meta.shape()[2];
    assert_batch(batch);
    assert!(
        context > 0 && sequence >= context,
        "direct K/V sequence must cover its non-empty context copy"
    );
    let total_sequence = sequence
        .checked_add(context)
        .expect("packed K/V sequence length overflow");
    assert!(
        eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0,
        "epsilon must be finite, positive, and representable as f32"
    );
    let (precision, rope_f32) = direct_binding_precision(
        combined.dtype,
        qk_weight.dtype,
        rope_cos.dtype,
        rope_sin.dtype,
        ctx_kv.dtype,
    )
    .expect("direct K/V tensors must use homogeneous f32/f16 storage or f16 with f32 RoPE");

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
        if rope_f32 {
            KernelFloatPrecision::F32
        } else {
            precision
        },
        [sequence, HALF_HEAD_DIM],
        [HALF_HEAD_DIM, 1],
        "rope_cos",
    );
    assert_layout(
        &rope_sin,
        if rope_f32 {
            KernelFloatPrecision::F32
        } else {
            precision
        },
        [sequence, HALF_HEAD_DIM],
        [HALF_HEAD_DIM, 1],
        "rope_sin",
    );
    assert_layout(
        &ctx_kv,
        precision,
        [2, batch, context, NUM_HEADS, HEAD_DIM],
        [
            batch * context * MODEL_DIM,
            context * MODEL_DIM,
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
    let q_gate_bytes = q_bytes
        .checked_mul(2)
        .expect("packed Q/gate byte size overflow");
    let kv_bytes = kv_elements
        .checked_mul(precision.element_bytes())
        .expect("packed K/V byte size overflow");
    let device = combined.device.clone();
    let q_gate_handle = client.empty(q_gate_bytes);
    assert!(
        q_gate_handle.size_in_used() >= u64::try_from(q_gate_bytes).expect("Q/gate bytes fit u64"),
        "packed Q/gate allocation is smaller than requested"
    );
    let (q_handle, gate_handle) = split_leading_views(&q_gate_handle, q_bytes, q_bytes);
    let q = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([batch, NUM_HEADS, sequence, HEAD_DIM]),
        q_handle,
        precision.dtype(),
    );
    let gate = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([batch, sequence, MODEL_DIM]),
        gate_handle,
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
                rope_f32,
                batch: checked_u32(batch, "batch"),
                sequence: checked_u32(sequence, "sequence"),
                context: checked_u32(context, "context"),
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
        .with_buffer(q_gate_handle.binding())
        .with_buffer(k_all.handle.clone().binding())
        .with_buffer(v_all.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups_u32), bindings);

    DirectPackedKvOutput {
        q,
        k_all,
        v_all,
        gate,
    }
}

/// Project `[B*S,1280]` and materialize normalized/rotated Q, packed K/V, and
/// compact gate storage in one dispatch.
///
/// Callers must first use [`supports_projection_direct_packed_kv`]. Keeping
/// the support predicate separate lets the model retain all fallback inputs
/// until the candidate is known to be executable on the exact adapter.
#[allow(clippy::too_many_arguments)]
pub fn projection_direct_packed_kv_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    qk_weight: CubeTensor<WgpuRuntime>,
    rope_cos: CubeTensor<WgpuRuntime>,
    rope_sin: CubeTensor<WgpuRuntime>,
    ctx_kv: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    eps: f64,
    subgroup: bool,
) -> DirectPackedKvOutput {
    assert!(
        supports_projection_direct_packed_kv(
            &input, &weight, &qk_weight, &rope_cos, &rope_sin, &ctx_kv, batch, sequence, eps,
            subgroup,
        ),
        "projection-direct packed K/V launch contract must be validated"
    );
    let rows = batch
        .checked_mul(sequence)
        .expect("projection-direct row count overflow");
    let context = ctx_kv.meta.shape()[2];
    let total_sequence = sequence
        .checked_add(context)
        .expect("projection-direct total sequence overflow");
    let q_elements = rows
        .checked_mul(MODEL_DIM)
        .expect("projection-direct Q element count overflow");
    let kv_elements = batch
        .checked_mul(total_sequence)
        .and_then(|value| value.checked_mul(MODEL_DIM))
        .expect("projection-direct K/V element count overflow");
    checked_u32(q_elements, "projection-direct Q elements");
    checked_u32(kv_elements, "projection-direct K/V elements");
    let q_bytes = q_elements
        .checked_mul(size_of::<f32>())
        .expect("projection-direct Q byte size overflow");
    let q_gate_bytes = q_bytes
        .checked_mul(2)
        .expect("projection-direct Q/gate byte size overflow");
    let kv_bytes = kv_elements
        .checked_mul(size_of::<f32>())
        .expect("projection-direct K/V byte size overflow");

    let client = input.client.clone();
    let device = input.device.clone();
    let q_gate_handle = client.empty(q_gate_bytes);
    assert!(
        q_gate_handle.size_in_used()
            >= u64::try_from(q_gate_bytes).expect("projection-direct Q/gate bytes fit u64"),
        "projection-direct Q/gate allocation is smaller than requested"
    );
    let (q_handle, gate_handle) = split_leading_views(&q_gate_handle, q_bytes, q_bytes);
    let q = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([batch, NUM_HEADS, sequence, HEAD_DIM]),
        q_handle,
        burn::tensor::DType::F32,
    );
    let gate = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([batch, sequence, MODEL_DIM]),
        gate_handle,
        burn::tensor::DType::F32,
    );
    let make_kv = || {
        CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            Shape::from([batch, NUM_HEADS, total_sequence, HEAD_DIM]),
            client.empty(kv_bytes),
            burn::tensor::DType::F32,
        )
    };
    let k_all = make_kv();
    let v_all = make_kv();
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ProjectionDirectPackedKvKernel {
                batch: checked_u32(batch, "projection-direct batch"),
                sequence: checked_u32(sequence, "projection-direct sequence"),
                context: checked_u32(context, "projection-direct context"),
                eps,
                subgroup,
            },
            CubeDim::new_2d(32, 8),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            40,
            checked_u32(rows.div_ceil(64), "projection-direct row tiles"),
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(qk_weight.handle.binding())
            .with_buffer(rope_cos.handle.binding())
            .with_buffer(rope_sin.handle.binding())
            .with_buffer(ctx_kv.handle.binding())
            .with_buffer(q_gate_handle.binding())
            .with_buffer(k_all.handle.clone().binding())
            .with_buffer(v_all.handle.clone().binding()),
    );
    DirectPackedKvOutput {
        q,
        k_all,
        v_all,
        gate,
    }
}

/// Run the regular CubeK projection with an accumulator scatter that writes
/// compact Q/gate and head-major packed K/V directly.
///
/// This is two dispatches for the complete front end: one projection/scatter
/// and one in-place Q/K RMSNorm+RoPE. It is nevertheless a true direct store
/// from the matmul accumulator: no conventional `[B,S,4D]` output, copy, or
/// post-projection split exists.
#[allow(clippy::too_many_arguments)]
pub fn try_cubek_projection_direct_packed_kv(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    qk_weight: CubeTensor<WgpuRuntime>,
    rope_cos: CubeTensor<WgpuRuntime>,
    rope_sin: CubeTensor<WgpuRuntime>,
    ctx_kv: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    eps: f64,
) -> Option<DirectPackedKvOutput> {
    if !supports_cubek_projection_direct_packed_kv(
        &input, &weight, &qk_weight, &rope_cos, &rope_sin, &ctx_kv, batch, sequence, eps,
    ) {
        return None;
    }
    let rows = batch.checked_mul(sequence)?;
    let context = ctx_kv.meta.shape()[2];
    let total_sequence = sequence.checked_add(context)?;
    let q_elements = rows.checked_mul(MODEL_DIM)?;
    let kv_elements = batch.checked_mul(total_sequence)?.checked_mul(MODEL_DIM)?;
    let q_bytes = q_elements.checked_mul(size_of::<f32>())?;
    let q_gate_bytes = q_bytes.checked_mul(2)?;
    let kv_bytes = kv_elements.checked_mul(size_of::<f32>())?;
    let client = input.client.clone();
    let device = input.device.clone();

    let q_gate_handle = client.empty(q_gate_bytes);
    let q_gate_full = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([2 * q_elements]),
        q_gate_handle.clone(),
        burn::tensor::DType::F32,
    );
    let (q_handle, gate_handle) = split_leading_views(&q_gate_handle, q_bytes, q_bytes);
    let q = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([batch, NUM_HEADS, sequence, HEAD_DIM]),
        q_handle,
        burn::tensor::DType::F32,
    );
    let gate = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([batch, sequence, MODEL_DIM]),
        gate_handle,
        burn::tensor::DType::F32,
    );
    let make_kv = || {
        CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            Shape::from([batch, NUM_HEADS, total_sequence, HEAD_DIM]),
            client.empty(kv_bytes),
            burn::tensor::DType::F32,
        )
    };
    let k_all = make_kv();
    let v_all = make_kv();
    let placeholder = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([1]),
        client.empty(size_of::<f32>()),
        burn::tensor::DType::F32,
    );

    let make_view = |binding: cubecl::prelude::TensorBinding<WgpuRuntime>| {
        let layout = SimpleLayoutLaunch::from_handle(binding.clone(), 1);
        ViewArg::new_tensor::<SimpleLayout>(binding.into_tensor_arg(), layout)
    };
    let runtime_config = QkvScatterRuntimeArgsLaunch::new(
        make_view(q_gate_full.binding()),
        make_view(k_all.clone().binding()),
        make_view(v_all.clone().binding()),
        make_view(ctx_kv.binding()),
        checked_u32(batch, "CubeK scatter batch"),
        checked_u32(sequence, "CubeK scatter sequence"),
        checked_u32(context, "CubeK scatter context"),
        checked_u32(total_sequence, "CubeK scatter total sequence"),
    );
    let storage = dtype_to_storage_type(burn::tensor::DType::F32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });
    let strategy = BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
        tile_size: TileSizeSelection::MaxTileSize,
    });
    let launched = cubek_matmul::launch::launch_accumulator_scatter_ref::<
        WgpuRuntime,
        QkvScatterRuntimeArgs,
        QkvProjectionScatter,
    >(
        &client,
        InputBinding::new(input.binding(), storage),
        InputBinding::new(weight.binding(), storage),
        placeholder.binding(),
        runtime_config,
        &strategy,
        &mut dtypes,
    );
    #[cfg(feature = "profile")]
    if let Err(error) = &launched {
        tracing::debug!(
            target: "irodori_tts_burn::route",
            ?error,
            "CubeK direct QKV scatter launch rejected"
        );
    }
    launched.ok()?;

    let norm_workgroups = rows.checked_mul(NUM_HEADS)?;
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DirectNormRopeKernel {
                batch: checked_u32(batch, "CubeK scatter norm batch"),
                sequence: checked_u32(sequence, "CubeK scatter norm sequence"),
                context: checked_u32(context, "CubeK scatter norm context"),
                eps,
            },
            CubeDim::new_1d(DIRECT_WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d(checked_u32(
            norm_workgroups,
            "CubeK scatter norm workgroups",
        )),
        KernelArguments::new()
            .with_buffer(q_gate_handle.binding())
            .with_buffer(k_all.handle.clone().binding())
            .with_buffer(qk_weight.handle.binding())
            .with_buffer(rope_cos.handle.binding())
            .with_buffer(rope_sin.handle.binding()),
    );
    Some(DirectPackedKvOutput {
        q,
        k_all,
        v_all,
        gate,
    })
}

/// Fuse the mandatory SDPA output layout copy with the existing gate multiply.
///
/// `attention` must be contiguous `[B,H,S,64]`; `gate_source` is either the
/// compact contiguous gate `[B,S,1280]` or the accepted fallback combined
/// QKV+gate buffer `[B,S,5120]` after in-place sigmoid.
///
/// # Panics
///
/// Panics on any dtype/device/shape/stride mismatch, unsupported batch,
/// integer overflow, or insufficient device limits.
pub fn post_sdpa_layout_gate_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    gate_source: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    assert_eq!(attention.meta.num_dims(), 4, "attention must be rank 4");
    let batch = attention.meta.shape()[0];
    let sequence = attention.meta.shape()[2];
    assert_batch(batch);
    assert!(sequence > 0, "post-SDPA sequence must be nonzero");
    let precision = common_float_precision([attention.dtype, gate_source.dtype])
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
    let (gate_stride, gate_offset) = gate_layout(&gate_source, batch, sequence)
        .expect("gate source must be compact gate or combined QKV+gate storage");
    assert_eq!(
        gate_source.dtype,
        precision.dtype(),
        "gate source dtype mismatch"
    );
    attention.assert_is_on_same_device(&gate_source);

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
                gate_stride: checked_u32(gate_stride, "gate stride"),
                gate_offset: checked_u32(gate_offset, "gate offset"),
            },
            CubeDim::new_1d(POST_WORKGROUP_SIZE),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(attention.handle.binding())
        .with_buffer(gate_source.handle.binding())
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

/// Live bytes removed after direct materialization by retaining only Q+gate.
pub const fn compact_q_gate_saved_live_bytes(batch: usize, sequence: usize) -> usize {
    3 * batch * sequence * MODEL_DIM * size_of::<f32>()
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
        assert_eq!(compact_q_gate_saved_live_bytes(3, 489), 22_533_120);
    }

    #[test]
    fn post_sdpa_index_mapping_covers_each_source_once_for_b1_b2_b3() {
        for sequence in [13, 25, 50, 100, 200] {
            for batch_count in [1, 2, 3] {
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
            for context in [3, 11, 22] {
                if context > sequence {
                    continue;
                }
                let total_sequence = sequence + context;
                for batch_count in [1, 2, 3] {
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

                    for row in 0..batch_count * context * NUM_HEADS {
                        let head = row % NUM_HEADS;
                        let token = row / NUM_HEADS;
                        let batch = token / context;
                        let seq = token % context;
                        let base = ((batch * NUM_HEADS + head) * total_sequence + sequence + seq)
                            * HEAD_DIM;
                        for component in 0..HEAD_DIM {
                            assert!(!seen[base + component], "duplicate context K/V index");
                            seen[base + component] = true;
                        }
                    }
                    assert!(seen.into_iter().all(|value| value));
                }
            }
        }
    }

    #[test]
    fn post_sdpa_mapping_preserves_gate_multiply_order() {
        for sequence in [13, 25, 50, 100, 200] {
            for batch_count in [1, 2, 3] {
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
                "direct_norm_rope",
                include_str!("qkv_direct_norm_rope.wgsl"),
                5,
                &["batch", "sequence", "context", "eps"][..],
            ),
            (
                "post",
                include_str!("joint_attention_post_sdpa.wgsl"),
                3,
                &[
                    "elements",
                    "sequence",
                    "gate_stride",
                    "gate_offset",
                    "workgroup_size",
                ][..],
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

    #[test]
    #[ignore = "requires a WGPU adapter and compiles the CubeK scatter candidate"]
    fn cubek_scatter_writes_compact_outputs_without_a_projection_buffer() {
        use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
        use burn::tensor::{Tensor, TensorData};

        let raw_device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::wgpu_device_with_precision(
            &raw_device,
            crate::WgpuFloatPrecision::Fp32,
        )
        .unwrap();
        let (batch, sequence, context) = (1, 3, 3);
        let rows = batch * sequence;
        let input = Tensor::<2>::zeros([rows, MODEL_DIM], &device);
        let weight = Tensor::<2>::zeros([COMBINED_DIM, MODEL_DIM], &device).transpose();
        let qk_weight = Tensor::<3>::ones([2, NUM_HEADS, HEAD_DIM], &device);
        let rope_cos = Tensor::<2>::ones([sequence, HALF_HEAD_DIM], &device);
        let rope_sin = Tensor::<2>::zeros([sequence, HALF_HEAD_DIM], &device);
        let context_values = (0..2 * batch * context * MODEL_DIM)
            .map(|index| index as f32 * 1.0e-4)
            .collect::<Vec<_>>();
        let ctx_kv = Tensor::<1>::from_data(
            TensorData::new(context_values.clone(), [context_values.len()]),
            &device,
        )
        .reshape([2, batch, context, NUM_HEADS, HEAD_DIM]);

        let output = try_cubek_projection_direct_packed_kv(
            input.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            weight.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            qk_weight.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            rope_cos.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            rope_sin.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            ctx_kv.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            batch,
            sequence,
            1.0e-6,
        )
        .expect("exact CubeK scatter contract");
        let q = Tensor::<4>::from_primitive::<crate::WgpuRaw>(output.q)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let k = Tensor::<4>::from_primitive::<crate::WgpuRaw>(output.k_all)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let v = Tensor::<4>::from_primitive::<crate::WgpuRaw>(output.v_all)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let gate = Tensor::<3>::from_primitive::<crate::WgpuRaw>(output.gate)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(q.iter().all(|value| *value == 0.0));
        assert!(gate.iter().all(|value| *value == 0.5));
        for batch_index in 0..batch {
            for head in 0..NUM_HEADS {
                for self_sequence in 0..sequence {
                    let base = ((batch_index * NUM_HEADS + head) * (sequence + context)
                        + self_sequence)
                        * HEAD_DIM;
                    assert!(k[base..base + HEAD_DIM].iter().all(|value| *value == 0.0));
                    assert!(v[base..base + HEAD_DIM].iter().all(|value| *value == 0.0));
                }
                for context_sequence in 0..context {
                    let packed_base = ((batch_index * NUM_HEADS + head) * (sequence + context)
                        + sequence
                        + context_sequence)
                        * HEAD_DIM;
                    let source_base =
                        ((batch_index * context + context_sequence) * NUM_HEADS + head) * HEAD_DIM;
                    assert_eq!(
                        &k[packed_base..packed_base + HEAD_DIM],
                        &context_values[source_base..source_base + HEAD_DIM]
                    );
                    let value_source = batch * context * MODEL_DIM + source_base;
                    assert_eq!(
                        &v[packed_base..packed_base + HEAD_DIM],
                        &context_values[value_source..value_source + HEAD_DIM]
                    );
                }
            }
        }
    }
}
