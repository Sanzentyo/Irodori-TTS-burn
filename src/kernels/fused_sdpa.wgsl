// Fused Scaled Dot-Product Attention (SDPA) WGSL kernel.
//
// Algorithm: row-streaming with online softmax (FlashAttention-1 style, untiled).
// Each workgroup handles one (batch, head, query_position).
// 32 threads per workgroup, each owning D/32 dimensions of the output.
//
// Key advantage: avoids materializing the S_Q × S_KV attention score matrix.
// Single kernel launch replaces: Q@K^T → scale → mask → softmax → @V.
//
// Input layout: [B, H, S, D] (contiguous)
// Mask layout:  [B, S_KV] as f32 (1.0 = attend, 0.0 = mask-out)
//
// Template parameters:
//   workgroup_size:  threads per workgroup (32 = one NVIDIA warp)
//   head_dim:        D — head dimension (must be divisible by workgroup_size)
//   seq_kv:          S_KV — key/value sequence length
//   seq_q:           S_Q — query sequence length
//   num_heads:       H — number of attention heads
//   dims_per_thread: D / workgroup_size
//   scale:           1 / sqrt(D) — attention scaling factor
//   elem:            element type (f32)
//
// Platform notes:
// - Workgroup size 32 matches NVIDIA warp size — barriers are essentially
//   free within a single warp (lockstep execution). On AMD (wave64) and
//   Intel, barriers are slightly more expensive but still efficient.
// - Subgroup operations (subgroupAdd) would eliminate tree reduction:
//     enable subgroups;
//     partial = subgroupAdd(partial);  // replaces 5 barriers
//   WebGPU fallback: current shared-memory tree reduction works everywhere.
// - f16 variant (enable f16;) would halve K/V memory bandwidth:
//     enable f16;
//     var q_vals: array<f16, DPT>;  // f16 storage, f32 accumulation
//   WebGPU fallback: this f32 kernel.

// All bindings read_write: WGPU suballocator may pack tensors into shared
// physical buffers, requiring uniform usage flags across bindings.
@group(0) @binding(0) var<storage, read_write> q_buf:      array<{{ elem }}>;
@group(0) @binding(1) var<storage, read_write> k_buf:      array<{{ elem }}>;
@group(0) @binding(2) var<storage, read_write> v_buf:      array<{{ elem }}>;
@group(0) @binding(3) var<storage, read_write> mask_buf:   array<{{ elem }}>;
@group(0) @binding(4) var<storage, read_write> output_buf: array<{{ elem }}>;

const WG_SIZE: u32 = {{ workgroup_size }}u;
const D: u32 = {{ head_dim }}u;
const S_KV: u32 = {{ seq_kv }}u;
const S_Q: u32 = {{ seq_q }}u;
const H: u32 = {{ num_heads }}u;
const DPT: u32 = {{ dims_per_thread }}u;
const SCALE: {{ elem }} = {{ scale }};

// Sentinel values for online softmax with safe masking.
// INIT_MAX must be > MASKED_SCORE so that masked positions produce weight ≈ 0.
const INIT_MAX: {{ elem }} = -1.0e30;
const MASKED_SCORE: {{ elem }} = -1.0e38;

// Shared memory for dot-product tree reduction (one entry per thread).
var<workgroup> shared_partial: array<{{ elem }}, {{ workgroup_size }}>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id)  local_id: vec3<u32>,
    @builtin(workgroup_id)         group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let wg_idx = group_id.x;

    // Decode (batch, head, query_pos) from linear workgroup index.
    let b = wg_idx / (H * S_Q);
    let h = (wg_idx / S_Q) % H;
    let q_pos = wg_idx % S_Q;

    // Dimensions owned by this thread: [dim_start .. dim_start + DPT)
    let dim_start = tid * DPT;

    // ---- Load Q row into registers (once per workgroup lifetime) ----
    let q_offset = ((b * H + h) * S_Q + q_pos) * D + dim_start;
    var q_vals: array<{{ elem }}, {{ dims_per_thread }}>;
    for (var d = 0u; d < DPT; d = d + 1u) {
        q_vals[d] = q_buf[q_offset + d];
    }

    // ---- Online softmax accumulators (per-thread private state) ----
    var max_score: {{ elem }} = INIT_MAX;
    var sum_exp: {{ elem }} = 0.0;
    var out_accum: array<{{ elem }}, {{ dims_per_thread }}>;
    for (var d = 0u; d < DPT; d = d + 1u) {
        out_accum[d] = 0.0;
    }

    // Precompute base offsets
    let kv_head_base = (b * H + h) * S_KV * D;
    let mask_base = b * S_KV;

    // ---- Stream through all K/V positions ----
    for (var j = 0u; j < S_KV; j = j + 1u) {
        let kv_offset = kv_head_base + j * D + dim_start;

        // Partial dot product: DPT multiply-adds per thread
        var partial: {{ elem }} = 0.0;
        for (var d = 0u; d < DPT; d = d + 1u) {
            partial = partial + q_vals[d] * k_buf[kv_offset + d];
        }

        // Tree reduction to compute full dot product across all threads.
        // With workgroup_size=32 (one warp on NVIDIA), barriers are ~free.
        shared_partial[tid] = partial;
        workgroupBarrier();

        var stride = WG_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                shared_partial[tid] = shared_partial[tid] + shared_partial[tid + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        // Broadcast score to all threads (shared_partial[0] has full dot product)
        var score = shared_partial[0] * SCALE;

        // Apply mask: invalid positions get MASKED_SCORE → weight ≈ exp(-1e38) ≈ 0.
        // Using MASKED_SCORE (-1e38) distinct from INIT_MAX (-1e30) prevents
        // the NaN trap: exp(MASKED_SCORE - INIT_MAX) = exp(-1e38) = 0, not NaN.
        let mask_val = mask_buf[mask_base + j];
        score = select(MASKED_SCORE, score, mask_val > 0.5);

        // Online softmax recurrence (Milakov & Gimelshein, 2018):
        //   max_new = max(max_old, score)
        //   correction = exp(max_old - max_new)  [rescales previous accumulator]
        //   weight = exp(score - max_new)
        let prev_max = max_score;
        max_score = max(max_score, score);
        let correction = exp(prev_max - max_score);
        let weight = exp(score - max_score);

        sum_exp = sum_exp * correction + weight;

        // Accumulate weighted V with running-max correction
        for (var d = 0u; d < DPT; d = d + 1u) {
            out_accum[d] = out_accum[d] * correction + weight * v_buf[kv_offset + d];
        }
    }

    // ---- Final normalization and output ----
    let out_offset = ((b * H + h) * S_Q + q_pos) * D + dim_start;
    if (sum_exp > 0.0) {
        for (var d = 0u; d < DPT; d = d + 1u) {
            output_buf[out_offset + d] = out_accum[d] / sum_exp;
        }
    } else {
        // All positions masked — output zeros (matches safe_softmax behavior)
        for (var d = 0u; d < DPT; d = d + 1u) {
            output_buf[out_offset + d] = 0.0;
        }
    }
}
