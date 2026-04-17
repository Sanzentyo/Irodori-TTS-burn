// Tiled FlashAttention SDPA WGSL kernel.
//
// Algorithm: score-parallel 2D tiling with online softmax (FlashAttention-1 style).
// Each workgroup handles TILE_Q query rows, streaming through KV in tiles of TILE_KV.
// No N×N score matrix materialization — memory is O(TILE_Q × TILE_KV), not O(S_Q × S_KV).
//
// Thread mapping: tid → (row = tid / TILE_KV, sec = tid % TILE_KV)
//   Score phase:  sec = KV position index within the tile.
//                 Thread (row, kv) computes the FULL dot product Q[row] · K[kv] over D dims.
//                 No cross-thread reduction needed — each thread owns one score.
//   Output phase: sec = dimension chunk index.
//                 Thread (row, chunk) accumulates output[row][chunk*DPT .. chunk*DPT + DPT).
//
// Per-KV-block barriers: 5 (K load, scores, softmax, V load, output accum).
// Total barriers: ceil(S_KV / TILE_KV) × 5.
//
// Shared memory budget (TILE_Q=16, TILE_KV=8, D=128, PAD=1):
//   q_tile:  16 × 129 × 4 =  8,256 B
//   kv_tile:  8 × 129 × 4 =  4,128 B
//   scores:  16 × 8   × 4 =    512 B
//   row state:           3 × 16 × 4 =    192 B
//   Total: ~13,088 B < 16,384 B (WebGPU limit)  ✓
//
// Template parameters:
//   tile_q, tile_kv:     tile dimensions (query rows, KV rows per block)
//   head_dim:            D — head dimension
//   d_padded:            D + PAD — padded stride for bank conflict avoidance
//   seq_q, seq_kv:       sequence lengths
//   num_heads:           H — number of attention heads
//   scale:               1 / sqrt(D)
//   workgroup_size:      tile_q × tile_kv (must equal head_dim)
//   dims_per_thread:     D / tile_kv — output dims owned per thread
//   q_tile_size, kv_tile_size, scores_size: shared memory array sizes
//   elem:                element type (f32)
//
// Input layout: Q/K/V [B, H, S, D] (contiguous), mask [B, S_KV] as f32 (1.0 = attend)
//
// Platform notes:
// - All shared memory, no subgroup ops — WebGPU forward-compatible.
// - Subgroup ops (enable subgroups;) could replace the serial softmax loop:
//     let tile_max = subgroupMax(my_score);
//     let exp_score = exp(my_score - tile_max);
//     let tile_sum = subgroupAdd(exp_score);
//   This would make all threads participate in softmax, eliminating the serial bottleneck.
//   WebGPU fallback: current single-thread-per-row softmax works everywhere.
// - PAD=1 eliminates shared memory bank conflicts in the score phase:
//     stride 129 → 129 % 32 = 1 → consecutive KV rows hit consecutive banks.
// - f16 variant (enable f16;) would halve K/V bandwidth and double throughput
//   for score computation. WebGPU fallback: this f32 kernel.

// All bindings read_write: WGPU suballocator may pack tensors into shared
// physical buffers, requiring uniform usage flags across bindings.
@group(0) @binding(0) var<storage, read_write> q_buf:      array<{{ elem }}>;
@group(0) @binding(1) var<storage, read_write> k_buf:      array<{{ elem }}>;
@group(0) @binding(2) var<storage, read_write> v_buf:      array<{{ elem }}>;
@group(0) @binding(3) var<storage, read_write> mask_buf:   array<{{ elem }}>;
@group(0) @binding(4) var<storage, read_write> output_buf: array<{{ elem }}>;

const TILE_Q: u32 = {{ tile_q }}u;
const TILE_KV: u32 = {{ tile_kv }}u;
const D: u32 = {{ head_dim }}u;
const D_PAD: u32 = {{ d_padded }}u;
const S_Q: u32 = {{ seq_q }}u;
const S_KV: u32 = {{ seq_kv }}u;
const H: u32 = {{ num_heads }}u;
const SCALE: {{ elem }} = {{ scale }};
const WG_SIZE: u32 = {{ workgroup_size }}u;
const DPT: u32 = {{ dims_per_thread }}u;

// Derived constants
const NUM_Q_TILES: u32 = (S_Q + TILE_Q - 1u) / TILE_Q;
const NUM_KV_BLOCKS: u32 = (S_KV + TILE_KV - 1u) / TILE_KV;

// Online softmax sentinels
const INIT_MAX: {{ elem }} = -1.0e30;
const MASKED_SCORE: {{ elem }} = -1.0e38;

// Shared memory
var<workgroup> q_tile: array<{{ elem }}, {{ q_tile_size }}>;
var<workgroup> kv_tile: array<{{ elem }}, {{ kv_tile_size }}>;
var<workgroup> scores: array<{{ elem }}, {{ scores_size }}>;
var<workgroup> row_max_s: array<{{ elem }}, {{ tile_q }}>;
var<workgroup> row_sum_s: array<{{ elem }}, {{ tile_q }}>;
var<workgroup> rescale_s: array<{{ elem }}, {{ tile_q }}>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id)  local_id: vec3<u32>,
    @builtin(workgroup_id)         group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let wg_idx = group_id.x;

    // Decode (batch, head, q_tile_index) from linear workgroup index
    let b = wg_idx / (H * NUM_Q_TILES);
    let h = (wg_idx / NUM_Q_TILES) % H;
    let qt = wg_idx % NUM_Q_TILES;
    let q_base = qt * TILE_Q;

    // Thread role: (row, sec)
    let row = tid / TILE_KV;
    let sec = tid % TILE_KV;
    let gq = q_base + row;           // global query position
    let valid_q = gq < S_Q;

    // Global base offsets for this (batch, head)
    let q_head_off = (b * H + h) * S_Q * D;
    let kv_head_off = (b * H + h) * S_KV * D;
    let mask_off = b * S_KV;

    // Initialize per-row softmax state in shared memory
    if (sec == 0u) {
        row_max_s[row] = INIT_MAX;
        row_sum_s[row] = 0.0;
        rescale_s[row] = 1.0;
    }

    // Private output accumulator (DPT contiguous dims per thread)
    var out_accum: array<{{ elem }}, {{ dims_per_thread }}>;
    for (var d = 0u; d < DPT; d = d + 1u) {
        out_accum[d] = 0.0;
    }

    // ====== Load Q tile cooperatively ======
    // WG_SIZE == D: each thread loads exactly 1 element per row, TILE_Q rows
    for (var r = 0u; r < TILE_Q; r = r + 1u) {
        let gq_r = q_base + r;
        if (gq_r < S_Q) {
            q_tile[r * D_PAD + tid] = q_buf[q_head_off + gq_r * D + tid];
        } else {
            q_tile[r * D_PAD + tid] = 0.0;
        }
    }
    workgroupBarrier();

    // ====== Stream through KV blocks ======
    for (var kb = 0u; kb < NUM_KV_BLOCKS; kb = kb + 1u) {
        let kv_base = kb * TILE_KV;

        // ------ Step 1: Cooperative K tile load ------
        for (var r = 0u; r < TILE_KV; r = r + 1u) {
            let gkv = kv_base + r;
            if (gkv < S_KV) {
                kv_tile[r * D_PAD + tid] = k_buf[kv_head_off + gkv * D + tid];
            } else {
                kv_tile[r * D_PAD + tid] = 0.0;
            }
        }
        workgroupBarrier();  // A: K tile ready

        // ------ Step 2: Score computation ------
        // Thread (row, sec=kv) computes full dot product Q[row] · K[kv] — D FMAs, no reduction
        var score: {{ elem }} = MASKED_SCORE;
        let gkv = kv_base + sec;
        if (valid_q && gkv < S_KV) {
            var dot: {{ elem }} = 0.0;
            for (var d = 0u; d < D; d = d + 1u) {
                dot = dot + q_tile[row * D_PAD + d] * kv_tile[sec * D_PAD + d];
            }
            let raw_score = dot * SCALE;

            // Apply mask (1.0 = attend, 0.0 = mask-out)
            let mask_val = mask_buf[mask_off + gkv];
            score = select(MASKED_SCORE, raw_score, mask_val > 0.5);
        }
        scores[row * TILE_KV + sec] = score;
        workgroupBarrier();  // B: scores ready

        // ------ Step 3: Online softmax update (1 thread per row) ------
        // Subgroup alternative (DX12/Vulkan/Metal — not WebGPU):
        //   let tile_max = subgroupMax(score);
        //   let w = exp(score - tile_max);
        //   let tile_sum = subgroupAdd(w);
        //   // All threads participate — no serial bottleneck.
        if (sec == 0u) {
            if (valid_q) {
                let prev_max = row_max_s[row];

                // Find tile maximum
                var tile_max = INIT_MAX;
                for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                    tile_max = max(tile_max, scores[row * TILE_KV + kv]);
                }

                let new_max = max(prev_max, tile_max);
                // rescale ∈ (0, 1]: exp(prev - new) ≤ 1.0 since new_max ≥ prev_max
                let rv = exp(prev_max - new_max);

                // Convert scores to attention weights, accumulate sum
                var tile_sum: {{ elem }} = 0.0;
                for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                    let w = exp(scores[row * TILE_KV + kv] - new_max);
                    scores[row * TILE_KV + kv] = w;
                    tile_sum = tile_sum + w;
                }

                row_sum_s[row] = row_sum_s[row] * rv + tile_sum;
                row_max_s[row] = new_max;
                rescale_s[row] = rv;
            } else {
                // Invalid row: no rescaling, weights stay as MASKED_SCORE → exp ≈ 0
                rescale_s[row] = 1.0;
            }
        }
        workgroupBarrier();  // C: weights + rescale ready

        // ------ Step 4: Rescale output accumulator + Cooperative V load ------
        // Rescale previous accumulation for updated max (private state, no barrier)
        let my_rescale = rescale_s[row];
        for (var d = 0u; d < DPT; d = d + 1u) {
            out_accum[d] = out_accum[d] * my_rescale;
        }

        // Cooperative V tile load (reuses kv_tile buffer)
        for (var r = 0u; r < TILE_KV; r = r + 1u) {
            let gkv = kv_base + r;
            if (gkv < S_KV) {
                kv_tile[r * D_PAD + tid] = v_buf[kv_head_off + gkv * D + tid];
            } else {
                kv_tile[r * D_PAD + tid] = 0.0;
            }
        }
        workgroupBarrier();  // D: V tile ready

        // ------ Step 5: Output accumulation ------
        // Thread (row, sec=dim_chunk): accumulate weighted V into private output
        let dim_start = sec * DPT;
        if (valid_q) {
            for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                let weight = scores[row * TILE_KV + kv];
                for (var d = 0u; d < DPT; d = d + 1u) {
                    out_accum[d] = out_accum[d] + weight * kv_tile[kv * D_PAD + dim_start + d];
                }
            }
        }
        workgroupBarrier();  // E: done before next K load overwrites kv_tile
    }

    // ====== Final output: normalize and write ======
    if (valid_q) {
        let sum = row_sum_s[row];
        let dim_start = sec * DPT;
        let out_off = q_head_off + gq * D + dim_start;
        // Output layout is same as Q: [B, H, S_Q, D]
        if (sum > 0.0) {
            let inv_sum = 1.0 / sum;
            for (var d = 0u; d < DPT; d = d + 1u) {
                output_buf[out_off + d] = out_accum[d] * inv_sum;
            }
        } else {
            // All positions masked — output zeros (safe_softmax behavior)
            for (var d = 0u; d < DPT; d = d + 1u) {
                output_buf[out_off + d] = 0.0;
            }
        }
    }
}
