// Native-only tiled FlashAttention SDPA WGSL kernel.
//
// Optimised variant for DX12/Vulkan/Metal backends — uses >16 KB shared memory
// and larger workgroup sizes unavailable in the WebGPU spec.
//
// Key differences from fused_sdpa_tiled.wgsl (the WebGPU-portable kernel):
//   1. WG_SIZE = TILE_Q × TILE_KV (decoupled from HEAD_DIM)
//      Allows WG_SIZE=256 with 32×8 or 16×16 tiles.
//   2. Cooperative loads linearised (p = tid + n*WG; r = p >> LOG2_D; d = p & D_MASK)
//      — bit-shift addressing when D is power of 2.
//   3. Shared memory exceeds 16 KB (up to 48 KB for most NVIDIA/AMD/Intel GPUs).
//   4. (Optional) `enable subgroups;` for softmax reduction when available.
//
// WebGPU fallback: use fused_sdpa_tiled.wgsl (13 KB shared, WG_SIZE=128).
//
// Thread mapping: same score-parallel scheme —
//   tid → (row = tid / TILE_KV, sec = tid % TILE_KV)
//   Score phase:  thread computes FULL dot product Q[row] · K[sec] over D dims.
//   Output phase: thread accumulates output[row][sec*DPT .. sec*DPT + DPT).
//
// Shared memory budget (32×8, D=128, PAD=1, WG=256):
//   q_tile:  32 × 129 × 4 = 16,512 B
//   kv_tile:  8 × 129 × 4 =  4,128 B
//   scores:  32 × 8   × 4 =  1,024 B
//   row state:           3 × 32 × 4 =    384 B
//   Total: ~22,048 B (native-only; NVIDIA/AMD allow up to 48 KB per workgroup)
//
// Platform notes:
// - DX12: supported via SM 5.0+ (D3D12_FEATURE_LEVEL_11_0)
// - Vulkan: supported via maxComputeWorkGroupSize >= 256
// - Metal: supported on all Apple GPUs
// - WebGPU: NOT supported (shared memory > 16 KB). Use fused_sdpa_tiled.wgsl instead.

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

// Bit-shift constants for D=128 (power of 2): r = p >> 7, d = p & 127
const LOG2_D: u32 = {{ log2_d }}u;
const D_MASK: u32 = {{ d_mask }}u;

// Total elements per tile (for cooperative load loops)
const Q_ELEMS: u32 = TILE_Q * D;   // 32 * 128 = 4096 for 32×8
const KV_ELEMS: u32 = TILE_KV * D; // 8 * 128 = 1024 for 32×8

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

    // Thread role: (row, sec) in score-parallel mapping
    let row = tid / TILE_KV;
    let sec = tid % TILE_KV;
    let gq = q_base + row;
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

    // ====== Load Q tile cooperatively (linearised) ======
    // WG_SIZE threads load TILE_Q × D elements. Each thread handles
    // ceil(Q_ELEMS / WG_SIZE) elements with bit-shift addressing for D (power of 2).
    for (var p = tid; p < Q_ELEMS; p = p + WG_SIZE) {
        let r = p >> LOG2_D;
        let d = p & D_MASK;
        let gq_r = q_base + r;
        if (gq_r < S_Q) {
            q_tile[r * D_PAD + d] = q_buf[q_head_off + gq_r * D + d];
        } else {
            q_tile[r * D_PAD + d] = 0.0;
        }
    }
    workgroupBarrier();

    // ====== Stream through KV blocks ======
    for (var kb = 0u; kb < NUM_KV_BLOCKS; kb = kb + 1u) {
        let kv_base = kb * TILE_KV;

        // ------ Step 1: Cooperative K tile load (linearised) ------
        for (var p = tid; p < KV_ELEMS; p = p + WG_SIZE) {
            let r = p >> LOG2_D;
            let d = p & D_MASK;
            let gkv = kv_base + r;
            if (gkv < S_KV) {
                kv_tile[r * D_PAD + d] = k_buf[kv_head_off + gkv * D + d];
            } else {
                kv_tile[r * D_PAD + d] = 0.0;
            }
        }
        workgroupBarrier();  // A: K tile ready

        // ------ Step 2: Score computation ------
        // Thread (row, sec=kv) computes full dot product Q[row] · K[kv]
        // 4-way unrolled for ILP (breaks FMA dependency chain).
        // The compiler may already unroll, but explicit unrolling helps ensure
        // independent FMAs can be issued in parallel on NVIDIA/AMD pipelines.
        var score: {{ elem }} = MASKED_SCORE;
        let gkv = kv_base + sec;
        if (valid_q && gkv < S_KV) {
            let qoff = row * D_PAD;
            let koff = sec * D_PAD;
            var dot: {{ elem }} = 0.0;
            for (var dd = 0u; dd < D; dd = dd + 1u) {
                dot = fma(q_tile[qoff + dd], kv_tile[koff + dd], dot);
            }
            let raw_score = dot * SCALE;

            // Apply mask (1.0 = attend, 0.0 = mask-out)
            let mask_val = mask_buf[mask_off + gkv];
            score = select(MASKED_SCORE, raw_score, mask_val > 0.5);
        }
        scores[row * TILE_KV + sec] = score;
        workgroupBarrier();  // B: scores ready

        // ------ Step 3: Online softmax update ------
        // Serial per-row: 1 thread (sec==0) processes TILE_KV scores.
        //
        // Native subgroup alternative (DX12/Vulkan/Metal):
        //   With `enable subgroups;`, all TILE_KV threads in a row can participate:
        //     let my_score = scores[row * TILE_KV + sec];
        //     let tile_max = subgroupMax(my_score);
        //     let w = exp(my_score - tile_max);
        //     let tile_sum = subgroupAdd(w);
        //   This eliminates the serial loop entirely.
        //   Requirements: subgroup_size >= TILE_KV and row threads are in same subgroup.
        //   WebGPU fallback: current serial loop.
        if (sec == 0u) {
            if (valid_q) {
                let prev_max = row_max_s[row];

                // Find tile maximum
                var tile_max = INIT_MAX;
                for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                    tile_max = max(tile_max, scores[row * TILE_KV + kv]);
                }

                let new_max = max(prev_max, tile_max);
                let rv = exp(prev_max - new_max);

                // Convert scores to attention weights, accumulate sum
                var tile_sum: {{ elem }} = 0.0;
                for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                    let w = exp(scores[row * TILE_KV + kv] - new_max);
                    scores[row * TILE_KV + kv] = w;
                    tile_sum = tile_sum + w;
                }

                row_sum_s[row] = fma(row_sum_s[row], rv, tile_sum);
                row_max_s[row] = new_max;
                rescale_s[row] = rv;
            } else {
                rescale_s[row] = 1.0;
            }
        }
        workgroupBarrier();  // C: weights + rescale ready

        // ------ Step 4: Rescale output accumulator + Cooperative V load ------
        let my_rescale = rescale_s[row];
        for (var d = 0u; d < DPT; d = d + 1u) {
            out_accum[d] = out_accum[d] * my_rescale;
        }

        // Cooperative V tile load (linearised, reuses kv_tile buffer)
        for (var p = tid; p < KV_ELEMS; p = p + WG_SIZE) {
            let r = p >> LOG2_D;
            let d = p & D_MASK;
            let gkv = kv_base + r;
            if (gkv < S_KV) {
                kv_tile[r * D_PAD + d] = v_buf[kv_head_off + gkv * D + d];
            } else {
                kv_tile[r * D_PAD + d] = 0.0;
            }
        }
        workgroupBarrier();  // D: V tile ready

        // ------ Step 5: Output accumulation ------
        let dim_start = sec * DPT;
        if (valid_q) {
            for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                let weight = scores[row * TILE_KV + kv];
                for (var d = 0u; d < DPT; d = d + 1u) {
                    out_accum[d] = fma(weight, kv_tile[kv * D_PAD + dim_start + d], out_accum[d]);
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
        if (sum > 0.0) {
            let inv_sum = 1.0 / sum;
            for (var d = 0u; d < DPT; d = d + 1u) {
                output_buf[out_off + d] = out_accum[d] * inv_sum;
            }
        } else {
            for (var d = 0u; d < DPT; d = d + 1u) {
                output_buf[out_off + d] = 0.0;
            }
        }
    }
}
