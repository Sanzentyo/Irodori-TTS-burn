// Native-only tiled FlashAttention SDPA — f16 Q/K/V storage + f32 accumulation.
//
// Variant of fused_sdpa_native.wgsl where Q/K/V buffers and shared-memory tiles
// are stored as f16 (2× global memory bandwidth), while all arithmetic uses f32.
//
// Key differences from fused_sdpa_native.wgsl (f32 baseline):
//   1. Q/K/V input bindings are array<f16> (2 bytes per element vs 4)
//   2. q_tile/kv_tile shared memory use f16 (half the shared-memory capacity)
//   3. Dot products upcast f16 → f32 before FMA (f32 accumulation for precision)
//   4. Softmax state (row_max/sum/rescale) and scores[] remain f32
//   5. Output buffer is f32 (v1 — keeps output type simple)
//
// Shared memory layout (32×8, D=128, PAD=1):
//   q_tile:  32 × 129 × 2 =  8,256 B  (f16)
//   kv_tile:  8 × 129 × 2 =  2,064 B  (f16)
//   scores:  32 × 8   × 4 =  1,024 B  (f32)
//   row state:        3 × 32 × 4 =    384 B  (f32)
//   Total: ~11,728 B — 47% less than f32 baseline (~22,048 B)
//
// Bank-conflict note:
//   V-reads use the strided mapping: thread sec owns dims {sec, sec+TILE_KV, ...}.
//   With f16 elements (2 bytes) and 8 sec threads: each set of 8 accesses spans
//   16 bytes = 4 banks (2-way conflict). Not ideal but much better than the
//   unpadded contiguous layout.
//
// WebGPU note: This kernel requires `shader-f16` GPU feature; not WebGPU portable.
enable f16;

// All bindings read_write: WGPU suballocator may pack tensors into shared
// physical buffers, requiring uniform usage flags across bindings.
@group(0) @binding(0) var<storage, read_write> q_buf:      array<f16>;
@group(0) @binding(1) var<storage, read_write> k_buf:      array<f16>;
@group(0) @binding(2) var<storage, read_write> v_buf:      array<f16>;
@group(0) @binding(3) var<storage, read_write> mask_buf:   array<f32>;
@group(0) @binding(4) var<storage, read_write> output_buf: array<f32>;

const TILE_Q: u32 = {{ tile_q }}u;
const TILE_KV: u32 = {{ tile_kv }}u;
const D: u32 = {{ head_dim }}u;
const D_PAD: u32 = {{ d_padded }}u;
const S_Q: u32 = {{ seq_q }}u;
const S_KV: u32 = {{ seq_kv }}u;
const H: u32 = {{ num_heads }}u;
const SCALE: f32 = {{ scale }};
const WG_SIZE: u32 = {{ workgroup_size }}u;
const DPT: u32 = {{ dims_per_thread }}u;

// Bit-shift constants for D=128 (power of 2): r = p >> 7, d = p & 127
const LOG2_D: u32 = {{ log2_d }}u;
const D_MASK: u32 = {{ d_mask }}u;

// Total elements per tile (for cooperative load loops)
const Q_ELEMS: u32 = TILE_Q * D;
const KV_ELEMS: u32 = TILE_KV * D;

// Derived constants
const NUM_Q_TILES: u32 = (S_Q + TILE_Q - 1u) / TILE_Q;
const NUM_KV_BLOCKS: u32 = (S_KV + TILE_KV - 1u) / TILE_KV;

// Online softmax sentinels (f32)
const INIT_MAX: f32 = -1.0e30;
const MASKED_SCORE: f32 = -1.0e38;

// Shared memory: tiles in f16 (saves capacity), state in f32 (precision)
var<workgroup> q_tile:    array<f16, {{ q_tile_size }}>;
var<workgroup> kv_tile:   array<f16, {{ kv_tile_size }}>;
var<workgroup> scores:    array<f32, {{ scores_size }}>;
var<workgroup> row_max_s: array<f32, {{ tile_q }}>;
var<workgroup> row_sum_s: array<f32, {{ tile_q }}>;
var<workgroup> rescale_s: array<f32, {{ tile_q }}>;

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

    // Private output accumulator in f32 (precision-critical)
    var out_accum: array<f32, {{ dims_per_thread }}>;
    for (var d = 0u; d < DPT; d = d + 1u) {
        out_accum[d] = 0.0;
    }

    // ====== Load Q tile cooperatively (f16 from global, f16 in shared) ======
    for (var p = tid; p < Q_ELEMS; p = p + WG_SIZE) {
        let r = p >> LOG2_D;
        let d = p & D_MASK;
        let gq_r = q_base + r;
        if (gq_r < S_Q) {
            q_tile[r * D_PAD + d] = q_buf[q_head_off + gq_r * D + d];
        } else {
            q_tile[r * D_PAD + d] = f16(0.0);
        }
    }
    workgroupBarrier();

    // ====== Stream through KV blocks ======
    for (var kb = 0u; kb < NUM_KV_BLOCKS; kb = kb + 1u) {
        let kv_base = kb * TILE_KV;

        // ------ Step 1: Cooperative K tile load (f16) ------
        for (var p = tid; p < KV_ELEMS; p = p + WG_SIZE) {
            let r = p >> LOG2_D;
            let d = p & D_MASK;
            let gkv = kv_base + r;
            if (gkv < S_KV) {
                kv_tile[r * D_PAD + d] = k_buf[kv_head_off + gkv * D + d];
            } else {
                kv_tile[r * D_PAD + d] = f16(0.0);
            }
        }
        workgroupBarrier();  // A: K tile ready

        // ------ Step 2: Score computation (upcast f16 → f32, 4-way ILP) ------
        var score: f32 = MASKED_SCORE;
        let gkv = kv_base + sec;
        if (valid_q && gkv < S_KV) {
            let qoff = row * D_PAD;
            let koff = sec * D_PAD;
            var dot0: f32 = 0.0;
            var dot1: f32 = 0.0;
            var dot2: f32 = 0.0;
            var dot3: f32 = 0.0;
            // D is always a multiple of 4 (D=128 in this model)
            for (var dd = 0u; dd < D; dd = dd + 4u) {
                dot0 = fma(f32(q_tile[qoff + dd]),       f32(kv_tile[koff + dd]),       dot0);
                dot1 = fma(f32(q_tile[qoff + dd + 1u]),  f32(kv_tile[koff + dd + 1u]),  dot1);
                dot2 = fma(f32(q_tile[qoff + dd + 2u]),  f32(kv_tile[koff + dd + 2u]),  dot2);
                dot3 = fma(f32(q_tile[qoff + dd + 3u]),  f32(kv_tile[koff + dd + 3u]),  dot3);
            }
            let raw_score = (dot0 + dot1 + dot2 + dot3) * SCALE;

            // Apply mask (1.0 = attend, 0.0 = mask-out)
            let mask_val = mask_buf[mask_off + gkv];
            score = select(MASKED_SCORE, raw_score, mask_val > 0.5);
        }
        scores[row * TILE_KV + sec] = score;
        workgroupBarrier();  // B: scores ready

        // ------ Step 3: Online softmax update (f32) ------
        if (sec == 0u) {
            if (valid_q) {
                let prev_max = row_max_s[row];

                var tile_max: f32 = INIT_MAX;
                for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                    tile_max = max(tile_max, scores[row * TILE_KV + kv]);
                }

                let new_max = max(prev_max, tile_max);
                let rv = exp(prev_max - new_max);

                var tile_sum: f32 = 0.0;
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

        // ------ Step 4: Rescale output accumulator + Cooperative V load (f16) ------
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
                kv_tile[r * D_PAD + d] = f16(0.0);
            }
        }
        workgroupBarrier();  // D: V tile ready

        // ------ Step 5: Output accumulation (strided dim mapping, upcast f16 → f32) ------
        if (valid_q) {
            for (var kv = 0u; kv < TILE_KV; kv = kv + 1u) {
                let weight = scores[row * TILE_KV + kv];
                for (var d = 0u; d < DPT; d = d + 1u) {
                    out_accum[d] = fma(weight, f32(kv_tile[kv * D_PAD + sec + d * TILE_KV]), out_accum[d]);
                }
            }
        }
        workgroupBarrier();  // E: done before next K load overwrites kv_tile
    }

    // ====== Final output: normalize and write f32 (strided dim mapping) ======
    if (valid_q) {
        let sum = row_sum_s[row];
        let out_base = q_head_off + gq * D;
        if (sum > 0.0) {
            let inv_sum = 1.0 / sum;
            for (var d = 0u; d < DPT; d = d + 1u) {
                output_buf[out_base + sec + d * TILE_KV] = out_accum[d] * inv_sum;
            }
        } else {
            for (var d = 0u; d < DPT; d = d + 1u) {
                output_buf[out_base + sec + d * TILE_KV] = 0.0;
            }
        }
    }
}
