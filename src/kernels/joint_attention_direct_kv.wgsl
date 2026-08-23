// Production exact-shape JointAttention materialization kernel for v4-Small.
//
// This is the accepted QKV+gate post-process arithmetic with two layout-only
// changes: Q/K norm weights share one packed binding, and K/V are written
// directly as contiguous head-major [B,H,S+3,Dh] storage. Context K/V share
// one packed [2,B,3,H,Dh] binding. Q is likewise written as contiguous
// [B,H,S,Dh], removing the post-kernel transpose/materialization before SDPA.
// Q and sigmoid(gate) share one two-plane allocation, so the projection's
// 4*D source can die before SDPA without adding a ninth storage binding. The
// eight-storage-binding WebGPU guarantee is therefore preserved.
//
// Every storage binding is read_write because CubeCL's sliced allocator can
// place otherwise independent logical buffers in one physical WGPU buffer.

@group(0) @binding(0) var<storage, read_write> combined:  array<f32>;
@group(0) @binding(1) var<storage, read_write> qk_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> rope_cos:  array<f32>;
@group(0) @binding(3) var<storage, read_write> rope_sin:  array<f32>;
@group(0) @binding(4) var<storage, read_write> ctx_kv:    array<f32>;
@group(0) @binding(5) var<storage, read_write> q_gate:    array<f32>;
@group(0) @binding(6) var<storage, read_write> k_all:     array<f32>;
@group(0) @binding(7) var<storage, read_write> v_all:     array<f32>;

const BATCH: u32 = {{ batch }}u;
const S: u32 = {{ sequence }}u;
const CTX: u32 = 3u;
const TOTAL_S: u32 = {{ total_sequence }}u;
const H: u32 = 20u;
const DH: u32 = 64u;
const HALF_DH: u32 = 32u;
const KV_DIM: u32 = 1280u;
const INPUT_WIDTH: u32 = 5120u;
const Q_ELEMENTS: u32 = BATCH * S * KV_DIM;
const BLOCK_SIZE: u32 = 32u;
const EPS: f32 = {{ eps }};

var<workgroup> q_sum: array<f32, 32>;
var<workgroup> k_sum: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let row = group_id.x;
    let head = row % H;
    let token = row / H;
    let batch = token / S;
    let seq = token % S;

    let combined_base = token * INPUT_WIDTH;
    let head_offset = head * DH;
    let q_base = combined_base + head_offset;
    let k_base = combined_base + KV_DIM + head_offset;
    let v_base = combined_base + 2u * KV_DIM + head_offset;
    let q_out_base = ((batch * H + head) * S + seq) * DH;
    let all_out_base = ((batch * H + head) * TOTAL_S + seq) * DH;

    // Preserve the production shader's exact per-lane loads and tree order.
    var q_local = 0.0;
    var k_local = 0.0;
    for (var dim = tid; dim < DH; dim = dim + BLOCK_SIZE) {
        let q_value = combined[q_base + dim];
        let k_value = combined[k_base + dim];
        q_local = q_local + q_value * q_value;
        k_local = k_local + k_value * k_value;
    }
    q_sum[tid] = q_local;
    k_sum[tid] = k_local;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            q_sum[tid] = q_sum[tid] + q_sum[tid + stride];
            k_sum[tid] = k_sum[tid] + k_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let q_rms = sqrt(q_sum[0] / f32(DH) + EPS);
    let k_rms = sqrt(k_sum[0] / f32(DH) + EPS);
    let rotate = head < H / 2u;

    for (var pair = tid; pair < HALF_DH; pair = pair + BLOCK_SIZE) {
        let even = 2u * pair;
        let odd = even + 1u;
        let q_re = combined[q_base + even] / q_rms * qk_weight[head_offset + even];
        let q_im = combined[q_base + odd] / q_rms * qk_weight[head_offset + odd];
        let k_weight_base = KV_DIM + head_offset;
        let k_re = combined[k_base + even] / k_rms * qk_weight[k_weight_base + even];
        let k_im = combined[k_base + odd] / k_rms * qk_weight[k_weight_base + odd];

        if (rotate) {
            let rope_index = seq * HALF_DH + pair;
            let cos_value = rope_cos[rope_index];
            let sin_value = rope_sin[rope_index];
            q_gate[q_out_base + even] = q_re * cos_value - q_im * sin_value;
            q_gate[q_out_base + odd] = q_re * sin_value + q_im * cos_value;
            k_all[all_out_base + even] = k_re * cos_value - k_im * sin_value;
            k_all[all_out_base + odd] = k_re * sin_value + k_im * cos_value;
        } else {
            q_gate[q_out_base + even] = q_re;
            q_gate[q_out_base + odd] = q_im;
            k_all[all_out_base + even] = k_re;
            k_all[all_out_base + odd] = k_im;
        }

        v_all[all_out_base + even] = combined[v_base + even];
        v_all[all_out_base + odd] = combined[v_base + odd];

        let gate_base = combined_base + 3u * KV_DIM + head_offset;
        let gate_even = combined[gate_base + even];
        let gate_odd = combined[gate_base + odd];
        // Match Burn's sigmoid fallback and the production QKV+gate shader.
        let gate_out_base = Q_ELEMENTS + (batch * S + seq) * KV_DIM + head_offset;
        q_gate[gate_out_base + even] = exp(-log(exp(-gate_even) + 1.0));
        q_gate[gate_out_base + odd] = exp(-log(exp(-gate_odd) + 1.0));
    }

    // A subset of the already-dispatched self rows copies the cached tail.
    // These writes are disjoint from the self prefix and preserve K/V values.
    if (row < BATCH * CTX * H) {
        let ctx_head = row % H;
        let ctx_token = row / H;
        let ctx_batch = ctx_token / CTX;
        let ctx_seq = ctx_token % CTX;
        let ctx_base = (ctx_batch * CTX + ctx_seq) * KV_DIM + ctx_head * DH;
        let ctx_plane = BATCH * CTX * KV_DIM;
        let ctx_out_base = ((ctx_batch * H + ctx_head) * TOTAL_S + S + ctx_seq) * DH;

        for (var pair = tid; pair < HALF_DH; pair = pair + BLOCK_SIZE) {
            let even = 2u * pair;
            let odd = even + 1u;
            k_all[ctx_out_base + even] = ctx_kv[ctx_base + even];
            k_all[ctx_out_base + odd] = ctx_kv[ctx_base + odd];
            v_all[ctx_out_base + even] = ctx_kv[ctx_plane + ctx_base + even];
            v_all[ctx_out_base + odd] = ctx_kv[ctx_plane + ctx_base + odd];
        }
    }
}
