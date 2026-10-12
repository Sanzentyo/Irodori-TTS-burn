// Fused post-processing for either [B,S,3*H*Dh] QKV or the production
// [B,S,4*H*Dh] QKV+gate projection.
//
// Each workgroup owns one (batch, sequence, head) row. It performs two
// Dh-wide RMS reductions, writes unmodified V, and applies adjacent-pair RoPE
// to normalised Q/K for the first floor(H/2) heads. Outputs are independent
// contiguous [B,S,H,Dh] buffers. In the four-segment variant, each workgroup
// also applies sigmoid to its head's gate values in the input buffer.
//
// All bindings are read_write. CubeCL's sliced allocator may place otherwise
// independent tensor handles in one physical WGPU buffer, where mixing
// read-only and read-write storage usage would be invalid.

@group(0) @binding(0) var<storage, read_write> fused_qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_weight:  array<f32>;
@group(0) @binding(2) var<storage, read_write> k_weight:  array<f32>;
@group(0) @binding(3) var<storage, read_write> rope_cos:  array<f32>;
@group(0) @binding(4) var<storage, read_write> rope_sin:  array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out:     array<f32>;
@group(0) @binding(6) var<storage, read_write> k_out:     array<f32>;
@group(0) @binding(7) var<storage, read_write> v_out:     array<f32>;

const S: u32 = {{ seq_len }}u;
const H: u32 = {{ num_heads }}u;
const DH: u32 = {{ head_dim }}u;
const HALF_DH: u32 = {{ half_head_dim }}u;
const KV_DIM: u32 = {{ kv_dim }}u;
const INPUT_WIDTH: u32 = {{ input_width }}u;
const WRITES_GATE: bool = {{ writes_gate }};
const BLOCK_SIZE: u32 = {{ workgroup_size }}u;
const EPS: f32 = {{ eps }};

var<workgroup> q_sum: array<f32, {{ workgroup_size }}>;
var<workgroup> k_sum: array<f32, {{ workgroup_size }}>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let row = group_id.x;
    let head = row % H;
    let token = row / H;
    let seq = token % S;

    let fused_token_base = token * INPUT_WIDTH;
    let head_offset = head * DH;
    let q_base = fused_token_base + head_offset;
    let k_base = fused_token_base + KV_DIM + head_offset;
    let v_base = fused_token_base + 2u * KV_DIM + head_offset;
    let out_base = token * KV_DIM + head_offset;
    let weight_base = head_offset;

    // Reduce both normalisation rows together. BLOCK_SIZE is based on Dh/2,
    // so Dh=64 occupies one 32-lane workgroup and loads two values per lane.
    var q_local = 0.0;
    var k_local = 0.0;
    for (var dim = tid; dim < DH; dim = dim + BLOCK_SIZE) {
        let q_value = fused_qkv[q_base + dim];
        let k_value = fused_qkv[k_base + dim];
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

    // Assign adjacent component pairs to one invocation so no temporary
    // global buffer is needed between RMSNorm and complex multiplication.
    for (var pair = tid; pair < HALF_DH; pair = pair + BLOCK_SIZE) {
        let even = 2u * pair;
        let odd = even + 1u;
        let q_re = fused_qkv[q_base + even] / q_rms * q_weight[weight_base + even];
        let q_im = fused_qkv[q_base + odd] / q_rms * q_weight[weight_base + odd];
        let k_re = fused_qkv[k_base + even] / k_rms * k_weight[weight_base + even];
        let k_im = fused_qkv[k_base + odd] / k_rms * k_weight[weight_base + odd];

        if (rotate) {
            let rope_index = seq * HALF_DH + pair;
            let cos_value = rope_cos[rope_index];
            let sin_value = rope_sin[rope_index];
            q_out[out_base + even] = q_re * cos_value - q_im * sin_value;
            q_out[out_base + odd] = q_re * sin_value + q_im * cos_value;
            k_out[out_base + even] = k_re * cos_value - k_im * sin_value;
            k_out[out_base + odd] = k_re * sin_value + k_im * cos_value;
        } else {
            q_out[out_base + even] = q_re;
            q_out[out_base + odd] = q_im;
            k_out[out_base + even] = k_re;
            k_out[out_base + odd] = k_im;
        }

        v_out[out_base + even] = fused_qkv[v_base + even];
        v_out[out_base + odd] = fused_qkv[v_base + odd];

        if (WRITES_GATE) {
            let gate_base = fused_token_base + 3u * KV_DIM + head_offset;
            let gate_even = fused_qkv[gate_base + even];
            let gate_odd = fused_qkv[gate_base + odd];
            // Match Burn's sigmoid fallback: exp(-log(exp(-x) + 1)).
            fused_qkv[gate_base + even] = exp(-log(exp(-gate_even) + 1.0));
            fused_qkv[gate_base + odd] = exp(-log(exp(-gate_odd) + 1.0));
        }
    }
}
