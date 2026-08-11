// Isolated FP16 fused QKV post-processing candidate.
//
// Persistent projection output, RMS weights, and Q/K/V output are f16. RMS
// reductions and RoPE arithmetic are f32. The f16 normalization temporaries
// are a required semantic boundary: upstream RMSNorm returns half before
// apply_rotary_emb widens that half tensor back to f32. Removing this rounding
// point changes the model even if the fused result looks numerically better.
//
// RoPE cos/sin tables remain f32, matching precompute_freqs_cis (complex64) in
// the pinned Python source. They are not model parameters and must not be cast
// with the f16 model weights.
enable f16;

@group(0) @binding(0) var<storage, read_write> fused_qkv: array<f16>;
@group(0) @binding(1) var<storage, read_write> q_weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> k_weight: array<f16>;
@group(0) @binding(3) var<storage, read_write> rope_cos: array<f32>;
@group(0) @binding(4) var<storage, read_write> rope_sin: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out: array<f16>;
@group(0) @binding(6) var<storage, read_write> k_out: array<f16>;
@group(0) @binding(7) var<storage, read_write> v_out: array<f16>;

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

    var q_local: f32 = 0.0;
    var k_local: f32 = 0.0;
    for (var dim = tid; dim < DH; dim = dim + BLOCK_SIZE) {
        let q_value: f32 = f32(fused_qkv[q_base + dim]);
        let k_value: f32 = f32(fused_qkv[k_base + dim]);
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

    let q_inv_rms: f32 = inverseSqrt(q_sum[0] / f32(DH) + EPS);
    let k_inv_rms: f32 = inverseSqrt(k_sum[0] / f32(DH) + EPS);
    let rotate = head < H / 2u;

    for (var pair = tid; pair < HALF_DH; pair = pair + BLOCK_SIZE) {
        let even = 2u * pair;
        let odd = even + 1u;
        let weight_base = head_offset;

        // Required half boundary between Python RMSNorm and apply_rotary_emb.
        let q_norm_re_half: f16 = f16(
            (f32(fused_qkv[q_base + even]) * q_inv_rms) *
            f32(q_weight[weight_base + even])
        );
        let q_norm_im_half: f16 = f16(
            (f32(fused_qkv[q_base + odd]) * q_inv_rms) *
            f32(q_weight[weight_base + odd])
        );
        let k_norm_re_half: f16 = f16(
            (f32(fused_qkv[k_base + even]) * k_inv_rms) *
            f32(k_weight[weight_base + even])
        );
        let k_norm_im_half: f16 = f16(
            (f32(fused_qkv[k_base + odd]) * k_inv_rms) *
            f32(k_weight[weight_base + odd])
        );

        if (rotate) {
            let q_re: f32 = f32(q_norm_re_half);
            let q_im: f32 = f32(q_norm_im_half);
            let k_re: f32 = f32(k_norm_re_half);
            let k_im: f32 = f32(k_norm_im_half);
            let rope_index = seq * HALF_DH + pair;
            let cos_value: f32 = rope_cos[rope_index];
            let sin_value: f32 = rope_sin[rope_index];

            q_out[out_base + even] = f16(q_re * cos_value - q_im * sin_value);
            q_out[out_base + odd] = f16(q_re * sin_value + q_im * cos_value);
            k_out[out_base + even] = f16(k_re * cos_value - k_im * sin_value);
            k_out[out_base + odd] = f16(k_re * sin_value + k_im * cos_value);
        } else {
            q_out[out_base + even] = q_norm_re_half;
            q_out[out_base + odd] = q_norm_im_half;
            k_out[out_base + even] = k_norm_re_half;
            k_out[out_base + odd] = k_norm_im_half;
        }

        v_out[out_base + even] = fused_qkv[v_base + even];
        v_out[out_base + odd] = fused_qkv[v_base + odd];

        if (WRITES_GATE) {
            let gate_base = fused_token_base + 3u * KV_DIM + head_offset;
            let gate_even: f32 = f32(fused_qkv[gate_base + even]);
            let gate_odd: f32 = f32(fused_qkv[gate_base + odd]);
            // Match the existing Burn sigmoid fallback in f32, then restore the
            // half tensor boundary expected by the following gate multiply.
            let sigmoid_even: f32 = exp(-log(exp(-gate_even) + 1.0));
            let sigmoid_odd: f32 = exp(-log(exp(-gate_odd) + 1.0));
            fused_qkv[gate_base + even] = f16(sigmoid_even);
            fused_qkv[gate_base + odd] = f16(sigmoid_odd);
        }
    }
}
