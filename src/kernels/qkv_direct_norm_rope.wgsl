// In-place Q/K RMSNorm and adjacent-pair RoPE for CubeK direct QKV scatter.
// Q and gate share binding 0; only the Q prefix is touched here. Context K is
// already copied into the K tail and remains unchanged.

@group(0) @binding(0) var<storage, read_write> q_gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_all: array<f32>;
@group(0) @binding(2) var<storage, read_write> qk_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> rope_cos: array<f32>;
@group(0) @binding(4) var<storage, read_write> rope_sin: array<f32>;

const BATCH: u32 = {{ batch }}u;
const S: u32 = {{ sequence }}u;
const CTX: u32 = {{ context }}u;
const TOTAL_S: u32 = S + CTX;
const H: u32 = 20u;
const DH: u32 = 64u;
const HALF_DH: u32 = 32u;
const MODEL_DIM: u32 = H * DH;
const EPS: f32 = {{ eps }};

var<workgroup> q_sum: array<f32, 32>;
var<workgroup> k_sum: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let lane = local_id.x;
    let row = group_id.x;
    let head = row % H;
    let token = row / H;
    let batch = token / S;
    let sequence = token - batch * S;
    let q_base = ((batch * H + head) * S + sequence) * DH;
    let k_base = ((batch * H + head) * TOTAL_S + sequence) * DH;

    let q0 = q_gate[q_base + lane];
    let q1 = q_gate[q_base + lane + 32u];
    let k0 = k_all[k_base + lane];
    let k1 = k_all[k_base + lane + 32u];
    q_sum[lane] = q0 * q0 + q1 * q1;
    k_sum[lane] = k0 * k0 + k1 * k1;
    workgroupBarrier();

    var stride = 16u;
    while (stride > 0u) {
        if (lane < stride) {
            q_sum[lane] = q_sum[lane] + q_sum[lane + stride];
            k_sum[lane] = k_sum[lane] + k_sum[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let q_rms = sqrt(q_sum[0] / f32(DH) + EPS);
    let k_rms = sqrt(k_sum[0] / f32(DH) + EPS);
    let weight_base = head * DH;
    let even = 2u * lane;
    let odd = even + 1u;
    let q_re = q_gate[q_base + even] / q_rms * qk_weight[weight_base + even];
    let q_im = q_gate[q_base + odd] / q_rms * qk_weight[weight_base + odd];
    let k_weight_base = MODEL_DIM + weight_base;
    let k_re = k_all[k_base + even] / k_rms * qk_weight[k_weight_base + even];
    let k_im = k_all[k_base + odd] / k_rms * qk_weight[k_weight_base + odd];
    if (head < H / 2u) {
        let rope_index = sequence * HALF_DH + lane;
        let cos_value = rope_cos[rope_index];
        let sin_value = rope_sin[rope_index];
        q_gate[q_base + even] = q_re * cos_value - q_im * sin_value;
        q_gate[q_base + odd] = q_re * sin_value + q_im * cos_value;
        k_all[k_base + even] = k_re * cos_value - k_im * sin_value;
        k_all[k_base + odd] = k_re * sin_value + k_im * cos_value;
    } else {
        q_gate[q_base + even] = q_re;
        q_gate[q_base + odd] = q_im;
        k_all[k_base + even] = k_re;
        k_all[k_base + odd] = k_im;
    }
}
