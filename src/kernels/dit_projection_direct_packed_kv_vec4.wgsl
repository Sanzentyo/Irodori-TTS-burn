// One-dispatch strict-f32 JointAttention front end with vectorized K16 staging.
//
// The logical operation matches dit_projection_direct_packed_kv.wgsl. Input
// rows and row-major projection weights are loaded as vec4 values, reducing
// shared memory from 32 KiB to 20 KiB while preserving the scalar FMA order.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> qk_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> rope_cos: array<f32>;
@group(0) @binding(4) var<storage, read_write> rope_sin: array<f32>;
@group(0) @binding(5) var<storage, read_write> ctx_kv: array<f32>;
@group(0) @binding(6) var<storage, read_write> q_gate: array<f32>;
@group(0) @binding(7) var<storage, read_write> k_all: array<f32>;
@group(0) @binding(8) var<storage, read_write> v_all: array<f32>;

const BATCH: u32 = {{ batch }}u;
const S: u32 = {{ sequence }}u;
const ROWS: u32 = BATCH * S;
const CTX: u32 = {{ context }}u;
const TOTAL_S: u32 = S + CTX;
const K: u32 = 1280u;
const K_VECS: u32 = K / 4u;
const N: u32 = 5120u;
const N_VECS: u32 = N / 4u;
const H: u32 = 20u;
const DH: u32 = 64u;
const HALF_DH: u32 = 32u;
const MODEL_DIM: u32 = H * DH;
const Q_ELEMENTS: u32 = ROWS * MODEL_DIM;
const TILE_ROWS: u32 = 64u;
const TILE_COLUMNS: u32 = 128u;
const TILE_K: u32 = 16u;
const TILE_K_VECS: u32 = TILE_K / 4u;
const LOCAL_ROWS: u32 = 8u;
const LOCAL_COLUMN_VECS: u32 = 32u;
const EPS: f32 = {{ eps }};

var<workgroup> input_tile: array<vec4<f32>, 256>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;
var<workgroup> norm_partial: array<f32, 2048>;

fn store_frontend_value(
    component: u32,
    token: u32,
    head: u32,
    dim_base: u32,
    value: vec4<f32>,
    inv_rms: f32,
) {
    let batch = token / S;
    let seq = token - batch * S;
    let q_offset = ((batch * H + head) * S + seq) * DH + dim_base;
    let kv_offset = ((batch * H + head) * TOTAL_S + seq) * DH + dim_base;
    if (component == 0u || component == 1u) {
        let weight_offset = component * MODEL_DIM + head * DH + dim_base;
        let normalized = value * inv_rms * vec4<f32>(
            qk_weight[weight_offset],
            qk_weight[weight_offset + 1u],
            qk_weight[weight_offset + 2u],
            qk_weight[weight_offset + 3u],
        );
        var transformed = normalized;
        if (head < H / 2u) {
            let pair = dim_base / 2u;
            let rope_base = seq * HALF_DH + pair;
            let cos_0 = rope_cos[rope_base];
            let sin_0 = rope_sin[rope_base];
            let cos_1 = rope_cos[rope_base + 1u];
            let sin_1 = rope_sin[rope_base + 1u];
            transformed = vec4<f32>(
                normalized.x * cos_0 - normalized.y * sin_0,
                normalized.x * sin_0 + normalized.y * cos_0,
                normalized.z * cos_1 - normalized.w * sin_1,
                normalized.z * sin_1 + normalized.w * cos_1,
            );
        }
        if (component == 0u) {
            q_gate[q_offset] = transformed.x;
            q_gate[q_offset + 1u] = transformed.y;
            q_gate[q_offset + 2u] = transformed.z;
            q_gate[q_offset + 3u] = transformed.w;
        } else {
            k_all[kv_offset] = transformed.x;
            k_all[kv_offset + 1u] = transformed.y;
            k_all[kv_offset + 2u] = transformed.z;
            k_all[kv_offset + 3u] = transformed.w;
        }
    } else if (component == 2u) {
        v_all[kv_offset] = value.x;
        v_all[kv_offset + 1u] = value.y;
        v_all[kv_offset + 2u] = value.z;
        v_all[kv_offset + 3u] = value.w;
    } else {
        let gate_offset = Q_ELEMENTS + token * MODEL_DIM + head * DH + dim_base;
        let gate = exp(-log(exp(-value) + vec4<f32>(1.0)));
        q_gate[gate_offset] = gate.x;
        q_gate[gate_offset + 1u] = gate.y;
        q_gate[gate_offset + 2u] = gate.z;
        q_gate[gate_offset + 3u] = gate.w;
    }
}

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let row_base = group_id.y * TILE_ROWS;
    let column_vec_base = group_id.x * LOCAL_COLUMN_VECS;
    var acc_0 = vec4<f32>(0.0);
    var acc_1 = vec4<f32>(0.0);
    var acc_2 = vec4<f32>(0.0);
    var acc_3 = vec4<f32>(0.0);
    var acc_4 = vec4<f32>(0.0);
    var acc_5 = vec4<f32>(0.0);
    var acc_6 = vec4<f32>(0.0);
    var acc_7 = vec4<f32>(0.0);

    for (var k_base = 0u; k_base < K; k_base = k_base + TILE_K) {
        for (var load = local_index; load < TILE_ROWS * TILE_K_VECS; load = load + 256u) {
            let tile_row = load / TILE_K_VECS;
            let tile_k_vec = load - tile_row * TILE_K_VECS;
            let row = row_base + tile_row;
            var value = vec4<f32>(0.0);
            if (row < ROWS) {
                value = input[row * K_VECS + k_base / 4u + tile_k_vec];
            }
            input_tile[load] = value;
        }
        for (var load = local_index; load < TILE_K * LOCAL_COLUMN_VECS; load = load + 256u) {
            let tile_k = load / LOCAL_COLUMN_VECS;
            let tile_column_vec = load - tile_k * LOCAL_COLUMN_VECS;
            let column_vec = column_vec_base + tile_column_vec;
            var value = vec4<f32>(0.0);
            if (column_vec < N_VECS) {
                value = weight[(k_base + tile_k) * N_VECS + column_vec];
            }
            weight_tile[load] = value;
        }
        workgroupBarrier();
        for (var tile_k_vec = 0u; tile_k_vec < TILE_K_VECS; tile_k_vec = tile_k_vec + 1u) {
            let k0 = tile_k_vec * 4u;
            let w0 = weight_tile[k0 * LOCAL_COLUMN_VECS + local_id.x];
            let w1 = weight_tile[(k0 + 1u) * LOCAL_COLUMN_VECS + local_id.x];
            let w2 = weight_tile[(k0 + 2u) * LOCAL_COLUMN_VECS + local_id.x];
            let w3 = weight_tile[(k0 + 3u) * LOCAL_COLUMN_VECS + local_id.x];
            let r0 = local_id.y;
            let r1 = r0 + LOCAL_ROWS;
            let r2 = r1 + LOCAL_ROWS;
            let r3 = r2 + LOCAL_ROWS;
            let r4 = r3 + LOCAL_ROWS;
            let r5 = r4 + LOCAL_ROWS;
            let r6 = r5 + LOCAL_ROWS;
            let r7 = r6 + LOCAL_ROWS;
            let a0 = input_tile[r0 * TILE_K_VECS + tile_k_vec];
            let a1 = input_tile[r1 * TILE_K_VECS + tile_k_vec];
            let a2 = input_tile[r2 * TILE_K_VECS + tile_k_vec];
            let a3 = input_tile[r3 * TILE_K_VECS + tile_k_vec];
            let a4 = input_tile[r4 * TILE_K_VECS + tile_k_vec];
            let a5 = input_tile[r5 * TILE_K_VECS + tile_k_vec];
            let a6 = input_tile[r6 * TILE_K_VECS + tile_k_vec];
            let a7 = input_tile[r7 * TILE_K_VECS + tile_k_vec];
            acc_0 = fma(vec4<f32>(a0.x), w0, acc_0);
            acc_0 = fma(vec4<f32>(a0.y), w1, acc_0);
            acc_0 = fma(vec4<f32>(a0.z), w2, acc_0);
            acc_0 = fma(vec4<f32>(a0.w), w3, acc_0);
            acc_1 = fma(vec4<f32>(a1.x), w0, acc_1);
            acc_1 = fma(vec4<f32>(a1.y), w1, acc_1);
            acc_1 = fma(vec4<f32>(a1.z), w2, acc_1);
            acc_1 = fma(vec4<f32>(a1.w), w3, acc_1);
            acc_2 = fma(vec4<f32>(a2.x), w0, acc_2);
            acc_2 = fma(vec4<f32>(a2.y), w1, acc_2);
            acc_2 = fma(vec4<f32>(a2.z), w2, acc_2);
            acc_2 = fma(vec4<f32>(a2.w), w3, acc_2);
            acc_3 = fma(vec4<f32>(a3.x), w0, acc_3);
            acc_3 = fma(vec4<f32>(a3.y), w1, acc_3);
            acc_3 = fma(vec4<f32>(a3.z), w2, acc_3);
            acc_3 = fma(vec4<f32>(a3.w), w3, acc_3);
            acc_4 = fma(vec4<f32>(a4.x), w0, acc_4);
            acc_4 = fma(vec4<f32>(a4.y), w1, acc_4);
            acc_4 = fma(vec4<f32>(a4.z), w2, acc_4);
            acc_4 = fma(vec4<f32>(a4.w), w3, acc_4);
            acc_5 = fma(vec4<f32>(a5.x), w0, acc_5);
            acc_5 = fma(vec4<f32>(a5.y), w1, acc_5);
            acc_5 = fma(vec4<f32>(a5.z), w2, acc_5);
            acc_5 = fma(vec4<f32>(a5.w), w3, acc_5);
            acc_6 = fma(vec4<f32>(a6.x), w0, acc_6);
            acc_6 = fma(vec4<f32>(a6.y), w1, acc_6);
            acc_6 = fma(vec4<f32>(a6.z), w2, acc_6);
            acc_6 = fma(vec4<f32>(a6.w), w3, acc_6);
            acc_7 = fma(vec4<f32>(a7.x), w0, acc_7);
            acc_7 = fma(vec4<f32>(a7.y), w1, acc_7);
            acc_7 = fma(vec4<f32>(a7.z), w2, acc_7);
            acc_7 = fma(vec4<f32>(a7.w), w3, acc_7);
        }
        workgroupBarrier();
    }

    let component = group_id.x / 10u;
    let group_in_component = group_id.x - component * 10u;
    let head = group_in_component * 2u + local_id.x / 16u;
    let dim_base = (local_id.x % 16u) * 4u;
    let local_rows = array<u32, 8>(
        local_id.y,
        local_id.y + 8u,
        local_id.y + 16u,
        local_id.y + 24u,
        local_id.y + 32u,
        local_id.y + 40u,
        local_id.y + 48u,
        local_id.y + 56u,
    );
    let values = array<vec4<f32>, 8>(acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7);

    var norm_sums: array<f32, 8>;
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
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        let token = row_base + local_rows[i];
        if (token < ROWS) {
            var inv_rms = 1.0;
            if (component < 2u) {
                inv_rms = inverseSqrt(norm_sums[i] / f32(DH) + EPS);
            }
            store_frontend_value(component, token, head, dim_base, values[i], inv_rms);
        }
    }

    if (group_id.x == 0u && group_id.y == 0u) {
        let ctx_plane = BATCH * CTX * MODEL_DIM;
        for (var index = local_index; index < ctx_plane; index = index + 256u) {
            let dim = index % DH;
            let head_ctx = (index / DH) % H;
            let ctx_seq = (index / MODEL_DIM) % CTX;
            let batch_ctx = index / (CTX * MODEL_DIM);
            let dst = ((batch_ctx * H + head_ctx) * TOTAL_S + S + ctx_seq) * DH + dim;
            k_all[dst] = ctx_kv[index];
            v_all[dst] = ctx_kv[ctx_plane + index];
        }
    }
}
