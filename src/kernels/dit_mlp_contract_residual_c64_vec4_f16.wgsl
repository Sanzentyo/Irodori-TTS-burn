enable f16;

// F16-storage form of the 64x64 contract route. Cooperative loads are
// promoted and all accumulation remains F32 until the final output store.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> residual: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> gate: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f16>>;

const ROWS: u32 = {{ rows }}u;
const SEQUENCE: u32 = {{ sequence }}u;
const K: u32 = {{ inner }}u;
const K_VECS: u32 = K / 4u;
const INPUT_ROW_STRIDE_VECS: u32 = {{ input_row_stride }}u / 4u;
const N: u32 = 1280u;
const N_VECS: u32 = N / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_K: u32 = 32u;
const TILE_K_VECS: u32 = TILE_K / 4u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 16u;

var<workgroup> input_tile: array<vec4<f32>, 512>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

fn store_result(row: u32, column_vec: u32, branch: vec4<f32>) {
    if (row < ROWS && column_vec < N_VECS) {
        let index = row * N_VECS + column_vec;
        let batch = row / SEQUENCE;
        let residual_value = vec4<f32>(residual[index]);
        let gate_value = vec4<f32>(gate[batch * N_VECS + column_vec]);
        output[index] = vec4<f16>(residual_value + gate_value * branch);
    }
}

@compute @workgroup_size(16, 16, 1)
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

    for (var k_base = 0u; k_base < K; k_base = k_base + TILE_K) {
        for (var load = local_index; load < TILE_ROWS * TILE_K_VECS; load = load + 256u) {
            let tile_row = load / TILE_K_VECS;
            let tile_k_vec = load - tile_row * TILE_K_VECS;
            let row = row_base + tile_row;
            var value = vec4<f32>(0.0);
            if (row < ROWS) {
                value = vec4<f32>(input[row * INPUT_ROW_STRIDE_VECS + k_base / 4u + tile_k_vec]);
            }
            input_tile[load] = value;
        }
        for (var load = local_index; load < TILE_K * LOCAL_COLUMN_VECS; load = load + 256u) {
            let tile_k = load / LOCAL_COLUMN_VECS;
            let tile_column_vec = load - tile_k * LOCAL_COLUMN_VECS;
            let column_vec = column_vec_base + tile_column_vec;
            var value = vec4<f32>(0.0);
            if (column_vec < N_VECS) {
                value = vec4<f32>(weight[(k_base + tile_k) * N_VECS + column_vec]);
            }
            weight_tile[load] = value;
        }
        workgroupBarrier();

        let row_0 = local_id.y;
        let row_1 = row_0 + LOCAL_ROWS;
        let row_2 = row_1 + LOCAL_ROWS;
        let row_3 = row_2 + LOCAL_ROWS;
        for (var tile_k_vec = 0u; tile_k_vec < TILE_K_VECS; tile_k_vec = tile_k_vec + 1u) {
            let input_0 = input_tile[row_0 * TILE_K_VECS + tile_k_vec];
            let input_1 = input_tile[row_1 * TILE_K_VECS + tile_k_vec];
            let input_2 = input_tile[row_2 * TILE_K_VECS + tile_k_vec];
            let input_3 = input_tile[row_3 * TILE_K_VECS + tile_k_vec];
            let weight_base = tile_k_vec * 4u * LOCAL_COLUMN_VECS + local_id.x;

            let weight_0 = weight_tile[weight_base];
            acc_0 = fma(vec4<f32>(input_0.x), weight_0, acc_0);
            acc_1 = fma(vec4<f32>(input_1.x), weight_0, acc_1);
            acc_2 = fma(vec4<f32>(input_2.x), weight_0, acc_2);
            acc_3 = fma(vec4<f32>(input_3.x), weight_0, acc_3);

            let weight_1 = weight_tile[weight_base + LOCAL_COLUMN_VECS];
            acc_0 = fma(vec4<f32>(input_0.y), weight_1, acc_0);
            acc_1 = fma(vec4<f32>(input_1.y), weight_1, acc_1);
            acc_2 = fma(vec4<f32>(input_2.y), weight_1, acc_2);
            acc_3 = fma(vec4<f32>(input_3.y), weight_1, acc_3);

            let weight_2 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS];
            acc_0 = fma(vec4<f32>(input_0.z), weight_2, acc_0);
            acc_1 = fma(vec4<f32>(input_1.z), weight_2, acc_1);
            acc_2 = fma(vec4<f32>(input_2.z), weight_2, acc_2);
            acc_3 = fma(vec4<f32>(input_3.z), weight_2, acc_3);

            let weight_3 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS];
            acc_0 = fma(vec4<f32>(input_0.w), weight_3, acc_0);
            acc_1 = fma(vec4<f32>(input_1.w), weight_3, acc_1);
            acc_2 = fma(vec4<f32>(input_2.w), weight_3, acc_2);
            acc_3 = fma(vec4<f32>(input_3.w), weight_3, acc_3);
        }
        workgroupBarrier();
    }

    let column = column_vec_base + local_id.x;
    let row_0 = row_base + local_id.y;
    let row_1 = row_0 + LOCAL_ROWS;
    let row_2 = row_1 + LOCAL_ROWS;
    let row_3 = row_2 + LOCAL_ROWS;
    store_result(row_0, column, acc_0);
    store_result(row_1, column, acc_1);
    store_result(row_2, column, acc_2);
    store_result(row_3, column, acc_3);
}
