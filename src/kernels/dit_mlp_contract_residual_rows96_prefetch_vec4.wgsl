// Exact FP32 K16 MLP contract with a 96-row tile and register-prefetched
// global loads. A 256-invocation group computes 96x128 outputs, reducing the
// number of times each layer's weight is fetched for small-M workloads.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> residual: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

const ROWS: u32 = {{ rows }}u;
const SEQUENCE: u32 = {{ sequence }}u;
const K: u32 = {{ inner }}u;
const INPUT_ROW_STRIDE_VECS: u32 = {{ input_row_stride }}u / 4u;
const N_VECS: u32 = 320u;
const TILE_ROWS: u32 = 96u;
const TILE_K: u32 = 16u;
const TILE_K_VECS: u32 = 4u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 32u;
const INPUT_TILE_VECS: u32 = 384u;

var<workgroup> input_tile: array<vec4<f32>, 384>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

fn load_input_value(load: u32, k_base: u32, row_base: u32) -> vec4<f32> {
    let tile_row = load / TILE_K_VECS;
    let tile_k_vec = load - tile_row * TILE_K_VECS;
    let row = row_base + tile_row;
    if (load < INPUT_TILE_VECS && row < ROWS) {
        return input[row * INPUT_ROW_STRIDE_VECS + k_base / 4u + tile_k_vec];
    }
    return vec4<f32>(0.0);
}

fn load_weight_value(load: u32, k_base: u32, column_vec_base: u32) -> vec4<f32> {
    let tile_k = load / LOCAL_COLUMN_VECS;
    let tile_column_vec = load - tile_k * LOCAL_COLUMN_VECS;
    let column_vec = column_vec_base + tile_column_vec;
    if (column_vec < N_VECS) {
        return weight[(k_base + tile_k) * N_VECS + column_vec];
    }
    return vec4<f32>(0.0);
}

fn store_result(row: u32, column_vec: u32, branch: vec4<f32>) {
    if (row < ROWS && column_vec < N_VECS) {
        let index = row * N_VECS + column_vec;
        let batch = row / SEQUENCE;
        output[index] = residual[index] + gate[batch * N_VECS + column_vec] * branch;
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
    let input_load_0 = local_index;
    let input_load_1 = local_index + 256u;
    let weight_load_0 = local_index;
    let weight_load_1 = local_index + 256u;
    var prefetched_input_0 = load_input_value(input_load_0, 0u, row_base);
    var prefetched_input_1 = load_input_value(input_load_1, 0u, row_base);
    var prefetched_weight_0 = load_weight_value(weight_load_0, 0u, column_vec_base);
    var prefetched_weight_1 = load_weight_value(weight_load_1, 0u, column_vec_base);
    input_tile[input_load_0] = prefetched_input_0;
    if (input_load_1 < INPUT_TILE_VECS) {
        input_tile[input_load_1] = prefetched_input_1;
    }
    weight_tile[weight_load_0] = prefetched_weight_0;
    weight_tile[weight_load_1] = prefetched_weight_1;
    workgroupBarrier();

    var acc_00 = vec4<f32>(0.0);
    var acc_01 = vec4<f32>(0.0);
    var acc_02 = vec4<f32>(0.0);
    var acc_03 = vec4<f32>(0.0);
    var acc_04 = vec4<f32>(0.0);
    var acc_05 = vec4<f32>(0.0);
    var acc_10 = vec4<f32>(0.0);
    var acc_11 = vec4<f32>(0.0);
    var acc_12 = vec4<f32>(0.0);
    var acc_13 = vec4<f32>(0.0);
    var acc_14 = vec4<f32>(0.0);
    var acc_15 = vec4<f32>(0.0);

    for (var k_base = 0u; k_base < K; k_base = k_base + TILE_K) {
        let row_0 = local_id.y;
        let row_1 = row_0 + LOCAL_ROWS;
        let row_2 = row_1 + LOCAL_ROWS;
        let row_3 = row_2 + LOCAL_ROWS;
        let row_4 = row_3 + LOCAL_ROWS;
        let row_5 = row_4 + LOCAL_ROWS;
        for (var tile_k_vec = 0u; tile_k_vec < TILE_K_VECS; tile_k_vec = tile_k_vec + 1u) {
            let input_0 = input_tile[row_0 * TILE_K_VECS + tile_k_vec];
            let input_1 = input_tile[row_1 * TILE_K_VECS + tile_k_vec];
            let input_2 = input_tile[row_2 * TILE_K_VECS + tile_k_vec];
            let input_3 = input_tile[row_3 * TILE_K_VECS + tile_k_vec];
            let input_4 = input_tile[row_4 * TILE_K_VECS + tile_k_vec];
            let input_5 = input_tile[row_5 * TILE_K_VECS + tile_k_vec];
            let weight_base = tile_k_vec * 4u * LOCAL_COLUMN_VECS + local_id.x;

            let weight_00 = weight_tile[weight_base];
            let weight_10 = weight_tile[weight_base + 16u];
            acc_00 = fma(vec4<f32>(input_0.x), weight_00, acc_00);
            acc_01 = fma(vec4<f32>(input_1.x), weight_00, acc_01);
            acc_02 = fma(vec4<f32>(input_2.x), weight_00, acc_02);
            acc_03 = fma(vec4<f32>(input_3.x), weight_00, acc_03);
            acc_04 = fma(vec4<f32>(input_4.x), weight_00, acc_04);
            acc_05 = fma(vec4<f32>(input_5.x), weight_00, acc_05);
            acc_10 = fma(vec4<f32>(input_0.x), weight_10, acc_10);
            acc_11 = fma(vec4<f32>(input_1.x), weight_10, acc_11);
            acc_12 = fma(vec4<f32>(input_2.x), weight_10, acc_12);
            acc_13 = fma(vec4<f32>(input_3.x), weight_10, acc_13);
            acc_14 = fma(vec4<f32>(input_4.x), weight_10, acc_14);
            acc_15 = fma(vec4<f32>(input_5.x), weight_10, acc_15);

            let weight_01 = weight_tile[weight_base + LOCAL_COLUMN_VECS];
            let weight_11 = weight_tile[weight_base + LOCAL_COLUMN_VECS + 16u];
            acc_00 = fma(vec4<f32>(input_0.y), weight_01, acc_00);
            acc_01 = fma(vec4<f32>(input_1.y), weight_01, acc_01);
            acc_02 = fma(vec4<f32>(input_2.y), weight_01, acc_02);
            acc_03 = fma(vec4<f32>(input_3.y), weight_01, acc_03);
            acc_04 = fma(vec4<f32>(input_4.y), weight_01, acc_04);
            acc_05 = fma(vec4<f32>(input_5.y), weight_01, acc_05);
            acc_10 = fma(vec4<f32>(input_0.y), weight_11, acc_10);
            acc_11 = fma(vec4<f32>(input_1.y), weight_11, acc_11);
            acc_12 = fma(vec4<f32>(input_2.y), weight_11, acc_12);
            acc_13 = fma(vec4<f32>(input_3.y), weight_11, acc_13);
            acc_14 = fma(vec4<f32>(input_4.y), weight_11, acc_14);
            acc_15 = fma(vec4<f32>(input_5.y), weight_11, acc_15);

            let weight_02 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS];
            let weight_12 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS + 16u];
            acc_00 = fma(vec4<f32>(input_0.z), weight_02, acc_00);
            acc_01 = fma(vec4<f32>(input_1.z), weight_02, acc_01);
            acc_02 = fma(vec4<f32>(input_2.z), weight_02, acc_02);
            acc_03 = fma(vec4<f32>(input_3.z), weight_02, acc_03);
            acc_04 = fma(vec4<f32>(input_4.z), weight_02, acc_04);
            acc_05 = fma(vec4<f32>(input_5.z), weight_02, acc_05);
            acc_10 = fma(vec4<f32>(input_0.z), weight_12, acc_10);
            acc_11 = fma(vec4<f32>(input_1.z), weight_12, acc_11);
            acc_12 = fma(vec4<f32>(input_2.z), weight_12, acc_12);
            acc_13 = fma(vec4<f32>(input_3.z), weight_12, acc_13);
            acc_14 = fma(vec4<f32>(input_4.z), weight_12, acc_14);
            acc_15 = fma(vec4<f32>(input_5.z), weight_12, acc_15);

            let weight_03 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS];
            let weight_13 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS + 16u];
            acc_00 = fma(vec4<f32>(input_0.w), weight_03, acc_00);
            acc_01 = fma(vec4<f32>(input_1.w), weight_03, acc_01);
            acc_02 = fma(vec4<f32>(input_2.w), weight_03, acc_02);
            acc_03 = fma(vec4<f32>(input_3.w), weight_03, acc_03);
            acc_04 = fma(vec4<f32>(input_4.w), weight_03, acc_04);
            acc_05 = fma(vec4<f32>(input_5.w), weight_03, acc_05);
            acc_10 = fma(vec4<f32>(input_0.w), weight_13, acc_10);
            acc_11 = fma(vec4<f32>(input_1.w), weight_13, acc_11);
            acc_12 = fma(vec4<f32>(input_2.w), weight_13, acc_12);
            acc_13 = fma(vec4<f32>(input_3.w), weight_13, acc_13);
            acc_14 = fma(vec4<f32>(input_4.w), weight_13, acc_14);
            acc_15 = fma(vec4<f32>(input_5.w), weight_13, acc_15);
        }

        let next_k = k_base + TILE_K;
        if (next_k < K) {
            prefetched_input_0 = load_input_value(input_load_0, next_k, row_base);
            prefetched_input_1 = load_input_value(input_load_1, next_k, row_base);
            prefetched_weight_0 = load_weight_value(weight_load_0, next_k, column_vec_base);
            prefetched_weight_1 = load_weight_value(weight_load_1, next_k, column_vec_base);
            workgroupBarrier();
            input_tile[input_load_0] = prefetched_input_0;
            if (input_load_1 < INPUT_TILE_VECS) {
                input_tile[input_load_1] = prefetched_input_1;
            }
            weight_tile[weight_load_0] = prefetched_weight_0;
            weight_tile[weight_load_1] = prefetched_weight_1;
            workgroupBarrier();
        }
    }

    let column_0 = column_vec_base + local_id.x;
    let column_1 = column_0 + 16u;
    let row_0 = row_base + local_id.y;
    let row_1 = row_0 + LOCAL_ROWS;
    let row_2 = row_1 + LOCAL_ROWS;
    let row_3 = row_2 + LOCAL_ROWS;
    let row_4 = row_3 + LOCAL_ROWS;
    let row_5 = row_4 + LOCAL_ROWS;
    store_result(row_0, column_0, acc_00);
    store_result(row_1, column_0, acc_01);
    store_result(row_2, column_0, acc_02);
    store_result(row_3, column_0, acc_03);
    store_result(row_4, column_0, acc_04);
    store_result(row_5, column_0, acc_05);
    store_result(row_0, column_1, acc_10);
    store_result(row_1, column_1, acc_11);
    store_result(row_2, column_1, acc_12);
    store_result(row_3, column_1, acc_13);
    store_result(row_4, column_1, acc_14);
    store_result(row_5, column_1, acc_15);
}
