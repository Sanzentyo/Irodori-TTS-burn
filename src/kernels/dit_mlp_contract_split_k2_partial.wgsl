{{ enable_f16 }}

// Split-K2 partial MLP contract with register-prefetched K16 tiles.
//
// The Z workgroup axis owns one disjoint half of K. Each group writes an F32
// partial; a second kernel combines both halves with the gated residual.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<{{ storage }}>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<{{ storage }}>>;
@group(0) @binding(2) var<storage, read_write> partial: array<vec4<f32>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = 3680u;
const SPLIT_K: u32 = K / 2u;
const INPUT_ROW_STRIDE_VECS: u32 = {{ input_row_stride }}u / 4u;
const N: u32 = 1280u;
const N_VECS: u32 = N / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_K: u32 = 16u;
const TILE_K_VECS: u32 = TILE_K / 4u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 32u;
const INPUT_TILE_VECS: u32 = 256u;
const WEIGHT_TILE_VECS: u32 = 512u;

var<workgroup> input_tile: array<vec4<f32>, 256>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

fn load_input_value(load: u32, k_base: u32, row_base: u32) -> vec4<f32> {
    let tile_row = load / TILE_K_VECS;
    let tile_k_vec = load - tile_row * TILE_K_VECS;
    let row = row_base + tile_row;
    if (row < ROWS) {
        return vec4<f32>(input[row * INPUT_ROW_STRIDE_VECS + k_base / 4u + tile_k_vec]);
    }
    return vec4<f32>(0.0);
}

fn load_weight_value(load: u32, k_base: u32, column_vec_base: u32) -> vec4<f32> {
    let tile_k = load / LOCAL_COLUMN_VECS;
    let tile_column_vec = load - tile_k * LOCAL_COLUMN_VECS;
    let column_vec = column_vec_base + tile_column_vec;
    if (column_vec < N_VECS) {
        return vec4<f32>(weight[(k_base + tile_k) * N_VECS + column_vec]);
    }
    return vec4<f32>(0.0);
}

fn store_partial(split: u32, row: u32, column_vec: u32, branch: vec4<f32>) {
    if (row < ROWS && column_vec < N_VECS) {
        partial[(split * ROWS + row) * N_VECS + column_vec] = branch;
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
    let split = group_id.z;
    let split_k_offset = split * SPLIT_K;
    let input_load = local_index;
    let weight_load_0 = local_index;
    let weight_load_1 = local_index + 256u;
    var prefetched_input = load_input_value(input_load, split_k_offset, row_base);
    var prefetched_weight_0 = load_weight_value(weight_load_0, split_k_offset, column_vec_base);
    var prefetched_weight_1 = load_weight_value(weight_load_1, split_k_offset, column_vec_base);
    input_tile[input_load] = prefetched_input;
    weight_tile[weight_load_0] = prefetched_weight_0;
    weight_tile[weight_load_1] = prefetched_weight_1;
    workgroupBarrier();

    var acc_0 = vec4<f32>(0.0);
    var acc_1 = vec4<f32>(0.0);
    var acc_2 = vec4<f32>(0.0);
    var acc_3 = vec4<f32>(0.0);
    var acc_4 = vec4<f32>(0.0);
    var acc_5 = vec4<f32>(0.0);
    var acc_6 = vec4<f32>(0.0);
    var acc_7 = vec4<f32>(0.0);

    for (var k_base = 0u; k_base < SPLIT_K; k_base = k_base + TILE_K) {
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

            let weight_00 = weight_tile[weight_base];
            let weight_10 = weight_tile[weight_base + 16u];
            acc_0 = fma(vec4<f32>(input_0.x), weight_00, acc_0);
            acc_1 = fma(vec4<f32>(input_1.x), weight_00, acc_1);
            acc_2 = fma(vec4<f32>(input_2.x), weight_00, acc_2);
            acc_3 = fma(vec4<f32>(input_3.x), weight_00, acc_3);
            acc_4 = fma(vec4<f32>(input_0.x), weight_10, acc_4);
            acc_5 = fma(vec4<f32>(input_1.x), weight_10, acc_5);
            acc_6 = fma(vec4<f32>(input_2.x), weight_10, acc_6);
            acc_7 = fma(vec4<f32>(input_3.x), weight_10, acc_7);

            let weight_01 = weight_tile[weight_base + LOCAL_COLUMN_VECS];
            let weight_11 = weight_tile[weight_base + LOCAL_COLUMN_VECS + 16u];
            acc_0 = fma(vec4<f32>(input_0.y), weight_01, acc_0);
            acc_1 = fma(vec4<f32>(input_1.y), weight_01, acc_1);
            acc_2 = fma(vec4<f32>(input_2.y), weight_01, acc_2);
            acc_3 = fma(vec4<f32>(input_3.y), weight_01, acc_3);
            acc_4 = fma(vec4<f32>(input_0.y), weight_11, acc_4);
            acc_5 = fma(vec4<f32>(input_1.y), weight_11, acc_5);
            acc_6 = fma(vec4<f32>(input_2.y), weight_11, acc_6);
            acc_7 = fma(vec4<f32>(input_3.y), weight_11, acc_7);

            let weight_02 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS];
            let weight_12 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS + 16u];
            acc_0 = fma(vec4<f32>(input_0.z), weight_02, acc_0);
            acc_1 = fma(vec4<f32>(input_1.z), weight_02, acc_1);
            acc_2 = fma(vec4<f32>(input_2.z), weight_02, acc_2);
            acc_3 = fma(vec4<f32>(input_3.z), weight_02, acc_3);
            acc_4 = fma(vec4<f32>(input_0.z), weight_12, acc_4);
            acc_5 = fma(vec4<f32>(input_1.z), weight_12, acc_5);
            acc_6 = fma(vec4<f32>(input_2.z), weight_12, acc_6);
            acc_7 = fma(vec4<f32>(input_3.z), weight_12, acc_7);

            let weight_03 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS];
            let weight_13 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS + 16u];
            acc_0 = fma(vec4<f32>(input_0.w), weight_03, acc_0);
            acc_1 = fma(vec4<f32>(input_1.w), weight_03, acc_1);
            acc_2 = fma(vec4<f32>(input_2.w), weight_03, acc_2);
            acc_3 = fma(vec4<f32>(input_3.w), weight_03, acc_3);
            acc_4 = fma(vec4<f32>(input_0.w), weight_13, acc_4);
            acc_5 = fma(vec4<f32>(input_1.w), weight_13, acc_5);
            acc_6 = fma(vec4<f32>(input_2.w), weight_13, acc_6);
            acc_7 = fma(vec4<f32>(input_3.w), weight_13, acc_7);
        }

        let next_k = k_base + TILE_K;
        if (next_k < SPLIT_K) {
            prefetched_input = load_input_value(input_load, split_k_offset + next_k, row_base);
            prefetched_weight_0 = load_weight_value(weight_load_0, split_k_offset + next_k, column_vec_base);
            prefetched_weight_1 = load_weight_value(weight_load_1, split_k_offset + next_k, column_vec_base);
            workgroupBarrier();
            input_tile[input_load] = prefetched_input;
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
    store_partial(split, row_0, column_0, acc_0);
    store_partial(split, row_1, column_0, acc_1);
    store_partial(split, row_2, column_0, acc_2);
    store_partial(split, row_3, column_0, acc_3);
    store_partial(split, row_0, column_1, acc_4);
    store_partial(split, row_1, column_1, acc_5);
    store_partial(split, row_2, column_1, acc_6);
    store_partial(split, row_3, column_1, acc_7);
}
