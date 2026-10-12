// Exact FP32 C128/K16 projection with register-prefetched global tiles.
//
// Each invocation fetches one next input vec4 and two next weight vec4s
// before the shared-page overwrite barrier. The incumbent 12-KiB workgroup
// footprint and ordered FMA sequence are unchanged.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = {{ inner }}u;
const K_VECS: u32 = K / 4u;
const N: u32 = {{ columns }}u;
const N_VECS: u32 = N / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_K: u32 = 16u;
const TILE_K_VECS: u32 = TILE_K / 4u;
const LOCAL_ROWS: u32 = 8u;
const LOCAL_COLUMN_VECS: u32 = 32u;

var<workgroup> input_tile: array<vec4<f32>, 256>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

fn load_input_value(load: u32, k_base: u32, row_base: u32) -> vec4<f32> {
    let tile_row = load / TILE_K_VECS;
    let tile_k_vec = load - tile_row * TILE_K_VECS;
    let row = row_base + tile_row;
    if (row < ROWS) {
        return input[row * K_VECS + k_base / 4u + tile_k_vec];
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

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let row_base = group_id.y * TILE_ROWS;
    let column_vec_base = group_id.x * LOCAL_COLUMN_VECS;
    let input_load = local_index;
    let weight_load_0 = local_index;
    let weight_load_1 = local_index + 256u;
    var prefetched_input = load_input_value(input_load, 0u, row_base);
    var prefetched_weight_0 = load_weight_value(weight_load_0, 0u, column_vec_base);
    var prefetched_weight_1 = load_weight_value(weight_load_1, 0u, column_vec_base);
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

    for (var k_base = 0u; k_base < K; k_base = k_base + TILE_K) {
        let row_0 = local_id.y;
        let row_1 = row_0 + LOCAL_ROWS;
        let row_2 = row_1 + LOCAL_ROWS;
        let row_3 = row_2 + LOCAL_ROWS;
        let row_4 = row_3 + LOCAL_ROWS;
        let row_5 = row_4 + LOCAL_ROWS;
        let row_6 = row_5 + LOCAL_ROWS;
        let row_7 = row_6 + LOCAL_ROWS;
        for (var tile_k_vec = 0u; tile_k_vec < TILE_K_VECS; tile_k_vec = tile_k_vec + 1u) {
            let input_0 = input_tile[row_0 * TILE_K_VECS + tile_k_vec];
            let input_1 = input_tile[row_1 * TILE_K_VECS + tile_k_vec];
            let input_2 = input_tile[row_2 * TILE_K_VECS + tile_k_vec];
            let input_3 = input_tile[row_3 * TILE_K_VECS + tile_k_vec];
            let input_4 = input_tile[row_4 * TILE_K_VECS + tile_k_vec];
            let input_5 = input_tile[row_5 * TILE_K_VECS + tile_k_vec];
            let input_6 = input_tile[row_6 * TILE_K_VECS + tile_k_vec];
            let input_7 = input_tile[row_7 * TILE_K_VECS + tile_k_vec];
            let weight_base = tile_k_vec * 4u * LOCAL_COLUMN_VECS + local_id.x;

            let weight_0 = weight_tile[weight_base];
            acc_0 = fma(vec4<f32>(input_0.x), weight_0, acc_0);
            acc_1 = fma(vec4<f32>(input_1.x), weight_0, acc_1);
            acc_2 = fma(vec4<f32>(input_2.x), weight_0, acc_2);
            acc_3 = fma(vec4<f32>(input_3.x), weight_0, acc_3);
            acc_4 = fma(vec4<f32>(input_4.x), weight_0, acc_4);
            acc_5 = fma(vec4<f32>(input_5.x), weight_0, acc_5);
            acc_6 = fma(vec4<f32>(input_6.x), weight_0, acc_6);
            acc_7 = fma(vec4<f32>(input_7.x), weight_0, acc_7);

            let weight_1 = weight_tile[weight_base + LOCAL_COLUMN_VECS];
            acc_0 = fma(vec4<f32>(input_0.y), weight_1, acc_0);
            acc_1 = fma(vec4<f32>(input_1.y), weight_1, acc_1);
            acc_2 = fma(vec4<f32>(input_2.y), weight_1, acc_2);
            acc_3 = fma(vec4<f32>(input_3.y), weight_1, acc_3);
            acc_4 = fma(vec4<f32>(input_4.y), weight_1, acc_4);
            acc_5 = fma(vec4<f32>(input_5.y), weight_1, acc_5);
            acc_6 = fma(vec4<f32>(input_6.y), weight_1, acc_6);
            acc_7 = fma(vec4<f32>(input_7.y), weight_1, acc_7);

            let weight_2 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS];
            acc_0 = fma(vec4<f32>(input_0.z), weight_2, acc_0);
            acc_1 = fma(vec4<f32>(input_1.z), weight_2, acc_1);
            acc_2 = fma(vec4<f32>(input_2.z), weight_2, acc_2);
            acc_3 = fma(vec4<f32>(input_3.z), weight_2, acc_3);
            acc_4 = fma(vec4<f32>(input_4.z), weight_2, acc_4);
            acc_5 = fma(vec4<f32>(input_5.z), weight_2, acc_5);
            acc_6 = fma(vec4<f32>(input_6.z), weight_2, acc_6);
            acc_7 = fma(vec4<f32>(input_7.z), weight_2, acc_7);

            let weight_3 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS];
            acc_0 = fma(vec4<f32>(input_0.w), weight_3, acc_0);
            acc_1 = fma(vec4<f32>(input_1.w), weight_3, acc_1);
            acc_2 = fma(vec4<f32>(input_2.w), weight_3, acc_2);
            acc_3 = fma(vec4<f32>(input_3.w), weight_3, acc_3);
            acc_4 = fma(vec4<f32>(input_4.w), weight_3, acc_4);
            acc_5 = fma(vec4<f32>(input_5.w), weight_3, acc_5);
            acc_6 = fma(vec4<f32>(input_6.w), weight_3, acc_6);
            acc_7 = fma(vec4<f32>(input_7.w), weight_3, acc_7);
        }

        let next_k = k_base + TILE_K;
        if (next_k < K) {
            prefetched_input = load_input_value(input_load, next_k, row_base);
            prefetched_weight_0 = load_weight_value(weight_load_0, next_k, column_vec_base);
            prefetched_weight_1 = load_weight_value(weight_load_1, next_k, column_vec_base);
            workgroupBarrier();
            input_tile[input_load] = prefetched_input;
            weight_tile[weight_load_0] = prefetched_weight_0;
            weight_tile[weight_load_1] = prefetched_weight_1;
            workgroupBarrier();
        }
    }

    let output_column_vec = column_vec_base + local_id.x;
    let output_row_0 = row_base + local_id.y;
    let output_row_1 = output_row_0 + LOCAL_ROWS;
    let output_row_2 = output_row_1 + LOCAL_ROWS;
    let output_row_3 = output_row_2 + LOCAL_ROWS;
    let output_row_4 = output_row_3 + LOCAL_ROWS;
    let output_row_5 = output_row_4 + LOCAL_ROWS;
    let output_row_6 = output_row_5 + LOCAL_ROWS;
    let output_row_7 = output_row_6 + LOCAL_ROWS;
    if (output_row_0 < ROWS && output_column_vec < N_VECS) { output[output_row_0 * N_VECS + output_column_vec] = acc_0; }
    if (output_row_1 < ROWS && output_column_vec < N_VECS) { output[output_row_1 * N_VECS + output_column_vec] = acc_1; }
    if (output_row_2 < ROWS && output_column_vec < N_VECS) { output[output_row_2 * N_VECS + output_column_vec] = acc_2; }
    if (output_row_3 < ROWS && output_column_vec < N_VECS) { output[output_row_3 * N_VECS + output_column_vec] = acc_3; }
    if (output_row_4 < ROWS && output_column_vec < N_VECS) { output[output_row_4 * N_VECS + output_column_vec] = acc_4; }
    if (output_row_5 < ROWS && output_column_vec < N_VECS) { output[output_row_5 * N_VECS + output_column_vec] = acc_5; }
    if (output_row_6 < ROWS && output_column_vec < N_VECS) { output[output_row_6 * N_VECS + output_column_vec] = acc_6; }
    if (output_row_7 < ROWS && output_column_vec < N_VECS) { output[output_row_7 * N_VECS + output_column_vec] = acc_7; }
}
