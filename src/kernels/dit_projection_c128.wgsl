// Exact-shape production GEMMs for measured long DiT latent sequences.
// A[M,K] and B[K,N] are row-major f32. Each 16x16 workgroup
// produces a 64x128 output tile; each thread owns four rows and eight columns
// represented as two vec4 accumulators. K advances strictly from 0 to K-1.

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = {{ inner }}u;
const N: u32 = {{ columns }}u;
const N_VECS: u32 = N / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_COLUMNS: u32 = 128u;
const TILE_K: u32 = 16u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 32u;

var<workgroup> input_tile: array<f32, 1024>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

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
    var acc_4 = vec4<f32>(0.0);
    var acc_5 = vec4<f32>(0.0);
    var acc_6 = vec4<f32>(0.0);
    var acc_7 = vec4<f32>(0.0);

    for (var k_base = 0u; k_base < K; k_base = k_base + TILE_K) {
        for (var load = local_index; load < TILE_ROWS * TILE_K; load = load + 256u) {
            let tile_row = load / TILE_K;
            let tile_k = load - tile_row * TILE_K;
            let row = row_base + tile_row;
            var value = 0.0;
            if (row < ROWS) {
                value = input[row * K + k_base + tile_k];
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

        for (var tile_k_index = 0u; tile_k_index < TILE_K; tile_k_index = tile_k_index + 1u) {
            let weight_value_0 = weight_tile[tile_k_index * LOCAL_COLUMN_VECS + local_id.x];
            let weight_value_1 = weight_tile[tile_k_index * LOCAL_COLUMN_VECS + local_id.x + 16u];
            let row_0 = local_id.y;
            let row_1 = row_0 + LOCAL_ROWS;
            let row_2 = row_1 + LOCAL_ROWS;
            let row_3 = row_2 + LOCAL_ROWS;
            let input_0 = vec4<f32>(input_tile[row_0 * TILE_K + tile_k_index]);
            let input_1 = vec4<f32>(input_tile[row_1 * TILE_K + tile_k_index]);
            let input_2 = vec4<f32>(input_tile[row_2 * TILE_K + tile_k_index]);
            let input_3 = vec4<f32>(input_tile[row_3 * TILE_K + tile_k_index]);
            acc_0 = fma(input_0, weight_value_0, acc_0);
            acc_1 = fma(input_1, weight_value_0, acc_1);
            acc_2 = fma(input_2, weight_value_0, acc_2);
            acc_3 = fma(input_3, weight_value_0, acc_3);
            acc_4 = fma(input_0, weight_value_1, acc_4);
            acc_5 = fma(input_1, weight_value_1, acc_5);
            acc_6 = fma(input_2, weight_value_1, acc_6);
            acc_7 = fma(input_3, weight_value_1, acc_7);
        }
        workgroupBarrier();
    }

    let output_column_vec_0 = column_vec_base + local_id.x;
    let output_column_vec_1 = output_column_vec_0 + 16u;
    let output_row_0 = row_base + local_id.y;
    let output_row_1 = output_row_0 + LOCAL_ROWS;
    let output_row_2 = output_row_1 + LOCAL_ROWS;
    let output_row_3 = output_row_2 + LOCAL_ROWS;
    if (output_row_0 < ROWS && output_column_vec_0 < N_VECS) {
        output[output_row_0 * N_VECS + output_column_vec_0] = acc_0;
    }
    if (output_row_1 < ROWS && output_column_vec_0 < N_VECS) {
        output[output_row_1 * N_VECS + output_column_vec_0] = acc_1;
    }
    if (output_row_2 < ROWS && output_column_vec_0 < N_VECS) {
        output[output_row_2 * N_VECS + output_column_vec_0] = acc_2;
    }
    if (output_row_3 < ROWS && output_column_vec_0 < N_VECS) {
        output[output_row_3 * N_VECS + output_column_vec_0] = acc_3;
    }
    if (output_row_0 < ROWS && output_column_vec_1 < N_VECS) {
        output[output_row_0 * N_VECS + output_column_vec_1] = acc_4;
    }
    if (output_row_1 < ROWS && output_column_vec_1 < N_VECS) {
        output[output_row_1 * N_VECS + output_column_vec_1] = acc_5;
    }
    if (output_row_2 < ROWS && output_column_vec_1 < N_VECS) {
        output[output_row_2 * N_VECS + output_column_vec_1] = acc_6;
    }
    if (output_row_3 < ROWS && output_column_vec_1 < N_VECS) {
        output[output_row_3 * N_VECS + output_column_vec_1] = acc_7;
    }
}
