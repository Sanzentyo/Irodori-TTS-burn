// Exact-shape production GEMM for DiT MLP w1||w3 at latent sequence 200.
// A[M,1280] and B[1280,7360] are row-major f32. Each 16x16 workgroup
// produces a 64x64 output tile; each thread owns four rows and four contiguous
// columns. K advances strictly from 0 to 1279. B and C are viewed as vec4.

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = 1280u;
const N: u32 = 7360u;
const N_VECS: u32 = N / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_COLUMNS: u32 = 64u;
const TILE_K: u32 = 16u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 16u;

var<workgroup> input_tile: array<f32, 1024>;
var<workgroup> weight_tile: array<vec4<f32>, 256>;

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

        let tile_k = local_index / LOCAL_COLUMN_VECS;
        let tile_column_vec = local_index - tile_k * LOCAL_COLUMN_VECS;
        let column_vec = column_vec_base + tile_column_vec;
        weight_tile[local_index] = weight[(k_base + tile_k) * N_VECS + column_vec];
        workgroupBarrier();

        for (var tile_k_index = 0u; tile_k_index < TILE_K; tile_k_index = tile_k_index + 1u) {
            let weight_value = weight_tile[tile_k_index * LOCAL_COLUMN_VECS + local_id.x];
            let row_0 = local_id.y;
            let row_1 = row_0 + LOCAL_ROWS;
            let row_2 = row_1 + LOCAL_ROWS;
            let row_3 = row_2 + LOCAL_ROWS;
            acc_0 = fma(vec4<f32>(input_tile[row_0 * TILE_K + tile_k_index]), weight_value, acc_0);
            acc_1 = fma(vec4<f32>(input_tile[row_1 * TILE_K + tile_k_index]), weight_value, acc_1);
            acc_2 = fma(vec4<f32>(input_tile[row_2 * TILE_K + tile_k_index]), weight_value, acc_2);
            acc_3 = fma(vec4<f32>(input_tile[row_3 * TILE_K + tile_k_index]), weight_value, acc_3);
        }
        workgroupBarrier();
    }

    let output_column_vec = column_vec_base + local_id.x;
    let output_row_0 = row_base + local_id.y;
    let output_row_1 = output_row_0 + LOCAL_ROWS;
    let output_row_2 = output_row_1 + LOCAL_ROWS;
    let output_row_3 = output_row_2 + LOCAL_ROWS;
    if (output_row_0 < ROWS) {
        output[output_row_0 * N_VECS + output_column_vec] = acc_0;
    }
    if (output_row_1 < ROWS) {
        output[output_row_1 * N_VECS + output_column_vec] = acc_1;
    }
    if (output_row_2 < ROWS) {
        output[output_row_2 * N_VECS + output_column_vec] = acc_2;
    }
    if (output_row_3 < ROWS) {
        output[output_row_3 * N_VECS + output_column_vec] = acc_3;
    }
}
