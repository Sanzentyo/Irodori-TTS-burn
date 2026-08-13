enable f16;

// Released compact duration input projection with bias-last arithmetic.
// A[ROWS,512] is row-major. The logical B[512,1024] view directly retains the
// checkpoint-native physical [1024,512] output-major rows. Each workgroup
// produces a 64x64 output tile; each thread owns four rows and four columns.

@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> bias: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = 512u;
const N: u32 = 1024u;
const N_VECS: u32 = N / 4u;
const K_VECS: u32 = K / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_K: u32 = 16u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 16u;

var<workgroup> input_tile: array<f32, 1024>;
var<workgroup> weight_tile: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let row_base = group_id.y * TILE_ROWS;
    let column_base = group_id.x * 64u;
    let column_vec_base = column_base / 4u;
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
                value = f32(input[row * K + k_base + tile_k]);
            }
            input_tile[load] = value;
        }

        // Load the checkpoint-native output-major rows as contiguous K vec4s,
        // then transpose the 16x64 tile in shared memory for the FMA loop.
        let load_output = local_index / 4u;
        let load_k_vec = local_index - load_output * 4u;
        let weight_value = vec4<f32>(weight[(column_base + load_output) * K_VECS + k_base / 4u + load_k_vec]);
        let shared_k = load_k_vec * 4u;
        weight_tile[(shared_k + 0u) * 64u + load_output] = weight_value.x;
        weight_tile[(shared_k + 1u) * 64u + load_output] = weight_value.y;
        weight_tile[(shared_k + 2u) * 64u + load_output] = weight_value.z;
        weight_tile[(shared_k + 3u) * 64u + load_output] = weight_value.w;
        workgroupBarrier();

        for (var tile_k_index = 0u; tile_k_index < TILE_K; tile_k_index = tile_k_index + 1u) {
            let tile_output = local_id.x * 4u;
            let weight_base = tile_k_index * 64u + tile_output;
            let weight_value = vec4<f32>(
                weight_tile[weight_base + 0u],
                weight_tile[weight_base + 1u],
                weight_tile[weight_base + 2u],
                weight_tile[weight_base + 3u],
            );
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
    let bias_value = vec4<f32>(bias[output_column_vec]);
    let output_row_0 = row_base + local_id.y;
    let output_row_1 = output_row_0 + LOCAL_ROWS;
    let output_row_2 = output_row_1 + LOCAL_ROWS;
    let output_row_3 = output_row_2 + LOCAL_ROWS;
    if (output_row_0 < ROWS) {
        output[output_row_0 * N_VECS + output_column_vec] = vec4<f16>(acc_0 + bias_value);
    }
    if (output_row_1 < ROWS) {
        output[output_row_1 * N_VECS + output_column_vec] = vec4<f16>(acc_1 + bias_value);
    }
    if (output_row_2 < ROWS) {
        output[output_row_2 * N_VECS + output_column_vec] = vec4<f16>(acc_2 + bias_value);
    }
    if (output_row_3 < ROWS) {
        output[output_row_3 * N_VECS + output_column_vec] = vec4<f16>(acc_3 + bias_value);
    }
}
