enable f16;

// F16 storage variant. Products and accumulation remain F32 and round once
// at the output store, matching the other handwritten projection kernels.

@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> bias: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;

const ROWS: u32 = {{ rows }}u;
const BATCH: u32 = {{ batch }}u;
const K: u32 = 32u;
const N: u32 = 1280u;
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

        let load_k = local_index / LOCAL_COLUMN_VECS;
        let load_output_vec = local_index - load_k * LOCAL_COLUMN_VECS;
        let weight_value = vec4<f32>(weight[(k_base + load_k) * N_VECS + column_vec_base + load_output_vec]);
        let shared_output = load_output_vec * 4u;
        weight_tile[load_k * 64u + shared_output + 0u] = weight_value.x;
        weight_tile[load_k * 64u + shared_output + 1u] = weight_value.y;
        weight_tile[load_k * 64u + shared_output + 2u] = weight_value.z;
        weight_tile[load_k * 64u + shared_output + 3u] = weight_value.w;
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
    let output_rows = array<u32, 4>(
        row_base + local_id.y,
        row_base + local_id.y + LOCAL_ROWS,
        row_base + local_id.y + 2u * LOCAL_ROWS,
        row_base + local_id.y + 3u * LOCAL_ROWS,
    );
    let values = array<vec4<f32>, 4>(
        acc_0 + bias_value,
        acc_1 + bias_value,
        acc_2 + bias_value,
        acc_3 + bias_value,
    );
    for (var owned_row = 0u; owned_row < 4u; owned_row = owned_row + 1u) {
        let row = output_rows[owned_row];
        if (row < ROWS) {
            for (var batch = 0u; batch < BATCH; batch = batch + 1u) {
                output[(batch * ROWS + row) * N_VECS + output_column_vec] = vec4<f16>(values[owned_row]);
            }
        }
    }
}
