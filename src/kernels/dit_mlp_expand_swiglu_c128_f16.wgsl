enable f16;

// Exact released DiT MLP expansion with a SwiGLU epilogue.
//
// A[rows,1280] and B[1280,7360] are row-major f32. Each workgroup
// computes 64 rows and 64 logical hidden columns. The first 64-column weight
// half supplies gate values; the matching columns at +3680 supply values.
// K advances strictly from 0 to 1279, matching the separate production
// projection, before the exact production SwiGLU expression is evaluated.

@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f16>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = 1280u;
const EXPANDED: u32 = 7360u;
const HIDDEN: u32 = 3680u;
const EXPANDED_VECS: u32 = EXPANDED / 4u;
const HIDDEN_VECS: u32 = HIDDEN / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_K: u32 = 32u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 16u;

var<workgroup> input_tile: array<f32, 2048>;
var<workgroup> weight_tile: array<vec4<f32>, 1024>;

fn swiglu(gate: vec4<f32>, value: vec4<f32>) -> vec4<f32> {
    return gate / (vec4<f32>(1.0) + exp(-gate)) * value;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let row_base = group_id.y * TILE_ROWS;
    let hidden_vec_base = group_id.x * LOCAL_COLUMN_VECS;
    var gate_0 = vec4<f32>(0.0);
    var gate_1 = vec4<f32>(0.0);
    var gate_2 = vec4<f32>(0.0);
    var gate_3 = vec4<f32>(0.0);
    var value_0 = vec4<f32>(0.0);
    var value_1 = vec4<f32>(0.0);
    var value_2 = vec4<f32>(0.0);
    var value_3 = vec4<f32>(0.0);

    for (var k_base = 0u; k_base < K; k_base = k_base + TILE_K) {
        for (var load = local_index; load < TILE_ROWS * TILE_K; load = load + 256u) {
            let tile_row = load / TILE_K;
            let tile_k = load - tile_row * TILE_K;
            let row = row_base + tile_row;
            var element = 0.0;
            if (row < ROWS) {
                element = f32(input[row * K + k_base + tile_k]);
            }
            input_tile[load] = element;
        }

        for (var load = local_index; load < TILE_K * LOCAL_COLUMN_VECS * 2u; load = load + 256u) {
            let half = load / (TILE_K * LOCAL_COLUMN_VECS);
            let within_half = load - half * TILE_K * LOCAL_COLUMN_VECS;
            let tile_k = within_half / LOCAL_COLUMN_VECS;
            let tile_column_vec = within_half - tile_k * LOCAL_COLUMN_VECS;
            let hidden_vec = hidden_vec_base + tile_column_vec;
            var element = vec4<f32>(0.0);
            if (hidden_vec < HIDDEN_VECS) {
                let expanded_vec = hidden_vec + half * HIDDEN_VECS;
                element = vec4<f32>(weight[(k_base + tile_k) * EXPANDED_VECS + expanded_vec]);
            }
            weight_tile[load] = element;
        }
        workgroupBarrier();

        for (var tile_k_index = 0u; tile_k_index < TILE_K; tile_k_index = tile_k_index + 1u) {
            let gate_weight = weight_tile[tile_k_index * LOCAL_COLUMN_VECS + local_id.x];
            let value_weight = weight_tile[
                TILE_K * LOCAL_COLUMN_VECS + tile_k_index * LOCAL_COLUMN_VECS + local_id.x
            ];
            let row_0 = local_id.y;
            let row_1 = row_0 + LOCAL_ROWS;
            let row_2 = row_1 + LOCAL_ROWS;
            let row_3 = row_2 + LOCAL_ROWS;
            let input_0 = vec4<f32>(input_tile[row_0 * TILE_K + tile_k_index]);
            let input_1 = vec4<f32>(input_tile[row_1 * TILE_K + tile_k_index]);
            let input_2 = vec4<f32>(input_tile[row_2 * TILE_K + tile_k_index]);
            let input_3 = vec4<f32>(input_tile[row_3 * TILE_K + tile_k_index]);
            gate_0 = fma(input_0, gate_weight, gate_0);
            gate_1 = fma(input_1, gate_weight, gate_1);
            gate_2 = fma(input_2, gate_weight, gate_2);
            gate_3 = fma(input_3, gate_weight, gate_3);
            value_0 = fma(input_0, value_weight, value_0);
            value_1 = fma(input_1, value_weight, value_1);
            value_2 = fma(input_2, value_weight, value_2);
            value_3 = fma(input_3, value_weight, value_3);
        }
        workgroupBarrier();
    }

    let output_column_vec = hidden_vec_base + local_id.x;
    let output_row_0 = row_base + local_id.y;
    let output_row_1 = output_row_0 + LOCAL_ROWS;
    let output_row_2 = output_row_1 + LOCAL_ROWS;
    let output_row_3 = output_row_2 + LOCAL_ROWS;
    if (output_row_0 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_0 * HIDDEN_VECS + output_column_vec] = vec4<f16>(swiglu(gate_0, value_0));
    }
    if (output_row_1 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_1 * HIDDEN_VECS + output_column_vec] = vec4<f16>(swiglu(gate_1, value_1));
    }
    if (output_row_2 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_2 * HIDDEN_VECS + output_column_vec] = vec4<f16>(swiglu(gate_2, value_2));
    }
    if (output_row_3 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_3 * HIDDEN_VECS + output_column_vec] = vec4<f16>(swiglu(gate_3, value_3));
    }
}
