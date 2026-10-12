// Exact DiT MLP expansion with a SwiGLU epilogue and vectorized K staging.
//
// A[rows,1280] and B[1280,7360] are row-major f32. The arithmetic and
// accumulator order match dit_mlp_expand_swiglu_c128.wgsl, but four adjacent
// input scalars enter shared memory through one vec4 load and are consumed by
// an explicitly unrolled four-FMA group.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = 1280u;
const EXPANDED: u32 = 7360u;
const HIDDEN: u32 = 3680u;
const K_VECS: u32 = K / 4u;
const EXPANDED_VECS: u32 = EXPANDED / 4u;
const HIDDEN_VECS: u32 = HIDDEN / 4u;
const TILE_ROWS: u32 = 64u;
const TILE_K: u32 = {{ tile_k }}u;
const TILE_K_VECS: u32 = TILE_K / 4u;
const LOCAL_ROWS: u32 = 16u;
const LOCAL_COLUMN_VECS: u32 = 16u;

var<workgroup> input_tile: array<vec4<f32>, {{ input_tile_vecs }}>;
var<workgroup> weight_tile: array<vec4<f32>, {{ weight_tile_vecs }}>;

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
        for (var load = local_index; load < TILE_ROWS * TILE_K_VECS; load = load + 256u) {
            let tile_row = load / TILE_K_VECS;
            let tile_k_vec = load - tile_row * TILE_K_VECS;
            let row = row_base + tile_row;
            var element = vec4<f32>(0.0);
            if (row < ROWS) {
                element = input[row * K_VECS + k_base / 4u + tile_k_vec];
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
                element = weight[(k_base + tile_k) * EXPANDED_VECS + expanded_vec];
            }
            weight_tile[load] = element;
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

            let gate_weight_0 = weight_tile[weight_base];
            let value_weight_0 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base];
            gate_0 = fma(vec4<f32>(input_0.x), gate_weight_0, gate_0);
            gate_1 = fma(vec4<f32>(input_1.x), gate_weight_0, gate_1);
            gate_2 = fma(vec4<f32>(input_2.x), gate_weight_0, gate_2);
            gate_3 = fma(vec4<f32>(input_3.x), gate_weight_0, gate_3);
            value_0 = fma(vec4<f32>(input_0.x), value_weight_0, value_0);
            value_1 = fma(vec4<f32>(input_1.x), value_weight_0, value_1);
            value_2 = fma(vec4<f32>(input_2.x), value_weight_0, value_2);
            value_3 = fma(vec4<f32>(input_3.x), value_weight_0, value_3);

            let gate_weight_1 = weight_tile[weight_base + LOCAL_COLUMN_VECS];
            let value_weight_1 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base + LOCAL_COLUMN_VECS];
            gate_0 = fma(vec4<f32>(input_0.y), gate_weight_1, gate_0);
            gate_1 = fma(vec4<f32>(input_1.y), gate_weight_1, gate_1);
            gate_2 = fma(vec4<f32>(input_2.y), gate_weight_1, gate_2);
            gate_3 = fma(vec4<f32>(input_3.y), gate_weight_1, gate_3);
            value_0 = fma(vec4<f32>(input_0.y), value_weight_1, value_0);
            value_1 = fma(vec4<f32>(input_1.y), value_weight_1, value_1);
            value_2 = fma(vec4<f32>(input_2.y), value_weight_1, value_2);
            value_3 = fma(vec4<f32>(input_3.y), value_weight_1, value_3);

            let gate_weight_2 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS];
            let value_weight_2 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base + 2u * LOCAL_COLUMN_VECS];
            gate_0 = fma(vec4<f32>(input_0.z), gate_weight_2, gate_0);
            gate_1 = fma(vec4<f32>(input_1.z), gate_weight_2, gate_1);
            gate_2 = fma(vec4<f32>(input_2.z), gate_weight_2, gate_2);
            gate_3 = fma(vec4<f32>(input_3.z), gate_weight_2, gate_3);
            value_0 = fma(vec4<f32>(input_0.z), value_weight_2, value_0);
            value_1 = fma(vec4<f32>(input_1.z), value_weight_2, value_1);
            value_2 = fma(vec4<f32>(input_2.z), value_weight_2, value_2);
            value_3 = fma(vec4<f32>(input_3.z), value_weight_2, value_3);

            let gate_weight_3 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS];
            let value_weight_3 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base + 3u * LOCAL_COLUMN_VECS];
            gate_0 = fma(vec4<f32>(input_0.w), gate_weight_3, gate_0);
            gate_1 = fma(vec4<f32>(input_1.w), gate_weight_3, gate_1);
            gate_2 = fma(vec4<f32>(input_2.w), gate_weight_3, gate_2);
            gate_3 = fma(vec4<f32>(input_3.w), gate_weight_3, gate_3);
            value_0 = fma(vec4<f32>(input_0.w), value_weight_3, value_0);
            value_1 = fma(vec4<f32>(input_1.w), value_weight_3, value_1);
            value_2 = fma(vec4<f32>(input_2.w), value_weight_3, value_2);
            value_3 = fma(vec4<f32>(input_3.w), value_weight_3, value_3);
        }
        workgroupBarrier();
    }

    let output_column_vec = hidden_vec_base + local_id.x;
    let output_row_0 = row_base + local_id.y;
    let output_row_1 = output_row_0 + LOCAL_ROWS;
    let output_row_2 = output_row_1 + LOCAL_ROWS;
    let output_row_3 = output_row_2 + LOCAL_ROWS;
    if (output_row_0 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_0 * HIDDEN_VECS + output_column_vec] = swiglu(gate_0, value_0);
    }
    if (output_row_1 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_1 * HIDDEN_VECS + output_column_vec] = swiglu(gate_1, value_1);
    }
    if (output_row_2 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_2 * HIDDEN_VECS + output_column_vec] = swiglu(gate_2, value_2);
    }
    if (output_row_3 < ROWS && output_column_vec < HIDDEN_VECS) {
        output[output_row_3 * HIDDEN_VECS + output_column_vec] = swiglu(gate_3, value_3);
    }
}
