// Subgroup-aligned FP32 DiT MLP expansion with a SwiGLU epilogue.
//
// The logical output tile remains 64 rows by 64 hidden columns. Compared with
// the 16x16 vec4 route, each 32-lane subgroup now follows one logical row and
// each lane owns two adjacent hidden columns across eight rows. Global bytes,
// shared bytes, output scalars, and K reduction order remain unchanged.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;

const ROWS: u32 = {{ rows }}u;
const K: u32 = 1280u;
const EXPANDED: u32 = 7360u;
const HIDDEN: u32 = 3680u;
const K_VECS: u32 = K / 4u;
const EXPANDED_VECS: u32 = EXPANDED / 2u;
const HIDDEN_VECS: u32 = HIDDEN / 2u;
const TILE_ROWS: u32 = {{ tile_rows }}u;
const TILE_K: u32 = 32u;
const TILE_K_VECS: u32 = TILE_K / 4u;
const LOCAL_ROWS: u32 = {{ local_rows }}u;
const LOCAL_COLUMN_VECS: u32 = 32u;

var<workgroup> input_tile: array<vec4<f32>, {{ input_tile_vecs }}>;
var<workgroup> weight_tile: array<vec2<f32>, 2048>;

fn swiglu(gate: vec2<f32>, value: vec2<f32>) -> vec2<f32> {
    return gate / (vec2<f32>(1.0) + exp(-gate)) * value;
}

@compute @workgroup_size(32, {{ workgroup_y }}, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let row_base = group_id.y * TILE_ROWS;
    let hidden_vec_base = group_id.x * LOCAL_COLUMN_VECS;
    var gate_0 = vec2<f32>(0.0);
    var gate_1 = vec2<f32>(0.0);
    var gate_2 = vec2<f32>(0.0);
    var gate_3 = vec2<f32>(0.0);
    var gate_4 = vec2<f32>(0.0);
    var gate_5 = vec2<f32>(0.0);
    var gate_6 = vec2<f32>(0.0);
    var gate_7 = vec2<f32>(0.0);
    var value_0 = vec2<f32>(0.0);
    var value_1 = vec2<f32>(0.0);
    var value_2 = vec2<f32>(0.0);
    var value_3 = vec2<f32>(0.0);
    var value_4 = vec2<f32>(0.0);
    var value_5 = vec2<f32>(0.0);
    var value_6 = vec2<f32>(0.0);
    var value_7 = vec2<f32>(0.0);

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
            var element = vec2<f32>(0.0);
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

            let gate_weight_0 = weight_tile[weight_base];
            let value_weight_0 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base];
            gate_0 = fma(vec2<f32>(input_0.x), gate_weight_0, gate_0);
            gate_1 = fma(vec2<f32>(input_1.x), gate_weight_0, gate_1);
            gate_2 = fma(vec2<f32>(input_2.x), gate_weight_0, gate_2);
            gate_3 = fma(vec2<f32>(input_3.x), gate_weight_0, gate_3);
            gate_4 = fma(vec2<f32>(input_4.x), gate_weight_0, gate_4);
            gate_5 = fma(vec2<f32>(input_5.x), gate_weight_0, gate_5);
            gate_6 = fma(vec2<f32>(input_6.x), gate_weight_0, gate_6);
            gate_7 = fma(vec2<f32>(input_7.x), gate_weight_0, gate_7);
            value_0 = fma(vec2<f32>(input_0.x), value_weight_0, value_0);
            value_1 = fma(vec2<f32>(input_1.x), value_weight_0, value_1);
            value_2 = fma(vec2<f32>(input_2.x), value_weight_0, value_2);
            value_3 = fma(vec2<f32>(input_3.x), value_weight_0, value_3);
            value_4 = fma(vec2<f32>(input_4.x), value_weight_0, value_4);
            value_5 = fma(vec2<f32>(input_5.x), value_weight_0, value_5);
            value_6 = fma(vec2<f32>(input_6.x), value_weight_0, value_6);
            value_7 = fma(vec2<f32>(input_7.x), value_weight_0, value_7);

            let gate_weight_1 = weight_tile[weight_base + LOCAL_COLUMN_VECS];
            let value_weight_1 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base + LOCAL_COLUMN_VECS];
            gate_0 = fma(vec2<f32>(input_0.y), gate_weight_1, gate_0);
            gate_1 = fma(vec2<f32>(input_1.y), gate_weight_1, gate_1);
            gate_2 = fma(vec2<f32>(input_2.y), gate_weight_1, gate_2);
            gate_3 = fma(vec2<f32>(input_3.y), gate_weight_1, gate_3);
            gate_4 = fma(vec2<f32>(input_4.y), gate_weight_1, gate_4);
            gate_5 = fma(vec2<f32>(input_5.y), gate_weight_1, gate_5);
            gate_6 = fma(vec2<f32>(input_6.y), gate_weight_1, gate_6);
            gate_7 = fma(vec2<f32>(input_7.y), gate_weight_1, gate_7);
            value_0 = fma(vec2<f32>(input_0.y), value_weight_1, value_0);
            value_1 = fma(vec2<f32>(input_1.y), value_weight_1, value_1);
            value_2 = fma(vec2<f32>(input_2.y), value_weight_1, value_2);
            value_3 = fma(vec2<f32>(input_3.y), value_weight_1, value_3);
            value_4 = fma(vec2<f32>(input_4.y), value_weight_1, value_4);
            value_5 = fma(vec2<f32>(input_5.y), value_weight_1, value_5);
            value_6 = fma(vec2<f32>(input_6.y), value_weight_1, value_6);
            value_7 = fma(vec2<f32>(input_7.y), value_weight_1, value_7);

            let gate_weight_2 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS];
            let value_weight_2 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base + 2u * LOCAL_COLUMN_VECS];
            gate_0 = fma(vec2<f32>(input_0.z), gate_weight_2, gate_0);
            gate_1 = fma(vec2<f32>(input_1.z), gate_weight_2, gate_1);
            gate_2 = fma(vec2<f32>(input_2.z), gate_weight_2, gate_2);
            gate_3 = fma(vec2<f32>(input_3.z), gate_weight_2, gate_3);
            gate_4 = fma(vec2<f32>(input_4.z), gate_weight_2, gate_4);
            gate_5 = fma(vec2<f32>(input_5.z), gate_weight_2, gate_5);
            gate_6 = fma(vec2<f32>(input_6.z), gate_weight_2, gate_6);
            gate_7 = fma(vec2<f32>(input_7.z), gate_weight_2, gate_7);
            value_0 = fma(vec2<f32>(input_0.z), value_weight_2, value_0);
            value_1 = fma(vec2<f32>(input_1.z), value_weight_2, value_1);
            value_2 = fma(vec2<f32>(input_2.z), value_weight_2, value_2);
            value_3 = fma(vec2<f32>(input_3.z), value_weight_2, value_3);
            value_4 = fma(vec2<f32>(input_4.z), value_weight_2, value_4);
            value_5 = fma(vec2<f32>(input_5.z), value_weight_2, value_5);
            value_6 = fma(vec2<f32>(input_6.z), value_weight_2, value_6);
            value_7 = fma(vec2<f32>(input_7.z), value_weight_2, value_7);

            let gate_weight_3 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS];
            let value_weight_3 = weight_tile[TILE_K * LOCAL_COLUMN_VECS + weight_base + 3u * LOCAL_COLUMN_VECS];
            gate_0 = fma(vec2<f32>(input_0.w), gate_weight_3, gate_0);
            gate_1 = fma(vec2<f32>(input_1.w), gate_weight_3, gate_1);
            gate_2 = fma(vec2<f32>(input_2.w), gate_weight_3, gate_2);
            gate_3 = fma(vec2<f32>(input_3.w), gate_weight_3, gate_3);
            gate_4 = fma(vec2<f32>(input_4.w), gate_weight_3, gate_4);
            gate_5 = fma(vec2<f32>(input_5.w), gate_weight_3, gate_5);
            gate_6 = fma(vec2<f32>(input_6.w), gate_weight_3, gate_6);
            gate_7 = fma(vec2<f32>(input_7.w), gate_weight_3, gate_7);
            value_0 = fma(vec2<f32>(input_0.w), value_weight_3, value_0);
            value_1 = fma(vec2<f32>(input_1.w), value_weight_3, value_1);
            value_2 = fma(vec2<f32>(input_2.w), value_weight_3, value_2);
            value_3 = fma(vec2<f32>(input_3.w), value_weight_3, value_3);
            value_4 = fma(vec2<f32>(input_4.w), value_weight_3, value_4);
            value_5 = fma(vec2<f32>(input_5.w), value_weight_3, value_5);
            value_6 = fma(vec2<f32>(input_6.w), value_weight_3, value_6);
            value_7 = fma(vec2<f32>(input_7.w), value_weight_3, value_7);
        }
        workgroupBarrier();
    }

    let output_column_vec = hidden_vec_base + local_id.x;
    let output_row_0 = row_base + local_id.y;
    let output_row_1 = output_row_0 + LOCAL_ROWS;
    let output_row_2 = output_row_1 + LOCAL_ROWS;
    let output_row_3 = output_row_2 + LOCAL_ROWS;
    let output_row_4 = output_row_3 + LOCAL_ROWS;
    let output_row_5 = output_row_4 + LOCAL_ROWS;
    let output_row_6 = output_row_5 + LOCAL_ROWS;
    let output_row_7 = output_row_6 + LOCAL_ROWS;
    if (output_column_vec < HIDDEN_VECS) {
        if (output_row_0 < ROWS) { output[output_row_0 * HIDDEN_VECS + output_column_vec] = swiglu(gate_0, value_0); }
        if (output_row_1 < ROWS) { output[output_row_1 * HIDDEN_VECS + output_column_vec] = swiglu(gate_1, value_1); }
        if (output_row_2 < ROWS) { output[output_row_2 * HIDDEN_VECS + output_column_vec] = swiglu(gate_2, value_2); }
        if (output_row_3 < ROWS) { output[output_row_3 * HIDDEN_VECS + output_column_vec] = swiglu(gate_3, value_3); }
        if (output_row_4 < ROWS) { output[output_row_4 * HIDDEN_VECS + output_column_vec] = swiglu(gate_4, value_4); }
        if (output_row_5 < ROWS) { output[output_row_5 * HIDDEN_VECS + output_column_vec] = swiglu(gate_5, value_5); }
        if (output_row_6 < ROWS) { output[output_row_6 * HIDDEN_VECS + output_column_vec] = swiglu(gate_6, value_6); }
        if (output_row_7 < ROWS) { output[output_row_7 * HIDDEN_VECS + output_column_vec] = swiglu(gate_7, value_7); }
    }
}
