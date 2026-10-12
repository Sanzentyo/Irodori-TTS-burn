enable f16;

// F16-storage form of the subgroup-aligned SwiGLU expansion. Global values
// are promoted once and the entire reduction and epilogue remain F32.

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> weight: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f16>>;

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
    var gate: array<vec2<f32>, 8>;
    var value: array<vec2<f32>, 8>;
    for (var row_group = 0u; row_group < 8u; row_group = row_group + 1u) {
        gate[row_group] = vec2<f32>(0.0);
        value[row_group] = vec2<f32>(0.0);
    }

    for (var k_base = 0u; k_base < K; k_base = k_base + TILE_K) {
        for (var load = local_index; load < TILE_ROWS * TILE_K_VECS; load = load + 256u) {
            let tile_row = load / TILE_K_VECS;
            let tile_k_vec = load - tile_row * TILE_K_VECS;
            let row = row_base + tile_row;
            var element = vec4<f32>(0.0);
            if (row < ROWS) {
                element = vec4<f32>(input[row * K_VECS + k_base / 4u + tile_k_vec]);
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
                element = vec2<f32>(weight[(k_base + tile_k) * EXPANDED_VECS + expanded_vec]);
            }
            weight_tile[load] = element;
        }
        workgroupBarrier();

        for (var tile_k_vec = 0u; tile_k_vec < TILE_K_VECS; tile_k_vec = tile_k_vec + 1u) {
            let weight_base = tile_k_vec * 4u * LOCAL_COLUMN_VECS + local_id.x;
            let gate_weight_0 = weight_tile[weight_base];
            let gate_weight_1 = weight_tile[weight_base + LOCAL_COLUMN_VECS];
            let gate_weight_2 = weight_tile[weight_base + 2u * LOCAL_COLUMN_VECS];
            let gate_weight_3 = weight_tile[weight_base + 3u * LOCAL_COLUMN_VECS];
            let value_base = TILE_K * LOCAL_COLUMN_VECS + weight_base;
            let value_weight_0 = weight_tile[value_base];
            let value_weight_1 = weight_tile[value_base + LOCAL_COLUMN_VECS];
            let value_weight_2 = weight_tile[value_base + 2u * LOCAL_COLUMN_VECS];
            let value_weight_3 = weight_tile[value_base + 3u * LOCAL_COLUMN_VECS];
            for (var row_group = 0u; row_group < 8u; row_group = row_group + 1u) {
                let tile_row = local_id.y + row_group * LOCAL_ROWS;
                let input_value = input_tile[tile_row * TILE_K_VECS + tile_k_vec];
                gate[row_group] = fma(vec2<f32>(input_value.x), gate_weight_0, gate[row_group]);
                gate[row_group] = fma(vec2<f32>(input_value.y), gate_weight_1, gate[row_group]);
                gate[row_group] = fma(vec2<f32>(input_value.z), gate_weight_2, gate[row_group]);
                gate[row_group] = fma(vec2<f32>(input_value.w), gate_weight_3, gate[row_group]);
                value[row_group] = fma(vec2<f32>(input_value.x), value_weight_0, value[row_group]);
                value[row_group] = fma(vec2<f32>(input_value.y), value_weight_1, value[row_group]);
                value[row_group] = fma(vec2<f32>(input_value.z), value_weight_2, value[row_group]);
                value[row_group] = fma(vec2<f32>(input_value.w), value_weight_3, value[row_group]);
            }
        }
        workgroupBarrier();
    }

    let output_column_vec = hidden_vec_base + local_id.x;
    if (output_column_vec < HIDDEN_VECS) {
        for (var row_group = 0u; row_group < 8u; row_group = row_group + 1u) {
            let row = row_base + local_id.y + row_group * LOCAL_ROWS;
            if (row < ROWS) {
                output[row * HIDDEN_VECS + output_column_vec] = vec2<f16>(swiglu(gate[row_group], value[row_group]));
            }
        }
    }
}
