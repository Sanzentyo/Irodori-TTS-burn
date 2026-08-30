{{ enable_f16 }}

// Combine two F32 MLP-contract partials and apply the gated residual.

@group(0) @binding(0) var<storage, read_write> partial: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> residual: array<vec4<{{ storage }}>>;
@group(0) @binding(2) var<storage, read_write> gate: array<vec4<{{ storage }}>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<{{ storage }}>>;

const ROWS: u32 = {{ rows }}u;
const SEQUENCE: u32 = {{ sequence }}u;
const N_VECS: u32 = 320u;
const OUTPUT_VECS: u32 = ROWS * N_VECS;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= OUTPUT_VECS) {
        return;
    }
    let row = index / N_VECS;
    let column_vec = index - row * N_VECS;
    let batch = row / SEQUENCE;
    let branch = partial[index] + partial[OUTPUT_VECS + index];
    let residual_value = vec4<f32>(residual[index]);
    let gate_value = vec4<f32>(gate[batch * N_VECS + column_vec]);
    output[index] = vec4<{{ storage }}>(residual_value + gate_value * branch);
}
