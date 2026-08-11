// Released no-aux duration block: residual + gate * branch.

@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read_write> branch: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const ELEMENTS: u32 = {{ elements }}u;
const DIM: u32 = 1024u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < ELEMENTS) {
        output[index] = fma(gate[index % DIM], branch[index], residual[index]);
    }
}
