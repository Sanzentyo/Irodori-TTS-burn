enable f16;

// Released no-aux duration block: residual + gate * branch.

@group(0) @binding(0) var<storage, read_write> residual: array<f16>;
@group(0) @binding(1) var<storage, read_write> branch: array<f16>;
@group(0) @binding(2) var<storage, read_write> gate: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

const ELEMENTS: u32 = {{ elements }}u;
const DIM: u32 = 1024u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < ELEMENTS) {
        output[index] = f16(fma(f32(gate[index % DIM]), f32(branch[index]), f32(residual[index])));
    }
}
