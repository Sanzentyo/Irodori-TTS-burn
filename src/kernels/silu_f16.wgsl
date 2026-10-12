enable f16;

@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= {{ elements }}u) { return; }
    let x = f32(input[id.x]);
    // Match PyTorch's opmath: round once after the complete SiLU expression.
    output[id.x] = f16(x / (1.0 + exp(-x)));
}
