// SwiGLU activation for two independently projected gate/value tensors.
//
// Inputs and output are contiguous [rows, hidden]. Keeping the two dense
// projections separate lets the backend tune the narrower GEMMs independently,
// while this dispatch avoids materialising a high-level SiLU/multiply graph.
// All bindings are read_write because CubeCL's sliced allocator can bind
// disjoint logical tensors from one physical WGPU buffer in a single task.
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const ELEMENTS: u32 = {{ elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= ELEMENTS) {
        return;
    }

    let gate_value = gate[index];
    output[index] = gate_value / (1.0 + exp(-gate_value)) * value[index];
}
