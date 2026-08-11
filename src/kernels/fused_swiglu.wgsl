// Fused SwiGLU activation for an already fused w1||w3 projection.
//
// Input is [rows, 2 * hidden], laid out as [gate | value] per row.
// Output is [rows, hidden]: silu(gate) * value.
// One dispatch replaces two slices, sigmoid/SiLU, and multiplication.

// SourceKernel supplies raw WGSL without CubeCL's generated-kernel binding
// metadata. CubeCL's sliced allocator may therefore bind disjoint logical
// tensors from the same physical buffer in one Tasks compute pass. WGPU tracks
// storage usage for that whole buffer, so mixing read-only input and read-write
// output bindings is invalid even when their byte ranges do not overlap.
// Declare every storage binding read_write; the kernel still treats input as
// logically immutable.
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const HIDDEN: u32 = {{ hidden }}u;
const ELEMENTS: u32 = {{ elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= ELEMENTS) {
        return;
    }

    let row = index / HIDDEN;
    let column = index - row * HIDDEN;
    let input_row = row * HIDDEN * 2u;
    let gate = input[input_row + column];
    let value = input[input_row + HIDDEN + column];
    output[index] = gate / (1.0 + exp(-gate)) * value;
}
