// In-place SwiGLU for a contiguous [rows, gate | value] projection.
//
// The activated result occupies the first half of each physical row. Its
// logical view is therefore pitched with row stride 2 * HIDDEN.

@group(0) @binding(0) var<storage, read_write> projection: array<f32>;

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
    let physical_row = row * HIDDEN * 2u;
    let gate = projection[physical_row + column];
    let value = projection[physical_row + HIDDEN + column];
    projection[physical_row + column] = gate / (1.0 + exp(-gate)) * value;
}
