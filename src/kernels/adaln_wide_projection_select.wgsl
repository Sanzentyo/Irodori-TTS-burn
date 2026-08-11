// Isolated LowRankAdaLN wide-projection diagonal selector.
//
// Input rows are the flattened [batch, branch] dimension. A wide GEMM writes
// every branch projection for every row; only the segment matching row % 3 is
// useful. This copy keeps the GEMM reduction untouched.
//
// SourceKernel supplies raw WGSL without CubeCL binding metadata. The sliced
// WGPU allocator can bind logical tensors from one physical buffer, so every
// storage binding must use the same read_write access class.

@group(0) @binding(0) var<storage, read_write> wide_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const ROWS: u32 = {{ rows }}u;
const BRANCHES: u32 = {{ branches }}u;
const BRANCH_WIDTH: u32 = {{ branch_width }}u;
const ELEMENTS: u32 = {{ elements }}u;
const WIDE_WIDTH: u32 = BRANCHES * BRANCH_WIDTH;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    if (output_index >= ELEMENTS) {
        return;
    }

    let row = output_index / BRANCH_WIDTH;
    if (row >= ROWS) {
        return;
    }
    let column = output_index - row * BRANCH_WIDTH;
    let branch = row % BRANCHES;
    let input_index = row * WIDE_WIDTH + branch * BRANCH_WIDTH + column;
    output[output_index] = wide_input[input_index];
}
