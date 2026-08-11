// Isolated LowRankAdaLN wide-up diagonal selector and residual epilogue.
//
// This folds the two existing elementwise kernels into the mandatory diagonal
// select. Keep the two f32 additions explicit and left associative so the
// operation remains (selected_up + bias) + raw.
//
// SourceKernel buffers uniformly use read_write for sliced-buffer validity;
// the first three bindings remain logically immutable.

@group(0) @binding(0) var<storage, read_write> wide_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> raw: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

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
    let wide_index = row * WIDE_WIDTH + branch * BRANCH_WIDTH + column;
    let bias_index = branch * BRANCH_WIDTH + column;
    let biased: f32 = wide_up[wide_index] + bias[bias_index];
    output[output_index] = biased + raw[output_index];
}
