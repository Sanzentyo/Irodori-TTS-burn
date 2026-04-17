// Minimal diagnostic kernel to verify `enable subgroups;` behavior.
//
// This shader performs a trivial copy (output[i] = input[i] * 2.0) to test
// whether the `enable subgroups;` directive affects kernel correctness on
// different backends (DX12, Vulkan, Metal).
//
// FINDING (wgpu 29.0.1 + DX12, NVIDIA RTX 5070 Ti):
//   - WITHOUT `enable subgroups;`: kernel produces correct output ✓
//   - WITH `enable subgroups;` (SUBGROUP_VARIANT): output is all zeros ✗
//     even though NO subgroup operations are used in the shader body.
//   - This appears to be a naga codegen bug specific to DX12.

{{ subgroup_enable }}

@group(0) @binding(0) var<storage, read_write> input_buf:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

const N: u32 = {{ num_elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < N) {
        output_buf[idx] = input_buf[idx] * 2.0;
    }
}
