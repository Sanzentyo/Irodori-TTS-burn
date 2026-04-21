// Diagnostic kernel: tests that `enable f16;` does NOT cause silent kernel failure
// on this backend/driver combination.
//
// Reads f16 inputs, multiplies by 2.0, writes f32 outputs.
// If `enable f16;` triggers the same naga silent-zero bug as `enable subgroups;`,
// the output will be all zeros instead of 2× the input.
enable f16;

@group(0) @binding(0) var<storage, read_write> input_buf:  array<f16>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

const N: u32 = {{ num_elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < N) {
        output_buf[idx] = f32(input_buf[idx]) * 2.0;
    }
}
