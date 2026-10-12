// Snake1d on contiguous NHWC storage, preserving physical layout.
@group(0) @binding(0) var<storage, read_write> input_nhwc: array<f32>;
@group(0) @binding(1) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_nhwc: array<f32>;

const CHANNELS: u32 = {{ channels }}u;
const ELEMENTS: u32 = {{ elements }}u;
const DISPATCH_X: u32 = {{ dispatch_x }}u;
const WORKGROUP_SIZE: u32 = {{ workgroup_size }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let workgroup_linear = group_id.y * DISPATCH_X + group_id.x;
    let index = workgroup_linear * WORKGROUP_SIZE + local_id.x;
    if (index >= ELEMENTS) { return; }
    let channel = index % CHANNELS;
    let x = input_nhwc[index];
    let a = alpha[channel];
    let sine = sin(a * x);
    output_nhwc[index] = x + (sine * sine) / (a + 1e-9);
}
