// Fused DACVAE Snake1d activation.
// output[b,c,t] = x + sin(alpha[c] * x)^2 / (alpha[c] + 1e-9)
//
// SourceKernel shaders use uniform read_write storage bindings because
// CubeCL's sliced allocator may place distinct logical tensors in the same
// physical WGPU buffer. WGPU validates access at physical-buffer granularity.
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const CHANNELS: u32 = {{ channels }}u;
const TIME: u32 = {{ time }}u;
const ELEMENTS: u32 = {{ elements }}u;
const DISPATCH_X: u32 = {{ dispatch_x }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let linear_group = group_id.y * DISPATCH_X + group_id.x;
    let index = linear_group * {{ workgroup_size }}u + local_id.x;
    if (index >= ELEMENTS) {
        return;
    }

    let channel = (index / TIME) % CHANNELS;
    let x = input[index];
    let a = alpha[channel];
    let sine = sin(a * x);
    output[index] = x + (sine * sine) / (a + 1e-9);
}
