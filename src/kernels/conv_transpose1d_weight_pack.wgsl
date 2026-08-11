// One-time layout conversion for DACVAE ConvTranspose1d weights.
//
// Source layout: [Cin, Cout, 2 * stride]
// Packed layout: [phase, Cout, Cin, tap]
//   packed[phase, oc, ic, 0] = source[ic, oc, phase]
//   packed[phase, oc, ic, 1] = source[ic, oc, phase + stride]
//
// Threads read the checkpoint-native tensor contiguously and scatter once to
// the persistent phase-major cache. All SourceKernel storage declarations are
// read_write because CubeCL may slice one physical allocator buffer into
// otherwise disjoint logical tensors.

@group(0) @binding(0) var<storage, read_write> source_weight: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_weight: array<f32>;

const INPUT_CHANNELS: u32 = {{ input_channels }}u;
const OUTPUT_CHANNELS: u32 = {{ output_channels }}u;
const STRIDE: u32 = {{ stride }}u;
const KERNEL_SIZE: u32 = {{ kernel_size }}u;
const ELEMENTS: u32 = {{ elements }}u;
const DISPATCH_X: u32 = {{ dispatch_x }}u;
const WORKGROUP_SIZE: u32 = {{ workgroup_size }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let workgroup_index = group_id.y * DISPATCH_X + group_id.x;
    let source_index = workgroup_index * WORKGROUP_SIZE + local_id.x;
    if (source_index >= ELEMENTS) {
        return;
    }

    let kernel_index = source_index % KERNEL_SIZE;
    let source_outer = source_index / KERNEL_SIZE;
    let output_channel = source_outer % OUTPUT_CHANNELS;
    let input_channel = source_outer / OUTPUT_CHANNELS;
    let phase = kernel_index % STRIDE;
    let tap = kernel_index / STRIDE;
    let packed_index =
        (((phase * OUTPUT_CHANNELS + output_channel) * INPUT_CHANNELS + input_channel) * 2u)
        + tap;
    packed_weight[packed_index] = source_weight[source_index];
}
