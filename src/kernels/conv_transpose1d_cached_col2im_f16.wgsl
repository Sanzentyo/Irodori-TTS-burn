enable f16;

// Exact 1D col2im epilogue for released DACVAE ConvTranspose1d cases 1--3.
//
// Burn first computes contiguous columns [Cout * kernel, Lin]. Its generic
// col2im kernel initializes zero, visits contributing input times in ascending
// order, adds each column value, and adds bias last. For the exact released
// contract k=2*stride, padding=stride/2, dilation=1, and padding_out=0, there
// are at most two contributors. The two guarded statements below preserve
// that order without the generic loop or rank-4 index decomposition.
//
// All SourceKernel storage declarations are read_write because CubeCL may
// place disjoint logical tensors in slices of one physical allocator buffer.

@group(0) @binding(0) var<storage, read_write> columns_buf: array<f16>;
@group(0) @binding(1) var<storage, read_write> bias_buf: array<f16>;
@group(0) @binding(2) var<storage, read_write> output_buf: array<f16>;

const OUTPUT_CHANNELS: u32 = {{ output_channels }}u;
const INPUT_LENGTH: u32 = {{ input_length }}u;
const OUTPUT_LENGTH: u32 = {{ output_length }}u;
const STRIDE: u32 = {{ stride }}u;
const KERNEL_SIZE: u32 = {{ kernel_size }}u;
const PADDING: u32 = {{ padding }}u;
const OUTPUT_ELEMENTS: u32 = {{ output_elements }}u;
const DISPATCH_X: u32 = {{ dispatch_x }}u;
const WORKGROUP_SIZE: u32 = {{ workgroup_size }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let workgroup_linear = group_id.y * DISPATCH_X + group_id.x;
    let output_index = workgroup_linear * WORKGROUP_SIZE + local_id.x;
    if (output_index >= OUTPUT_ELEMENTS) {
        return;
    }

    let output_time = output_index % OUTPUT_LENGTH;
    let output_channel = (output_index / OUTPUT_LENGTH) % OUTPUT_CHANNELS;
    let padded_time = output_time + PADDING;
    var input_start = 0u;
    if (padded_time >= KERNEL_SIZE) {
        input_start = (padded_time - KERNEL_SIZE) / STRIDE + 1u;
    }
    let input_end = min(padded_time / STRIDE + 1u, INPUT_LENGTH);

    var value = 0.0;
    if (input_start < input_end) {
        let first_kernel_index = padded_time - input_start * STRIDE;
        let first_column_index =
            (output_channel * KERNEL_SIZE + first_kernel_index) * INPUT_LENGTH
            + input_start;
        value = value + f32(columns_buf[first_column_index]);
    }

    let second_input = input_start + 1u;
    if (second_input < input_end) {
        let second_kernel_index = padded_time - second_input * STRIDE;
        let second_column_index =
            (output_channel * KERNEL_SIZE + second_kernel_index) * INPUT_LENGTH
            + second_input;
        value = value + f32(columns_buf[second_column_index]);
    }

    output_buf[output_index] = f16(value + f32(bias_buf[output_channel]));
}
