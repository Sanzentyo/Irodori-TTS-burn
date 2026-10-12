// Exact col2im finalizer with a second post-storage-cast Snake output.
// A padded workgroup tile coalesces both the raw NCL and activated NHWC stores.

@group(0) @binding(0) var<storage, read_write> columns_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> bias_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> alpha_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> raw_ncl_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> activated_nhwc_buf: array<f32>;

const OUTPUT_CHANNELS: u32 = {{ output_channels }}u;
const INPUT_LENGTH: u32 = {{ input_length }}u;
const OUTPUT_LENGTH: u32 = {{ output_length }}u;
const STRIDE: u32 = {{ stride }}u;
const KERNEL_SIZE: u32 = {{ kernel_size }}u;
const PADDING: u32 = {{ padding }}u;
const TILE: u32 = {{ tile }}u;
const TILE_STRIDE: u32 = {{ tile_stride }}u;

var<workgroup> activated_tile: array<f32, {{ tile }} * {{ tile_stride }}>;

@compute @workgroup_size({{ tile }}, {{ tile }}, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let output_time = group_id.x * TILE + local_id.x;
    let output_channel = group_id.y * TILE + local_id.y;
    if (output_time < OUTPUT_LENGTH && output_channel < OUTPUT_CHANNELS) {
        let padded_time = output_time + PADDING;
        var input_start = 0u;
        if (padded_time >= KERNEL_SIZE) {
            input_start = (padded_time - KERNEL_SIZE) / STRIDE + 1u;
        }
        let input_end = min(padded_time / STRIDE + 1u, INPUT_LENGTH);

        var value = 0.0;
        if (input_start < input_end) {
            let kernel_index = padded_time - input_start * STRIDE;
            let column_index =
                (output_channel * KERNEL_SIZE + kernel_index) * INPUT_LENGTH + input_start;
            value = value + columns_buf[column_index];
        }
        let second_input = input_start + 1u;
        if (second_input < input_end) {
            let kernel_index = padded_time - second_input * STRIDE;
            let column_index =
                (output_channel * KERNEL_SIZE + kernel_index) * INPUT_LENGTH + second_input;
            value = value + columns_buf[column_index];
        }

        let raw = value + bias_buf[output_channel];
        raw_ncl_buf[output_channel * OUTPUT_LENGTH + output_time] = raw;
        let alpha = alpha_buf[output_channel];
        let sine = sin(alpha * raw);
        activated_tile[local_id.y * TILE_STRIDE + local_id.x] =
            raw + (sine * sine) / (alpha + 1e-9);
    }

    workgroupBarrier();

    let activated_time = group_id.x * TILE + local_id.y;
    let activated_channel = group_id.y * TILE + local_id.x;
    if (activated_time < OUTPUT_LENGTH && activated_channel < OUTPUT_CHANNELS) {
        activated_nhwc_buf[activated_time * OUTPUT_CHANNELS + activated_channel] =
            activated_tile[local_id.x * TILE_STRIDE + local_id.y];
    }
}
