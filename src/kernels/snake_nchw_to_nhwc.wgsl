// Snake1d while materializing contiguous NCHW as contiguous NHWC.
@group(0) @binding(0) var<storage, read_write> input_nchw: array<f32>;
@group(0) @binding(1) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_nhwc: array<f32>;

const BATCH: u32 = {{ batch }}u;
const CHANNELS: u32 = {{ channels }}u;
const TIME: u32 = {{ time }}u;
const TILE: u32 = {{ tile }}u;
const TILE_STRIDE: u32 = {{ tile_stride }}u;

var<workgroup> activated_tile: array<f32, 1056>;

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let time = group_id.x * TILE + local_id.x;
    let channel_base = group_id.y * TILE;
    for (var row = local_id.y; row < TILE; row += 8u) {
        let channel = channel_base + row;
        var activated = 0.0;
        if (channel < CHANNELS && time < TIME) {
            let input_index =
                (group_id.z * CHANNELS + channel) * TIME + time;
            let x = input_nchw[input_index];
            let a = alpha[channel];
            let sine = sin(a * x);
            activated = x + (sine * sine) / (a + 1e-9);
        }
        activated_tile[row * TILE_STRIDE + local_id.x] = activated;
    }

    workgroupBarrier();

    let output_channel = channel_base + local_id.x;
    for (var row = local_id.y; row < TILE; row += 8u) {
        let output_time = group_id.x * TILE + row;
        if (output_channel < CHANNELS && output_time < TIME) {
            let output_index =
                (group_id.z * TIME + output_time) * CHANNELS + output_channel;
            output_nhwc[output_index] =
                activated_tile[local_id.x * TILE_STRIDE + row];
        }
    }
}
