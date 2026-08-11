// Production WmHead Stage A: exact Snake1d plus NCL-to-NLC layout write.
//
// The input is contiguous [1, 96, 96000] NCL. The output allocation is
// contiguous [1, 96000, 96] NLC, so Burn's unchanged Conv1d NHWC entry can
// consume it without its ordinary NCL-to-NLC materialization.
//
// A 32x32 shared tile makes both sides coalesced. The 32x8 workgroup loads four
// channel rows per invocation in NCL order, evaluates the production Snake
// expression, then writes four time rows per invocation in NLC order.
//
// All SourceKernel storage declarations are read_write because CubeCL may
// place disjoint logical tensors in slices of one physical allocator buffer.

@group(0) @binding(0) var<storage, read_write> input_ncl: array<f32>;
@group(0) @binding(1) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_nlc: array<f32>;

const CHANNELS: u32 = 96u;
const TIME: u32 = 96000u;
const TILE: u32 = 32u;
const LOCAL_TIME_ROWS: u32 = 8u;
const TILE_STRIDE: u32 = 33u;

var<workgroup> activated_tile: array<f32, 1056>;

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TILE;
    let channel_base = group_id.y * TILE;

    var repeat = 0u;
    loop {
        if (repeat >= 4u) {
            break;
        }
        let channel_lane = local_id.y + repeat * LOCAL_TIME_ROWS;
        let input_channel = channel_base + channel_lane;
        let input_time = time_base + local_id.x;
        let input_index = input_channel * TIME + input_time;
        let x = input_ncl[input_index];
        let a = alpha[input_channel];
        let sine = sin(a * x);
        activated_tile[channel_lane * TILE_STRIDE + local_id.x] =
            x + (sine * sine) / (a + 1e-9);
        repeat += 1u;
    }

    workgroupBarrier();

    repeat = 0u;
    loop {
        if (repeat >= 4u) {
            break;
        }
        let time_lane = local_id.y + repeat * LOCAL_TIME_ROWS;
        let output_time = time_base + time_lane;
        let output_channel = channel_base + local_id.x;
        let output_index = output_time * CHANNELS + output_channel;
        output_nlc[output_index] =
            activated_tile[local_id.x * TILE_STRIDE + time_lane];
        repeat += 1u;
    }
}
