// Tiled polyphase ConvTranspose1d for the released DACVAE decoder.
//
// Exact operation:
//   B=1, k=2*stride, padding=stride/2, output_padding=0,
//   dilation=1, groups=1, bias=true.
//
// Packed weights are [phase, Cout, Cin, 2]. For a phase p, logical output
// time t, and shift=(p < padding):
//   a = t + shift
//   y = t*stride + (p + stride - padding) % stride
//   out[oc,y] = bias[oc]
//             + sum_ic input[ic,a]   * packed[p,oc,ic,0]
//             + sum_ic input[ic,a-1] * packed[p,oc,ic,1]
// Invalid input times are zero. This is exactly the scatter definition
// y = n*stride - padding + kernel_index, regrouped by kernel_index % stride.
//
// The portable tile uses an 8x16 workgroup, 32 times, and Cin=16. Exact
// released case 0 uses the measured 16x16 workgroup, 64 times, and Cin=32 when
// device limits allow it; otherwise it falls back to the portable tile. Each
// invocation owns four time positions and one output channel. Separate tap
// accumulators preserve the accepted floating-point evaluation order.
//
// All SourceKernel storage declarations are read_write because CubeCL may
// slice one physical allocator buffer into disjoint logical tensors.

@group(0) @binding(0) var<storage, read_write> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_weight_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_buf: array<f32>;

const INPUT_CHANNELS: u32 = {{ input_channels }}u;
const OUTPUT_CHANNELS: u32 = {{ output_channels }}u;
const INPUT_LENGTH: u32 = {{ input_length }}u;
const OUTPUT_LENGTH: u32 = {{ output_length }}u;
const STRIDE: u32 = {{ stride }}u;
const PADDING: u32 = {{ padding }}u;

const LOCAL_TIME_LANES: u32 = {{ local_time_lanes }}u;
const OUTPUT_CHANNEL_TILE: u32 = {{ output_channel_tile }}u;
const TIME_REPEATS: u32 = {{ time_repeats }}u;
const TIME_TILE: u32 = {{ time_tile }}u;
const INPUT_CHANNEL_TILE: u32 = {{ input_channel_tile }}u;
const INPUT_SPAN: u32 = {{ input_span }}u;
const WORKGROUP_SIZE: u32 = {{ workgroup_size }}u;
const INPUT_TILE_ELEMENTS: u32 = {{ input_tile_elements }}u;
const WEIGHT_TILE_ELEMENTS: u32 = {{ weight_tile_elements }}u;

var<workgroup> input_tile: array<f32, {{ input_tile_elements }}>;
var<workgroup> weight_tile: array<f32, {{ weight_tile_elements }}>;

@compute @workgroup_size({{ local_time_lanes }}, {{ output_channel_tile }}, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TIME_TILE;
    let output_channel_base = group_id.y * OUTPUT_CHANNEL_TILE;
    let phase = group_id.z;
    let phase_shift = select(0u, 1u, phase < PADDING);
    let output_phase = (phase + STRIDE - PADDING) % STRIDE;
    let source_time_base = i32(time_base + phase_shift) - 1;
    let output_channel = output_channel_base + local_id.y;

    var accumulator_main_0 = 0.0;
    var accumulator_main_1 = 0.0;
    var accumulator_main_2 = 0.0;
    var accumulator_main_3 = 0.0;
    var accumulator_previous_0 = 0.0;
    var accumulator_previous_1 = 0.0;
    var accumulator_previous_2 = 0.0;
    var accumulator_previous_3 = 0.0;

    var input_channel_base = 0u;
    loop {
        if (input_channel_base >= INPUT_CHANNELS) {
            break;
        }

        var tile_index = local_index;
        loop {
            if (tile_index >= INPUT_TILE_ELEMENTS) {
                break;
            }
            let tile_input_channel = tile_index / INPUT_SPAN;
            let tile_time = tile_index - tile_input_channel * INPUT_SPAN;
            let source_time = source_time_base + i32(tile_time);
            var value = 0.0;
            if (source_time >= 0 && source_time < i32(INPUT_LENGTH)) {
                let input_channel = input_channel_base + tile_input_channel;
                let input_index = input_channel * INPUT_LENGTH + u32(source_time);
                value = input_buf[input_index];
            }
            input_tile[tile_index] = value;
            tile_index += WORKGROUP_SIZE;
        }

        tile_index = local_index;
        loop {
            if (tile_index >= WEIGHT_TILE_ELEMENTS) {
                break;
            }
            let output_stride = INPUT_CHANNEL_TILE * 2u;
            let tile_output_channel = tile_index / output_stride;
            let remainder = tile_index - tile_output_channel * output_stride;
            let tile_input_channel = remainder / 2u;
            let tap = remainder - tile_input_channel * 2u;
            let packed_index =
                (((phase * OUTPUT_CHANNELS + output_channel_base + tile_output_channel)
                    * INPUT_CHANNELS
                    + input_channel_base
                    + tile_input_channel)
                    * 2u)
                + tap;
            weight_tile[tile_index] = packed_weight_buf[packed_index];
            tile_index += WORKGROUP_SIZE;
        }

        workgroupBarrier();

        var tile_input_channel = 0u;
        loop {
            if (tile_input_channel >= INPUT_CHANNEL_TILE) {
                break;
            }
            let input_base = tile_input_channel * INPUT_SPAN + local_id.x;
            let weight_base =
                (local_id.y * INPUT_CHANNEL_TILE + tile_input_channel) * 2u;
            let weight_main = weight_tile[weight_base];
            let weight_previous = weight_tile[weight_base + 1u];

            accumulator_main_0 = fma(
                input_tile[input_base + 1u],
                weight_main,
                accumulator_main_0,
            );
            accumulator_previous_0 = fma(
                input_tile[input_base],
                weight_previous,
                accumulator_previous_0,
            );
            accumulator_main_1 = fma(
                input_tile[input_base + LOCAL_TIME_LANES + 1u],
                weight_main,
                accumulator_main_1,
            );
            accumulator_previous_1 = fma(
                input_tile[input_base + LOCAL_TIME_LANES],
                weight_previous,
                accumulator_previous_1,
            );
            accumulator_main_2 = fma(
                input_tile[input_base + 2u * LOCAL_TIME_LANES + 1u],
                weight_main,
                accumulator_main_2,
            );
            accumulator_previous_2 = fma(
                input_tile[input_base + 2u * LOCAL_TIME_LANES],
                weight_previous,
                accumulator_previous_2,
            );
            accumulator_main_3 = fma(
                input_tile[input_base + 3u * LOCAL_TIME_LANES + 1u],
                weight_main,
                accumulator_main_3,
            );
            accumulator_previous_3 = fma(
                input_tile[input_base + 3u * LOCAL_TIME_LANES],
                weight_previous,
                accumulator_previous_3,
            );
            tile_input_channel += 1u;
        }

        workgroupBarrier();
        input_channel_base += INPUT_CHANNEL_TILE;
    }

    let bias = bias_buf[output_channel];
    let time_0 = time_base + local_id.x;
    let time_1 = time_0 + LOCAL_TIME_LANES;
    let time_2 = time_1 + LOCAL_TIME_LANES;
    let time_3 = time_2 + LOCAL_TIME_LANES;
    if (time_0 < INPUT_LENGTH) {
        let output_index = output_channel * OUTPUT_LENGTH + time_0 * STRIDE + output_phase;
        output_buf[output_index] = accumulator_main_0 + accumulator_previous_0 + bias;
    }
    if (time_1 < INPUT_LENGTH) {
        let output_index = output_channel * OUTPUT_LENGTH + time_1 * STRIDE + output_phase;
        output_buf[output_index] = accumulator_main_1 + accumulator_previous_1 + bias;
    }
    if (time_2 < INPUT_LENGTH) {
        let output_index = output_channel * OUTPUT_LENGTH + time_2 * STRIDE + output_phase;
        output_buf[output_index] = accumulator_main_2 + accumulator_previous_2 + bias;
    }
    if (time_3 < INPUT_LENGTH) {
        let output_index = output_channel * OUTPUT_LENGTH + time_3 * STRIDE + output_phase;
        output_buf[output_index] = accumulator_main_3 + accumulator_previous_3 + bias;
    }
}
