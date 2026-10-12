enable f16;

// Dynamic released-decoder stem direct T64/O32/Cin16 f32 convolution.

@group(0) @binding(0) var<storage, read_write> input_buf: array<f16>;
@group(0) @binding(1) var<storage, read_write> weight_buf: array<f16>;
@group(0) @binding(2) var<storage, read_write> bias_buf: array<f16>;
@group(0) @binding(3) var<storage, read_write> output_buf: array<f16>;

const INPUT_CHANNELS: u32 = 1024u;
const OUTPUT_CHANNELS: u32 = 1536u;
const LENGTH: u32 = {{ length }}u;
const KERNEL_SIZE: u32 = 7u;
const PADDING: i32 = 3;
const WORKGROUP_SIZE: u32 = 256u;
const LOCAL_TIME_LANES: u32 = 16u;
const LOCAL_CHANNEL_LANES: u32 = 16u;
const TIME_TILE: u32 = 64u;
const OUTPUT_CHANNEL_TILE: u32 = 32u;
const INPUT_CHANNEL_TILE: u32 = 16u;
const INPUT_SPAN: u32 = 70u;
const INPUT_TILE_SIZE: u32 = 1120u;
const WEIGHT_TILE_SIZE: u32 = 3584u;

var<workgroup> input_tile: array<f32, 1120>;
var<workgroup> weight_tile: array<f32, 3584>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TIME_TILE;
    let output_channel_base = group_id.y * OUTPUT_CHANNEL_TILE;
    let time_0 = time_base + local_id.x;
    let time_1 = time_0 + LOCAL_TIME_LANES;
    let time_2 = time_1 + LOCAL_TIME_LANES;
    let time_3 = time_2 + LOCAL_TIME_LANES;
    let output_channel_0 = output_channel_base + local_id.y;
    let output_channel_1 = output_channel_0 + LOCAL_CHANNEL_LANES;

    var accumulator_00 = 0.0;
    var accumulator_01 = 0.0;
    var accumulator_10 = 0.0;
    var accumulator_11 = 0.0;
    var accumulator_20 = 0.0;
    var accumulator_21 = 0.0;
    var accumulator_30 = 0.0;
    var accumulator_31 = 0.0;

    var input_channel_base = 0u;
    loop {
        if (input_channel_base >= INPUT_CHANNELS) {
            break;
        }
        var tile_index = local_index;
        loop {
            if (tile_index >= INPUT_TILE_SIZE) {
                break;
            }
            let tile_channel = tile_index / INPUT_SPAN;
            let tile_time = tile_index - tile_channel * INPUT_SPAN;
            let source_time = i32(time_base + tile_time) - PADDING;
            var value = 0.0;
            if (source_time >= 0 && source_time < i32(LENGTH)) {
                let input_channel = input_channel_base + tile_channel;
                value = f32(input_buf[input_channel * LENGTH + u32(source_time)]);
            }
            input_tile[tile_index] = value;
            tile_index += WORKGROUP_SIZE;
        }

        tile_index = local_index;
        loop {
            if (tile_index >= WEIGHT_TILE_SIZE) {
                break;
            }
            let output_stride = INPUT_CHANNEL_TILE * KERNEL_SIZE;
            let tile_output_channel = tile_index / output_stride;
            let remainder = tile_index - tile_output_channel * output_stride;
            let tile_input_channel = remainder / KERNEL_SIZE;
            let kernel_index = remainder - tile_input_channel * KERNEL_SIZE;
            let output_channel = output_channel_base + tile_output_channel;
            let input_channel = input_channel_base + tile_input_channel;
            let weight_index =
                (output_channel * INPUT_CHANNELS + input_channel) * KERNEL_SIZE + kernel_index;
            weight_tile[tile_index] = f32(weight_buf[weight_index]);
            tile_index += WORKGROUP_SIZE;
        }

        workgroupBarrier();
        var tile_input_channel = 0u;
        loop {
            if (tile_input_channel >= INPUT_CHANNEL_TILE) {
                break;
            }
            let input_base = tile_input_channel * INPUT_SPAN + local_id.x;
            let weight_base_0 =
                (local_id.y * INPUT_CHANNEL_TILE + tile_input_channel) * KERNEL_SIZE;
            let weight_base_1 =
                ((local_id.y + LOCAL_CHANNEL_LANES) * INPUT_CHANNEL_TILE
                    + tile_input_channel) * KERNEL_SIZE;
            var kernel_index = 0u;
            loop {
                if (kernel_index >= KERNEL_SIZE) {
                    break;
                }
                let input_0 = input_tile[input_base + kernel_index];
                let input_1 = input_tile[input_base + LOCAL_TIME_LANES + kernel_index];
                let input_2 = input_tile[input_base + 2u * LOCAL_TIME_LANES + kernel_index];
                let input_3 = input_tile[input_base + 3u * LOCAL_TIME_LANES + kernel_index];
                let weight_0 = weight_tile[weight_base_0 + kernel_index];
                let weight_1 = weight_tile[weight_base_1 + kernel_index];
                accumulator_00 = fma(input_0, weight_0, accumulator_00);
                accumulator_01 = fma(input_0, weight_1, accumulator_01);
                accumulator_10 = fma(input_1, weight_0, accumulator_10);
                accumulator_11 = fma(input_1, weight_1, accumulator_11);
                accumulator_20 = fma(input_2, weight_0, accumulator_20);
                accumulator_21 = fma(input_2, weight_1, accumulator_21);
                accumulator_30 = fma(input_3, weight_0, accumulator_30);
                accumulator_31 = fma(input_3, weight_1, accumulator_31);
                kernel_index += 1u;
            }
            tile_input_channel += 1u;
        }
        workgroupBarrier();
        input_channel_base += INPUT_CHANNEL_TILE;
    }

    let output_base_0 = output_channel_0 * LENGTH;
    let output_base_1 = output_channel_1 * LENGTH;
    if (time_0 < LENGTH) {
        output_buf[output_base_0 + time_0] = f16(accumulator_00 + f32(bias_buf[output_channel_0]));
        output_buf[output_base_1 + time_0] = f16(accumulator_01 + f32(bias_buf[output_channel_1]));
    }
    if (time_1 < LENGTH) {
        output_buf[output_base_0 + time_1] = f16(accumulator_10 + f32(bias_buf[output_channel_0]));
        output_buf[output_base_1 + time_1] = f16(accumulator_11 + f32(bias_buf[output_channel_1]));
    }
    if (time_2 < LENGTH) {
        output_buf[output_base_0 + time_2] = f16(accumulator_20 + f32(bias_buf[output_channel_0]));
        output_buf[output_base_1 + time_2] = f16(accumulator_21 + f32(bias_buf[output_channel_1]));
    }
    if (time_3 < LENGTH) {
        output_buf[output_base_0 + time_3] = f16(accumulator_30 + f32(bias_buf[output_channel_0]));
        output_buf[output_base_1 + time_3] = f16(accumulator_31 + f32(bias_buf[output_channel_1]));
    }
}
