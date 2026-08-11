// Isolated T=32 tile candidates for the DACVAE residual k=7 Conv1d.
//
// Physical layouts:
//   input:  contiguous NCHW [1, C, L]
//   weight: contiguous OIK  [C, C, 7]
//   bias:                    [C]
//   output: contiguous NCHW [1, C, L]
//
// A 16x{8,16} workgroup produces 32 time positions by {16,32} output
// channels. Each invocation owns two time positions and two output channels
// (four f32 accumulators). Input and weight reduction tiles cover 16 input
// channels. The fma order is input channel first, kernel tap second, exactly
// matching the accepted T64/O16 shader and scalar benchmark reference.
//
// SourceKernel buffers must all be read_write because CubeCL's sliced
// allocator can bind otherwise disjoint tensors to one physical buffer.

@group(0) @binding(0) var<storage, read_write> input_buf:  array<f32>;
@group(0) @binding(1) var<storage, read_write> weight_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias_buf:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output_buf: array<f32>;

const CHANNELS: u32 = {{ channels }}u;
const LENGTH: u32 = {{ length }}u;
const DILATION: u32 = {{ dilation }}u;
const PADDING: i32 = {{ padding }};

const KERNEL_SIZE: u32 = 7u;
const WORKGROUP_SIZE: u32 = {{ workgroup_size }}u;
const LOCAL_TIME_LANES: u32 = 16u;
const LOCAL_CHANNEL_LANES: u32 = {{ local_channel_lanes }}u;
const TIME_TILE: u32 = 32u;
const OUTPUT_CHANNEL_TILE: u32 = {{ output_channel_tile }}u;
const INPUT_CHANNEL_TILE: u32 = 16u;
const INPUT_SPAN: u32 = {{ input_span }}u;
const INPUT_TILE_SIZE: u32 = {{ input_tile_size }}u;
const WEIGHT_TILE_SIZE: u32 = {{ weight_tile_size }}u;

var<workgroup> input_tile: array<f32, {{ input_tile_size }}>;
var<workgroup> weight_tile: array<f32, {{ weight_tile_size }}>;

@compute @workgroup_size(16, {{ local_channel_lanes }}, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TIME_TILE;
    let output_channel_base = group_id.y * OUTPUT_CHANNEL_TILE;
    let batch_index = group_id.z;

    let time_0 = time_base + local_id.x;
    let time_1 = time_0 + LOCAL_TIME_LANES;
    let output_channel_0 = output_channel_base + local_id.y;
    let output_channel_1 = output_channel_0 + LOCAL_CHANNEL_LANES;

    var accumulator_00 = bias_buf[output_channel_0];
    var accumulator_01 = bias_buf[output_channel_1];
    var accumulator_10 = accumulator_00;
    var accumulator_11 = accumulator_01;

    var input_channel_base = 0u;
    loop {
        if input_channel_base >= CHANNELS {
            break;
        }

        var tile_index = local_index;
        loop {
            if tile_index >= INPUT_TILE_SIZE {
                break;
            }
            let tile_channel = tile_index / INPUT_SPAN;
            let tile_time = tile_index - tile_channel * INPUT_SPAN;
            let source_time = i32(time_base + tile_time) - PADDING;
            var value = 0.0;
            if source_time >= 0 && source_time < i32(LENGTH) {
                let input_channel = input_channel_base + tile_channel;
                let input_index =
                    (batch_index * CHANNELS + input_channel) * LENGTH + u32(source_time);
                value = input_buf[input_index];
            }
            input_tile[tile_index] = value;
            tile_index += WORKGROUP_SIZE;
        }

        tile_index = local_index;
        loop {
            if tile_index >= WEIGHT_TILE_SIZE {
                break;
            }
            let output_tile_stride = INPUT_CHANNEL_TILE * KERNEL_SIZE;
            let tile_output_channel = tile_index / output_tile_stride;
            let output_remainder = tile_index - tile_output_channel * output_tile_stride;
            let tile_input_channel = output_remainder / KERNEL_SIZE;
            let kernel_index = output_remainder - tile_input_channel * KERNEL_SIZE;
            let output_channel = output_channel_base + tile_output_channel;
            let input_channel = input_channel_base + tile_input_channel;
            let weight_index =
                (output_channel * CHANNELS + input_channel) * KERNEL_SIZE + kernel_index;
            weight_tile[tile_index] = weight_buf[weight_index];
            tile_index += WORKGROUP_SIZE;
        }

        workgroupBarrier();

        var tile_input_channel = 0u;
        loop {
            if tile_input_channel >= INPUT_CHANNEL_TILE {
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
                if kernel_index >= KERNEL_SIZE {
                    break;
                }
                let input_offset = kernel_index * DILATION;
                let input_0 = input_tile[input_base + input_offset];
                let input_1 = input_tile[input_base + LOCAL_TIME_LANES + input_offset];
                let weight_0 = weight_tile[weight_base_0 + kernel_index];
                let weight_1 = weight_tile[weight_base_1 + kernel_index];

                accumulator_00 = fma(input_0, weight_0, accumulator_00);
                accumulator_01 = fma(input_0, weight_1, accumulator_01);
                accumulator_10 = fma(input_1, weight_0, accumulator_10);
                accumulator_11 = fma(input_1, weight_1, accumulator_11);
                kernel_index += 1u;
            }
            tile_input_channel += 1u;
        }

        workgroupBarrier();
        input_channel_base += INPUT_CHANNEL_TILE;
    }

    let output_base_0 = (batch_index * CHANNELS + output_channel_0) * LENGTH;
    let output_base_1 = (batch_index * CHANNELS + output_channel_1) * LENGTH;
    if time_0 < LENGTH {
        output_buf[output_base_0 + time_0] = accumulator_00;
        output_buf[output_base_1 + time_0] = accumulator_01;
    }
    if time_1 < LENGTH {
        output_buf[output_base_0 + time_1] = accumulator_10;
        output_buf[output_base_1 + time_1] = accumulator_11;
    }
}
