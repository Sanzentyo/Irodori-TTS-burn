// Production T256/O32 DACVAE residual k=7 Conv1d + act1 Snake vec4 store.
//
// Everything through the accepted T256 convolution reduction is source-identical
// to production. Each output component retains the production scalar Snake order;
// four adjacent NCL time values are then committed by one vec4 store.
//
// Physical layouts:
//   input:  contiguous NCL [1, C, L]
//   weight: logical [Cin, 7, Cout], physical invocation-owned vec4 groups
//   bias:                   [C]
//   output: contiguous NCL [1, C, L], viewed as array<vec4<f32>>
//   alpha:  contiguous      [1, C, 1]
//
// The host contract requires L % 4 == 0 and a 16-byte aligned output binding.
// SourceKernel buffers must all be read_write because CubeCL's sliced allocator
// can bind otherwise disjoint tensors to one physical buffer.

@group(0) @binding(0) var<storage, read_write> input_buf:  array<f32>;
@group(0) @binding(1) var<storage, read_write> weight_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bias_buf:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output_buf: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> alpha_buf:  array<f32>;

const CHANNELS: u32 = {{ channels }}u;
const LENGTH: u32 = {{ length }}u;
const DILATION: u32 = {{ dilation }}u;
const PADDING: i32 = {{ padding }};

const KERNEL_SIZE: u32 = 7u;
const WORKGROUP_SIZE: u32 = 256u;
const LOCAL_TIME_LANES: u32 = 16u;
const LOCAL_CHANNEL_LANES: u32 = 16u;
const TIME_TILE: u32 = 256u;
const OUTPUT_CHANNEL_TILE: u32 = 32u;
const OUTPUT_VECTORS: u32 = CHANNELS / 4u;
const INPUT_CHANNEL_TILE: u32 = {{ input_channel_tile }}u;
const INPUT_SPAN: u32 = {{ input_span }}u;
const INPUT_TILE_SIZE: u32 = {{ input_tile_size }}u;
const WEIGHT_VECTOR_TILE_SIZE: u32 = {{ weight_vector_tile_size }}u;

var<workgroup> input_tile: array<f32, {{ input_tile_size }}>;
var<workgroup> weight_tile: array<vec4<f32>, {{ weight_vector_tile_size }}>;

fn load_input_vec4(index: u32) -> vec4<f32> {
    return vec4<f32>(
        input_tile[index],
        input_tile[index + 1u],
        input_tile[index + 2u],
        input_tile[index + 3u],
    );
}

fn select_weight_pair(weight_vector: vec4<f32>, lane: u32) -> vec2<f32> {
    if lane < 8u {
        return vec2<f32>(weight_vector.x, weight_vector.z);
    }
    return vec2<f32>(weight_vector.y, weight_vector.w);
}

// Keep this scalar operation sequence identical to the production snake.wgsl.
// Each component reaches it only after the unchanged T256 convolution reduction.
fn snake_epilogue(x: f32, output_channel: u32) -> f32 {
    let a = alpha_buf[output_channel];
    let sine = sin(a * x);
    return x + (sine * sine) / (a + 1e-9);
}

// Host invariants make output_base, time, and LENGTH multiples of four. Thus a
// single tail guard is sufficient and the division maps the scalar NCL index to
// its exact array<vec4<f32>> element.
fn store_output_vec4(output_base: u32, time: u32, output_channel: u32, value: vec4<f32>) {
    if time < LENGTH {
        let output_x = snake_epilogue(value.x, output_channel);
        let output_y = snake_epilogue(value.y, output_channel);
        let output_z = snake_epilogue(value.z, output_channel);
        let output_w = snake_epilogue(value.w, output_channel);
        let output_vec_index = (output_base + time) / 4u;
        output_buf[output_vec_index] = vec4<f32>(output_x, output_y, output_z, output_w);
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TIME_TILE;
    let output_channel_base = group_id.y * OUTPUT_CHANNEL_TILE;
    let batch_index = group_id.z;

    let local_time = local_id.x * 4u;
    let time_0 = time_base + local_time;
    let time_1 = time_0 + 64u;
    let time_2 = time_0 + 128u;
    let time_3 = time_0 + 192u;
    let output_channel_0 = output_channel_base + local_id.y;
    let output_channel_1 = output_channel_0 + LOCAL_CHANNEL_LANES;

    let bias_0 = bias_buf[output_channel_0];
    let bias_1 = bias_buf[output_channel_1];
    var accumulator_00 = vec4<f32>(bias_0);
    var accumulator_01 = vec4<f32>(bias_1);
    var accumulator_10 = vec4<f32>(bias_0);
    var accumulator_11 = vec4<f32>(bias_1);
    var accumulator_20 = vec4<f32>(bias_0);
    var accumulator_21 = vec4<f32>(bias_1);
    var accumulator_30 = vec4<f32>(bias_0);
    var accumulator_31 = vec4<f32>(bias_1);

    var input_channel_base = 0u;
    loop {
        if input_channel_base >= CHANNELS {
            break;
        }

        // Consecutive invocations cooperatively stage consecutive scalar NCL
        // elements. Out-of-range halo and final-tail positions become zero.
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

        // Stage the invocation-owned vec4 output groups prepared once at load.
        tile_index = local_index;
        loop {
            if tile_index >= WEIGHT_VECTOR_TILE_SIZE {
                break;
            }
            let output_vector_stride = INPUT_CHANNEL_TILE * KERNEL_SIZE;
            let tile_output_vector = tile_index / output_vector_stride;
            let output_remainder = tile_index - tile_output_vector * output_vector_stride;
            let tile_input_channel = output_remainder / KERNEL_SIZE;
            let kernel_index = output_remainder - tile_input_channel * KERNEL_SIZE;
            let input_channel = input_channel_base + tile_input_channel;
            let output_vector = group_id.y * (OUTPUT_CHANNEL_TILE / 4u) + tile_output_vector;
            let weight_index =
                (input_channel * KERNEL_SIZE + kernel_index) * OUTPUT_VECTORS + output_vector;
            weight_tile[tile_index] = weight_buf[weight_index];
            tile_index += WORKGROUP_SIZE;
        }

        workgroupBarrier();

        var tile_input_channel = 0u;
        loop {
            if tile_input_channel >= INPUT_CHANNEL_TILE {
                break;
            }
            let input_base_0 = tile_input_channel * INPUT_SPAN + local_time;
            let input_base_1 = input_base_0 + 64u;
            let input_base_2 = input_base_0 + 128u;
            let input_base_3 = input_base_0 + 192u;
            let weight_base =
                ((local_id.y % 8u) * INPUT_CHANNEL_TILE + tile_input_channel) * KERNEL_SIZE;

            // tap 0
            var weight_pair = select_weight_pair(weight_tile[weight_base], local_id.y);
            var weight_0 = weight_pair.x;
            var weight_1 = weight_pair.y;
            var input_vector = load_input_vec4(input_base_0);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 1
            weight_pair = select_weight_pair(weight_tile[weight_base + 1u], local_id.y);
            weight_0 = weight_pair.x;
            weight_1 = weight_pair.y;
            input_vector = load_input_vec4(input_base_0 + DILATION);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + DILATION);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + DILATION);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + DILATION);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 2
            weight_pair = select_weight_pair(weight_tile[weight_base + 2u], local_id.y);
            weight_0 = weight_pair.x;
            weight_1 = weight_pair.y;
            input_vector = load_input_vec4(input_base_0 + 2u * DILATION);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 2u * DILATION);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 2u * DILATION);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 2u * DILATION);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 3
            weight_pair = select_weight_pair(weight_tile[weight_base + 3u], local_id.y);
            weight_0 = weight_pair.x;
            weight_1 = weight_pair.y;
            input_vector = load_input_vec4(input_base_0 + 3u * DILATION);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 3u * DILATION);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 3u * DILATION);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 3u * DILATION);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 4
            weight_pair = select_weight_pair(weight_tile[weight_base + 4u], local_id.y);
            weight_0 = weight_pair.x;
            weight_1 = weight_pair.y;
            input_vector = load_input_vec4(input_base_0 + 4u * DILATION);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 4u * DILATION);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 4u * DILATION);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 4u * DILATION);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 5
            weight_pair = select_weight_pair(weight_tile[weight_base + 5u], local_id.y);
            weight_0 = weight_pair.x;
            weight_1 = weight_pair.y;
            input_vector = load_input_vec4(input_base_0 + 5u * DILATION);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 5u * DILATION);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 5u * DILATION);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 5u * DILATION);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 6
            weight_pair = select_weight_pair(weight_tile[weight_base + 6u], local_id.y);
            weight_0 = weight_pair.x;
            weight_1 = weight_pair.y;
            input_vector = load_input_vec4(input_base_0 + 6u * DILATION);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 6u * DILATION);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 6u * DILATION);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 6u * DILATION);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            tile_input_channel += 1u;
        }

        workgroupBarrier();
        input_channel_base += INPUT_CHANNEL_TILE;
    }

    let output_base_0 = (batch_index * CHANNELS + output_channel_0) * LENGTH;
    let output_base_1 = (batch_index * CHANNELS + output_channel_1) * LENGTH;
    store_output_vec4(output_base_0, time_0, output_channel_0, accumulator_00);
    store_output_vec4(output_base_1, time_0, output_channel_1, accumulator_01);
    store_output_vec4(output_base_0, time_1, output_channel_0, accumulator_10);
    store_output_vec4(output_base_1, time_1, output_channel_1, accumulator_11);
    store_output_vec4(output_base_0, time_2, output_channel_0, accumulator_20);
    store_output_vec4(output_base_1, time_2, output_channel_1, accumulator_21);
    store_output_vec4(output_base_0, time_3, output_channel_0, accumulator_30);
    store_output_vec4(output_base_1, time_3, output_channel_1, accumulator_31);
}
