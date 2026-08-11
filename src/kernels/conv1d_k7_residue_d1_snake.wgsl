// Production residue-class dilation-one k=7 Conv1d + scalar Snake.
//
// packed_input is compact [residue][channel][q]. Each workgroup evaluates one
// residue with ordinary d=1/pad=3 shared-memory addressing. The convolution
// reduction remains input-channel outer, tap 0..6 inner, exactly matching the
// production scalar FMA order. Results scatter directly to contiguous logical
// NCL at t = residue + q * original_dilation; no output unpack is materialized.

// SourceKernel buffers are read_write because CubeCL sliced allocations can
// place otherwise disjoint logical tensors in one physical buffer.
@group(0) @binding(0) var<storage, read_write> packed_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight_buf:   array<f32>;
@group(0) @binding(2) var<storage, read_write> bias_buf:     array<f32>;
@group(0) @binding(3) var<storage, read_write> output_buf:   array<f32>;
@group(0) @binding(4) var<storage, read_write> alpha_buf:    array<f32>;

const CHANNELS: u32 = {{ channels }}u;
const LENGTH: u32 = {{ length }}u;
const DILATION: u32 = {{ dilation }}u;
const BASE_LENGTH: u32 = {{ base_length }}u;
const REMAINDER: u32 = {{ remainder }}u;

const KERNEL_SIZE: u32 = 7u;
const PADDING: i32 = 3;
const WORKGROUP_SIZE: u32 = 256u;
const LOCAL_TIME_LANES: u32 = 16u;
const LOCAL_CHANNEL_LANES: u32 = 16u;
const TIME_TILE: u32 = 256u;
const OUTPUT_CHANNEL_TILE: u32 = 32u;
const INPUT_CHANNEL_TILE: u32 = 16u;
const INPUT_SPAN: u32 = {{ input_span }}u;
const INPUT_TILE_SIZE: u32 = {{ input_tile_size }}u;
const WEIGHT_TILE_SIZE: u32 = {{ weight_tile_size }}u;

var<workgroup> input_tile: array<f32, {{ input_tile_size }}>;
var<workgroup> weight_tile: array<f32, {{ weight_tile_size }}>;

fn residue_length(residue: u32) -> u32 {
    return BASE_LENGTH + select(0u, 1u, residue < REMAINDER);
}

fn residue_element_base(residue: u32) -> u32 {
    let residue_prefix_q = residue * BASE_LENGTH + min(residue, REMAINDER);
    return residue_prefix_q * CHANNELS;
}

fn load_input_vec4(index: u32) -> vec4<f32> {
    return vec4<f32>(
        input_tile[index],
        input_tile[index + 1u],
        input_tile[index + 2u],
        input_tile[index + 3u],
    );
}

// Keep this scalar sequence identical to the production Snake implementation.
fn snake_epilogue(x: f32, output_channel: u32) -> f32 {
    let a = alpha_buf[output_channel];
    let sine = sin(a * x);
    return x + (sine * sine) / (a + 1e-9);
}

fn store_output_vec4(
    output_base: u32,
    q: u32,
    output_channel: u32,
    residue: u32,
    q_length: u32,
    value: vec4<f32>,
) {
    if q < q_length {
        let output_time = residue + q * DILATION;
        output_buf[output_base + output_time] = snake_epilogue(value.x, output_channel);
    }
    if q + 1u < q_length {
        let output_time = residue + (q + 1u) * DILATION;
        output_buf[output_base + output_time] = snake_epilogue(value.y, output_channel);
    }
    if q + 2u < q_length {
        let output_time = residue + (q + 2u) * DILATION;
        output_buf[output_base + output_time] = snake_epilogue(value.z, output_channel);
    }
    if q + 3u < q_length {
        let output_time = residue + (q + 3u) * DILATION;
        output_buf[output_base + output_time] = snake_epilogue(value.w, output_channel);
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let q_base = group_id.x * TIME_TILE;
    let output_channel_base = group_id.y * OUTPUT_CHANNEL_TILE;
    let residue = group_id.z;
    let q_length = residue_length(residue);
    let packed_residue_base = residue_element_base(residue);

    let local_time = local_id.x * 4u;
    let q_0 = q_base + local_time;
    let q_1 = q_0 + 64u;
    let q_2 = q_0 + 128u;
    let q_3 = q_0 + 192u;
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

        // Stage one compact residue as a conventional d=1/pad=3 input tile.
        var tile_index = local_index;
        loop {
            if tile_index >= INPUT_TILE_SIZE {
                break;
            }
            let tile_channel = tile_index / INPUT_SPAN;
            let tile_q = tile_index - tile_channel * INPUT_SPAN;
            let source_q = i32(q_base + tile_q) - PADDING;
            var value = 0.0;
            if source_q >= 0 && source_q < i32(q_length) {
                let input_channel = input_channel_base + tile_channel;
                let packed_index = packed_residue_base
                    + input_channel * q_length
                    + u32(source_q);
                value = packed_input[packed_index];
            }
            input_tile[tile_index] = value;
            tile_index += WORKGROUP_SIZE;
        }

        // Retain contiguous checkpoint OIK staging from the production core.
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
            let input_base_0 = tile_input_channel * INPUT_SPAN + local_time;
            let input_base_1 = input_base_0 + 64u;
            let input_base_2 = input_base_0 + 128u;
            let input_base_3 = input_base_0 + 192u;
            let weight_base_0 =
                (local_id.y * INPUT_CHANNEL_TILE + tile_input_channel) * KERNEL_SIZE;
            let weight_base_1 =
                ((local_id.y + LOCAL_CHANNEL_LANES) * INPUT_CHANNEL_TILE
                    + tile_input_channel) * KERNEL_SIZE;

            // tap 0
            var weight_0 = weight_tile[weight_base_0];
            var weight_1 = weight_tile[weight_base_1];
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
            weight_0 = weight_tile[weight_base_0 + 1u];
            weight_1 = weight_tile[weight_base_1 + 1u];
            input_vector = load_input_vec4(input_base_0 + 1u);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 1u);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 1u);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 1u);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 2
            weight_0 = weight_tile[weight_base_0 + 2u];
            weight_1 = weight_tile[weight_base_1 + 2u];
            input_vector = load_input_vec4(input_base_0 + 2u);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 2u);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 2u);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 2u);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 3
            weight_0 = weight_tile[weight_base_0 + 3u];
            weight_1 = weight_tile[weight_base_1 + 3u];
            input_vector = load_input_vec4(input_base_0 + 3u);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 3u);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 3u);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 3u);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 4
            weight_0 = weight_tile[weight_base_0 + 4u];
            weight_1 = weight_tile[weight_base_1 + 4u];
            input_vector = load_input_vec4(input_base_0 + 4u);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 4u);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 4u);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 4u);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 5
            weight_0 = weight_tile[weight_base_0 + 5u];
            weight_1 = weight_tile[weight_base_1 + 5u];
            input_vector = load_input_vec4(input_base_0 + 5u);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 5u);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 5u);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 5u);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            // tap 6
            weight_0 = weight_tile[weight_base_0 + 6u];
            weight_1 = weight_tile[weight_base_1 + 6u];
            input_vector = load_input_vec4(input_base_0 + 6u);
            accumulator_00 = fma(input_vector, vec4<f32>(weight_0), accumulator_00);
            accumulator_01 = fma(input_vector, vec4<f32>(weight_1), accumulator_01);
            input_vector = load_input_vec4(input_base_1 + 6u);
            accumulator_10 = fma(input_vector, vec4<f32>(weight_0), accumulator_10);
            accumulator_11 = fma(input_vector, vec4<f32>(weight_1), accumulator_11);
            input_vector = load_input_vec4(input_base_2 + 6u);
            accumulator_20 = fma(input_vector, vec4<f32>(weight_0), accumulator_20);
            accumulator_21 = fma(input_vector, vec4<f32>(weight_1), accumulator_21);
            input_vector = load_input_vec4(input_base_3 + 6u);
            accumulator_30 = fma(input_vector, vec4<f32>(weight_0), accumulator_30);
            accumulator_31 = fma(input_vector, vec4<f32>(weight_1), accumulator_31);

            tile_input_channel += 1u;
        }

        workgroupBarrier();
        input_channel_base += INPUT_CHANNEL_TILE;
    }

    let output_base_0 = output_channel_0 * LENGTH;
    let output_base_1 = output_channel_1 * LENGTH;
    store_output_vec4(output_base_0, q_0, output_channel_0, residue, q_length, accumulator_00);
    store_output_vec4(output_base_1, q_0, output_channel_1, residue, q_length, accumulator_01);
    store_output_vec4(output_base_0, q_1, output_channel_0, residue, q_length, accumulator_10);
    store_output_vec4(output_base_1, q_1, output_channel_1, residue, q_length, accumulator_11);
    store_output_vec4(output_base_0, q_2, output_channel_0, residue, q_length, accumulator_20);
    store_output_vec4(output_base_1, q_2, output_channel_1, residue, q_length, accumulator_21);
    store_output_vec4(output_base_0, q_3, output_channel_0, residue, q_length, accumulator_30);
    store_output_vec4(output_base_1, q_3, output_channel_1, residue, q_length, accumulator_31);
}
