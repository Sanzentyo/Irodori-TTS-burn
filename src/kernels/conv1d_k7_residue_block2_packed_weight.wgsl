// Isolated C192/L48000 compact-residue k7 Conv1d + Snake candidate.
//
// Physical layouts:
//   packed_input: [residue][channel][q], scalar f32
//   packed_weight: [input_channel][tap][output_channel / 4], vec4<f32>
//   output: contiguous NCL [1, 192, 48000], scalar f32
//
// A 32x8 workgroup produces Q256 x O32.  One subgroup owns four adjacent
// output channels and 32 adjacent q positions.  Its shared-input accesses are
// contiguous, while each packed weight read is subgroup-uniform.  Reduction is
// still input-channel outer and tap 0..6 inner.

@group(0) @binding(0) var<storage, read_write> packed_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bias_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> alpha_buf: array<f32>;

const CHANNELS: u32 = 192u;
const LENGTH: u32 = 48000u;
const DILATION: u32 = {{ dilation }}u;
const BASE_LENGTH: u32 = {{ base_length }}u;
const REMAINDER: u32 = {{ remainder }}u;
const KERNEL_SIZE: u32 = 7u;
const PADDING: i32 = 3;
const TIME_TILE: u32 = 256u;
const OUTPUT_CHANNEL_TILE: u32 = 32u;
const OUTPUT_VECTOR_TILE: u32 = 8u;
const INPUT_CHANNEL_TILE: u32 = 16u;
const INPUT_SPAN: u32 = 262u;
const INPUT_TILE_ELEMENTS: u32 = 4192u;
const OUTPUT_VECTORS: u32 = 48u;
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> input_tile: array<f32, 4192>;

fn residue_length(residue: u32) -> u32 {
    return BASE_LENGTH + select(0u, 1u, residue < REMAINDER);
}

fn residue_element_base(residue: u32) -> u32 {
    let residue_prefix_q = residue * BASE_LENGTH + min(residue, REMAINDER);
    return residue_prefix_q * CHANNELS;
}

fn snake_epilogue(x: f32, output_channel: u32) -> f32 {
    let a = alpha_buf[output_channel];
    let sine = sin(a * x);
    return x + (sine * sine) / (a + 1e-9);
}

fn store_output4(value: vec4<f32>, q: u32, output_channel: u32, residue: u32, q_length: u32) {
    if q >= q_length {
        return;
    }
    let output_time = residue + q * DILATION;
    output_buf[(output_channel + 0u) * LENGTH + output_time] =
        snake_epilogue(value.x, output_channel + 0u);
    output_buf[(output_channel + 1u) * LENGTH + output_time] =
        snake_epilogue(value.y, output_channel + 1u);
    output_buf[(output_channel + 2u) * LENGTH + output_time] =
        snake_epilogue(value.z, output_channel + 2u);
    output_buf[(output_channel + 3u) * LENGTH + output_time] =
        snake_epilogue(value.w, output_channel + 3u);
}

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let q_base = group_id.x * TIME_TILE;
    let output_channel_base =
        group_id.y * OUTPUT_CHANNEL_TILE + local_id.y * 4u;
    let output_vector = group_id.y * OUTPUT_VECTOR_TILE + local_id.y;
    let residue = group_id.z;
    let q_length = residue_length(residue);
    let packed_residue_base = residue_element_base(residue);

    let q_0 = q_base + local_id.x;
    let q_1 = q_0 + 32u;
    let q_2 = q_0 + 64u;
    let q_3 = q_0 + 96u;
    let q_4 = q_0 + 128u;
    let q_5 = q_0 + 160u;
    let q_6 = q_0 + 192u;
    let q_7 = q_0 + 224u;

    let initial = vec4<f32>(
        bias_buf[output_channel_base + 0u],
        bias_buf[output_channel_base + 1u],
        bias_buf[output_channel_base + 2u],
        bias_buf[output_channel_base + 3u],
    );
    var accumulator_0 = initial;
    var accumulator_1 = initial;
    var accumulator_2 = initial;
    var accumulator_3 = initial;
    var accumulator_4 = initial;
    var accumulator_5 = initial;
    var accumulator_6 = initial;
    var accumulator_7 = initial;

    var input_channel_base = 0u;
    loop {
        if input_channel_base >= CHANNELS {
            break;
        }

        var tile_index = local_index;
        loop {
            if tile_index >= INPUT_TILE_ELEMENTS {
                break;
            }
            let tile_input_channel = tile_index / INPUT_SPAN;
            let tile_q = tile_index - tile_input_channel * INPUT_SPAN;
            let source_q = i32(q_base + tile_q) - PADDING;
            var value = 0.0;
            if source_q >= 0 && source_q < i32(q_length) {
                let input_channel = input_channel_base + tile_input_channel;
                let packed_index = packed_residue_base
                    + input_channel * q_length
                    + u32(source_q);
                value = packed_input[packed_index];
            }
            input_tile[tile_index] = value;
            tile_index += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        var tile_input_channel = 0u;
        loop {
            if tile_input_channel >= INPUT_CHANNEL_TILE {
                break;
            }
            let input_channel = input_channel_base + tile_input_channel;
            let input_base = tile_input_channel * INPUT_SPAN + local_id.x;
            let weight_base =
                (input_channel * KERNEL_SIZE) * OUTPUT_VECTORS + output_vector;

            // tap 0
            var weight = packed_weight[weight_base];
            accumulator_0 = fma(vec4<f32>(input_tile[input_base + 0u]), weight, accumulator_0);
            accumulator_1 = fma(vec4<f32>(input_tile[input_base + 32u]), weight, accumulator_1);
            accumulator_2 = fma(vec4<f32>(input_tile[input_base + 64u]), weight, accumulator_2);
            accumulator_3 = fma(vec4<f32>(input_tile[input_base + 96u]), weight, accumulator_3);
            accumulator_4 = fma(vec4<f32>(input_tile[input_base + 128u]), weight, accumulator_4);
            accumulator_5 = fma(vec4<f32>(input_tile[input_base + 160u]), weight, accumulator_5);
            accumulator_6 = fma(vec4<f32>(input_tile[input_base + 192u]), weight, accumulator_6);
            accumulator_7 = fma(vec4<f32>(input_tile[input_base + 224u]), weight, accumulator_7);

            // tap 1
            weight = packed_weight[weight_base + OUTPUT_VECTORS];
            accumulator_0 = fma(vec4<f32>(input_tile[input_base + 1u]), weight, accumulator_0);
            accumulator_1 = fma(vec4<f32>(input_tile[input_base + 33u]), weight, accumulator_1);
            accumulator_2 = fma(vec4<f32>(input_tile[input_base + 65u]), weight, accumulator_2);
            accumulator_3 = fma(vec4<f32>(input_tile[input_base + 97u]), weight, accumulator_3);
            accumulator_4 = fma(vec4<f32>(input_tile[input_base + 129u]), weight, accumulator_4);
            accumulator_5 = fma(vec4<f32>(input_tile[input_base + 161u]), weight, accumulator_5);
            accumulator_6 = fma(vec4<f32>(input_tile[input_base + 193u]), weight, accumulator_6);
            accumulator_7 = fma(vec4<f32>(input_tile[input_base + 225u]), weight, accumulator_7);

            // tap 2
            weight = packed_weight[weight_base + 2u * OUTPUT_VECTORS];
            accumulator_0 = fma(vec4<f32>(input_tile[input_base + 2u]), weight, accumulator_0);
            accumulator_1 = fma(vec4<f32>(input_tile[input_base + 34u]), weight, accumulator_1);
            accumulator_2 = fma(vec4<f32>(input_tile[input_base + 66u]), weight, accumulator_2);
            accumulator_3 = fma(vec4<f32>(input_tile[input_base + 98u]), weight, accumulator_3);
            accumulator_4 = fma(vec4<f32>(input_tile[input_base + 130u]), weight, accumulator_4);
            accumulator_5 = fma(vec4<f32>(input_tile[input_base + 162u]), weight, accumulator_5);
            accumulator_6 = fma(vec4<f32>(input_tile[input_base + 194u]), weight, accumulator_6);
            accumulator_7 = fma(vec4<f32>(input_tile[input_base + 226u]), weight, accumulator_7);

            // tap 3
            weight = packed_weight[weight_base + 3u * OUTPUT_VECTORS];
            accumulator_0 = fma(vec4<f32>(input_tile[input_base + 3u]), weight, accumulator_0);
            accumulator_1 = fma(vec4<f32>(input_tile[input_base + 35u]), weight, accumulator_1);
            accumulator_2 = fma(vec4<f32>(input_tile[input_base + 67u]), weight, accumulator_2);
            accumulator_3 = fma(vec4<f32>(input_tile[input_base + 99u]), weight, accumulator_3);
            accumulator_4 = fma(vec4<f32>(input_tile[input_base + 131u]), weight, accumulator_4);
            accumulator_5 = fma(vec4<f32>(input_tile[input_base + 163u]), weight, accumulator_5);
            accumulator_6 = fma(vec4<f32>(input_tile[input_base + 195u]), weight, accumulator_6);
            accumulator_7 = fma(vec4<f32>(input_tile[input_base + 227u]), weight, accumulator_7);

            // tap 4
            weight = packed_weight[weight_base + 4u * OUTPUT_VECTORS];
            accumulator_0 = fma(vec4<f32>(input_tile[input_base + 4u]), weight, accumulator_0);
            accumulator_1 = fma(vec4<f32>(input_tile[input_base + 36u]), weight, accumulator_1);
            accumulator_2 = fma(vec4<f32>(input_tile[input_base + 68u]), weight, accumulator_2);
            accumulator_3 = fma(vec4<f32>(input_tile[input_base + 100u]), weight, accumulator_3);
            accumulator_4 = fma(vec4<f32>(input_tile[input_base + 132u]), weight, accumulator_4);
            accumulator_5 = fma(vec4<f32>(input_tile[input_base + 164u]), weight, accumulator_5);
            accumulator_6 = fma(vec4<f32>(input_tile[input_base + 196u]), weight, accumulator_6);
            accumulator_7 = fma(vec4<f32>(input_tile[input_base + 228u]), weight, accumulator_7);

            // tap 5
            weight = packed_weight[weight_base + 5u * OUTPUT_VECTORS];
            accumulator_0 = fma(vec4<f32>(input_tile[input_base + 5u]), weight, accumulator_0);
            accumulator_1 = fma(vec4<f32>(input_tile[input_base + 37u]), weight, accumulator_1);
            accumulator_2 = fma(vec4<f32>(input_tile[input_base + 69u]), weight, accumulator_2);
            accumulator_3 = fma(vec4<f32>(input_tile[input_base + 101u]), weight, accumulator_3);
            accumulator_4 = fma(vec4<f32>(input_tile[input_base + 133u]), weight, accumulator_4);
            accumulator_5 = fma(vec4<f32>(input_tile[input_base + 165u]), weight, accumulator_5);
            accumulator_6 = fma(vec4<f32>(input_tile[input_base + 197u]), weight, accumulator_6);
            accumulator_7 = fma(vec4<f32>(input_tile[input_base + 229u]), weight, accumulator_7);

            // tap 6
            weight = packed_weight[weight_base + 6u * OUTPUT_VECTORS];
            accumulator_0 = fma(vec4<f32>(input_tile[input_base + 6u]), weight, accumulator_0);
            accumulator_1 = fma(vec4<f32>(input_tile[input_base + 38u]), weight, accumulator_1);
            accumulator_2 = fma(vec4<f32>(input_tile[input_base + 70u]), weight, accumulator_2);
            accumulator_3 = fma(vec4<f32>(input_tile[input_base + 102u]), weight, accumulator_3);
            accumulator_4 = fma(vec4<f32>(input_tile[input_base + 134u]), weight, accumulator_4);
            accumulator_5 = fma(vec4<f32>(input_tile[input_base + 166u]), weight, accumulator_5);
            accumulator_6 = fma(vec4<f32>(input_tile[input_base + 198u]), weight, accumulator_6);
            accumulator_7 = fma(vec4<f32>(input_tile[input_base + 230u]), weight, accumulator_7);

            tile_input_channel += 1u;
        }
        workgroupBarrier();
        input_channel_base += INPUT_CHANNEL_TILE;
    }

    store_output4(accumulator_0, q_0, output_channel_base, residue, q_length);
    store_output4(accumulator_1, q_1, output_channel_base, residue, q_length);
    store_output4(accumulator_2, q_2, output_channel_base, residue, q_length);
    store_output4(accumulator_3, q_3, output_channel_base, residue, q_length);
    store_output4(accumulator_4, q_4, output_channel_base, residue, q_length);
    store_output4(accumulator_5, q_5, output_channel_base, residue, q_length);
    store_output4(accumulator_6, q_6, output_channel_base, residue, q_length);
    store_output4(accumulator_7, q_7, output_channel_base, residue, q_length);
}
