// T64/O96 WG32x8 vector f32 pointwise + bias + residual. K is templated.
@group(0) @binding(0) var<storage, read_write> input_ncl: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> residual_ncl: array<f32>;
@group(0) @binding(4) var<storage, read_write> raw_ncl: array<f32>;

const CHANNELS: u32 = {{ channels }}u;
const LENGTH: u32 = {{ length }}u;
const K_TILE: u32 = {{ k_tile }}u;
const INPUT_STRIDE: u32 = {{ input_stride }}u;
const INPUT_TILE_ELEMENTS: u32 = {{ input_tile_elements }}u;
const WEIGHT_VECTOR_TILE_ELEMENTS: u32 = {{ weight_vector_tile_elements }}u;
const TIME_TILE: u32 = 64u;
const OUTPUT_TILE: u32 = 96u;
const OUTPUT_VECTORS: u32 = 24u;
const VECTORS_PER_THREAD: u32 = 3u;
const WEIGHT_VECTORS_PER_ROW: u32 = CHANNELS / 4u;
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> input_tile: array<f32, {{ input_tile_storage_elements }}>;
var<workgroup> weight_tile: array<vec4<f32>, {{ weight_vector_tile_elements }}>;

fn finish_raw(accumulator: f32, output_channel: u32, time: u32) -> f32 {
    let biased = accumulator + bias[output_channel];
    return biased + residual_ncl[output_channel * LENGTH + time];
}

fn store_raw4(accumulator: vec4<f32>, output_channel: u32, time: u32) {
    if (output_channel < CHANNELS) {
        raw_ncl[(output_channel + 0u) * LENGTH + time] =
            finish_raw(accumulator.x, output_channel + 0u, time);
        raw_ncl[(output_channel + 1u) * LENGTH + time] =
            finish_raw(accumulator.y, output_channel + 1u, time);
        raw_ncl[(output_channel + 2u) * LENGTH + time] =
            finish_raw(accumulator.z, output_channel + 2u, time);
        raw_ncl[(output_channel + 3u) * LENGTH + time] =
            finish_raw(accumulator.w, output_channel + 3u, time);
    }
}

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TIME_TILE;
    let output_channel_base =
        group_id.y * OUTPUT_TILE + local_id.y * VECTORS_PER_THREAD * 4u;
    let local_time_base = local_id.x * 2u;
    var accumulator_00 = vec4<f32>(0.0);
    var accumulator_01 = vec4<f32>(0.0);
    var accumulator_02 = vec4<f32>(0.0);
    var accumulator_10 = vec4<f32>(0.0);
    var accumulator_11 = vec4<f32>(0.0);
    var accumulator_12 = vec4<f32>(0.0);

    var input_channel_base = 0u;
    loop {
        if (input_channel_base >= CHANNELS) {
            break;
        }
        var load_index = local_index;
        loop {
            if (load_index >= INPUT_TILE_ELEMENTS) {
                break;
            }
            let tile_input_channel = load_index / TIME_TILE;
            let tile_time = load_index - tile_input_channel * TIME_TILE;
            let input_channel = input_channel_base + tile_input_channel;
            input_tile[tile_time * INPUT_STRIDE + tile_input_channel] =
                input_ncl[input_channel * LENGTH + time_base + tile_time];
            load_index += WORKGROUP_SIZE;
        }
        var load_vector_index = local_index;
        loop {
            if (load_vector_index >= WEIGHT_VECTOR_TILE_ELEMENTS) {
                break;
            }
            let tile_input_channel = load_vector_index / OUTPUT_VECTORS;
            let tile_output_vector =
                load_vector_index - tile_input_channel * OUTPUT_VECTORS;
            let input_channel = input_channel_base + tile_input_channel;
            let output_channel = group_id.y * OUTPUT_TILE + tile_output_vector * 4u;
            var weight_value = vec4<f32>(0.0);
            if (output_channel < CHANNELS) {
                let output_vector = group_id.y * OUTPUT_VECTORS + tile_output_vector;
                weight_value =
                    packed_weight[input_channel * WEIGHT_VECTORS_PER_ROW + output_vector];
            }
            weight_tile[load_vector_index] = weight_value;
            load_vector_index += WORKGROUP_SIZE;
        }

        workgroupBarrier();
        var tile_input_channel = 0u;
        loop {
            if (tile_input_channel >= K_TILE) {
                break;
            }
            let weight_base = tile_input_channel * OUTPUT_VECTORS + local_id.y * VECTORS_PER_THREAD;
            let weight_0 = weight_tile[weight_base + 0u];
            let weight_1 = weight_tile[weight_base + 1u];
            let weight_2 = weight_tile[weight_base + 2u];
            let input_0 = input_tile[(local_time_base + 0u) * INPUT_STRIDE + tile_input_channel];
            let input_1 = input_tile[(local_time_base + 1u) * INPUT_STRIDE + tile_input_channel];
            accumulator_00 = fma(vec4<f32>(input_0), weight_0, accumulator_00);
            accumulator_01 = fma(vec4<f32>(input_0), weight_1, accumulator_01);
            accumulator_02 = fma(vec4<f32>(input_0), weight_2, accumulator_02);
            accumulator_10 = fma(vec4<f32>(input_1), weight_0, accumulator_10);
            accumulator_11 = fma(vec4<f32>(input_1), weight_1, accumulator_11);
            accumulator_12 = fma(vec4<f32>(input_1), weight_2, accumulator_12);
            tile_input_channel += 1u;
        }
        workgroupBarrier();
        input_channel_base += K_TILE;
    }

    let time_0 = time_base + local_time_base + 0u;
    let time_1 = time_base + local_time_base + 1u;
    store_raw4(accumulator_00, output_channel_base + 0u, time_0);
    store_raw4(accumulator_01, output_channel_base + 4u, time_0);
    store_raw4(accumulator_02, output_channel_base + 8u, time_0);
    store_raw4(accumulator_10, output_channel_base + 0u, time_1);
    store_raw4(accumulator_11, output_channel_base + 4u, time_1);
    store_raw4(accumulator_12, output_channel_base + 8u, time_1);
}
