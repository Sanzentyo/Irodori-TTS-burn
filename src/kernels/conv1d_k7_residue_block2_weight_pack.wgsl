// One-time exact OIK -> [Cin, K7, Cout] vec4 pack for Block2 C192 weights.

@group(0) @binding(0) var<storage, read_write> weight_oik: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_weight: array<vec4<f32>>;

const CHANNELS: u32 = 192u;
const KERNEL_SIZE: u32 = 7u;
const OUTPUT_VECTORS: u32 = 48u;
const PACKED_VECTORS: u32 = CHANNELS * KERNEL_SIZE * OUTPUT_VECTORS;

fn source_weight(output_channel: u32, input_channel: u32, tap: u32) -> f32 {
    let source_index =
        (output_channel * CHANNELS + input_channel) * KERNEL_SIZE + tap;
    return weight_oik[source_index];
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let packed_vector_index = global_id.x;
    if packed_vector_index >= PACKED_VECTORS {
        return;
    }

    let input_tap = packed_vector_index / OUTPUT_VECTORS;
    let output_vector = packed_vector_index - input_tap * OUTPUT_VECTORS;
    let input_channel = input_tap / KERNEL_SIZE;
    let tap = input_tap - input_channel * KERNEL_SIZE;
    let output_channel = output_vector * 4u;
    packed_weight[packed_vector_index] = vec4<f32>(
        source_weight(output_channel + 0u, input_channel, tap),
        source_weight(output_channel + 1u, input_channel, tap),
        source_weight(output_channel + 2u, input_channel, tap),
        source_weight(output_channel + 3u, input_channel, tap),
    );
}
