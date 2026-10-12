enable f16;

// One-time OIK -> [Cin, tap, output-vector] pack for the residue k7 core.
//
// Each vec4 stores the four output channels owned by one production invocation:
// [tile*32 + lane, +8, +16, +24]. The convolution reduction order is unchanged;
// this layout quarters scalar weight load instructions.

@group(0) @binding(0) var<storage, read_write> source_oik: array<f16>;
@group(0) @binding(1) var<storage, read_write> packed_vectors: array<vec4<f16>>;

const CHANNELS: u32 = {{ channels }}u;
const OUTPUT_VECTORS: u32 = CHANNELS / 4u;
const VECTORS_PER_TILE: u32 = 8u;
const KERNEL_SIZE: u32 = 7u;
const VECTOR_ELEMENTS: u32 = {{ vector_elements }}u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let packed_index = global_id.x;
    if packed_index >= VECTOR_ELEMENTS {
        return;
    }

    let vector = packed_index % OUTPUT_VECTORS;
    let reduction = packed_index / OUTPUT_VECTORS;
    let tap = reduction % KERNEL_SIZE;
    let input_channel = reduction / KERNEL_SIZE;
    let output_tile = vector / VECTORS_PER_TILE;
    let output_lane = vector % VECTORS_PER_TILE;
    let output_channel_0 = output_tile * 32u + output_lane;
    let output_channel_1 = output_channel_0 + 8u;
    let output_channel_2 = output_channel_0 + 16u;
    let output_channel_3 = output_channel_0 + 24u;
    let source_0 = (output_channel_0 * CHANNELS + input_channel) * KERNEL_SIZE + tap;
    let source_1 = (output_channel_1 * CHANNELS + input_channel) * KERNEL_SIZE + tap;
    let source_2 = (output_channel_2 * CHANNELS + input_channel) * KERNEL_SIZE + tap;
    let source_3 = (output_channel_3 * CHANNELS + input_channel) * KERNEL_SIZE + tap;
    packed_vectors[packed_index] = vec4<f16>(vec4<f32>(
        f32(source_oik[source_0]),
        f32(source_oik[source_1]),
        f32(source_oik[source_2]),
        f32(source_oik[source_3]),
    ));
}
