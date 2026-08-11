// One-time OIK -> [Cin, tap, output-pair] pack for the residue k7 core.
//
// Each vec2 stores the two output channels owned by one production invocation:
// [tile*32 + lane, tile*32 + lane + 16]. The convolution reduction order is
// unchanged; this layout only halves scalar weight load instructions.

@group(0) @binding(0) var<storage, read_write> source_oik: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_pairs: array<vec2<f32>>;

const CHANNELS: u32 = {{ channels }}u;
const OUTPUT_PAIRS: u32 = CHANNELS / 2u;
const PAIRS_PER_TILE: u32 = 16u;
const KERNEL_SIZE: u32 = 7u;
const PAIR_ELEMENTS: u32 = {{ pair_elements }}u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let packed_index = global_id.x;
    if packed_index >= PAIR_ELEMENTS {
        return;
    }

    let pair = packed_index % OUTPUT_PAIRS;
    let reduction = packed_index / OUTPUT_PAIRS;
    let tap = reduction % KERNEL_SIZE;
    let input_channel = reduction / KERNEL_SIZE;
    let output_tile = pair / PAIRS_PER_TILE;
    let output_lane = pair % PAIRS_PER_TILE;
    let output_channel_0 = output_tile * 32u + output_lane;
    let output_channel_1 = output_channel_0 + 16u;
    let source_0 = (output_channel_0 * CHANNELS + input_channel) * KERNEL_SIZE + tap;
    let source_1 = (output_channel_1 * CHANNELS + input_channel) * KERNEL_SIZE + tap;
    packed_pairs[packed_index] = vec2<f32>(source_oik[source_0], source_oik[source_1]);
}
