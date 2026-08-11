// Compact exact C192/L48000 NCL into [residue][channel][q].
//
// Residue blocks are ragged rather than padded. For L=48000, d=3 has three
// Q=16000 blocks; d=9 has Q=[5334,5334,5334,5333,5333,5333,5333,5333,5333].
// Every source f32 is copied once and the destination contains exactly the
// original 9,216,000 elements (36,864,000 bytes).

// SourceKernel buffers are read_write because CubeCL sliced allocations can
// place otherwise disjoint logical tensors in one physical buffer.
@group(0) @binding(0) var<storage, read_write> input_buf:  array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_buf: array<f32>;

const CHANNELS: u32 = {{ channels }}u;
const LENGTH: u32 = {{ length }}u;
const ELEMENTS: u32 = {{ elements }}u;
const DILATION: u32 = {{ dilation }}u;
const BASE_LENGTH: u32 = {{ base_length }}u;
const REMAINDER: u32 = {{ remainder }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_index = global_id.x;
    if input_index >= ELEMENTS {
        return;
    }

    let channel = input_index / LENGTH;
    let time = input_index - channel * LENGTH;
    let residue = time % DILATION;
    let q = time / DILATION;
    let residue_length = BASE_LENGTH + select(0u, 1u, residue < REMAINDER);
    let residue_prefix_q = residue * BASE_LENGTH + min(residue, REMAINDER);
    let packed_index =
        residue_prefix_q * CHANNELS + channel * residue_length + q;
    packed_buf[packed_index] = input_buf[input_index];
}
