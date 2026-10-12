// Fused gated residual update used by every DiT attention and MLP branch.
// output[b,s,d] = residual[b,s,d] + gate[b,d] * branch[b,s,d]

// SourceKernel supplies raw WGSL without CubeCL's generated-kernel binding
// metadata. CubeCL's sliced allocator may therefore bind disjoint logical
// tensors from the same physical buffer in one Tasks compute pass. WGPU tracks
// storage usage for that whole buffer, so mixing read-only inputs and a
// read-write output is invalid even when their byte ranges do not overlap.
// Declare every storage binding read_write; the kernel still treats its three
// inputs as logically immutable.
@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read_write> branch: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const DIM: u32 = {{ dim }}u;
const SEQ_LEN: u32 = {{ seq_len }}u;
const ELEMENTS: u32 = {{ elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= ELEMENTS) {
        return;
    }

    let row = index / DIM;
    let batch = row / SEQ_LEN;
    let column = index - row * DIM;
    output[index] = residual[index] + gate[batch * DIM + column] * branch[index];
}
