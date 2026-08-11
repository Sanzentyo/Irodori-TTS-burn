// Isolated FP16 AdaLN normalization/modulation candidate.
//
// The low-rank projections that produce scale and shift are outside this
// kernel. Their persistent weights, inputs, and results stay f16 while their
// GEMM accumulators are f32. This kernel begins at the exact upstream
// `x = x.float()` boundary and rounds only its final modulated result to f16.
enable f16;

@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> scale: array<f16>;
@group(0) @binding(2) var<storage, read_write> shift: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

const BLOCK_SIZE: u32 = {{ workgroup_size }}u;
const DIM: u32 = {{ dim }}u;
const SEQ_LEN: u32 = {{ seq_len }}u;
const EPS: f32 = {{ eps }};

// Caching f16 is lossless here because input is already an f16 tensor. All
// arithmetic after a cache load is explicitly widened to f32.
var<workgroup> shared_input: array<f16, {{ dim }}>;
var<workgroup> shared_sum: array<f32, {{ workgroup_size }}>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let row = group_id.x;
    let batch = row / SEQ_LEN;
    let row_offset = row * DIM;
    let modulation_offset = batch * DIM;

    var local_sum: f32 = 0.0;
    for (var dim = tid; dim < DIM; dim = dim + BLOCK_SIZE) {
        let value_half = input[row_offset + dim];
        let value: f32 = f32(value_half);
        shared_input[dim] = value_half;
        local_sum = local_sum + value * value;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let inv_rms: f32 = inverseSqrt(shared_sum[0] / f32(DIM) + EPS);
    for (var dim = tid; dim < DIM; dim = dim + BLOCK_SIZE) {
        let normalized: f32 = f32(shared_input[dim]) * inv_rms;
        let scale_f32: f32 = f32(scale[modulation_offset + dim]);
        let shift_f32: f32 = f32(shift[modulation_offset + dim]);
        let modulated: f32 = normalized * (1.0 + scale_f32) + shift_f32;
        output[row_offset + dim] = f16(modulated);
    }
}
