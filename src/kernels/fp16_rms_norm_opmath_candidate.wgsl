// Isolated FP16 RMSNorm candidate.
//
// PyTorch reference boundary:
//   half input + half weight -> f32 square/reduce/rsqrt/multiply -> half output.
// The single terminal f16 conversion is intentional. In particular, do not
// round the reciprocal RMS to f16 before multiplying the input and weight.
enable f16;

// Keep every binding read_write because CubeCL may suballocate logical tensors
// from one physical storage buffer with a common usage declaration.
@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

const BLOCK_SIZE: u32 = {{ workgroup_size }}u;
const DIM: u32 = {{ dim }}u;
const EPS: f32 = {{ eps }};

// Reduction state must remain f32 even though persistent tensors are f16.
var<workgroup> shared_sum: array<f32, {{ workgroup_size }}>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let row_offset = group_id.x * DIM;

    var local_sum: f32 = 0.0;
    for (var dim = tid; dim < DIM; dim = dim + BLOCK_SIZE) {
        let value: f32 = f32(input[row_offset + dim]);
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

    // The upstream implementation uses torch.rsqrt after x.float().
    let inv_rms: f32 = inverseSqrt(shared_sum[0] / f32(DIM) + EPS);
    for (var dim = tid; dim < DIM; dim = dim + BLOCK_SIZE) {
        let normalized: f32 = f32(input[row_offset + dim]) * inv_rms;
        output[row_offset + dim] = f16(normalized * f32(weight[dim]));
    }
}
