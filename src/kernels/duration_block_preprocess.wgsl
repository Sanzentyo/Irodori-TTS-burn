// Released no-aux duration block: RMSNorm(x) * scale_plus_one + shift.

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> norm_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> shift: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const DIM: u32 = 1024u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let row = group_id.x * DIM;
    var sum_square = 0.0;
    for (var feature = tid; feature < DIM; feature = feature + 256u) {
        let value = input[row + feature];
        sum_square = fma(value, value, sum_square);
    }
    partial[tid] = sum_square;
    workgroupBarrier();

    var width = 128u;
    while (width > 0u) {
        if (tid < width) {
            partial[tid] = partial[tid] + partial[tid + width];
        }
        workgroupBarrier();
        width = width / 2u;
    }
    let inv_rms = inverseSqrt(partial[0] / f32(DIM) + {{ eps }});
    for (var feature = tid; feature < DIM; feature = feature + 256u) {
        let index = row + feature;
        output[index] = input[index] * inv_rms * norm_weight[feature] * scale[feature]
            + shift[feature];
    }
}
