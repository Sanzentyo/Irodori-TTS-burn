// Isolated v4 ModernBERT residual-add + LayerNorm candidate.
//
// Exact specialized shape: B=1, S=3, D=768, f32. One workgroup owns one
// token row. The variance deliberately follows Burn's biased two-pass
// definition:
//
//   mean = sum(x) / D
//   variance = sum((x - mean) * (x - mean)) / D
//
// Normalization preserves Burn's operation order and has gamma but no beta:
//
//   ((x - mean) / sqrt(variance + 1e-5)) * gamma
//
// All storage bindings are read_write. CubeCL can suballocate distinct logical
// tensors into one physical WGPU buffer, for which mixed read/read_write
// bindings would fail WebGPU validation.

@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read_write> branch: array<f32>;
@group(0) @binding(2) var<storage, read_write> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> updated_residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> normalized: array<f32>;

const WIDTH: u32 = {{ width }}u;
const WORKGROUP_SIZE: u32 = {{ workgroup_size }}u;
const EPSILON: f32 = {{ epsilon }};

var<workgroup> partial: array<f32, {{ workgroup_size }}>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let lane = local_id.x;
    let row_offset = group_id.x * WIDTH;
    let column_0 = lane;
    let column_1 = lane + WORKGROUP_SIZE;
    let column_2 = lane + 2u * WORKGROUP_SIZE;
    let index_0 = row_offset + column_0;
    let index_1 = row_offset + column_1;
    let index_2 = row_offset + column_2;

    // Preserve the production residual ownership/order: residual + branch.
    let value_0 = residual[index_0] + branch[index_0];
    let value_1 = residual[index_1] + branch[index_1];
    let value_2 = residual[index_2] + branch[index_2];
    updated_residual[index_0] = value_0;
    updated_residual[index_1] = value_1;
    updated_residual[index_2] = value_2;

    partial[lane] = (value_0 + value_1) + value_2;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean = partial[0] / f32(WIDTH);
    // Every invocation must capture mean before lane zero reuses partial[0].
    workgroupBarrier();

    let centered_0 = value_0 - mean;
    let centered_1 = value_1 - mean;
    let centered_2 = value_2 - mean;
    partial[lane] =
        (centered_0 * centered_0 + centered_1 * centered_1) +
        centered_2 * centered_2;
    workgroupBarrier();

    stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let variance = partial[0] / f32(WIDTH);
    let standard_deviation = sqrt(variance + EPSILON);
    normalized[index_0] = (centered_0 / standard_deviation) * gamma[column_0];
    normalized[index_1] = (centered_1 / standard_deviation) * gamma[column_1];
    normalized[index_2] = (centered_2 / standard_deviation) * gamma[column_2];
}
