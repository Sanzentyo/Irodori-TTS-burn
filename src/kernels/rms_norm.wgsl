// Custom RMSNorm kernel for WGPU backend.
//
// Computes: output[i] = (x[i] / rms) * weight[i % dim]
// where rms = sqrt(mean(x^2) + eps)
//
// Layout: input is [num_rows, dim], weight is [dim].
// Each workgroup handles one row (one token position).
// All dimensions are baked in as template parameters for optimal codegen.
//
// Platform notes:
// - Uses workgroup shared memory (WGSL core, supported on all backends).
// - Subgroup operations (subgroupAdd) would speed up the reduction but
//   require the "subgroups" WGSL extension (not yet stable in WebGPU).

// All bindings are declared read_write to avoid WGPU validation errors when
// the memory manager suballocates multiple tensors from the same physical buffer.
// WGPU forbids mixing STORAGE_READ_ONLY and STORAGE_READ_WRITE on the same buffer.
@group(0) @binding(0) var<storage, read_write> input:  array<{{ elem }}>;
@group(0) @binding(1) var<storage, read_write> weight: array<{{ elem }}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{ elem }}>;

const BLOCK_SIZE: u32 = {{ workgroup_size_x }}u;
const DIM: u32 = {{ dim }}u;

var<workgroup> shared_sum: array<{{ elem }}, {{ workgroup_size_x }}>;

@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         group_id:  vec3<u32>,
) {
    let tid = local_id.x;
    let row = group_id.x;
    let row_offset = row * DIM;

    // Phase 1: compute sum of squares using parallel reduction.
    var local_sum: {{ elem }} = 0.0;
    for (var i = tid; i < DIM; i = i + BLOCK_SIZE) {
        let val = input[row_offset + i];
        local_sum = local_sum + val * val;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction in shared memory.
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Phase 2: normalize. Thread 0 has the total sum in shared_sum[0].
    let rms = sqrt(shared_sum[0] / {{ elem }}(DIM) + {{ eps }});

    // Phase 3: write output = (x / rms) * weight
    for (var i = tid; i < DIM; i = i + BLOCK_SIZE) {
        output[row_offset + i] = (input[row_offset + i] / rms) * weight[i];
    }
}
