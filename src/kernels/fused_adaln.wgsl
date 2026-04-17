// Fused AdaLN kernel for WGPU backend.
//
// Computes: output[i] = (x[i] / rms) * (1.0 + scale[i]) + shift[i]
// where rms = sqrt(mean(x^2) + eps)
//
// This fuses the RMSNorm + modulation steps of LowRankAdaLn::forward,
// eliminating 3 separate kernel launches (powf, mean+sqrt, div+mul+add)
// into a single pass.
//
// Input layout:
//   x:     [B * S, D] (flattened from [B, S, D])
//   scale: [B, D]     (from [B, 1, D], broadcast over seq dim)
//   shift: [B, D]     (from [B, 1, D], broadcast over seq dim)
//
// Dispatch: 1D grid — one workgroup per row (total_rows = B * S).
// batch_idx derived from row index and seq_len template parameter.
//
// Template parameters (baked at compile time):
//   dim:              hidden dimension D (fixed per model)
//   seq_len:          sequence length S (varies per utterance, not per step)
//   workgroup_size_x: threads per workgroup (min(256, next_pow2(dim)))
//   shared_mem_size:  max(dim, workgroup_size_x) for row cache
//   eps:              RMSNorm epsilon (fixed per model)
//   elem:             element type (f32)
//
// Platform notes:
// - Uses workgroup shared memory only (WGSL core, all backends).
// - Caches the input row in shared memory — single global read for both
//   reduction and output phases.
// - Subgroup operations (subgroupAdd) would speed up the tree reduction
//   but require the "subgroups" WGSL extension (W3C CR, Chrome v145+).
//   Fallback: current shared-memory tree reduction works everywhere.
// - f16 variant would need "shader-f16" extension (Chrome 120+, wgpu
//   experimental). Fallback: use this f32 kernel with f32 data.

// All bindings declared read_write: WGPU suballocator may pack multiple
// tensors into one physical buffer, forbidding mixed usage flags.
@group(0) @binding(0) var<storage, read_write> input:  array<{{ elem }}>;
@group(0) @binding(1) var<storage, read_write> scale:  array<{{ elem }}>;
@group(0) @binding(2) var<storage, read_write> shift:  array<{{ elem }}>;
@group(0) @binding(3) var<storage, read_write> output: array<{{ elem }}>;

const BLOCK_SIZE: u32 = {{ workgroup_size_x }}u;
const DIM: u32 = {{ dim }}u;
const SEQ_LEN: u32 = {{ seq_len }}u;

// Row cache: holds one full row of x for reuse in both reduction and output.
var<workgroup> shared_data: array<{{ elem }}, {{ shared_mem_size }}>;
// Reduction buffer: one partial sum per thread.
var<workgroup> shared_sum: array<{{ elem }}, {{ workgroup_size_x }}>;

@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(
    @builtin(local_invocation_id)  local_id: vec3<u32>,
    @builtin(workgroup_id)         group_id: vec3<u32>,
) {
    let tid = local_id.x;
    // 1D dispatch: one workgroup per row in [B*S, D].
    let row = group_id.x;
    let batch_idx = row / SEQ_LEN;

    let row_offset = row * DIM;
    let batch_offset = batch_idx * DIM;

    // Phase 1: cache input row in shared memory + accumulate sum of squares.
    var local_sum: {{ elem }} = 0.0;
    for (var i = tid; i < DIM; i = i + BLOCK_SIZE) {
        let val = input[row_offset + i];
        shared_data[i] = val;
        local_sum = local_sum + val * val;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum of squares.
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // RMS = sqrt(mean(x²) + eps). shared_sum[0] is broadcast to all threads.
    let rms = sqrt(shared_sum[0] / {{ elem }}(DIM) + {{ eps }});

    // Phase 2: fused normalize + modulate from cached row.
    // output = (x / rms) * (1 + scale) + shift
    for (var i = tid; i < DIM; i = i + BLOCK_SIZE) {
        let x_norm = shared_data[i] / rms;
        output[row_offset + i] = x_norm * (1.0 + scale[batch_offset + i]) + shift[batch_offset + i];
    }
}
