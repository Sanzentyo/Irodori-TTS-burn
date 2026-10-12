enable f16;

// Production codec pointwise residual finalizer.
//
// branch_nlc is the contiguous [1,L,C] output of the packed pointwise GEMM.
// residual_ncl and output_ncl are contiguous [1,C,L]. The two f32 additions
// are intentionally sequenced as (branch + bias) + residual.
//
// SourceKernel bindings are uniformly read_write because CubeCL's sliced
// allocator may place distinct logical tensors in one physical WGPU buffer.
@group(0) @binding(0) var<storage, read_write> branch_nlc: array<f16>;
@group(0) @binding(1) var<storage, read_write> bias: array<f16>;
@group(0) @binding(2) var<storage, read_write> residual_ncl: array<f16>;
@group(0) @binding(3) var<storage, read_write> output_ncl: array<f16>;

const CHANNELS: u32 = {{ channels }}u;
const LENGTH: u32 = {{ length }}u;
const ELEMENTS: u32 = {{ elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    if (output_index >= ELEMENTS) {
        return;
    }

    let channel = output_index / LENGTH;
    let time = output_index - channel * LENGTH;
    let branch_index = time * CHANNELS + channel;
    let biased = f32(branch_nlc[branch_index]) + f32(bias[channel]);
    output_ncl[output_index] = f16(biased + f32(residual_ncl[output_index]));
}
