enable f16;

// Production codec prepared pointwise-residual and next-Snake pair.
//
// branch_nlc is the contiguous [1,L,C] packed pointwise GEMM output.
// residual_ncl, raw_ncl, and activated_ncl are contiguous [1,C,L]. The
// finalizer keeps the production f32 order `(branch + bias) + residual`, then
// applies the exact production Snake formula to that rounded f32 `raw` value.
//
// SourceKernel bindings are uniformly read_write because CubeCL's sliced
// allocator may place distinct logical tensors in one physical WGPU buffer.
@group(0) @binding(0) var<storage, read_write> branch_nlc: array<f16>;
@group(0) @binding(1) var<storage, read_write> bias: array<f16>;
@group(0) @binding(2) var<storage, read_write> residual_ncl: array<f16>;
@group(0) @binding(3) var<storage, read_write> alpha: array<f16>;
@group(0) @binding(4) var<storage, read_write> raw_ncl: array<f16>;
@group(0) @binding(5) var<storage, read_write> activated_ncl: array<f16>;

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
    let raw = biased + f32(residual_ncl[output_index]);
    raw_ncl[output_index] = f16(raw);

    let a = f32(alpha[channel]);
    let sine = sin(a * raw);
    activated_ncl[output_index] = f16(raw + (sine * sine) / (a + 1e-9));
}
