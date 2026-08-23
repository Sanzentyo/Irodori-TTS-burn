// Production exact-shape JointAttention post-SDPA kernel for v4-Small.
//
// The tuned SDPA output is contiguous [B,H,S,64]. The accepted combined
// gate source is either compact [B,S,1280] storage or the fallback contiguous
// [B,S,4*1280] QKV+gate allocation. This shader performs the metadata swap, mandatory
// contiguous reshape, and gate multiplication directly into token-major
// [B,S,1280] storage in one dispatch.

@group(0) @binding(0) var<storage, read_write> attention: array<f32>;
@group(0) @binding(1) var<storage, read_write> gate_source: array<f32>;
@group(0) @binding(2) var<storage, read_write> output:    array<f32>;

const S: u32 = {{ sequence }}u;
const H: u32 = 20u;
const DH: u32 = 64u;
const D: u32 = 1280u;
const GATE_STRIDE: u32 = {{ gate_stride }}u;
const GATE_OFFSET: u32 = {{ gate_offset }}u;
const ELEMENTS: u32 = {{ elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    if (output_index >= ELEMENTS) {
        return;
    }

    let token = output_index / D;
    let dim = output_index % D;
    let batch = token / S;
    let seq = token % S;
    let head = dim / DH;
    let component = dim % DH;
    let attention_index = ((batch * H + head) * S + seq) * DH + component;
    let gate_index = token * GATE_STRIDE + GATE_OFFSET + dim;

    // Operand order matches the current `gate * out` expression.
    output[output_index] = gate_source[gate_index] * attention[attention_index];
}
