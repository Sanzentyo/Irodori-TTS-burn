enable f16;

// F16-storage counterpart of fused_swiglu_pair.wgsl. The nonlinear epilogue
// remains F32 so storage precision does not silently weaken the accumulator.
@group(0) @binding(0) var<storage, read_write> gate: array<f16>;
@group(0) @binding(1) var<storage, read_write> value: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

const ELEMENTS: u32 = {{ elements }}u;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= ELEMENTS) {
        return;
    }

    let gate_value = f32(gate[index]);
    let value_value = f32(value[index]);
    output[index] = f16(gate_value / (1.0 + exp(-gate_value)) * value_value);
}
