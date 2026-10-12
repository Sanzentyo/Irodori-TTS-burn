enable f16;

// Independent single-signal CFG combine and Euler update. Explicit f16 casts
// preserve the storage-rounding boundary of the five-dispatch reference path.

@group(0) @binding(0) var<storage, read_write> x_t: array<f16>;
@group(0) @binding(1) var<storage, read_write> velocities: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

const ELEMENTS: u32 = {{ elements }}u;
const CFG_SCALE: f32 = {{ cfg_scale }};
const DT: f32 = {{ dt }};

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= ELEMENTS) {
        return;
    }
    let conditioned = velocities[index];
    let unconditional = velocities[ELEMENTS + index];
    // Burn converts scalar operands to the tensor element type before the
    // elementwise multiplication. Preserve that boundary as well as every
    // intermediate tensor-store boundary below.
    let cfg_scale = f16(CFG_SCALE);
    let dt = f16(DT);
    let delta = f16(f32(conditioned) - f32(unconditional));
    let scaled = f16(f32(delta) * f32(cfg_scale));
    let guided = f16(f32(conditioned) + f32(scaled));
    let step = f16(f32(guided) * f32(dt));
    output[index] = f16(f32(x_t[index]) + f32(step));
}
