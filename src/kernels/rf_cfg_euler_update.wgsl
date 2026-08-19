// Independent single-signal CFG combine and Euler update.
// The output shape is [B, S, D]; velocities contains conditioned rows followed
// by the matching unconditional rows and therefore has shape [2B, S, D].

@group(0) @binding(0) var<storage, read_write> x_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

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
    let delta = conditioned - unconditional;
    let scaled = delta * CFG_SCALE;
    let guided = conditioned + scaled;
    let step = guided * DT;
    output[index] = x_t[index] + step;
}
