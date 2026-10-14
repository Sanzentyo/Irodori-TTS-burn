{{ enable }}
alias F = {{ scalar }};
@group(0) @binding(0) var<storage, read_write> input: array<F>;
@group(0) @binding(1) var<storage, read_write> output: array<F>;
@group(0) @binding(2) var<storage, read_write> erf_input: array<F>;
{{ activation_binding }}
@group(0) @binding({{ metadata_binding }}) var<uniform> geometry: vec4<u32>;
const HIDDEN: u32 = {{ hidden }}u;
var<workgroup> stage_a: array<F, 256>;
var<workgroup> stage_b: array<F, 256>;

fn load_half(index: u32, half_index: u32) -> F {
    if (index >= geometry.x) { return F(0.0); }
    return input[(index / HIDDEN) * (2u * HIDDEN) + index % HIDDEN + half_index * HIDDEN];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = id.x + id.y * (geometry.y * 256u);
    // Exchange neighboring lanes at each materialized storage boundary. This
    // prevents elimination of the half conversions, including subnormals.
    stage_a[lane] = F(0.0);
    if (index < geometry.x) { stage_a[lane] = erf_input[index] + F(1.0); }
    workgroupBarrier();
    stage_b[lane] = load_half(index ^ 1u, 0u) * stage_a[lane ^ 1u];
    workgroupBarrier();
    stage_a[lane] = stage_b[lane ^ 1u] / F(2.0);
    workgroupBarrier();
    let destination = index ^ 1u;
    if (destination < geometry.x) {
        {{ activation_write }}
        output[destination] = stage_a[lane ^ 1u] * load_half(destination, 1u);
    }
}
