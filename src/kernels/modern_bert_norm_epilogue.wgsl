{{ enable }}
alias F = {{ scalar }};
@group(0) @binding(0) var<storage, read_write> centered: array<f32>;
@group(0) @binding(1) var<storage, read_write> variance: array<f32>;
@group(0) @binding(2) var<storage, read_write> gamma: array<F>;
@group(0) @binding(3) var<storage, read_write> output: array<F>;
@group(0) @binding(4) var<uniform> geometry: vec4<u32>;
const WIDTH: u32 = {{ width }}u;
var<workgroup> first: array<f32, 256>;
var<workgroup> second: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = id.x + id.y * (geometry.y * 256u);
    first[lane] = 1.0;
    if (index < geometry.x) { first[lane] = variance[index / WIDTH] + 1.0e-5; }
    workgroupBarrier();
    second[lane] = sqrt(first[lane ^ 1u]);
    workgroupBarrier();
    first[lane] = 0.0;
    if (index < geometry.x) { first[lane] = centered[index] / second[lane ^ 1u]; }
    workgroupBarrier();
    let partner = index ^ 1u;
    second[lane] = 0.0;
    if (partner < geometry.x) { second[lane] = first[lane ^ 1u] * f32(gamma[partner % WIDTH]); }
    workgroupBarrier();
    if (index < geometry.x) { output[index] = F(second[lane ^ 1u]); }
}
