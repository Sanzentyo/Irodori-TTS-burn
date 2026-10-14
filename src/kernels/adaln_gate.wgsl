{{ enable }}
alias F = {{ scalar }};
@group(0) @binding(0) var<storage, read_write> up: array<F>;
@group(0) @binding(1) var<storage, read_write> bias: array<F>;
@group(0) @binding(2) var<storage, read_write> raw: array<F>;
@group(0) @binding(3) var<storage, read_write> output: array<F>;
@group(0) @binding(4) var<uniform> geometry: vec4<u32>;
const DIM: u32 = {{ dim }}u;
const SLOTS: u32 = {{ slots }}u;
var<workgroup> biased: array<F, 256>;
var<workgroup> summed: array<F, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = id.x + id.y * (geometry.y * 256u);
    biased[lane] = F(0.0);
    if (index < geometry.x) {
        biased[lane] = up[index] + bias[index % (SLOTS * DIM)];
    }
    workgroupBarrier();
    let partner = index ^ 1u;
    summed[lane] = F(0.0);
    if (partner < geometry.x) {
        let batch = partner / (SLOTS * DIM);
        summed[lane] = biased[lane ^ 1u] + raw[batch * (3u * DIM) + partner % (3u * DIM)];
    }
    workgroupBarrier();
    if (index < geometry.x) {
        let value = summed[lane ^ 1u];
        if ((index / DIM) % 3u == 2u) { output[index] = tanh(value); }
        else { output[index] = value; }
    }
}
