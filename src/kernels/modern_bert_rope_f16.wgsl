enable f16;
@group(0) @binding(0) var<storage, read_write> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> cosine: array<f16>;
@group(0) @binding(2) var<storage, read_write> sine: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
var<workgroup> products: array<vec2<f16>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = id.x + id.y * ({{ groups_x }}u * 256u);
    if (index < {{ elements }}u) {
        let column = index % {{ width }}u;
        let row = index / {{ width }}u;
        let head = row % {{ heads }}u;
        let position = (row / {{ heads }}u) % {{ sequence }}u;
        let batch = row / ({{ heads }}u * {{ sequence }}u);
        let base = batch * {{ batch_stride }}u + position * {{ sequence_stride }}u + head * {{ head_stride }}u;
        let half_width = {{ width }}u / 2u;
        let is_second_half = column >= half_width;
        let partner = select(column + half_width, column - half_width, is_second_half);
        let partner_value = input[base + partner * {{ width_stride }}u];
        let rotated = select(-partner_value, partner_value, is_second_half);
        let coefficient = position * {{ width }}u + column;
        let value = input[base + column * {{ width_stride }}u];
        // Materialize both half products before summing. A cross-lane read after
        // the barrier prevents the driver from removing intermediate conversions.
        products[lane] = vec2<f16>(value * cosine[coefficient], rotated * sine[coefficient]);
    } else {
        products[lane] = vec2<f16>(0.0h);
    }
    workgroupBarrier();
    let destination = index ^ 1u;
    if (destination < {{ elements }}u) {
        let pair = products[lane ^ 1u];
        output[destination] = pair.x + pair.y;
    }
}
