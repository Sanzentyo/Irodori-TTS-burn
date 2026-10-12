// Strict-F32 in-place masked softmax with lane-private score/mask storage.
// Reduction lanes and addition order match the streaming variant. Only the
// final probabilities reach storage; masked logits are never read or exponentiated.

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> masked_out: array<u32>;

const ROWS: u32 = {{ rows }}u;
const HEADS: u32 = {{ heads }}u;
const SQ: u32 = {{ sequence_q }}u;
const SKV: u32 = {{ sequence_kv }}u;
const ITEMS: u32 = {{ items_per_lane }}u;
const SCALE: f32 = {{ scale }};
const BLOCK_SIZE: u32 = {{ workgroup_size }}u;
const MIN_FINITE: f32 = -3.4028234663852886e38;
const MIN_POSITIVE: f32 = 1.1754943508222875e-38;

var<workgroup> partial_max: array<f32, {{ workgroup_size }}>;
var<workgroup> partial_sum: array<f32, {{ workgroup_size }}>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let lane = local_id.x;
    let row = group_id.x;
    if (row >= ROWS) {
        return;
    }
    let batch = row / (HEADS * SQ);
    let row_base = row * SKV;
    let mask_base = batch * SKV;
    var values: array<f32, {{ items_per_lane }}>;
    var valid: array<bool, {{ items_per_lane }}>;

    var local_max = MIN_FINITE;
    for (var item = 0u; item < ITEMS; item = item + 1u) {
        let key = lane + item * BLOCK_SIZE;
        if (key < SKV) {
            valid[item] = masked_out[mask_base + key] == 0u;
            if (valid[item]) {
                let scaled = scores[row_base + key] * SCALE;
                values[item] = scaled;
                local_max = max(local_max, scaled);
            }
        }
    }
    partial_max[lane] = local_max;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            partial_max[lane] = max(partial_max[lane], partial_max[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = max(partial_max[0], MIN_FINITE);

    var local_sum = 0.0;
    for (var item = 0u; item < ITEMS; item = item + 1u) {
        var numerator = 0.0;
        if (valid[item]) {
            numerator = exp(values[item] - row_max);
        }
        values[item] = numerator;
        local_sum = local_sum + numerator;
    }
    partial_sum[lane] = local_sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let denominator = max(partial_sum[0], MIN_POSITIVE);
    for (var item = 0u; item < ITEMS; item = item + 1u) {
        let key = lane + item * BLOCK_SIZE;
        if (key < SKV) {
            scores[row_base + key] = values[item] / denominator;
        }
    }
}
