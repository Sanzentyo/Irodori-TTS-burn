// In-place strict-F32 score scaling, key-padding mask, and NaN-safe softmax.
// One 256-invocation workgroup owns one [Skv] row of [B,H,Sq,Skv].

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> masked_out: array<u32>;

const ROWS: u32 = {{ rows }}u;
const HEADS: u32 = {{ heads }}u;
const SQ: u32 = {{ sequence_q }}u;
const SKV: u32 = {{ sequence_kv }}u;
const SCALE: f32 = {{ scale }};
const BLOCK_SIZE: u32 = 256u;
const MIN_FINITE: f32 = -3.4028234663852886e38;
const MIN_POSITIVE: f32 = 1.1754943508222875e-38;

var<workgroup> partial_max: array<f32, 256>;
var<workgroup> partial_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
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

    var local_max = MIN_FINITE;
    for (var key = lane; key < SKV; key = key + BLOCK_SIZE) {
        let index = row_base + key;
        let mask_index = mask_base + key;
        if (masked_out[mask_index] != 0u) {
            scores[index] = MIN_FINITE;
        } else {
            let scaled = scores[index] * SCALE;
            scores[index] = scaled;
            local_max = max(local_max, scaled);
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
    for (var key = lane; key < SKV; key = key + BLOCK_SIZE) {
        let index = row_base + key;
        let mask_index = mask_base + key;
        var numerator = 0.0;
        if (masked_out[mask_index] == 0u) {
            numerator = exp(scores[index] - row_max);
        }
        scores[index] = numerator;
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
    for (var key = lane; key < SKV; key = key + BLOCK_SIZE) {
        let index = row_base + key;
        scores[index] = scores[index] / denominator;
    }
}
