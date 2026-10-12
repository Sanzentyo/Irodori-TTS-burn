// Native wgpu/Naga subgroup builtins require the SUBGROUP device feature.
// Do not add `enable subgroups`: Naga 30 accepts the builtins, not that directive.
// Each lane owns eight virtual lanes of the portable 256-lane reduction tree.
// Its first three tree levels stay in registers; the last five use shuffles.

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> masked_out: array<u32>;

const ROWS: u32 = {{ rows }}u;
const HEADS: u32 = {{ heads }}u;
const SQ: u32 = {{ sequence_q }}u;
const SKV: u32 = {{ sequence_kv }}u;
const ITEMS: u32 = {{ items_per_lane }}u;
const SCALE: f32 = {{ scale }};
const MIN_FINITE: f32 = -3.4028234663852886e38;
const MIN_POSITIVE: f32 = 1.1754943508222875e-38;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_id) subgroup: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let row = group_id.x * 4u + subgroup;
    let batch = row / (HEADS * SQ);
    let row_base = row * SKV;
    let mask_base = batch * SKV;
    var values: array<f32, 8 * {{ items_per_lane }}>;
    var valid: array<bool, 8 * {{ items_per_lane }}>;
    var sums: array<f32, 8>;
    var local_max = MIN_FINITE;
    for (var v = 0u; v < 8u; v = v + 1u) {
        for (var item = 0u; item < ITEMS; item = item + 1u) {
            let index = v * ITEMS + item;
            let key = lane + v * 32u + item * 256u;
            if (row < ROWS && key < SKV) {
                valid[index] = masked_out[mask_base + key] == 0u;
                if (valid[index]) {
                    let scaled = scores[row_base + key] * SCALE;
                    values[index] = scaled;
                    local_max = max(local_max, scaled);
                }
            }
        }
    }
    let row_max = max(subgroupMax(local_max), MIN_FINITE);
    for (var v = 0u; v < 8u; v = v + 1u) {
        var local_sum = 0.0;
        for (var item = 0u; item < ITEMS; item = item + 1u) {
            let index = v * ITEMS + item;
            var numerator = 0.0;
            if (valid[index]) {
                numerator = exp(values[index] - row_max);
            }
            values[index] = numerator;
            local_sum = local_sum + numerator;
        }
        sums[v] = local_sum;
    }
    // Match strides 128, 64 and 32 of the original 256-lane tree exactly.
    let s0 = (sums[0] + sums[4]) + (sums[2] + sums[6]);
    let s1 = (sums[1] + sums[5]) + (sums[3] + sums[7]);
    var sum = s0 + s1;
    for (var stride = 16u; stride > 0u; stride = stride / 2u) {
        let other = subgroupShuffleDown(sum, stride);
        if (lane < stride) {
            sum = sum + other;
        }
    }
    let denominator = max(subgroupBroadcastFirst(sum), MIN_POSITIVE);
    for (var v = 0u; v < 8u; v = v + 1u) {
        for (var item = 0u; item < ITEMS; item = item + 1u) {
            let key = lane + v * 32u + item * 256u;
            if (row < ROWS && key < SKV) {
                scores[row_base + key] = values[v * ITEMS + item] / denominator;
            }
        }
    }
}
