enable f16;

// Profile-only decoder tail fusion:
// pointwise(96 -> 96) + bias + residual + F16 storage boundary + Snake
// + Conv1d(96 -> 1, k=7, same padding) + tanh.

@group(0) @binding(0) var<storage, read_write> input_nhwc: array<f16>;
@group(0) @binding(1) var<storage, read_write> pointwise_weight_oik: array<f16>;
@group(0) @binding(2) var<storage, read_write> pointwise_bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> residual_ncl: array<f16>;
@group(0) @binding(4) var<storage, read_write> alpha: array<f16>;
@group(0) @binding(5) var<storage, read_write> head_weight_oik: array<f16>;
@group(0) @binding(6) var<storage, read_write> head_bias: array<f16>;
@group(0) @binding(7) var<storage, read_write> output_ncl: array<f16>;

const CHANNELS: u32 = 96u;
const TIME: u32 = {{ time }}u;
const TIME_TILE: u32 = 64u;
const PADDING: i32 = 3;
const HALO: u32 = 70u;
const HALO_ELEMENTS: u32 = 6720u;
const POINTWISE_WEIGHT_ELEMENTS: u32 = 9216u;
const HEAD_WEIGHT_ELEMENTS: u32 = 672u;
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> input_tile: array<f16, 6720>;
var<workgroup> pointwise_weight_all: array<f16, 9216>;
var<workgroup> pointwise_bias_all: array<f16, 96>;
var<workgroup> activated_tile: array<f16, 6720>;
var<workgroup> alpha_all: array<f16, 96>;
var<workgroup> head_weight_all: array<f16, 672>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TIME_TILE;

    var index = local_index;
    loop {
        if (index >= POINTWISE_WEIGHT_ELEMENTS) {
            break;
        }
        pointwise_weight_all[index] = pointwise_weight_oik[index];
        index += WORKGROUP_SIZE;
    }
    index = local_index;
    loop {
        if (index >= HEAD_WEIGHT_ELEMENTS) {
            break;
        }
        head_weight_all[index] = head_weight_oik[index];
        index += WORKGROUP_SIZE;
    }
    if (local_index < CHANNELS) {
        pointwise_bias_all[local_index] = pointwise_bias[local_index];
        alpha_all[local_index] = alpha[local_index];
    }

    index = local_index;
    loop {
        if (index >= HALO_ELEMENTS) {
            break;
        }
        let halo_time = index / CHANNELS;
        let channel = index - halo_time * CHANNELS;
        let source_time = i32(time_base + halo_time) - PADDING;
        var value = f16(0.0);
        if (source_time >= 0 && source_time < i32(TIME)) {
            value = input_nhwc[u32(source_time) * CHANNELS + channel];
        }
        input_tile[index] = value;
        index += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    index = local_index;
    loop {
        if (index >= HALO_ELEMENTS) {
            break;
        }
        let halo_time = index / CHANNELS;
        let output_channel = index - halo_time * CHANNELS;
        let source_time = i32(time_base + halo_time) - PADDING;
        var activated = f16(0.0);
        if (source_time >= 0 && source_time < i32(TIME)) {
            var accumulator = f32(pointwise_bias_all[output_channel]);
            let weight_base = output_channel * CHANNELS;
            let input_base = halo_time * CHANNELS;
            var input_channel = 0u;
            loop {
                if (input_channel >= CHANNELS) {
                    break;
                }
                accumulator = fma(
                    f32(input_tile[input_base + input_channel]),
                    f32(pointwise_weight_all[weight_base + input_channel]),
                    accumulator,
                );
                input_channel += 1u;
            }
            accumulator += f32(
                residual_ncl[output_channel * TIME + u32(source_time)],
            );
            let raw = f16(accumulator);
            let x = f32(raw);
            let a = f32(alpha_all[output_channel]);
            let sine = sin(a * x);
            activated = f16(x + (sine * sine) / (a + 1e-9));
        }
        activated_tile[index] = activated;
        index += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    if (local_index < TIME_TILE) {
        var accumulator = f32(head_bias[0u]);
        var input_channel = 0u;
        loop {
            if (input_channel >= CHANNELS) {
                break;
            }
            let weight_base = input_channel * 7u;
            var kernel_index = 0u;
            loop {
                if (kernel_index >= 7u) {
                    break;
                }
                accumulator = fma(
                    f32(activated_tile[(local_index + kernel_index) * CHANNELS + input_channel]),
                    f32(head_weight_all[weight_base + kernel_index]),
                    accumulator,
                );
                kernel_index += 1u;
            }
            input_channel += 1u;
        }
        output_ncl[time_base + local_index] = f16(tanh(accumulator));
    }
}
