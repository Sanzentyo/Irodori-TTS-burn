// Production released-WmHead fast path: Snake1d + Conv1d(96 -> 1, k=7) + tanh.
//
// One 240-invocation workgroup owns 240 consecutive output samples. Six Cin=16
// reduction tiles stage a 246-sample activated halo, so every valid input is
// evaluated by the exact production Snake expression before the ascending
// channel/kernel f32 FMA reduction.
//
// SourceKernel buffers are uniformly read_write because CubeCL's sliced
// allocator can bind otherwise-disjoint logical tensors into one physical
// storage buffer.

@group(0) @binding(0) var<storage, read_write> input_ncl: array<f32>;
@group(0) @binding(1) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> weight_oik: array<f32>;
@group(0) @binding(3) var<storage, read_write> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_ncl: array<f32>;

const CHANNELS: u32 = 96u;
const TIME: u32 = {{ time }}u;
const KERNEL_SIZE: u32 = 7u;
const PADDING: i32 = 3;
const TIME_TILE: u32 = 240u;
const INPUT_CHANNEL_TILE: u32 = 16u;
const INPUT_SPAN: u32 = 246u;
const INPUT_TILE_ELEMENTS: u32 = 3936u;
const WEIGHT_ELEMENTS: u32 = 672u;

var<workgroup> activated_tile: array<f32, 3936>;
var<workgroup> weight_all: array<f32, 672>;
var<workgroup> alpha_all: array<f32, 96>;

@compute @workgroup_size(240, 1, 1)
fn main(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let time_base = group_id.x * TIME_TILE;
    let output_time = time_base + local_index;
    var accumulator = bias[0u];

    if (local_index < CHANNELS) {
        alpha_all[local_index] = alpha[local_index];
    }
    var preload_weight_index = local_index;
    loop {
        if (preload_weight_index >= WEIGHT_ELEMENTS) {
            break;
        }
        weight_all[preload_weight_index] = weight_oik[preload_weight_index];
        preload_weight_index += TIME_TILE;
    }
    workgroupBarrier();

    var input_channel_base = 0u;
    loop {
        if (input_channel_base >= CHANNELS) {
            break;
        }

        var tile_index = local_index;
        loop {
            if (tile_index >= INPUT_TILE_ELEMENTS) {
                break;
            }
            let tile_channel = tile_index / INPUT_SPAN;
            let tile_time = tile_index - tile_channel * INPUT_SPAN;
            let source_time = i32(time_base + tile_time) - PADDING;
            var activated = 0.0;
            if (source_time >= 0 && source_time < i32(TIME)) {
                let input_channel = input_channel_base + tile_channel;
                let x = input_ncl[input_channel * TIME + u32(source_time)];
                let a = alpha_all[input_channel];
                let sine = sin(a * x);
                activated = x + (sine * sine) / (a + 1e-9);
            }
            activated_tile[tile_index] = activated;
            tile_index += TIME_TILE;
        }
        workgroupBarrier();

        var tile_channel = 0u;
        loop {
            if (tile_channel >= INPUT_CHANNEL_TILE) {
                break;
            }
            let input_base = tile_channel * INPUT_SPAN + local_index;
            let weight_base = (input_channel_base + tile_channel) * KERNEL_SIZE;
            var kernel_index = 0u;
            loop {
                if (kernel_index >= KERNEL_SIZE) {
                    break;
                }
                accumulator = fma(
                    activated_tile[input_base + kernel_index],
                    weight_all[weight_base + kernel_index],
                    accumulator,
                );
                kernel_index += 1u;
            }
            tile_channel += 1u;
        }
        workgroupBarrier();
        input_channel_base += INPUT_CHANNEL_TILE;
    }

    output_ncl[output_time] = tanh(accumulator);
}
