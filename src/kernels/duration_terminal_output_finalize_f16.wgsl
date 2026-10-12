enable f16;

// Released compact duration path: fold the final block residual update into
// RMSNorm, scalar projection, Softplus, token sum, and log1p. This preserves
// the existing fma and reduction order while avoiding a [S, 1024] write/read.

@group(0) @binding(0) var<storage, read_write> residual: array<f16>;
@group(0) @binding(1) var<storage, read_write> branch: array<f16>;
@group(0) @binding(2) var<storage, read_write> gate: array<f16>;
@group(0) @binding(3) var<storage, read_write> norm_weight: array<f16>;
@group(0) @binding(4) var<storage, read_write> output_weight: array<f16>;
@group(0) @binding(5) var<storage, read_write> output_bias: array<f16>;
@group(0) @binding(6) var<storage, read_write> output: array<f16>;

const DIM: u32 = 1024u;
const SEQUENCE: u32 = {{ sequence }}u;
const LANES: u32 = 4u;

var<workgroup> square_parts: array<f32, 256>;
var<workgroup> dot_parts: array<f32, 256>;
var<workgroup> token_frames: array<f32, 64>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let token = tid / LANES;
    let lane = tid - token * LANES;
    var square_sum = 0.0;
    var weighted_sum = 0.0;
    if (token < SEQUENCE) {
        let row = token * DIM;
        for (var feature = lane; feature < DIM; feature = feature + LANES) {
            let index = row + feature;
            let value = fma(f32(gate[feature]), f32(branch[index]), f32(residual[index]));
            square_sum = fma(value, value, square_sum);
            weighted_sum = fma(
                value,
                f32(norm_weight[feature]) * f32(output_weight[feature]),
                weighted_sum,
            );
        }
    }
    square_parts[tid] = square_sum;
    dot_parts[tid] = weighted_sum;
    workgroupBarrier();

    if (lane == 0u && token < SEQUENCE) {
        let base = token * LANES;
        let sum_square = square_parts[base]
            + square_parts[base + 1u]
            + square_parts[base + 2u]
            + square_parts[base + 3u];
        let sum_weighted = dot_parts[base]
            + dot_parts[base + 1u]
            + dot_parts[base + 2u]
            + dot_parts[base + 3u];
        let inv_rms = inverseSqrt(sum_square / f32(DIM) + {{ eps }});
        let logit = sum_weighted * inv_rms + f32(output_bias[0]);
        var frame = logit;
        if (logit <= 20.0) {
            frame = log(1.0 + exp(logit));
        }
        token_frames[token] = frame;
    }
    workgroupBarrier();

    if (tid < 64u) {
        square_parts[tid] = select(0.0, token_frames[tid], tid < SEQUENCE);
    }
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (tid < stride) {
            square_parts[tid] = square_parts[tid] + square_parts[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        output[0] = f16(log(1.0 + max(square_parts[0], 0.0)));
    }
}
