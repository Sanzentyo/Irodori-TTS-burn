// Fused activation and w2 projection for the released duration predictor.

@group(0) @binding(0) var<storage, read_write> projected: array<f32>;
@group(0) @binding(1) var<storage, read_write> w2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SEQUENCE: u32 = {{ sequence }}u;
const DIM: u32 = 1024u;
const TILE_ROWS: u32 = 16u;
const TILE_OUTPUTS: u32 = 32u;
const TILE_K: u32 = 32u;

var<workgroup> activated_tile: array<f32, 512>;
var<workgroup> weight_tile: array<f32, 1024>;

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let output_channel = group_id.x * TILE_OUTPUTS + local_id.x;
    let row_base = group_id.y * TILE_ROWS + local_id.y;
    let flat_tid = local_id.y * TILE_OUTPUTS + local_id.x;
    var acc: array<f32, 2>;
    for (var row_slot = 0u; row_slot < 2u; row_slot = row_slot + 1u) {
        acc[row_slot] = 0.0;
    }

    for (var k_base = 0u; k_base < DIM; k_base = k_base + TILE_K) {
        for (var load = flat_tid; load < TILE_ROWS * TILE_K; load = load + 256u) {
            let activation_row = load / TILE_K;
            let activation_k = load % TILE_K;
            var activated = 0.0;
            if (group_id.y * TILE_ROWS + activation_row < SEQUENCE) {
                let source_row = (group_id.y * TILE_ROWS + activation_row) * (DIM * 2u);
                let gate = projected[source_row + k_base + activation_k];
                let value = projected[source_row + DIM + k_base + activation_k];
                activated = (gate / (1.0 + exp(-gate))) * value;
            }
            activated_tile[load] = activated;
        }

        for (var load = flat_tid; load < TILE_K * TILE_OUTPUTS; load = load + 256u) {
            let local_k = load / TILE_OUTPUTS;
            let local_output = load % TILE_OUTPUTS;
            let global_output = group_id.x * TILE_OUTPUTS + local_output;
            weight_tile[load] = w2[(k_base + local_k) * DIM + global_output];
        }
        workgroupBarrier();

        if (output_channel < DIM) {
            for (var row_slot = 0u; row_slot < 2u; row_slot = row_slot + 1u) {
                let row = row_base + row_slot * 8u;
                if (row < SEQUENCE) {
                    for (var local_k = 0u; local_k < TILE_K; local_k = local_k + 1u) {
                        acc[row_slot] = fma(
                            activated_tile[(local_id.y + row_slot * 8u) * TILE_K + local_k],
                            weight_tile[local_k * TILE_OUTPUTS + local_id.x],
                            acc[row_slot],
                        );
                    }
                }
            }
        }
        workgroupBarrier();
    }
    for (var row_slot = 0u; row_slot < 2u; row_slot = row_slot + 1u) {
        let row = row_base + row_slot * 8u;
        if (row < SEQUENCE && output_channel < DIM) {
            output[row * DIM + output_channel] = acc[row_slot];
        }
    }
}
