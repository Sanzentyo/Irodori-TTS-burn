// Long-sequence duration SwiGLU and w2 projection with the block residual
// epilogue in the final store. K reduction and residual fma order are unchanged.

@group(0) @binding(0) var<storage, read_write> projected: array<f32>;
@group(0) @binding(1) var<storage, read_write> w2: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> residual: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

const SEQUENCE: u32 = {{ sequence }}u;
const DIM: u32 = 1024u;
const TILE_ROWS: u32 = 16u;
const TILE_OUTPUTS: u32 = 64u;
const TILE_OUTPUT_VECS: u32 = TILE_OUTPUTS / 4u;
const TILE_K: u32 = 128u;
const DIM_VECS: u32 = DIM / 4u;

var<workgroup> activated_tile: array<f32, 2048>;
var<workgroup> weight_tile: array<vec4<f32>, 2048>;

@compute @workgroup_size(16, 8, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let output_vector = group_id.x * TILE_OUTPUT_VECS + local_id.x;
    let row_base = group_id.y * TILE_ROWS + local_id.y;
    let flat_tid = local_id.y * TILE_OUTPUT_VECS + local_id.x;
    var acc: array<vec4<f32>, 2>;
    for (var row_slot = 0u; row_slot < 2u; row_slot = row_slot + 1u) {
        acc[row_slot] = vec4<f32>(0.0);
    }

    for (var k_base = 0u; k_base < DIM; k_base = k_base + TILE_K) {
        for (var load = flat_tid; load < TILE_ROWS * TILE_K; load = load + 128u) {
            let activation_row = load / TILE_K;
            let activation_k = load % TILE_K;
            var activated = 0.0;
            if (group_id.y * TILE_ROWS + activation_row < SEQUENCE) {
                let source_row = (group_id.y * TILE_ROWS + activation_row) * (DIM * 2u);
                let gate_value = projected[source_row + k_base + activation_k];
                let value = projected[source_row + DIM + k_base + activation_k];
                activated = (gate_value / (1.0 + exp(-gate_value))) * value;
            }
            activated_tile[load] = activated;
        }

        for (var load = flat_tid; load < TILE_K * TILE_OUTPUT_VECS; load = load + 128u) {
            let local_k = load / TILE_OUTPUT_VECS;
            let local_output_vector = load % TILE_OUTPUT_VECS;
            let global_output_vector = group_id.x * TILE_OUTPUT_VECS + local_output_vector;
            weight_tile[load] = w2[(k_base + local_k) * DIM_VECS + global_output_vector];
        }
        workgroupBarrier();

        for (var row_slot = 0u; row_slot < 2u; row_slot = row_slot + 1u) {
            let row = row_base + row_slot * 8u;
            if (row < SEQUENCE) {
                for (var local_k = 0u; local_k < TILE_K; local_k = local_k + 1u) {
                    acc[row_slot] = fma(
                        vec4<f32>(activated_tile[(local_id.y + row_slot * 8u) * TILE_K + local_k]),
                        weight_tile[local_k * TILE_OUTPUT_VECS + local_id.x],
                        acc[row_slot],
                    );
                }
            }
        }
        workgroupBarrier();
    }
    for (var row_slot = 0u; row_slot < 2u; row_slot = row_slot + 1u) {
        let row = row_base + row_slot * 8u;
        if (row < SEQUENCE) {
            let index = row * DIM_VECS + output_vector;
            output[index] = fma(gate[output_vector], acc[row_slot], residual[index]);
        }
    }
}
