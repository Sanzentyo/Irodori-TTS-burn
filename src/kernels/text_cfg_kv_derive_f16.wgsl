enable f16;

// Exact-v4 text-only CFG packed K/V derivation.
//
// Input:  [K/V, B=1, T=3, H=20, Dh=64]
// Output: [K/V, B=2, T=3, H=20, Dh=64]
//
// Batch row zero is copied bit-for-bit. Batch row one is numeric zero. The
// launcher is reachable only after the sampler supplies its text-only host
// proof (conditioned first, zero-unconditional second, no active aux context).
//
// Both bindings are read_write because CubeCL may suballocate logically
// distinct tensors from one physical WGPU buffer.

@group(0) @binding(0) var<storage, read_write> conditional: array<f16>;
@group(0) @binding(1) var<storage, read_write> batched_cfg: array<f16>;

const PLANES: u32 = 2u;
const CFG_BATCH: u32 = 2u;
const CONTEXT: u32 = 3u;
const HEADS: u32 = 20u;
const HEAD_DIM: u32 = 64u;
const CONTEXT_ELEMENTS: u32 = CONTEXT * HEADS * HEAD_DIM;
const OUTPUT_PLANE_ELEMENTS: u32 = CFG_BATCH * CONTEXT_ELEMENTS;
const OUTPUT_ELEMENTS: u32 = PLANES * OUTPUT_PLANE_ELEMENTS;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    if (output_index >= OUTPUT_ELEMENTS) {
        return;
    }

    let plane = output_index / OUTPUT_PLANE_ELEMENTS;
    let within_plane = output_index % OUTPUT_PLANE_ELEMENTS;
    let batch = within_plane / CONTEXT_ELEMENTS;
    let context_index = within_plane % CONTEXT_ELEMENTS;

    if (batch == 0u) {
        batched_cfg[output_index] =
            f16(f32(conditional[plane * CONTEXT_ELEMENTS + context_index]));
    } else {
        batched_cfg[output_index] = f16(0.0);
    }
}
