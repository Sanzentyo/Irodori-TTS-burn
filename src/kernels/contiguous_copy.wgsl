// Bit-preserving 16-byte GPU copy for stable graph inputs.

@group(0) @binding(0) var<storage, read> input: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<u32>>;

@compute @workgroup_size({{ workgroup_size }}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < {{ vectors }}u) {
        output[index] = input[index];
    }
}

