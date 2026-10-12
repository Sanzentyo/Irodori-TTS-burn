// WGSLPack v1: exactly 96 bytes, aligned to 8 bytes in storage buffers.
// Bind each selected payload range separately; offsets refer to the archive,
// not a promise that an entire multi-gigabyte file fits one GPU buffer.
struct WgslPackTensor {
    dtype: u32,
    rank: u32,
    name_offset: u32,
    name_length: u32,
    data_offset: vec2<u32>,
    data_length: vec2<u32>,
    shape: array<u32, 8>,
    strides: array<u32, 8>,
}

// A scalar decoder for word-addressed payloads; no shader-f16 feature required.
// F32/I32/U32 use one word per element; F16/BF16 use two elements per word.
fn wgslpack_float(word: u32, dtype: u32, half_index: u32) -> f32 {
    switch dtype {
        case 1u: { return bitcast<f32>(word); }
        case 2u: { return unpack2x16float(word)[half_index & 1u]; }
        case 3u: { return bitcast<f32>(((word >> ((half_index & 1u) * 16u)) & 65535u) << 16u); }
        case 4u: { return f32(bitcast<i32>(word)); }
        case 5u: { return f32(word); }
        default: { return 0.0; } // Host validation rejects unknown dtypes.
    }
}
