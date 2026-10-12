# WGSLPack wire format, version 1

All integers and scalar payload words are little-endian. Offsets are absolute byte
offsets unless stated otherwise. A file is exactly `total_len` bytes; a prefix
reader supplies that independently observed length to `Index::parse`.

## Header (64 bytes)

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | ASCII `WGSLPK01` |
| 8 | u32 | Version, exactly 1 |
| 12 | u32 | Descriptor size, exactly 96 |
| 16 | u32 | Tensor count |
| 20 | u32 | Names blob length |
| 24 | u32 | Opaque metadata length |
| 28 | u32 | Reserved, zero |
| 32 | u64 | Prefix length / first payload offset |
| 40 | u64 | Total archive length |
| 48 | 16 bytes | Reserved, zero |

Descriptors follow the header, then the names blob, opaque metadata and zero
padding to the next multiple of 256. `prefix_len` must equal that rounded size.
An empty archive still contains a 256-byte prefix.

## Descriptor (96 bytes)

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | u32 | Dtype: 1=F32, 2=F16, 3=BF16, 4=I32, 5=U32 |
| 4 | u32 | Rank, 0 through 8 |
| 8 | u32 | Name offset relative to names blob |
| 12 | u32 | Name byte length |
| 16 | u64 | Payload offset |
| 24 | u64 | Logical payload byte length |
| 32 | 8 × u32 | Shape; unused axes zero |
| 64 | 8 × u32 | Element strides; unused axes zero |

The WGSL representation encodes each u64 as `vec2<u32>` (low word first).
Its storage alignment is 8 and array stride is 96 under the
[WGSL memory layout rules](https://www.w3.org/TR/WGSL/#memory-layout).
Rust reads fields with little-endian decoding; it does not cast native structs.

Names are nonempty UTF-8, have no NUL, are tightly concatenated, and appear in
strict bytewise lexicographic order. Payloads follow descriptor order and each
starts at the next 256-byte boundary. Overlaps, unaccounted holes, unknown dtypes,
noncanonical strides and surplus trailing bytes are rejected.

Axes are nonzero and row-major contiguous; the last active stride is one.
The element count must fit u32. A scalar has rank zero, one element and all-zero
shape/stride slots. Zero-sized tensors are not represented in v1.
Payload length equals element count times scalar size (two bytes for F16/BF16,
four for the other types). The final payload is padded to 256 bytes too.

Payload padding is nonsemantic and readers do not scan it. Writers should emit
zeros. `storage_range()` rounds the logical end up to four bytes, staying inside
that padding, for word-based FP16/BF16 shaders. Only logical elements may be read.

## Reader contract

`Header::parse` validates the fixed header only. `Index::parse` validates the
complete prefix and every descriptor against the supplied total length.
`Archive::parse` additionally binds those validated ranges to the actual slice.
All borrowed views stay valid only while their input allocation is unchanged and
alive, enforced by Rust lifetimes. Prefix readers must keep their external payload
source consistent with the index they validated.

No compression, implicit dtype conversion, model naming convention, authentication,
file system access, allocator policy or GPU runtime belongs to this format.
Applications must bound prefix/payload allocations before performing I/O, and
validate their own opaque metadata version before interpreting model semantics.
