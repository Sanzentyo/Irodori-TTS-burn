# WGSLPack

A small, Sans-I/O tensor archive for borrowed Rust access and WGSL storage buffers.
The core has **no dependencies**, uses `no_std`, forbids unsafe code, and never
opens files, allocates GPU resources or copies tensor payloads.

* `Archive<'a>` validates and borrows a complete `&[u8]`.
* `Header` and `Index<'a>` support prefix/range readers without loading payloads.
* `Layout` (the default `alloc` feature) creates a deterministic prefix and ranges.
  The application supplies the bytes, I/O, atomic publication and integrity policy.
* `WGSL` contains the matching 96-byte tensor descriptor and scalar decoders.

FP32, FP16, BF16, I32 and U32 have explicit little-endian representations. Shapes
have up to eight dimensions; payloads are contiguous and begin at 256-byte offsets.
Names are sorted, unique UTF-8; lookup is binary search with no allocated map.

```rust
use wgslpack::{Archive, Dtype, Layout, TensorSpec};

let layout = Layout::new(&[
    TensorSpec { name: "weight", dtype: Dtype::F32, shape: &[2] },
], b"application metadata")?;
let mut bytes = vec![0; layout.total_len() as usize];
bytes[..layout.prefix().len()].copy_from_slice(layout.prefix());
let range = layout.index().get("weight").unwrap().data_range();
bytes[range.start as usize..range.end as usize]
    .copy_from_slice(&[1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat());

let archive = Archive::parse(&bytes)?;
assert_eq!(archive.index().get("weight").unwrap().shape().collect::<Vec<_>>(), [2]);
// This slice borrows the original allocation; nothing materializes the weights.
let payload = archive.data(0).unwrap();
assert_eq!(f32::from_le_bytes(payload[..4].try_into().unwrap()), 1.0);
# Ok::<(), wgslpack::Error>(())
```

For an allocation-free reader, disable default features. Typed Rust slices can
require additional pointer-alignment and host-endianness checks; the API returns
bytes so unaligned inputs remain safe. Decoding little-endian scalars works on
either host byte order.

For WGSL, upload `Index::descriptor_bytes()` directly to an
`array<WgslPackTensor>` storage binding. Upload each selected `Archive::storage(i)`
payload as a separate binding; its four-byte padded length permits `array<u32>`
access even for an odd count of FP16 values. Concatenate `wgslpack::WGSL` with the
application shader. Descriptor offsets refer to the archive. They do not require
one giant GPU buffer or imply that every device accepts every tensor size.
Check binding limits and offset alignment against the actual device.

Parsing validates structure and ranges, not payload integrity or floating-point
finiteness. Payload padding is ignored. Writers should zero it for reproducible
artifacts. Applications own hashes, signatures, size limits and model semantics.
GPU staging/upload copies depend on the backend; zero-copy parsing does not mean
zero-copy transfer to a discrete GPU.

See [SPEC.md](SPEC.md) for the versioned wire format. This is an initial 0.1 format;
consumers must validate both the wire version and their own metadata schema.

```sh
cargo nextest run --no-tests fail
cargo check --no-default-features
cargo test --doc
cargo clippy --all-targets -- -D warnings
```

The Irodori integration and GPU/load benchmarks live in the separate
[Irodori-TTS-burn project](https://github.com/Sanzentyo/Irodori-TTS-burn).

License: MIT.
