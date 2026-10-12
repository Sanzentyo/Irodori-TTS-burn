# Sans-I/O native checkpoint loading — 2026-10-13

WGSLPack model loading is accepted as an optional `packed-weights` feature.
It reduces measured MF model loading by 2.7–11.6× across the two adapters and
precisions below. FP32 remains the default. Conversion preserves the exact
parameters produced by the existing loader at each requested precision.

The format is independently available at
[Sanzentyo/wgslpack](https://github.com/Sanzentyo/wgslpack), commit `bb4bd67`.
`crates/wgslpack` is the same standalone source, included as a workspace member
and optional path dependency, without a submodule. It has no dependencies,
forbids unsafe code, and supports a `no_std`, allocation-free reader. Only the
layout writer needs the default `alloc` feature. The Irodori adapter owns file
I/O, model semantics and GPU uploads. Atomic conversion is separately gated by
`packed-export`; the only new package added by `packed-weights` is `wgslpack`.
The existing safetensors dependency already brings `tempfile` transitively;
the export feature declares its additional direct use explicitly.

## Final model-load measurements

Median milliseconds of four independent processes per cell, warm OS file cache.
The clock starts immediately before model loading and stops after `Device::sync`.
It includes file reads, index/config validation, model construction, parameter
installation and completed GPU uploads. Device creation is reported separately;
codec loading, kernel preparation, source hashing and synthesis are excluded.

| Adapter | Precision | Indexed safetensors | WGSLPack file | Speedup | Peak host RSS, old / packed MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5070 Ti Laptop | FP32 | 4741.257 | 947.280 | 5.0× | 1077 / 1114 |
| RTX 5070 Ti Laptop | FP16 | 5449.531 | 469.934 | 11.6× | 994 / 765 |
| Intel integrated | FP32 | 4893.399 | 1842.634 | 2.7× | 980 / 972 |
| Intel integrated | FP16 | 5269.392 | 962.326 | 5.5× | 763 / 567 |

The supplementary RF FP32 load comparison on RTX (two fresh processes per path)
measures 4732.962 → 969.324 ms, a 4.9× improvement. It uses the distinct v4.1
Small RF checkpoint and is not pooled with the MF timings above.

RSS is Linux `VmHWM`, sampled before validation readbacks; it is host process
memory, not device VRAM. The native files contain 717 tensors: 3,093,153,792 bytes
for FP32 and 1,546,631,168 for FP16. The prefix is 108,032 bytes. Parsing that
prefix 10,000 times averaged 8.928 microseconds per parse on this host.

The principal improvement comes from storing native names, transposed layout and
precision once. Binary header parsing alone does not explain these speedups.
The existing indexed safetensors loader already parses its header once per
materialization pass. Its remaining transposes and precision conversions occur
on every model load; the native path removes them. Conversion itself took
7.12 seconds for FP32 and 6.84 seconds for FP16, excluding device initialization.
The destination is published atomically and never overwritten.

## File and memory paths

The public adapter offers a bounded file reader, a borrowed `&[u8]` reader and
an owned `Bytes` reader. The last shares payload windows with the upload queue,
avoiding per-tensor host copies. Backend staging and device transfers still occur.

Each memory cell below is **load from existing memory / total including initial
file read** in milliseconds, medians of four processes. The borrowed form copies
each payload to satisfy the upload API's ownership; the owned form retains the
allocation until the upload consumes it.

| Adapter | Precision | Borrowed bytes | Shared owned bytes | Shared peak RSS MiB |
| --- | --- | ---: | ---: | ---: |
| RTX | FP32 | 953.480 / 1776.573 | 616.612 / 1443.809 | 3304 |
| RTX | FP16 | 502.267 / 929.586 | 337.447 / 770.009 | 1867 |
| Intel | FP32 | 1940.736 / 2770.496 | 1544.976 / 2374.290 | 3109 |
| Intel | FP16 | 956.414 / 1383.720 | 812.413 / 1243.051 | 1636 |

The file path remains the default: it overlaps per-tensor reads with uploads and
uses less host memory than reading the entire file first. The owned memory API
helps applications that already have the bytes. There is no runtime heuristic or
fallback chain choosing among these ownership contracts.

An initial 64-process campaign exposed a redundant allocation realignment in
`Bytes::from_bytes_vec`. The direct upload path never requires a typed CPU slice,
so it now retains the original byte allocation with `Bytes::from_elems`.
For example, RTX FP32 shared-memory loading fell from about 1.62 seconds to
0.62 seconds; peak host memory fell from about 6.1 GiB to 3.2 GiB. The final
campaign repeats all 64 cells after this change. Both campaigns retain their
raw results rather than mixing samples across implementations.

## Accuracy and structural validation

The benchmark reads every installed parameter back after timing, in canonical
name order, and hashes its name, shape, dtype and bytes. All four loading modes
produce the same digest on both adapters:

| Precision | All-parameter SHA-256 |
| --- | --- |
| FP32 | `b206bc88637c13b29a4c96157934ce377012349a582b9b6077ee34263037c21e` |
| FP16 | `2493d6579e81c748dbcddb9ea48ab138ab2045fc303353b9fa0e1615cd9d6335` |

This verifies bit preservation relative to the existing precision path, including
FP16 rounding. It does not claim that FP16 equals the FP32 reference.

Two-second MF synthesis also produces bit-identical latent and audio files through
the old and new loaders in all four adapter/precision combinations. The unchanged
FP32-oracle audio SNRs are 93.738 / 31.188 dB on RTX and 96.156 / 43.042 dB on
Intel (FP32 / FP16). A separate four-second caption-conditioned RF case on RTX
FP32 likewise preserves every parameter, latent and audio bit, with 97.670 dB
audio SNR against its original oracle. These are fixed-input numerical checks,
not a new perceptual or intelligibility evaluation.

The independent parser tests cover every truncated prefix/payload length,
overlap, duplicate names, bad ranks/strides/dtypes, overflow, canonical ordering,
borrowed pointer identity and 10,000 deterministic corruptions without panics.
The adapter rejects incompatible model schemas and precision requests. It bounds
file-prefix allocation before reading, and preserves the exact-only loader's
duration-predictor omission.

A real WGSL test uploads the descriptors and payload bytes directly, checks all
five supported dtypes, array stride, shape/stride fields and 64-bit offset layout,
and passes on both GPUs. CubeCL can place disjoint bindings in one storage
allocation; the test declares matching storage access modes to respect WGPU's
buffer usage rules. The production parser itself is independent of CubeCL.

Cargo metadata and reverse dependencies identify the root crate as the sole
consumer of `wgslpack`. The selected parser, weights, inference, synthesis and
backend configuration slice passes all 63 tests. A separate real-model exact-only
test passes. Clippy passes with all relevant optional components, including
worker, HTTP, LoRA and profiling. Minimal builds pass both with and without
`packed-weights`; the feature-minimal root retains its existing dead-code
warnings. The standalone package, no-default-feature check and doctest pass.
Stable Rust and the declared Rust 1.85 MSRV passed in the completed
[validation run](https://github.com/Sanzentyo/wgslpack/actions/runs/38273190448).
The CI workflow has since been removed from both the public repository and the
workspace copy. Repeating FP16 conversion with the final export feature produces
a byte-identical archive (SHA-256
`ab3b29dc8e43a539cf0b2eb9004897c6acc3f873a92da03b7350cf643b4b95fd`).

## Reproduction and measurement limits

Hardware is the RTX 5070 Ti Laptop GPU (12 GiB, NVIDIA 595.99.02) and Intel ARL
integrated GPU (8086:7d67, Mesa 26.0.8). The host has 30 GiB usable RAM and runs
Linux 7.0.0-38-generic. Rust is 1.95.0; Burn is 0.22.0-pre.2,
CubeCL 0.11.0-pre.2 and WGPU 30.
The source MF checkpoint is Aratako/Irodori-TTS-v4.1-Small-MF, SHA-256
`a3f204b3ee06058f4e639a1af38408e7e3af5b6c75176897ecb8bfaaebb4fd54`.

All GPU work is serialized. Each repetition reverses the preceding mode order;
there are no builds during timing campaigns. Inputs are read once to warm the
OS cache. No privileged cache drop was used, so these are not cold-SSD results.
Fresh-process device initialization in the final run was roughly 0.35 seconds
on RTX and 0.10 seconds on Intel, with driver caches already warm; first startup
can cost more. Do not interpret model-load speedups as complete CLI speedups.

```sh
cargo build --release --features cli,synthesis,packed-export \
  --bin pack_checkpoint --bin bench_checkpoint_load --bin validate_synthesis

target/release/pack_checkpoint --checkpoint MODEL.safetensors \
  --output PACKED_DIR/mf-fp32.wgslpack --precision fp32 --device discrete:0
target/release/pack_checkpoint --checkpoint MODEL.safetensors \
  --output PACKED_DIR/mf-fp16.wgslpack --precision fp16 --device discrete:0

uv run scripts/bench_packed_loading.py --checkpoint MODEL.safetensors \
  --packed-dir PACKED_DIR --output NEW_CAMPAIGN_DIR --repeats 4

uv run scripts/validate_packed_loading.py --campaign NEW_CAMPAIGN_DIR/results.json \
  --fixture ORIGINAL_ORACLE.safetensors --codec CODEC.safetensors --output NEW_E2E_DIR
```

The validation script first requires equal full-parameter digests, then binds a
copy of the original oracle to the packed file's actual SHA. Every oracle tensor
byte is preserved and the original checkpoint SHA is retained as provenance.
The existing synthesis validator still checks the hash of the file it loads.

The [review follow-up](../review/2026-10-13-wgslpack.md) adds checkpoint-file
identities to the benchmark and requires them before oracle replay. New runs use
campaign schema 2. Historical results below remain the original measurements;
they cannot be passed to the stricter replay script without a fresh campaign.

Raw artifacts are in
`/home/sanzentyo/benchmark-artifacts/irodori-packed-loading-20261013`:
`campaign/` is the realignment experiment; `final-campaign/results.json` contains
all accepted load samples, memory peaks, parameter digests and the binary hash.
`prefix-parse.log`, `rdeps-tests-final.log`, `wgsl-{rtx,intel}.log`,
`exact-load.log` and `clippy-final.log` retain focused verification.
`export-rdeps-tests.log` and `clippy-before-commit.log` verify the final optional
export boundary. `repack-proof.json` records deterministic conversion, and
`dependency-audit.json` verifies the standalone source mirror and dependency
change. The measured loading code is unchanged by that feature split.
`e2e/results.json` and `rf-e2e/results.json` record the bit-identical synthesis
comparisons. [Curated samples and digests](wgslpack-loading-2026-10-13.json) are
checked into the repository so the numerical evidence is reviewable without
access to the local artifact directory.
