# Optimization preparation without new measurements

This is the historical preparation record. The user subsequently authorized
device measurements; see the [acceptance campaign](../benchmarks/optimization-acceptance-2026-10-14.md)
for measured results and the later review. The pending items below describe the
state at the end of preparation.

The October 14 request limits this change to work that can proceed without new
performance measurements. No GPU tests, model inference, tuning sweeps or new
accuracy campaigns were run. CPU correctness tests and compilation are distinct
from device acceptance. Default features, precision and kernel choices remain
unchanged. No additional unsafe code or dependency version was introduced.

## Status of each candidate

| Candidate | Implemented preparation | Acceptance still required |
|---|---|---|
| Generated SPIR-V persistence | Independent `spirv-cache` feature and `CUBECL_WGPU_SPIRV_CACHE=0\|1` startup switch | Fresh-process 2×2 shader/driver-cache comparison, lifecycle/concurrency and Intel DB stability |
| Intel RF/codec bottlenecks | `validate_synthesis --device-profile` collects RF stages and the exact production codec path; reports timing source and verifies output equality | RTX/Intel stage diagnosis, then targeted route/kernel experiments |
| Ordinary synthesis graph reuse | `StreamingDacVaeDecoder`, bounded LRU, `SynthesisEngine::with_codec_graph_cache`, opt-in CLI | Same-input PCM parity, cold capture, warm replay, cancellation, eviction and memory peaks on both devices |
| Long-text/FP16 diagnosis | Checksummed stage manifest and CPU-only comparison tool | Capture matched long/short RF/MF fixtures; locate error before deciding which arithmetic needs FP32 |
| Packed codec loading | CPU FP32 converter, versioned schema, file/borrowed/owned loaders, prepared WGSL layouts and existing voice identity | Same-output GPU checks and complete load/first-audio/memory comparison |

The profiling entry point is diagnostic. The extra request is separate from the
ordinary latency loop, its receipt has `diagnostic_only: true` and
`latency_results_valid: false`, and hardware timestamps are distinguished from
synchronized host timing. The codec receipt groups projection and the full
cross-block-fused decoder: independent per-block timing would describe a
different execution graph. Existing detailed profiling tools remain available
for isolated candidates. No speculative kernel tile or mixed-precision default
was selected without the requested evidence.

## Cache and graph contracts

`spirv-cache` enables upstream `cubecl-wgpu/persistence`; the compilation-cache
configuration must also be enabled. The selected backend must support the
SPIR-V compiler. On macOS that requires the separate `vulkan-spirv` feature and
Vulkan SDK; ordinary Metal is unaffected. Set both cache switches before device
creation. Invalid SPIR-V switch values disable that store with a diagnostic.
Both layers use the existing CubeCL environment and its invalidation rules.
No cache database or fallback path was duplicated.

The earlier native-OFF/SPIR-V-ON prototype hit a Turso shared-reader-slot
assertion on Intel. Its cause remains unresolved; adding an independent switch
does not fix it. The published [native-only results](../benchmarks/native-pipeline-cache-2026-10-14.md)
cannot establish this new combination's performance or reliability.

Graph reuse requires both `streaming-graphs` and explicit activation. Immutable
decoder weights and their graphs have one owner; graphs drop before the weights.
Mutable replay produces owned PCM, preventing output storage from escaping to
the next request. Keys include input frame count and the exact output crop.
Entry count uses LRU eviction before capture; frame limits are validated for
the entire request before any audio is emitted. CubeCL environment-generation
changes clear the cache. Geometry/count limits are not a hard VRAM byte budget.
Capture/replay failures propagate without silently switching execution routes.
The ordinary decoder and graph decoder share window planning and PCM validation.

## Codec archive contract

`irodori-dacvae-f32-v1` is an application schema inside the unchanged standalone
WGSLPack wire format. It contains only production decoder tensors, optional
biases and three existing prepared layouts:

- ConvTranspose `[Cin, Cout, 2*stride]` to `[phase, Cout, Cin, tap]`.
- Pointwise OIK to contiguous `[1, Cin, Cout]`.
- C96/C192/C384 k7 vectors with output lanes `[lane, lane+8, lane+16, lane+24]`.

Conversion copies raw 32-bit words, preserving signed zero and payload bits;
it neither creates a device nor performs floating-point arithmetic. Source
dtype and topology are validated, and a same-directory temporary file is
published without overwriting an existing destination. Prepared data increases
file size in exchange for removing repeated layout preparation. Runtime-selected
ConvTranspose copies for blocks 1–3 are uploaded only when that codec route is
enabled. Required names, shapes, dtype and metadata are validated before upload.
The existing grouped uploader and bounded file staging are reused. Shared-owned
memory loading avoids per-tensor CPU copies, not GPU allocation or driver staging.

The original decoder digest is recorded by the converter so saved voices retain
their identity across formats. Like the packed model's source hash, this is a
declared identity, not authentication of payloads. Packed decoder archives do
not contain an encoder; requests to initialize one are rejected explicitly.
No format-crate API/version, submodule or CI was changed.

## Precision diagnosis from saved stages

Build `validate_synthesis` with `cli,synthesis` and use `--dump-stages` during a
future authorized GPU campaign. The manifest records input noise, text state,
each sampler forward, final latent and audio, with shapes and SHA-256 hashes.
It also binds to the input fixture and codec file. Traced and ordinary final
latents must match bit for bit before the manifest is published.

```sh
uv run scripts/compare_synthesis_stages.py REFERENCE_DIR CANDIDATE_DIR \
  --atol 0 --rtol 0
```

The reader rejects mismatched checkpoint/fixture/codec/sampler contracts,
different stage order or shapes, truncated/corrupted payloads and nonfinite
values. It reports exact equality, absolute error, RMSE, SNR and the first
observed tolerance failure. That first observed stage is not proof of cause.
This is not an accuracy gate, and defaults of zero express exact comparison,
not an approved FP16 tolerance. Only reports with this manifest are accepted.

## Future device acceptance

Build and run the ignored `codec_preparation_gpu` integration test explicitly
with `streaming-graphs,packed-export`, `IRODORI_CODEC_TEST_CHECKPOINT` set to the
original FP32 codec, and optionally `IRODORI_PACK_TEST_INTEGRATED=1` for Intel.
It compares file/borrowed/owned packed decoding and bounded graph replay with
ordinary PCM, changes noise/geometry/crops, forces eviction, checks cancellation
and rejects oversized windows before emission. It was compiled, not executed.

Keep shader-cache OFF/ON and driver-cache OFF/ON independently selectable in the
same binary. Freeze adapter, driver, checkpoint, fixture, precision, routes and
environment for each comparison. Include cold conversion/load/capture, fresh
process restoration, warm requests and persistent-cache size/peak memory.
FP32 and FP16 model comparisons keep the codec FP32. Profile diagnostics must
not substitute for uninstrumented latency. Promote defaults only after parity,
memory and repeated performance results support doing so.

CPU test selection, feature checks and review findings are recorded in the
[review note](../review/optimization-preparation-2026-10-14.md).
