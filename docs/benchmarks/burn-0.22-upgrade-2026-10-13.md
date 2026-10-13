# Burn 0.22 stable upgrade — 2026-10-13

Burn 0.22.0, CubeCL 0.11.0, CubeK 0.3.0 and tokenizers 0.23.2 replace
their previous pinned versions (the compute stack used `-pre.2`). Rust 1.95
is now the minimum. WGPU remains 30.0.0 in the lockfile so this campaign does
not also change the graphics API library. The production backend remains raw
WGPU with handwritten WGSL and generated native shaders.

The final candidate reduces RTX RF FP16 median latency by 8.08% and MF by
2–4% on these fixtures. RTX RF FP32 changes by 1.39%, which is inconclusive
with this sample size. All four FP32 cases meet the existing
`latent max_abs < 1e-3` and `waveform SNR >= 85 dB` acceptance gates.
Intel FP16 has a numerical regression against the fixed FP32 reference and
essentially no speed advantage over Intel FP32. It remains an explicit
experimental precision; use FP32 for the measured Intel device. This result
does not establish FP16 parity or perceptual speech quality.

## Reproduction and measurement boundary

The base is `ad11d15c0aed1c6841f7bde8f23fa1b3bd45dc1f` on `master`.
Saved old and new release executables, manifests, lockfiles, input hashes,
logs and raw float outputs are under
`/home/sanzentyo/benchmark-artifacts/irodori-burn-upgrade-20261013/`.
The [machine-readable receipts](data/burn-0.22-upgrade-2026-10-13.json)
retain per-request times, reference errors, input/executable SHA-256 and
the exact inference commands.

- NVIDIA GeForce RTX 5070 Ti Laptop GPU, driver 595.99.02, 12 GiB VRAM.
- Intel Graphics ARL, PCI 8086:7d67, Mesa 26.0.8-1ubuntu0.3, Vulkan.
- Rust/Cargo 1.95, release build, with `cli,synthesis,packed-weights,profile`
  for the final candidate.
- MF: v4.1 Small MeanFlow, two-second fixed-noise fixture, four steps.
- RF: v4.1 Small, four-second caption-conditioned fixture, 40 steps.
- Both use the earlier official CUDA FP32 oracle; the codec stays FP32.

Each inference process uses `validate_synthesis --autotune --warmups 2
--repeat 5`, with explicit device and precision. GPU jobs run serially and
compilation is excluded from timing runs. Medians end after CPU waveform
readback. They include condition encoding, sampling and decoding, and exclude
loading, tokenization and file output. `rf_seconds` ends after latent readback;
it is not a GPU timestamp or a pure transformer measurement.

The initial baseline was measured earlier on the same day. After the candidate
campaign, the saved original executable was rerun in `before-confirm`; that
rerun is the primary baseline below. The final candidate was measured after
the additional U32-mask and convolution repairs; the earlier `after` run is
retained as preliminary evidence. Both baselines and all samples are retained.
Differences near one percent should be treated as inconclusive.

The validation executable uses `RuntimeOptions::default()`: this changes from
the former SubSlices allocator to the new Adaptive allocator. These results
compare the complete old/new stacks, not an isolated compiler change. The
resident production runtime still defaults to ExclusivePages. New software
cache namespaces keep all old measurements intact and force independent tuning.

## Warm synthesis

Audio-complete milliseconds; negative changes mean less time. SNR is against
the fixed FP32 oracle, not against the old executable.

| Model / device / precision | Before ms | After ms | Change | Waveform SNR before → after dB |
|---|---:|---:|---:|---:|
| MF / RTX / FP32 | 79.114 | 77.443 | -2.11% | 93.738 → 93.738 |
| MF / RTX / FP16 | 53.881 | 51.593 | -4.25% | 31.188 → 31.188 |
| MF / Intel / FP32 | 729.805 | 699.917 | -4.10% | 96.156 → 99.190 |
| MF / Intel / FP16 | 720.004 | 689.413 | -4.25% | 43.042 → 39.136 |
| RF / RTX / FP32 | 724.276 | 714.191 | -1.39% | 97.670 → 98.193 |
| RF / RTX / FP16 | 355.986 | 327.214 | -8.08% | 31.980 → 31.980 |
| RF / Intel / FP32 | 8042.597 | 7865.134 | -2.21% | 101.556 → 90.937 |
| RF / Intel / FP16 | 8208.868 | 7841.101 | -4.48% | 43.960 → 33.610 |

RTX MF latent and audio buffers are bit-identical between executable versions
in both precisions; RTX RF FP16 is also bit-identical to its old FP16 output.
All eight candidate cases retain bit-identical streamed/full and trimmed/full
waveform prefixes. FP32 numerical changes on
Intel remain within the unchanged FP32 gates.

## Intel FP16 investigation

The Intel RF waveform RMSE changes from 0.000772 to 0.002541; maximum latent
absolute error changes from 0.01626 to 0.09291. A single forward's SNR stays
near 59 dB, so single-operation validation alone is insufficient to approve a
40-step trajectory. Waveform SNR is a numerical agreement metric; no listening,
MOS, or ASR-based intelligibility assessment was performed here.

Two controls were run on the preliminary candidate, before the U32-mask repair,
without modifying the user's tuning database:

1. A private SQLite backup selected the previously used WGSL winners from the
   new stack's already validated candidates. MF attention changed from
   `q32-kv16` back to `q32-kv8`; the separate RF 100-row SwiGLU changed from
   `vector-k16` back to `prefetch-k16`. MF SNR rose to 37.694 dB, still below
   43.042. RF output was unchanged at 29.638 dB. The main RF CFG routes already
   used the same WGSL winners in both versions.
2. The new MF portable graph produced 39.889 dB, taking about 964 ms instead
   of 700 ms. It was not installed as an automatic fallback.

The final U32-mask policy makes the previously excluded custom attention
candidates available and invalidates the preliminary selections. The final MF
and RF SNRs are 39.136 and 33.610 dB; both remain below the old FP16 baseline.
The preliminary controls show that route selection can change the numerical
result, but do not isolate the final regression to a particular operation.
The precise operation responsible for the remaining difference is
not yet identified. No acceptance tolerance was relaxed, and no FP32 path
was changed to half arithmetic. Further FP16 acceptance requires a per-layer
comparison and repeated full-trajectory validation across lengths and voices.

## Checkpoint loading

Two fresh processes per cell, alternating source/packed order, warm OS page
cache. `ready_ms` includes model construction, file reads and completed GPU
uploads, but excludes device initialization, validation readback, codec and
synthesis. The first process for each path also verifies every parameter after
the timing boundary. This small sample establishes the observed difference,
not a broad startup-latency distribution. These loader measurements use the
earlier stable candidate, before the mask and convolution repairs; those
repairs do not change loading. Its separate executable SHA-256 is retained.

| Device | Precision | Format | Before ms | After ms |
|---|---|---|---:|---:|
| RTX | FP32 | indexed safetensors | 5011.443 | 5130.983 |
| RTX | FP32 | WGSLPack file | 962.994 | 978.997 |
| RTX | FP16 | indexed safetensors | 5356.940 | 5466.949 |
| RTX | FP16 | WGSLPack file | 478.981 | 487.764 |
| Intel | FP32 | indexed safetensors | 4907.424 | 4761.457 |
| Intel | FP32 | WGSLPack file | 1916.539 | 1291.594 |
| Intel | FP16 | indexed safetensors | 5285.972 | 5298.737 |
| Intel | FP16 | WGSLPack file | 984.287 | 629.095 |

All before/after, source/packed and RTX/Intel MF parameter digests match:

- FP32: `b206bc88637c13b29a4c96157934ce377012349a582b9b6077ee34263037c21e`
- FP16: `2493d6579e81c748dbcddb9ea48ab138ab2045fc303353b9fa0e1615cd9d6335`

The distinct RF FP32 packed checkpoint also retains its earlier canonical
parameter digest, `7af2d366ebf0d35e79c523d883b3614b6d9b4b7b3118c50f9b1ce51bad2ecaa3`.
That supplementary check is a compatibility check, not a paired timing result.

The converter's existing `irodori-burn-v1` files remain valid; this upgrade
does not require repacking. Use `scripts/bench_packed_loading.py --repeats 2
--modes safetensors file` with each saved `--binary`, the pinned MF checkpoint
and the existing packed directory to repeat this comparison.

## Integration changes

- Migrated the runtime-erased tensor, client, kernel, graph and tuning APIs.
- Pinned U32 mask storage at device configuration so handwritten WGSL attention
  remains eligible. FP32 v7 / FP16 v8 cache namespaces invalidate older choices.
- Corrected convolution bias shared-memory accounting and the scalar k7 fan-out
  view that otherwise emits an invalid SPIR-V workgroup pointer bitcast.
- Ported the existing convolution/Snake and matmul epilogue patches onto the
  pinned stable sources, retaining the Intel subgroup correctness checks.
- Fixed AutoCompiler handling of explicitly tagged WGSL source. Each WGSL
  pipeline uses its language-specific compiler; generated IR keeps the native
  device compiler. Mixed graph replay has a regression test.
- Replaced the local WGPU capture implementation with upstream graph capture.
- Kept deferred indexed loading and native WGSLPack loading; new tensor-byte
  constructors validate their metadata.
- Moved the optional Fusion provider to an independent experimental workspace.
  Burn's new global Fusion feature otherwise replaces the production dispatch
  backend through Cargo feature unification.

The former `fusion-provider` root feature is replaced by
`experiments/fusion-provider`. Allocator tools call the new alternative
`adaptive`; CubeCL no longer provides the old `sub-slices` preset. Existing
WGSLPack files are validated by full parameter digests in the load comparison.

## Next measurements

The two workloads have different bottlenecks. On Intel, MF spends roughly
302 ms through latent readback and another 398 ms through audio; RF spends
about 7.05 seconds through latent readback and 0.82 seconds decoding/readback.
Optimize the codec for short MF and the transformer/conditioning paths for RF,
using device timestamps before adding candidates. These differences of medians
are only stage estimates, not isolated kernel measurements.

The existing PyTorch XPU environment already works with explicit math SDPA.
Repeat its matched fixtures after any kernel change instead of adding Python
dependencies to the Rust library. Other worthwhile experiments are longer
capture/replay regions on RTX, persistent handwritten-WGSL pipeline compilation
where WGPU supports it, and owned-buffer upload batching for callers that
already hold WGSLPack bytes. Accept them only with end-to-end timing, restart
cache validation, numerical gates and a measured memory budget.

The existing strict-FP32 route-approval machinery already seals full-trajectory
evidence and the selected cache vector. Extend that owner to the normal
MF/FP16 operation-tuning path and a length/voice corpus instead of introducing
a second approval cache. Include first-audio latency, repeated fresh-session
medians and p95, peak VRAM, and listening/intelligibility checks when accepting
future precision changes.
