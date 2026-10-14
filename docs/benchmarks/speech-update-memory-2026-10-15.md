# speech.cpp 0.8.4 → 0.8.7: memory, speed and decoder context

Baseline here: `dbdc448d6aa09803f776d0137725bc0841007289`.
Artifacts: `/home/sanzentyo/benchmark-artifacts/irodori-speech-update-20261015`.
The [machine-readable evidence](data/speech-update-memory-2026-10-15.json)
contains individual worker samples, process memory peaks and accuracy results.

## Upstream changes

The external baseline checkout is speech.cpp
`d23c49f3233375e3ad4944c880807790b7f490be` (0.8.4), with ggml
`1d73413d60fd74538099958600ca111856473e59`. The new external worktree is
`6bd7b027674ca6d8aa8329436b56ff4ca3fab4f8` (0.8.7), with ggml
`a759356c321062853217d18659ac1893a2cf12e9`. Both use CMake Release, GCC 15,
Vulkan and OpenMP. Neither checkout is a submodule of this repository.

| Upstream change | Relevance here |
|---|---|
| [Snake fusion on Metal/Vulkan](https://github.com/nyosegawa/speech.cpp/commit/c87c5d44883706ba8cc44ae46de20f5b7070d69d) | Our WGSL already fuses Snake and several convolution epilogues. No duplicate shader is added. Upstream also reports unchanged RTX peak memory for this change alone. |
| [Decoder margin derived from receptive field](https://github.com/nyosegawa/speech.cpp/commit/dba8e267b0611263747d2714188902bf7313c136) | The supported codec needs eight frames per side instead of ten. This is the port evaluated below. |
| [DiT FlashAttention on Metal](https://github.com/nyosegawa/speech.cpp/commit/d50a90761a36faed30e56b6cd55da1dbafbb0a8a) | Upstream keeps matrix products on Vulkan: flash attention was effectively neutral in speed and less accurate there. The half mask contains only zero/infinity; our custom path already uses compact masking. No attention policy changes. |
| [v4-Large / T5Gemma 2 support](https://github.com/nyosegawa/speech.cpp/commit/c8e3b5c) | A different model topology and tokenizer, not a memory improvement to v4.1-Small. Not part of this matched-model port. |
| [Real-input FFT](https://github.com/nyosegawa/speech.cpp/commit/cff2a1dab67d50292aaa39fecb0808881065bdf6), FastConformer and Qwen changes | ASR/other-model work, outside the measured Irodori synthesis path. |

The published [memory table](https://github.com/nyosegawa/speech.cpp/blob/6bd7b027674ca6d8aa8329436b56ff4ca3fab4f8/docs/models/irodori-tts.md)
includes different quantizations, devices and workloads. Its smaller Q4/Q8
figures must not be attributed to upgrading the same F16 model. This comparison
holds the GGUF file constant within each old/new pair.

## Measurement boundary

- RTX 5070 Ti **Laptop**, 12 GiB, NVIDIA driver 595.99.02, Linux Vulkan.
  Intel Arrow Lake (`8086:7d67`, Mesa 26.0.8) is used separately for our WGSL
  port. Host: Core Ultra 9 275HX, Linux 7.0.0-38-generic.
- Actual resident workers, no reference voice, fixed text, seed 1 and explicit
  2/12-second output. MeanFlow uses four steps and RF uses forty.
  speech.cpp F32 disables Vulkan F16, COOPMAT and COOPMAT2; F16 uses its default
  Vulkan policy. The codec weights remain F32 in both cases.
- Two fresh worker processes per variant/case, order AB then BA; each process
  receives two warmups and three timed requests. Timings are medians of six
  requests per variant, from host request submission to first PCM / final
  response, including CPU readback and worker transport but excluding loading.
- NVML is polled every 20 ms. Compute and graphics process lists are unioned by
  the worker PID, never summed twice. Peak VRAM includes startup and warmups;
  loaded, warm, measured and idle peaks are also retained. This is sampled
  process VRAM, not an allocator-exact transient high-water mark or the entire
  desktop's memory. MiB means 2^20 bytes.
- All GPU execution holds the shared OS `flock`; the SAM3 session receives
  immediate planned/start/release notifications. No competing heavy CPU build
  runs during accepted speed measurements. Correctness batches may overlap CPU
  builds; their latency figures are excluded from performance conclusions.

### speech.cpp worker results

| Model | Precision | Audio | Time before → after | Change | TTFA before → after | Peak VRAM before → after |
|---|---|---:|---:|---:|---:|---:|
| MF | F32 | 2 s | 163.30 → 144.31 ms | -11.6% | 97.83 → 91.99 ms | 3500.3 → 3493.2 MiB |
| MF | F32 | 12 s | 744.35 → 598.98 ms | -19.5% | 156.38 → 150.16 ms | 3588.3 → 3574.2 MiB |
| MF | F16 | 2 s | 83.38 → 75.20 ms | -9.8% | 41.35 → 40.80 ms | 2101.2 → 2093.5 MiB |
| MF | F16 | 12 s | 455.99 → 348.11 ms | -23.7% | 80.74 → 69.57 ms | 2194.8 → 2179.3 MiB |
| RF | F32 | 2 s | 750.98 → 733.62 ms | -2.3% | 687.45 → 680.82 ms | 3474.7 → 3468.7 MiB |
| RF | F32 | 12 s | 2274.69 → 2152.96 ms | -5.4% | 1714.40 → 1711.54 ms | 3580.7 → 3566.2 MiB |
| RF | F16 | 2 s | 279.40 → 259.28 ms | -7.2% | 240.29 → 230.38 ms | 2089.4 → 2081.7 MiB |
| RF | F16 | 12 s | 1093.29 → 983.82 ms | -10.0% | 712.18 → 710.48 ms | 2200.9 → 2185.0 MiB |

Loaded VRAM is identical before/after: MF F32 3311.91 MiB, MF F16
1837.47 MiB, RF F32 3284.41 MiB and RF F16 1823.72 MiB. The update saves
approximately 6–16 MiB at peak on these cases; it does not materially reduce
weight residency. The aggregate update improves runtime, particularly the
codec-heavy MeanFlow cases. This old/new experiment does not isolate each
upstream commit's contribution.

### Accuracy of the upstream update

Fixed-noise fixtures cover MF at 50/250 latent frames and RF at 100 frames,
each in F32/F16. The repository's C++ comparison harness links each unmodified
speech.cpp library; the new harness changes only the required `Dit` constructor
argument for upstream's new API.

All six sampled latents are bitwise identical between versions. For both
versions, ordinary and streamed raw FP32 waveform samples are bitwise identical
within each case. Old/new raw audio differs slightly: SNR is 120.9–126.0 dB in
F32 and 67.2–72.5 dB in F16, consistent with changed codec arithmetic. These are
numerical comparisons, not a listening study or CER measurement.

The existing difference from the official FP32 reference remains: for example,
long MF F16 audio is about 21.04 dB from that reference in both versions. This
update does not make FP16 equivalent to FP32 or establish universal precision
acceptance for long utterances.

## Port into our decoder

`DacVaeDecoder::context_frames()` derives the margin from the loaded convolution
widths, dilations and upsampling strides. The loader supports centered odd-width
convolutions and even strides with transposed-convolution width twice the stride.
Tracing the output radius backward yields eight latent frames per side.
Elementwise activations and residual shortcuts do not enlarge that dependency.

Ordinary and captured streaming share the same `DecodePlan`. The emitted chunk
schedule remains 12 then 96 frames, but an interior 96-frame window decodes
112 instead of 116 frames; the initial 12-frame window decodes 20 instead of 22.
The validator's graph bound uses the same derived margin. Tail context, partial
sample trimming and cancellation behavior remain intact. No dtype, shader,
dependency, unsafe boundary or fallback is introduced.

### Port performance and memory

| Device / model | Precision | Audio | Time before → after | Change | TTFA before → after | Peak VRAM before → after |
|---|---|---:|---:|---:|---:|---:|
| RTX / MF | FP16 | 2 s | 62.62 → 61.62 ms | -1.6% | 37.48 → 37.81 ms | 3056.1 → 3046.5 MiB |
| RTX / MF | FP16 | 12 s | 251.87 → 249.32 ms | -1.0% | 60.18 → 56.04 ms | 3302.8 → 3286.9 MiB |
| RTX / MF | FP32 | 2 s | 88.57 → 88.63 ms | +0.1% | 63.83 → 63.76 ms | 5400.9 → 5390.8 MiB |
| RTX / MF | FP32 | 12 s | 329.21 → 317.47 ms | -3.6% | 116.64 → 115.25 ms | 5647.2 → 5631.1 MiB |
| RTX / RF | FP16 | 2 s | 205.80 → 211.51 ms | +2.8% | 180.42 → 187.12 ms | 3093.9 → 3081.9 MiB |
| RTX / RF | FP16 | 12 s | 666.36 → 653.43 ms | -1.9% | 465.01 → 456.94 ms | 3339.0 → 3323.0 MiB |
| RTX / RF | FP32 | 2 s | 433.43 → 434.29 ms | +0.2% | 408.55 → 409.34 ms | 5194.3 → 5157.4 MiB |
| RTX / RF | FP32 | 12 s | 1420.72 → 1397.05 ms | -1.7% | 1220.90 → 1202.60 ms | 5439.0 → 5423.0 MiB |
| INTEL / MF | FP16 | 2 s | 854.24 → 823.15 ms | -3.6% | 478.05 → 461.79 ms | shared memory; n/a |
| INTEL / MF | FP16 | 12 s | 4074.62 → 4025.14 ms | -1.2% | 1376.61 → 1361.98 ms | shared memory; n/a |
| INTEL / MF | FP32 | 2 s | 861.82 → 829.31 ms | -3.8% | 485.04 → 468.18 ms | shared memory; n/a |
| INTEL / MF | FP32 | 12 s | 4005.54 → 3975.14 ms | -0.8% | 1310.81 → 1295.13 ms | shared memory; n/a |

Our worker experiment uses separate clones of the same seed tuning cache for
each variant/device/precision. A complete first process per workload primes both
variants and is excluded from the summaries; two subsequent alternating
processes provide six measured requests per variant. Native/SPIR-V pipeline
caches are enabled and `CUBECL_WGPU_MAX_TASKS=32`. MF weights and the FP32 codec
use the packed format; RF weights use the original safetensors. Within a pair,
all files and request settings are identical. Our long request is one sentence,
with `keep_tail=true`. Equal numeric seeds across runtimes are not equal noise
arrays, so worker PCM is compared within a runtime only.

This is warm resident performance. Starting from the chosen older seed caches,
Intel's first candidate MF request took 18.6–20.5 s for the short case and
90.4–92.7 s for the long case while preparing new shapes; the corresponding
baseline priming requests took 2.3–2.5 s and 45.2–46.3 s. Subsequent fresh
processes reuse persisted preparation. This transition does not establish a
cold-start improvement, and is not a test with completely empty caches.
The initial twelve-second MF priming process also peaks about **73 MiB higher**
on RTX in both precisions, before those decisions are cached. The roughly
16 MiB saving in the table applies to fresh processes using the prepared cache,
not to the first tuning process's retained allocation pool. The prepared
candidate caches are under `burn-workers-{rtx,intel}/{device}-{precision}-new-cache`
in the artifact directory and can be selected with the existing `--cache-dir`.

Intel uses shared system memory: its RSS is retained, and dedicated VRAM is
reported as unavailable rather than zero. RSS and RTX VRAM are different metrics.
Observed Intel peak host RSS is 2–17 MiB higher in the candidate; these readings
do not establish savings in shared GPU allocations.

For RTX, the twelve-second workloads save about 16 MiB at peak and 1.0–3.6%
in latency. Short FP32 requests are effectively neutral in speed; short MF FP16
improves 1.6%. The initial short RF FP16 cohort is 2.8% slower, so that condition
is rechecked with one common already prepared cache below. Short-process peak
VRAM varies between runs; do not interpret every maximum difference as an exact
allocation saving. Our loaded weight residency remains essentially unchanged.

Intel's short MF improves 3.6–3.8%, while the long-case 0.8–1.2% difference is
small at this sample size. These cohorts contain two measured fresh processes
per variant; the three timed requests within each process are correlated.
All twelve device/model/precision/duration cases preserve worker PCM bytes.

The focused RF FP16 recheck seeds both binaries with byte-for-byte copies of
the same completed candidate cache. All six processes retain the same 139
logical tuning entries and digest before/after execution; PCM bytes are equal.
Three alternating fresh processes per variant, with two warmups and three
timed requests each, give median process times of **212.96 → 209.97 ms
(−1.4%)** and sampled peak VRAM **3094.0 → 3082.0 MiB**. The slowdown in the
first independently warmed cohort does not reproduce under common prepared
tuning. Both cohorts remain in the evidence; the first table is not replaced
with selected follow-up samples.

The derived margin is adopted for the smaller decoder windows and measured
warm benefits, with exact output preserved. The update does not reduce model
weight storage; first-use compilation and tuning remain visible costs. No
extra vendor/precision fallback or new attention/Snake kernel is needed.

## Validation and retained failures

- Five selected release `cargo nextest` tests cover the independent inversion
  of all 1920 output sample phases, window coverage/trimming, invalid geometry
  and graph-cache ownership. The inverted indices prove that seven context
  frames are insufficient and eight cover the supported topology.
- The expanded real-codec integration test passes on RTX and Intel, comparing
  ordinary/captured windows against whole decoding bit for bit. It covers
  short inputs, single-frame first chunks, 112-frame interior geometry,
  partial sample trimming, changed input, eviction, cancellation and
  environment-generation invalidation.
- Six additional real-fixture checks pass: long MF and female RF in FP32/FP16
  on RTX, and long MF in FP32/FP16 on Intel. Whole decoding, 17-frame windows,
  production 96-frame windows and a partial-frame trimmed tail agree bit for
  bit. These checks clone the already prepared candidate worker caches and
  report existing oracle differences separately from the exact window checks.
- All-features/all-targets Clippy with warnings denied, formatting, default
  library check, and no-default codec / codec-plus-streaming-graphs checks pass.
  Minimal-feature builds retain their existing dead-code warnings.
- `workers-v1` did not synthesize (missing required voice); `workers-v2` selected
  Intel through a changed physical Vulkan enumeration and is excluded from RTX
  results. NVIDIA-only ICD selection and an adapter-name preflight fix that
  measurement error. `workers-v3` stopped on a process-teardown `/proc` ESRCH;
  the monitor now accepts that normal exit condition. `speech-workers-v4`
  retained the sixteen completed paired processes, discarded the incomplete
  pair and measured the remaining sixteen. All original attempts are retained.

The [review record](../review/speech-update-memory-2026-10-15.md) describes the
final contract and review fixes.

Verification commands (GPU tests require the common lock and
`IRODORI_CODEC_TEST_CHECKPOINT` set to decoder safetensors; add
`IRODORI_PACK_TEST_INTEGRATED=1` for Intel):

```sh
cargo nextest list --locked --release --all-features --lib \
  -E 'test(codec::streaming::) | test(codec::streaming_graph::)'
cargo nextest run --locked --release --all-features --lib \
  -E 'test(codec::streaming::) | test(codec::streaming_graph::)' --no-tests fail
cargo nextest run --locked --release --all-features \
  --test codec_preparation_gpu --run-ignored all --test-threads 1 --no-tests fail
cargo clippy --locked --workspace --all-features --all-targets -- -D warnings
cargo fmt --all -- --check
```

The artifact directory retains `measure_speech_workers.py`,
`measure_burn_workers.py`, `check_speech_accuracy.py`, `check_burn_accuracy.py`,
the immediate-notification/lock wrapper and each process's complete command,
request, PCM, stderr and telemetry. Run these PEP 723 scripts with `uv run`.
The accepted speech cohort is `workers-v3` completed by `speech-workers-v4`;
our cohorts are `burn-workers-rtx`, `burn-workers-intel` and `burn-accuracy`.
