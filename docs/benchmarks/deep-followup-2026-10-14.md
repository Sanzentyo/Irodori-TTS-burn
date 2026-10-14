# Projection retuning, loading, conditioning graphs and FP16 rotary fusion

Base revision: `de764030e3f830f712bb684c7c8d1c936e6f1b40`.
Artifacts: `/home/sanzentyo/benchmark-artifacts/irodori-deep-followup-20261014`.
Measurements use RTX 5070 Ti Laptop and Intel ARL through Vulkan, Burn 0.22,
CubeCL 0.11 and WGPU 30. GPU batches hold the shared OS `flock`; the SAM3
session receives immediate start/release notifications. Both successful and
failed attempts remain in the artifact directory.

## Projection retuning

`prepare_projection_retune` snapshots a closed CubeCL environment and removes
only the current validated-projection namespaces from the candidate snapshot.
It uses CubeCL's export API and checks the namespace and retained entry counts.
Each transport bundle is then imported through CubeCL into a fresh writable
runtime environment; it is not placed directly at the active database path.
The original cache is untouched. `scripts/retune_projection_cache.py` seeds
every requested workload, freezes logical tuning values, and compares baseline
and candidate in alternating fresh processes. CubeCL remains the sole owner of
persisted selections; there is no second routing database.

The RTX FP32 campaign removed 13 projection entries and retained 6,170 other
entries. Three sessions per variant each contain two warmups and three timed
requests. The statistic below is the median of process medians, including
condition encoding, sampling, decoding and CPU PCM readback, after loading.

| Workload | Baseline | Retuned | Change |
|---|---:|---:|---:|
| MeanFlow short, 2 s audio | 75.993 ms | 75.778 ms | −0.3% |
| RF female, 6 s audio | 1,155.398 ms | 995.614 ms | −13.8% |

All trials meet the unchanged FP32 limits: latent maximum error ≤0.001 and audio
SNR ≥85 dB against the PyTorch FP32 fixture. Each variant also reproduces its
own latent/audio hashes across all sessions. RF retuning changes low-order
latent/audio bits between variants; acceptance uses those unchanged reference
limits. No tuning value changes during
measurement, and every measured process records zero fresh tuning decisions.
Adoption requires at least 2% geometric-mean improvement, no case
slower by more than 2%, and every accuracy gate passing. The measured geometric
ratio is 0.92696. This is a workload-specific cache result, not a universal
speedup from recompiling the library.

Worst observed latent errors are 5.746e-5 (MF) and 2.459e-4 (RF); minimum audio
SNRs are 93.74 and 94.30 dB. Other input geometries require their own acceptance
cases; the persisted result does not assert that every long-text case meets the
strict FP32 gate.

The accepted cache persists at
`/home/sanzentyo/benchmark-artifacts/irodori-deep-followup-20261014/retune-rtx-v4/candidate-cache`.
Use it through the existing `--cache-dir` option or
`IRODORI_TTS_BURN_CACHE_DIR`. It is not silently installed over a user cache.

```sh
cargo build --locked --release --all-features \
  --bin prepare_projection_retune --bin validate_synthesis
uv run scripts/retune_projection_cache.py SPEC.json NEW_OUTPUT_DIR \
  --sessions 3 --repeat 3
```

The spec supplies `binary`, `prepare_binary`, `precision`, a closed `seed_cache`,
its relative `environment_file`, and `cases` with unique names and complete
validator command arrays. `result.json` records the acceptance decision, original and frozen
hashes, accuracy gates and selected cache. The current namespace match is tied
to CubeCL 0.11's validated-projection-v1 tuner; a future namespace migration must
update the matcher explicitly. Missing target entries are an error.

The first apparent improvement in `retune-rtx-v2` was invalidated during review:
the directly placed exported database became read-only at runtime, causing new
tuning in every process. An unchanged logical digest alone did not detect failed
persistence. Acceptance now also requires an existing, empty fresh-decision
record in every measured process. The original failed run remains available.
Python SQLite could not inspect the runtime Turso file in the next attempt;
inspection now also uses CubeCL. The accepted `retune-rtx-v4` campaign verifies
both imported entry counts, unchanged source bytes, stable logical decisions,
empty measured tuning records and cross-process output reproducibility.

## Loading without a second full archive pass

Packed model loading hashes original file bytes while serving tensor payloads.
Header, gaps, padding and trailer are included, even for overlapping or
out-of-order reads. It retains an 8 KiB hashing scratch buffer rather than a
second archive. The identity remains the actual loaded file SHA-256, preserving
the saved-voice contract; converter-declared source identity is not substituted.
Raw decoder loading computes its existing sorted tensor identity from the
already loaded original bytes before materialization.

The actual worker CLI was measured from process launch through its `ready`
message. Both frozen binaries have the same `--all-features` configuration.
These are warm filesystem/driver caches, one discarded startup pair followed
by three alternating fresh processes per variant; they are not cold disk I/O
measurements. Model information and codec identity match in every process.

| Packed model | Base ready | New ready | Change | Peak RSS, base → new |
|---|---:|---:|---:|---:|
| RTX FP32 | 2.914 s | 2.445 s | −16.1% | 1,137,216 → 1,104,600 KiB |
| Intel FP32 | 2.848 s | 2.653 s | −6.8% | 881,884 → 843,992 KiB |
| RTX FP16 | 1.852 s | 1.763 s | −4.8% | 829,640 → 812,384 KiB |
| Intel FP16 | 1.829 s | 1.762 s | −3.7% | 756,748 → 754,020 KiB |

Interleaving identity hashing with raw model loading showed no reliable gain
(ratios 0.985–1.014) and was removed. A build overlapped the earliest raw startup
pair and the beginning of its first measured candidate process; no raw loading
speed claim is based on that campaign. Packed measurements occurred after the
build finished. Final raw model loading retains its existing sequential hash.

## Intel MLP storage experiment

FP16 workgroup tiles halve shared-memory storage while preserving FP32 FMA and
SwiGLU arithmetic. K16 and K32 variants were measured at 13/50/150/300 rows on
both adapters, rotating twelve distinct weights. All candidate outputs matched
bit for bit. Some comparisons against the corresponding K32 baseline improved
by 12–14%, but this baseline was often slower than the already selected route.
RTX's existing cooperative-matrix path remained much faster.

A frozen Intel trial replaced the selected 150-row vector K16 route with the
best half-storage candidate. Three alternating sessions, each with three timed
full RF requests, measured **12.233 → 12.317 s (+0.7%)**. Every latent/audio hash
matched the frozen output and no fresh tuning decisions contaminated timing.
The candidate and its extra production API/target were removed. Its source
patch, microbenchmark and binaries remain in the artifact directory.

## Mutable conditioning in captured sampling

`capture_conditioned_sampling` owns persistent text/speaker/caption buffers and
captures KV projection inside the sampler. Callers can replace encoded values
between requests with identical shapes, dtype, device, auxiliary variant and
exact attention masks. All validation precedes copying; a topology change is
an explicit error requiring a new capture. There is no implicit recapture or
graph cache. `encode_prepared_conditions` exposes the matching encoder path.

For the short MF fixture, two alternating text/noise requests match ordinary
sampling bit for bit on both devices and both precisions. Timing includes
encoding, exact-mask validation, sampling and latent CPU readback, excluding
the decoder. Three measured rounds follow one warmup.

| Case | Ordinary | Graph | Change |
|---|---:|---:|---:|
| RTX FP32 | 47.003 ms | 46.925 ms | −0.2% |
| RTX FP16 | 23.691 ms | 23.351 ms | −1.4% |
| Intel FP32 | 297.882 ms | 298.958 ms | +0.4% |
| Intel FP16 | 296.669 ms | 297.389 ms | +0.2% |

Capture costs 0.119/0.087/0.576/0.396 s respectively. This API remains explicit;
the results do not justify automatically enabling it in resident synthesis.

Final validation also uses the real MF reference-voice fixture: 150 speaker
tokens, five caption tokens and ten text tokens. On RTX/Intel × FP32/FP16,
changed speaker values, caption/text IDs and noise, followed by switching back,
all match ordinary sampling bit for bit. Both auxiliary contexts are verified
present in the actual sampler report.

## Locating long-text numerical differences

The profile-only `--dump-text-layers` validator saves CPU snapshots of actual
embedding, normalization, QKV, rotary, attention, MLP, residual and projection
boundaries. It verifies that the observed final state is bitwise equal to the
ordinary encoder. Production uses the same operations with a no-op observer.
`scripts/trace_text_encoder.py` uses the exact compacted IDs/mask and hooks the
official implementation, then verifies its observed and ordinary outputs match.

For 25 text tokens and 25 ModernBERT layers, the final FP32 text-state maximum
error is 3.01e-6 on RTX and 3.99e-6 on Intel. Differences accumulate from small
rounding errors; this does not identify a discrete FP32 algorithmic defect or
justify weakening the existing end-to-end acceptance limits.

The original FP16 path's first rotary Q has maximum error 0.013671875 and RMSE
0.0005023 on RTX. The official inference move also rounds inverse-frequency
buffers to FP16, promotes them for forming angles, and rounds the rotary result
only once. CPU reconstruction using the captured official Q reproduces its
first rotary output exactly with that order. FP32 frequencies and separately
rounded half products reproduce the original difference. This distinguishes
the cause from the earlier experiment that only promoted rotary intermediates.

A fused candidate implementing the official FP16 rounding made the first Q/K
rotary outputs bitwise equal on both GPUs. This did not consistently improve
whole-model accuracy: against the FP32 oracle, RTX male audio SNR fell from
25.38 to 16.45 dB and Intel short audio from 39.14 to 32.55 dB. Even comparisons
with the official FP16 audio were mixed. The coefficient/intermediate rounding
change was rejected; matching an individual layer is insufficient for adoption.

Intel's installed PyTorch XPU automatic SDPA failed before inference because its
oneDNN OpenCL primitive could not be created. The successful Intel oracle uses
explicit `--sdpa math`; there is no silent backend fallback. CUDA uses its normal
SDPA policy. These results are accuracy diagnostics, not PyTorch speed claims.

## FP16 rotary fusion with unchanged arithmetic

The adopted WGSL kernel reads the strided Q/K views directly and combines rotary
rotation, two products and addition into one dispatch. Two half products are
materialized in 1 KiB of workgroup memory before a neighboring lane adds them.
This preserves intermediate rounding while removing global intermediate buffers
and dispatches. All lanes participate in the barrier, including the tail.
No vendor-specific route or tuning database is added; FP32 keeps its original
arithmetic.

Two initially promising formulations were rejected: `quantizeToF16` flushed
subnormal products on RTX, and conversion through `f16` alone was eliminated on
Intel, changing its rounding. The materialized version preserves latent and
PCM bytes in all ten acceptance cases: short/long/male MF and female RF in FP16
on both GPUs, plus a short FP32 regression on both. This preserves existing FP16
accuracy; it does not eliminate the differences against PyTorch FP32 above.

`rope-exact-speed-v5` compares frozen production-equivalent baseline and final
fusion binaries. Each workload has three alternating fresh-process sessions per
variant, one warmup and three timed complete requests per process. Both variants
are seeded before measuring; all measured tuning records are empty, and every
latent/PCM hash matches the frozen reference. Native and SPIR-V caches are on,
with `CUBECL_WGPU_MAX_TASKS=32`. No local build overlaps these measurements.

| FP16 workload | Baseline | Fused | Change |
|---|---:|---:|---:|
| RTX, short MF | 54.146 ms | 53.470 ms | −1.2% |
| RTX, long female MF | 186.630 ms | 181.492 ms | −2.8% |
| Intel, short MF | 681.577 ms | 677.903 ms | −0.5% |
| Intel, long female MF | 3,009.167 ms | 3,019.638 ms | +0.3% |

RTX shows a modest benefit. Intel is effectively neutral at this sampling size;
these measurements do not establish an Intel speedup. An earlier faster RTX-only
variant is not the shipped kernel and is not used for this table.

## Verification and review

The [machine-readable evidence](data/deep-followup-2026-10-14.json) includes
raw process measurements, binary/source hashes, numerical comparisons and the
invalidated attempts. The [review note](../review/deep-followup-2026-10-14.md)
records the defects found and their resolutions.

The final release nextest selection contains 35 owner/boundary tests, all passing:
graph ownership and refresh, ModernBERT, strided/subnormal rotary, persisted
voices, decoder identity, identified-file reads, and cache inspection. It was
listed before execution and run serially under the shared GPU lock with
`--no-tests fail`. The five Python retune-policy tests pass, as do Ruff checks,
`cargo fmt --all --check`, and
`cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`.

Checks also cover default features, no-default core, core with profiling,
synthesis without packed weights, synthesis with packed weights, and the new
CLI tool with only `cli` enabled. Minimal configurations still emit existing
dead-code warnings; all-features Clippy is clean.

The actual CLI produces byte-identical short MF WAV files to the base binary on all four
device/precision combinations using packed weights. Model-bound saved voices
with the original archive SHA are accepted; a mismatched model SHA is rejected.
Raw-model worker startup also preserves the complete `ready` response, including
codec identity, and accepts the original model-bound voice. These final checks
use frozen `final-v8` binaries and private cache copies.

No dependency version, archive format, default precision or unsafe boundary is
changed. Core inference, optional packed loading, diagnostics, CLI tooling and
transport consumers are validated separately according to their feature gates.
