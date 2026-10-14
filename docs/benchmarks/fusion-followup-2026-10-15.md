# WGSL epilogue fusion: rounding, routing and performance

Base: `a36c9271ba625e611b82e1856723e0e0c7bd0db1`.
Local artifacts: `/home/sanzentyo/benchmark-artifacts/irodori-fusion-followup-20261015`.
Hardware: RTX 5070 Ti **Laptop** 12 GiB (NVIDIA 595.99.02) and Intel Arrow Lake
`8086:7d67` (Mesa 26.0.8), Linux Vulkan. Dependencies remain Burn 0.22.0 /
CubeCL 0.11.0. All benchmark binaries use the locked release/all-features build.

## Candidates and numerical decisions

| Candidate | Implementation / decision |
|---|---|
| ModernBERT GELU and gate | Keep Burn's division and erf, then combine add-one, multiply-input, divide-two and multiply-gate. The diagnostic specialization also writes the actual pre-gate value. |
| Cross-layer AdaLN gates | Combine projection bias, raw-condition addition and gate tanh. Shift/scale retain their original values. Packed schedules own activated gates; ordinary modulation producers obey the same internal contract. |
| More residual/LayerNorm shapes | Preserve the existing three-token FP32 text-only implementation. Auxiliary-conditioned input at every length, plus other admitted v4 shapes/FP16, preserves Burn's residual addition and tuned mean/variance reductions, then combines the normalization epilogue. |

The pointwise shaders preserve the original materialized arithmetic boundaries
with workgroup exchanges. This matters in FP16 even when an expression is
algebraically unchanged. Runtime bounds/dispatch geometry use CubeCL metadata
uniforms; pipeline identities do not depend on text length or CFG batch.
The existing native pipeline cache owns persistence; no second cache is added.

Two larger fusions were rejected on accuracy:

- A complete WGSL GELU polynomial matched RTX, but Intel at hidden width 3072
  differed in 962 FP32 values (maximum `5.96e-8`) and 207 FP16 values (maximum
  `4.88e-4`) in the focused input. The accepted direction retains the backend
  division/erf implementation. `experiments/full-gelu` retains the candidate.
- Reusing one LayerNorm reduction tree for all lengths/precisions changes the
  accumulation order selected by Burn. On the long female MF input, the direct
  extension gives only 83.53/84.98 dB audio SNR relative to the unchanged RTX/
  Intel FP32 output; FP16 gives 23.78/26.41 dB. It is not accepted. The first
  apparently exact long-input run did not exercise that kernel: auxiliary
  conditioning bypassed the specialized text route. Explicit route testing
  exposed the difference. `norm-routes-*-v3` retains these results.

Same-runtime bitwise preservation is separate from agreement with the official
FP32 oracle. Existing long-input/FP16 oracle discrepancies are not removed by
these fusions, and no tolerances are relaxed to accept them. These checks are
numerical comparisons, not listening tests or a CER study.

## Measurement protocol

Fixed-noise fixtures exercise MeanFlow short text (50 latent frames), long text
with a female caption (250 frames), and RF with auxiliary conditioning. Model
precision is explicitly FP32 or FP16; the codec remains FP32. Packed MF weights
and safetensors RF weights are held constant within each comparison.

`scripts/benchmark_synthesis_variants.py` primes every variant, then rotates
fresh processes against identical logical cache snapshots per workload. Each process
warms up once and measures five requests. Reported complete-audio time includes
condition encoding, sampling, latent readback, ordinary codec decoding and audio
readback; it excludes loading and is not a worker-transport/TTFA measurement.
Each timed process must reproduce frozen latent/audio files bit for bit and add
zero tuning decisions. Process medians, rather than treating requests in one
process as independent machines, are the comparison units.

The first Intel cohort (`speed-intel-v5`) failed in the **unchanged baseline**
with Turso 0.8.3's `reader slot released by non-owner` assertion, before any
accepted measured sample. The revised benchmark uses the existing CPU-only
`prepare_projection_retune` tool to export all namespaces and import them into
a new runtime environment for each process. It uses the full baseline bundle,
never the projection-filtered candidate bundle. Entry counts and logical tuning
hashes must match; measured processes must retain those hashes. Export/import
time is outside the inference timing. No shared-memory coordination file is
copied, no user cache is cleared, and no production error is swallowed. This is
an isolated measurement setup, not an upstream Turso repair.

The assertion recurred in `speed-intel-v6` with fresh imports, so that incomplete
cohort is excluded too. The accepted Intel `v7`, RTX `v8` and confirmation cohorts
explicitly disable **CubeCL's** native-pipeline and SPIR-V persistence for every
variant. OS/driver shader caches
keep their ordinary settings. Tuning decisions remain persistent and frozen.
Compilation/driver preparation is warmed before timed requests. These results
compare warm inference; they do not establish cold-start performance or resolve
the intermittent error with persistent compilation enabled. Application defaults
are unchanged. Both failed cohorts and their command/environment records remain
available for diagnosis.

Every GPU batch holds the shared OS flock and sends immediate planned/start/
release notifications to the SAM3 session. Accepted speed windows exclude heavy
CPU compilation on both sides. Correctness-only runs may overlap CPU builds;
their timing fields are not used as performance evidence.

The initial static-shape RTX cohort (`speed-rtx-v1`, two measured processes per
variant) gives baseline → GELU-plus-gate times of 76.146 → 75.211 ms for short
FP32, 234.183 → 233.186 ms for long FP32, 52.703 → 52.500 ms for short FP16 and
184.122 → 183.468 ms for long FP16. These small differences motivated the final
repeat comparisons. That version also compiled a new shader for each text
length; the revised runtime metadata removes this unnecessary specialization.

## Complete-audio results

The primary cohorts completed successfully: three measured processes per variant
and workload, with five warmed requests per process. Each variant occupies every
execution position once. Values below are medians of process medians in ms;
negative changes are faster. The [machine-readable record](data/fusion-followup-2026-10-15.json)
retains each process value, sampling times, hashes, binary provenance and lock
events. Small changes should not be interpreted as a statistically established
improvement from only three processes.

| Device | Workload | Precision | Baseline | GELU + gate | All three | All vs baseline |
|---|---|---|---:|---:|---:|---:|
| RTX | MF short | FP32 | 75.478 | 75.214 | 75.581 | +0.14% |
| RTX | MF long, female caption | FP32 | 236.384 | 235.807 | 239.423 | +1.29% |
| RTX | MF short | FP16 | 52.295 | 52.586 | 50.278 | −3.86% |
| RTX | MF long, female caption | FP16 | 183.658 | 183.121 | 181.000 | −1.45% |
| RTX | RF, female caption | FP32 | 1000.244 | 1000.526 | 996.669 | −0.36% |
| RTX | RF, female caption | FP16 | 437.145 | 439.083 | 435.558 | −0.36% |
| Intel | MF short | FP32 | 683.568 | 683.092 | 683.647 | +0.01% |
| Intel | MF long, female caption | FP32 | 3060.479 | 3053.362 | 3055.101 | −0.18% |
| Intel | MF short | FP16 | 684.058 | 681.063 | 683.148 | −0.13% |
| Intel | MF long, female caption | FP16 | 3049.717 | 3067.683 | 3071.031 | +0.70% |

RTX MF FP16 benefits most from the normalization epilogue: the GELU-plus-gate
ablation alone does not produce the short-input improvement. Intel gains remain
small. For RTX long FP32, sampling itself improves from 102.757 to 101.940 ms,
while complete-audio time is worse; the codec/readback interval and process
variation therefore matter to adoption. Sampling alone is not the acceptance
metric.

The two slower-looking conditions received a separate four-pair AB/BA
confirmation using the final `v8` binary. Each process again warms up once and
measures five requests, with identical logical tuning snapshots and the same
isolation rules. Both the primary and confirmation cohorts are retained.

| Workload | Baseline ms | All three ms | Change | Range of paired changes |
|---|---:|---:|---:|---:|
| RTX MF long FP32 | 237.316 | 238.080 | +0.32% | −1.43% to +2.24% |
| Intel MF long FP16 | 3058.213 | 3061.325 | +0.10% | −0.23% to +0.39% |

RTX is faster in three of the four pairs; Intel is faster in one. The original
slowdowns do not recur at the same magnitude, but these small cohorts cannot
exclude a small regression. No FP32 or Intel speedup is claimed. Sampling-only
medians change by −0.36% and +0.11%, respectively; complete-audio time remains the
acceptance metric.

**Selection:** retain the three refined epilogues and the conditioning-aware
reduction contract. The useful observed gain is RTX MF FP16: 52.295 → 50.278 ms
for short input and 183.658 → 181.000 ms for long input. The shared implementation
preserves the tested outputs and avoids introducing device-specific branches for
the small, variable differences above. This is a measured FP16 improvement with
bounded evidence on other workloads, not a claim that every GPU or input becomes
faster. The larger GELU/reduction fusions that changed output remain rejected.
The [review](../review/fusion-followup-2026-10-15.md) records the corrected findings
and verification boundaries.

## Correctness and review verification

The numerical `v5` implementation passed 16 short/long MF captures (both GPUs,
both precisions, GELU-plus-gate and all-fusion variants). Every latent and audio
file matched its unchanged baseline byte for byte. Another 12 captures compare
baseline/all-fusion on male MF for both GPUs and female RF for Intel, again in
both precisions: all six candidate comparisons were exact. Streamed and trimmed
decoding matched full decoding throughout.

Four graph-reuse cases (two GPUs × two precisions) passed changed-noise,
changed-condition and switch-back checks. The diagnostic route also verifies the
actual pre-gate activation and agrees with ordinary inference. Focused tests
exercise odd tails, F16 rounding, changing/revisited text lengths and CFG batches.

The `v7` change after these captures tightens only the dispatch-grid
preflight; all three numerical Rust/WGSL kernel files are identical to `v5`.
Its affected owner tests pass on RTX (10) and Intel (2). The wider `v5` owner run
passed 44 RTX tests and eight Intel tests; unchanged numerical tests and graph
captures are reused as such. All-features/all-targets Clippy with warnings denied
passes on `v7`; default/minimal feature checks passed on `v5`. Formatting, the five existing retuning
script tests and the exact source/binary snapshot checks also passed.

A final review then found an additional routing boundary: three-token FP32 text
with auxiliary conditioning had previously used Burn's reduction tree. `v7`
selected the old text-only fixed tree for it. A new short-caption fixture
reproduced the resulting latent/audio difference on both devices. The final
`v8` passes a typed reduction choice from the condition frontend through normal
execution and diagnostics. Auxiliary-conditioned input preserves backend order
even at three tokens. Both devices now match the unchanged baseline byte for
byte; the old failing candidate remains in `accuracy-short-caption-v8` as a
negative control. The fixture reuses fixed noise and changes the caption, so
this check compares to the unchanged Rust baseline, not its old oracle waveform.

The `v8` owner selection passes 12 RTX tests and one additional Intel test, with
test lists inspected first and empty selections treated as failures. Default and
minimal feature checks and all-features/all-targets Clippy pass on the final
source (the minimal build has 25 existing dead-code warnings). All three new
Rust/WGSL numerical kernel files are unchanged from `v5`.

Primary Intel timing uses the completed `v7` cohort. Its four workloads are
text-only at three tokens or auxiliary-conditioned at 25 tokens; their numerical
shaders and GPU execution paths are unchanged by the final conditioning
correction. These measurements are reused, not described as a rerun of the `v8`
executable. The Intel long-FP16 confirmation and all final RTX timing use `v8`.

The final confirmation passed all 16 measured processes and four priming
processes, including exact latent/audio, streamed/trimmed decoding and frozen
tuning checks. It held the common lock from 21:03:32 to 21:10:17 UTC on October 14
(06:03–06:10 JST on October 15) and released it with exit code zero. Source patch,
new kernel files and both release executables were checked against the measured
`v8` snapshot before committing. Input sizes/mtimes still match the eight-file
pre-measurement SHA-256 manifest. No Rust source changed after the final owner,
feature and Clippy checks.
