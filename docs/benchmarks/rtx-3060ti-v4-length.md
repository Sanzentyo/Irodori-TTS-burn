# RTX 3060 Ti v4 length sweep

Measured on 2026-08-11 after generalizing the production fused WmHead, all
twelve decoder residual pointwise routes, the measured K7 residual tile
policies, the decoder ConvTranspose routes, and the JointAttention
materialization tail across the supported audio lengths. The WGPU path retains
GPU-resident tensors between stages; CPU
readback is performed only for the separately reported readback-inclusive
boundary.

## Protocol

- Strict FP32 on both runtimes; TF32 and autocast disabled
- Lengths: 0.5, 1, 2, 4, and 8 seconds at 48 kHz
- Five fresh processes per runtime and length
- Two warmups excluded, ten measured repetitions per process
- Primary: pre-sync to device-complete, preserving GPU-resident output
- Secondary: the same start through owned contiguous FP32 CPU readback
- Accuracy, determinism, RF work count, pins, and timing manifests are enforced
  for every repetition
- Performance does not filter the sweep artifact

Current evidence:
`/tmp/irodori-v4-length-sweep-dynamic-stem-attempt1-20260811`.
Its 564-entry manifest verifies in full, every timing family contains exactly
50 measured samples per length and runtime, and the tree is frozen as files
0444/directories 0555 without symlinks.

## Device-complete medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 124.017 ms | 118.455 ms | 1.047x | 21.307 ms | 20.136 ms | 1.058x |
| 1 s | 126.531 ms | 125.324 ms | 1.010x | 34.595 ms | 24.704 ms | 1.400x |
| 2 s | 137.012 ms | 118.991 ms | 1.151x | 46.536 ms | 41.299 ms | 1.127x |
| 4 s | 166.248 ms | 168.850 ms | 0.985x | 90.712 ms | 88.259 ms | 1.028x |
| 8 s | 220.587 ms | 273.233 ms | 0.807x | 189.451 ms | 172.560 ms | 1.098x |

The codec now passes the strict all-sample WGPU-below-PyTorch-minimum gate at
all five lengths and at both timer boundaries. Its device-complete maxima are
20.458, 25.259, 41.815, 89.309, and 173.692 ms versus Python minima 21.230,
34.502, 46.290, 90.190, and 188.059 ms. RF passes the same strict gate at two
seconds. At 0.5 and 1 second it wins in median but not in every sample; at 4 and
8 seconds it remains slower in median. The overall multi-length goal is now
strictly a long-sequence RF problem rather than a codec problem.

## CPU-readback-inclusive medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 124.061 ms | 118.571 ms | 1.046x | 21.357 ms | 20.233 ms | 1.056x |
| 1 s | 126.577 ms | 125.455 ms | 1.009x | 34.657 ms | 24.810 ms | 1.397x |
| 2 s | 137.057 ms | 119.096 ms | 1.151x | 46.626 ms | 41.552 ms | 1.122x |
| 4 s | 166.294 ms | 168.953 ms | 0.984x | 90.852 ms | 88.503 ms | 1.027x |
| 8 s | 220.635 ms | 273.393 ms | 0.807x | 189.696 ms | 172.922 ms | 1.097x |

## Dynamic pointwise effect

Relative to the immediately preceding dynamic-WmHead sweep, device-complete
WGPU codec medians improved by 6.058, 13.115, 0.036, 53.013, and 93.032 ms for
0.5/1/2/4/8 seconds. This is direct evidence that the former two-second-only
pointwise selectors were forcing large generic fallbacks at other lengths.
All numerical gates and output hashes remain valid after the dynamic routing
change.

The change is retained because it materially improves four of five lengths and
is neutral at the already-specialized two-second length. It is not sufficient:
the resulting K7 work below is also required.

## Dynamic K7 effect

Relative to the dynamic-pointwise sweep, reusing the measured channel/dilation
K7 tile policies at decoder-valid lengths improves WGPU codec medians by 6.064,
14.491, -0.086, 61.207, and 124.508 ms for 0.5/1/2/4/8 seconds. The small
two-second regression is within run-to-run variation; the other four changes
are large and directionally consistent. Decoder length divisibility is checked
before selecting these routes, and every numerical/hash gate passes.

The dynamic K7 change is retained. It reduces the remaining codec median gaps
to 2.204 ms (0.5 s), 2.735 ms (1 s), 22.647 ms (4 s), and 31.252 ms (8 s).
At that point the next fixed decoder target was ConvTranspose; the resulting
measurement follows. RF sequence-length scaling remains separately open, with
tail latency—not merely median—as the acceptance criterion.

## Dynamic ConvTranspose effect

The first upsampler now reuses its GPU-resident packed polyphase weight at
non-reference lengths. The remaining three use a zero-copy checkpoint-weight
view feeding tuned GPU GEMM and an ordered col2im finalizer. The cached-column
route is admitted only from one second upward: an initial 0.5-second diagnostic
showed a fresh-process convergence tail up to 31.860 ms, worse than the stable
K7-only fallback maximum. The final policy keeps that short route disabled.

Relative to the dynamic-K7 sweep, final WGPU codec medians change by +0.218,
-4.823, +0.059, -1.424, and -1.090 ms for 0.5/1/2/4/8 seconds. The 0.5- and
2-second differences are ordinary run variation; 1, 4, and 8 seconds improve.
Peak WGPU memory remains below 7 GiB on the 8 GiB GPU. The change is retained,
but the long-length codec gap now points primarily to the remaining residual or
materialization work rather than ConvTranspose.

## Dynamic JointAttention materialization effect

The direct packed-K/V and post-SDPA layout-plus-gate kernels formerly selected
only latent sequence length 50. They now specialize on the runtime latent
length while retaining the same B1/B2, H20, Dh64, context-3, contiguous-layout,
device, and hardware-limit gates. Kernel cache identities include the sequence
length, and no host readback or intermediate CPU copy is added.

Four independent five-process-per-runtime campaigns validate the non-reference
lengths. All RF work-count, accuracy, determinism, source-pin, device-complete,
and readback-complete contracts pass:

- `/tmp/irodori-v4-stage-dynamic-attention-s0p5-attempt1-20260811`
- `/tmp/irodori-v4-stage-dynamic-attention-s1-attempt1-20260811`
- `/tmp/irodori-v4-stage-dynamic-attention-s4-attempt1-20260811`
- `/tmp/irodori-v4-stage-dynamic-attention-s8-attempt1-20260811`

Device-complete WGPU RF medians improve by 1.535, 1.339, 2.898, and
4.882 ms at 0.5, 1, 4, and 8 seconds respectively. The corresponding
readback-inclusive gains are 1.501, 1.310, 2.860, and 4.875 ms. The new
device-complete medians are 121.197, 127.926, 169.555, and 284.761 ms.

This closes median parity at 0.5 seconds but not its strict all-point tail gate.
The 1-second median gap falls to 0.645 ms; the 4- and 8-second gaps remain
3.664 and 65.918 ms. The residual long-sequence gap is therefore in SDPA or
other sequence-scaling DiT work, not the removed K/V/layout materializations.

## Head-major SDPA routing

The direct Q/K/V materialization now writes contiguous head-major buffers
`[B,H,S,Dh]` and `[B,H,S+3,Dh]`. This removes the attention-boundary layout
materialization without a host copy. A persistent f32 `1 = attend` mask is
prepared once with each exact text-CFG cache, so native SDPA does not cast or
concatenate masks in the four-step sampling loop.

The production-shape isolated A/B is sealed at
`/tmp/irodori-v4-sdpa-length-ab-attempt1-20260811`. It uses B1/B2, H20, Dh64,
S=13/25/50/100/200, rotating 10-warmup/100-iteration/5-trial paths, and a
pre-sync-to-device-complete primary timer. Full f32 CPU readback, SHA-256, and
accuracy comparison are outside the primary timer. All native outputs have
maximum absolute error at most `5.22e-8` versus production Burn attention.

The measured selector is intentionally narrow: Q16/KV16 at S13, Q8/KV32 at
S25 and S50, and Burn attention at S100/S200. The native candidates lose to
Burn at both longer lengths, so no long-sequence custom path is selected. Any
shape, dtype, stride, device, mask, binding, workgroup, shared-memory, or
dispatch-limit mismatch also falls back to Burn.

Five new end-to-end campaigns validate the production connection:

| audio | Python RF | WGPU RF | Python codec | WGPU codec | strict RF | strict codec |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0.5 s | 123.824 ms | 118.337 ms | 21.283 ms | 23.525 ms | yes | no |
| 1 s | 126.466 ms | 125.307 ms | 34.542 ms | 32.320 ms | no | yes |
| 2 s | 136.844 ms | 118.942 ms | 46.503 ms | 45.143 ms | yes | yes |
| 4 s | 166.114 ms | 168.963 ms | 90.519 ms | 111.296 ms | no | no |
| 8 s | 220.126 ms | 272.891 ms | 189.178 ms | 219.478 ms | no | no |

The corresponding artifacts are
`/tmp/irodori-v4-stage-native-sdpa-{s0p5,s1,s2,s4,s8}-attempt1-20260811`.
Each uses five fresh processes per runtime, two excluded warmups and ten
measured repetitions per process. Device-complete and CPU-readback-complete
boundaries, all ten numerical gates, deterministic hashes, and RF work counts
pass. The strict column requires every one of 50 WGPU points to be below the
global Python minimum, not merely a faster median.

Relative to the preceding dynamic-materialization results, WGPU RF medians
improve by 2.860, 2.619, 3.929, 0.592, and 11.870 ms from 0.5 through 8 seconds.
The remaining 4/8-second RF gaps and 4/8-second codec gaps therefore remain the
next optimization targets.

## Current long-codec profile

`profile_codec_decoder` now derives the latent and waveform extents from the
oracle instead of assuming two seconds. Its primary boundary is a pre-launch
device sync through decode device completion; CPU waveform readback is recorded
separately. With the current production source, the direct-oracle-latent codec
medians are:

| audio | device complete | CPU readback complete | readback increment |
|---:|---:|---:|---:|
| 4 s | 108.818 ms | 109.026 ms | 0.209 ms |
| 8 s | 212.024 ms | 212.468 ms | 0.444 ms |

The stage-synchronized medians identify the same scaling bottleneck at both
lengths:

| group | 4 s | 8 s |
|---|---:|---:|
| block 2 residual units | 35.649 ms | 70.756 ms |
| block 3 residual units | 19.557 ms | 38.740 ms |
| block 1 residual units | 17.563 ms | 37.486 ms |
| all four transposed convolutions | 17.087 ms | 32.507 ms |
| decoder stem | 10.248 ms | 17.101 ms |
| block 0 residual units | 9.582 ms | 15.790 ms |
| output head | 0.602 ms | 1.089 ms |

The 4-second workload is preserved in
`/tmp/irodori-v4-codec-length-profile-current-attempt2-20260811`; the 8-second
workload is preserved in the corresponding `attempt3` directory. Each workload
ran exactly once and passed all 15 waveform checks with deterministic hashes
and zero WGPU errors. Both wrappers deliberately report a nonzero campaign
status because their immediate next/post idle sample still observed NVML's
activity lag. The workload logs and frozen manifests are diagnostic evidence,
not successful formal campaigns. A delayed read-only query subsequently showed
GPU1 back at 38 MiB, 0% utilization, P8, with no compute process.

## Current long-RF profile

The eight-second RF path was profiled separately with a `profile`-feature-only
device synchronization around each of the six sub-stages in every diffusion
block. Normal release builds do not perform these synchronizations or inspect
the profiling environment variable. The aggregate below was originally
recorded from `/tmp/irodori-v4-rf-s8-stage-profile-attempt1-20260811`.

That temporary evidence path is no longer auditable. A later diagnostic
mistakenly reused the same output name and overwrote part of the tree after its
fresh-output check failed without errexit. The directory is now sealed with an
`INVALIDATED_BY_COLLIDING_DIAGNOSTIC.txt` marker; its original manifest is kept
to expose the mismatch, and its post-incident manifest is not a replacement
for the lost payload. The values below remain historical direction recorded in
Git, but they must not be used as acceptance evidence for a new optimization.

The two steady repetitions attribute nearly all stage-synchronized time to the
two large transformer branches:

| RF sub-stage | steady total | median call | share of measured branch work |
|---|---:|---:|---:|
| JointAttention | 259.471 ms | 2.702 ms | 46% |
| SwiGLU MLP | 270.130 ms | 2.787 ms | 48% |
| both AdaLN stages and residual updates | 33.455 ms | below 0.08 ms | 6% |

The four whole-model forwards retain batches `[2,2,1,1]`, so the profile spans
96 calls to each sub-stage across the two steady repetitions. All twelve blocks
are similar rather than exposing one pathological layer. Closing the roughly
54 ms eight-second RF gap therefore requires about a 20% combined reduction in
Attention and MLP, not a localized normalization or residual-update change.
The next diagnostic consequently measures the current and alternative
projection weight layouts at sequence length 200 before changing production.

## Long-sequence projection layouts

The four dominant DiT projections were measured at latent sequence lengths
13, 25, 50, 100, and 200. Each isolated variant used identical GPU-resident
inputs and logical weights, ten warmups, 100 operations per trial, five rotated
trials, a pre-sync-to-device-complete primary timer, and full owned contiguous
f32 CPU readback outside the timer. The S200 evidence is frozen at
`/tmp/irodori-v4-dit-layout-s200-attempt1-20260811`; the other four lengths are
frozen together at `/tmp/irodori-v4-dit-layout-multilength-attempt1-20260811`.

The short S13/S25/S50 routes remain unchanged. At S100/S200 the production
policy now retains a second, checkpoint-native column-major QKV+gate cache and
selects it only for the measured winning replay batches. The existing row-major
w2 and wo caches are also reused for long B2 projections. No hot-path host
copy or readback is introduced. The extra QKV+gate cache is 300 MiB for all
twelve released layers.

At S200, all 32 full-output comparisons were bit-identical. The isolated
four-projection workload estimated an 18.486 ms saving per request, of which
16.235 ms came from QKV+gate. A five-fresh-process-per-runtime production run
at `/tmp/irodori-v4-stage-long-projection-s8-attempt1-20260811` reproduced the
effect:

| boundary | Python RF | prior WGPU RF | current WGPU RF | current speed |
|---|---:|---:|---:|---:|
| device complete | 219.018 ms | 272.891 ms | 254.944 ms | 0.859x |
| full f32 CPU readback | 219.067 ms | 273.393 ms | 255.061 ms | 0.859x |

All 50 WGPU repetitions pass every latent/waveform, hash, determinism, and RF
work-count gate. The current codec also remains a strict all-sample winner at
171.236 ms device-complete versus Python 188.481 ms. RF is still about
35.9 ms behind Python at eight seconds, so projection layout is retained but
does not close the long-sequence objective by itself.

### Long-sequence command aggregation and rejected MLP fusion

The eight-second production binary was also screened with `tasks_max` values
16, 32, 64, and 128. Every arm used a fresh process, twelve repetitions with
the first two excluded, SubSlices, the same strict oracle, and both the
device-complete and full owned f32 CPU-readback boundaries. All latent and
waveform gates passed. The evidence is frozen at
`/tmp/irodori-v4-s8-tasks-sweep-attempt1-20260811`.

| `tasks_max` | RF device complete | RF + CPU readback | codec device complete | codec + CPU readback |
|---:|---:|---:|---:|---:|
| 16 | 255.335 ms | 255.468 ms | 169.911 ms | 170.182 ms |
| **32** | **251.714 ms** | **251.807 ms** | **168.911 ms** | **169.217 ms** |
| 64 | 255.737 ms | 255.854 ms | 169.617 ms | 169.902 ms |
| 128 | 265.232 ms | 265.329 ms | 170.470 ms | 170.718 ms |

The existing value 32 therefore remains the production policy at S200 as well
as the shorter lengths. Increasing aggregation does not recover the remaining
RF gap.

A production-disconnected one-dispatch WGSL candidate that fused SwiGLU with
the `w2` projection was also rejected. It was bit-identical, but B1/S200 took
2.911 ms versus 0.950 ms for the tuned production path, and B2/S200 took
5.864 ms versus 1.371 ms. Its evidence is frozen at
`/tmp/irodori-v4-dit-swiglu-w2-attempt1-20260811`; the rejected source was
removed rather than retained as dead production-adjacent code.

## Dynamic C192 residue decomposition

The compact residue-class d3/d9 path for decoder block 2 is now selected for
every admitted C192 decoder-family length, rather than only the two-second
shape. Large input packs use a checked two-dimensional dispatch when the
logical workgroup count exceeds Vulkan's per-dimension limit. The temporary
remains GPU-only and is consumed immediately by the convolution/Snake core;
there is no CPU transfer or persistent cache. Peak sequential temporary size
is 70.312 MiB at four seconds and 140.625 MiB at eight seconds.

The isolated rotating A/B is sealed at
`/tmp/irodori-v4-residue-dynamic-length-ab-attempt2-20260811`. It uses the same
input, weights, bias, and alpha for both variants, ten warmups, 50 operations
per trial, five trials, and a pre-sync-to-device-complete primary timer. Full
CPU readback and bitwise comparison happen after timing.

| audio | prior d3+d9 median | residue d3+d9 median | saving | speedup |
|---:|---:|---:|---:|---:|
| 4 s | 23.330 ms | 16.262 ms | 7.069 ms | 1.435x |
| 8 s | 46.900 ms | 32.859 ms | 14.040 ms | 1.427x |

Both dilations have strict non-overlapping timing ranges at both lengths. All
36,864,000 four-second outputs and all 73,728,000 eight-second outputs are
bit-identical to the preceding fused route, with zero uncaptured WGPU errors.

Fresh-process production validation with the exact release binary is preserved
at `/tmp/irodori-v4-residue-production-{s4,s8}-attempt1-20260811`. Each run has
two excluded warmups and ten measured repetitions. All twelve latent and
waveform comparisons pass the ten numerical gates, and each tensor family has
one deterministic hash.

| audio | prior WGPU codec | current device complete | CPU readback complete | Python device complete | remaining gap |
|---:|---:|---:|---:|---:|---:|
| 4 s | 111.296 ms | 103.487 ms | 103.729 ms | 90.519 ms | 12.968 ms |
| 8 s | 219.478 ms | 202.879 ms | 203.168 ms | 189.178 ms | 13.701 ms |

These production runs are single fresh-process validations, not replacements
for the five-process strict campaigns above. They demonstrate the expected
end-to-end direction and preserve the readback boundary, while the remaining
long-codec gap still requires optimization and a final five-process campaign.

## Dynamic C96 residue decomposition

The same checked pack/core implementation now covers the block-3 C96 d3/d9
calls. Channel count and length are both part of the compiled-kernel identity;
the selector admits only C96/C192, d3/d9, positive decoder-family lengths, and
the complete F32/shape/stride/device/resource contract.

The C96 isolated workloads are preserved separately because their wrappers
failed after the completed GPU work: attempt 2 exhausted its post-S4 NVML idle
settle, and attempt 3 hit a shell `PIPESTATUS` capture bug after S8. Neither
workload was repeated. Their frozen logs remain valid diagnostic evidence and
show bit-identical full outputs, strict timing non-overlap for each dilation,
and zero WGPU errors:

| audio | prior d3+d9 median | residue d3+d9 median | saving | speedup |
|---:|---:|---:|---:|---:|
| 4 s | 11.835 ms | 8.869 ms | 2.966 ms | 1.334x |
| 8 s | 23.432 ms | 17.558 ms | 5.874 ms | 1.335x |

The evidence paths are
`/tmp/irodori-v4-residue-c96-dynamic-ab-attempt2-20260811` for four seconds and
the corresponding `attempt3` for eight seconds.

Fresh-process production validation is complete at
`/tmp/irodori-v4-residue-c96-production-{s4,s8}-attempt1-20260811`. All twelve
latent and waveform comparisons pass, both tensor hashes are deterministic,
and the primary/secondary timer contract remains unchanged:

| audio | preceding device complete | current device complete | CPU readback complete | Python device complete | remaining gap |
|---:|---:|---:|---:|---:|---:|
| 4 s | 103.487 ms | 99.626 ms | 99.871 ms | 90.519 ms | 9.107 ms |
| 8 s | 202.879 ms | 195.959 ms | 196.266 ms | 189.178 ms | 6.781 ms |

This second promotion saves another 3.861 and 6.921 ms end to end. It remains
a single-process production validation; the final claim still requires a new
five-process campaign after the remaining long-codec work is optimized.

## Dynamic C384 residue decomposition

The compact residue path now also covers block-1 C384 d3/d9 calls. The C384
route preserves the preceding T128 convolution/Snake arithmetic order and uses
the same GPU-only packed temporary as the C96/C192 routes. Admission requires
an exact decoder-family length, F32 contiguous layouts, matching devices, and
all binding, alignment, shared-memory, and dispatch limits. Any mismatch falls
back to the preceding fused path.

The isolated A/B workloads compare the exact same inputs and parameters, with
ten warmups, 50 operations per trial, five rotating trials, device-complete
timing, and full output readback after timing:

| audio | prior d3+d9 median | residue d3+d9 median | saving | speedup |
|---:|---:|---:|---:|---:|
| 4 s | 12.238 ms | 8.021 ms | 4.217 ms | 1.526x |
| 8 s | 24.380 ms | 15.219 ms | 9.161 ms | 1.602x |

All outputs are bit-identical and all timing ranges are strictly non-overlapping.
The four-second diagnostic is frozen at
`/tmp/irodori-v4-residue-c384-dynamic-ab-attempt1-20260811`; its wrapper failed
only during the between-workload NVML idle settle, after the completed and
passing workload. The independently executed eight-second campaign is complete
at the corresponding `attempt2` directory. No failed workload was repeated.

Fresh-process production validation is sealed at
`/tmp/irodori-v4-residue-c384-production-{s4,s8}-attempt1-20260811`. Both
manifests verify, all twelve latent and waveform comparisons pass, and output
hashes are deterministic:

| audio | preceding device complete | current device complete | CPU readback complete | Python median | median gap |
|---:|---:|---:|---:|---:|---:|
| 4 s | 99.626 ms | 96.040 ms | 96.295 ms | 90.519 ms | 5.521 ms |
| 8 s | 195.959 ms | 186.060 ms | 186.418 ms | 189.178 ms | -3.118 ms |

The end-to-end reduction from this promotion is 3.586 ms at four seconds and
9.899 ms at eight seconds. Eight-second median parity is now closed. The
strict all-sample gate remains open: the current WGPU maximum is 187.495 ms
versus the prior five-process Python minimum of 185.204 ms. At four seconds,
the Python minimum is 88.883 ms and the current WGPU maximum is 97.316 ms.
Thus both lengths still require tail reduction and a new five-process campaign;
the eight-second result is a median win, not yet a final performance claim.

## Dynamic decoder stem

The direct T64/O32/Cin16 stem formerly admitted only latent length 50. Other
lengths fell back to the generic convolution even though the weights, channel
geometry, and padding were identical. The production kernel now templates the
runtime latent length, dispatches `ceil(L/64)` time workgroups, and includes the
length in its kernel-cache identity. The input stays GPU-resident; the existing
contiguity conversion is a no-op for the normal contiguous decoder input and
remains an on-device fallback for a view.

Exact-oracle-latent A/B workloads use checkpoint weights, the same input for
both variants, ten warmups, 100 executions per rotating trial, five trials, and
a pre-sync-to-device-complete primary timer. Full output readback and comparison
are outside timing:

| audio | latent L | Burn median | direct median | saving | speedup |
|---:|---:|---:|---:|---:|---:|
| 0.5 s | 13 | 2.029 ms | 1.003 ms | 1.026 ms | 2.022x |
| 1 s | 25 | 5.073 ms | 1.009 ms | 4.064 ms | 5.029x |
| 4 s | 100 | 9.732 ms | 1.415 ms | 8.317 ms | 6.879x |
| 8 s | 200 | 18.181 ms | 2.554 ms | 15.627 ms | 7.120x |

Every direct range is below the corresponding Burn minimum. All outputs are
finite; maximum absolute differences are at most `2.25e-5`, within the existing
stem screening contract. Evidence is sealed at
`/tmp/irodori-v4-stem-dynamic-{s0p5,s1,s4,s8}-attempt1-20260811`.

Fresh-process full-decoder validation for the two long lengths is sealed at
`/tmp/irodori-v4-stem-dynamic-production-{s4,s8}-attempt1-20260811`. All twelve
latent/waveform accuracy checks and deterministic hashes pass:

| audio | preceding device complete | current device complete | CPU readback complete | Python median | WGPU max | Python min |
|---:|---:|---:|---:|---:|---:|---:|
| 4 s | 96.040 ms | 86.950 ms | 87.154 ms | 90.519 ms | 87.534 ms | 88.883 ms |
| 8 s | 186.060 ms | 169.517 ms | 169.805 ms | 189.178 ms | 170.209 ms | 185.204 ms |

The final balanced five-process-per-runtime sweep of committed source is sealed
at `/tmp/irodori-v4-length-sweep-dynamic-stem-attempt1-20260811`. All five
codec lengths pass both the device-complete and readback-inclusive strict
all-point gates. The sweep used 50 measured points per runtime and length after
two warmups in every fresh process. Its top-level summary SHA-256 is
`951de406db60128ca492d38792eee02c1d8ac6d9bf7743921e3e770976964478` and
its full manifest SHA-256 is
`6fbd3a8f727f917e3bf01748d21e5d883eeb3a10c964423b3341e388f8622127`.

## S8 CubeCL autotune effort screen

The remaining eight-second RF gap was screened once with isolated CubeCL
autotune caches at `balanced`, `extensive`, and `full` effort. Each arm used a
fresh process, the same strict FP32 S8 fixture, `tasks_max=32`, two excluded
warmups, ten measured repetitions, and both the device-complete and owned-f32
CPU-readback boundaries. All RF work-count, latent, waveform, determinism, and
WGPU error gates passed. This is a single-process diagnostic, not a replacement
for the balanced five-process campaign.

| effort | RF device complete | RF + CPU readback | codec device complete |
|---|---:|---:|---:|
| **balanced** | **250.289 ms** | **250.397 ms** | **171.802 ms** |
| extensive | 251.962 ms | 252.091 ms | 173.005 ms |
| full | 252.299 ms | 252.420 ms | 173.361 ms |

Increasing autotune effort did not improve steady performance and made both
stages slightly slower. Production therefore retains CubeCL's balanced effort;
no runtime option or extra cache policy was added. The frozen evidence roots
are `/tmp/irodori-v4-s8-autotune-{balanced,extensive,full}-attempt1-20260811`.
Their `SHA256SUMS` digests are respectively `3987bedb...d58d5`,
`61a7b274...058e`, and `23528baf...a4ed`.
