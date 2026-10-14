# Cache, decoder and kernel acceptance, October 14, 2026

This campaign follows the CPU-only
[preparation](../planning/optimization-preparation-2026-10-14.md). Device
measurements were subsequently authorized. FP32 remains the default, and the
codec remains FP32 when the model uses FP16. The new features are independently
optional; no extra unsafe boundary, dependency version or format-crate change
is involved.

## Environment and measurement boundaries

- RTX 5070 Ti **Laptop** GPU, 12 GiB, NVIDIA 595.99.02, Vulkan.
- Intel Graphics (ARL), Vulkan, Mesa 26.0.8-1ubuntu0.3; vendor 32902, device 32103.
- Burn 0.22.0, CubeCL 0.11.0, CubeK 0.3.0; base `129c407` plus this change.
- Release binaries frozen separately for each arithmetic/route experiment.
  Runtime allocator: Adaptive. Command-buffer limit: 32 unless explicitly swept.
- Private copies of closed tuning caches; ordinary driver disk caching enabled.
  Timed startup runs reject fresh autotuning decisions. Profiled requests are
  diagnostics and are not used as production latency samples.
- The OS file cache was not flushed. Loading measurements cover fresh processes
  using warmed local files; cold-storage performance was not measured.

The artifact directory is
`/home/sanzentyo/benchmark-artifacts/irodori-optimization-acceptance-20261014/`.
It retains commands, binary hashes, adapter receipts, raw reports, output hashes,
private caches and unsuccessful trials. `scripts/benchmark_synthesis_variants.py`
executes explicit variants serially, seeds their caches, rotates their order and
checks every latent and PCM result against a frozen reference.

A concurrent sam3.cpp task was discovered during the RoPE experiment. Those
timings are excluded entirely; its numerical outputs remain useful. Subsequent
GPU batches use `/home/sanzentyo/benchmark-artifacts/gpu-benchmark.lock` across
the complete batch, with immediate start/release notifications to the other
task. The initial startup and stage-diagnostic campaigns preceded the observed
overlap. They were serial within this task but did not yet use the shared lock.

## Startup: generated shaders complement the native driver cache

MF, 50 frames / 2 seconds, one warmup and three ordinary requests per process.
Four device/precision pairs × five cache/format variants × one seed plus five
fresh measured processes: **120 successful processes, 100 timed**. All produced
bit-identical latent and audio; all timed runs had zero new tuning decisions.
The following values are medians in milliseconds, not streaming first-audio
latency. “First request” ends when the complete PCM is available.

| Device/model | Cache layers | Ready | First request | Resident request |
|---|---|---:|---:|---:|
| RTX FP32 | Off | 2281.11 | 4002.64 | 77.28 |
| RTX FP32 | Native | 2076.08 | 1437.44 | 77.39 |
| RTX FP32 | SPIR-V | 2059.79 | 2597.21 | 79.34 |
| RTX FP32 | Both | 1818.16 | 142.51 | 76.23 |
| RTX FP16 | Off | 1814.62 | 2763.82 | 57.51 |
| RTX FP16 | Native | 1589.24 | 1155.70 | 53.59 |
| RTX FP16 | SPIR-V | 1680.41 | 1728.27 | 54.18 |
| RTX FP16 | Both | 1397.81 | 124.13 | 54.43 |
| Intel FP32 | Off | 2259.75 | 1296.19 | 684.29 |
| Intel FP32 | Native | 2268.42 | 1292.49 | 684.97 |
| Intel FP32 | SPIR-V | 2030.16 | 712.46 | 684.60 |
| Intel FP32 | Both | 2127.43 | 708.45 | 683.63 |
| Intel FP16 | Off | 1549.82 | 1368.56 | 682.51 |
| Intel FP16 | Native | 1655.28 | 1390.40 | 682.53 |
| Intel FP16 | SPIR-V | 1476.12 | 725.66 | 682.20 |
| Intel FP16 | Both | 1416.16 | 722.48 | 683.87 |

The benefit is process startup/compilation. These data do not establish a
general resident-inference improvement. The archived startup JSON uses the old
name `first_audio_seconds` for **ready plus the first complete request**; the
harness now calls that quantity `ready_plus_first_request_seconds` to avoid
confusing it with actual streaming TTFA.

The eight generated-shader lifecycle checks cover native/SPIR-V OFF/ON on both
devices, environment A → B → A, changed inputs and durable entry presence.
All passed. The historical Intel Turso shared-reader assertion did not recur
in the 120-process startup campaign; this is evidence of this campaign's
behavior, not a root-cause fix for that earlier assertion.

The final binary also passed **16 correctness-only processes**: two concurrent
clients sharing one private cache, across both devices and both precisions, with initial
versus restored snapshots. Latent, full PCM, streamed PCM and trimmed PCM matched
the frozen outputs. This deliberate same-device concurrency checks cache use;
none of its timings are performance evidence. It still does not establish a
root-cause fix for the historical database assertion.

The combined-cache benefit was then rechecked under the **shared GPU lock**:
four device/precision pairs × off/both/both-packed × one seed plus three fresh
measured processes, **48 successful processes, 36 timed**, zero new tuning
decisions in timed runs and bit-identical latent/PCM throughout. Median
milliseconds, readiness plus the complete first request (still not stream TTFA):

| Device/model | Off | Both caches | Both + packed codec |
|---|---:|---:|---:|
| RTX FP32 | 6351.30 | 2054.86 | 1972.28 |
| RTX FP16 | 4603.53 | 1599.18 | 1509.51 |
| Intel FP32 | 3601.32 | 2879.63 | 2815.28 |
| Intel FP16 | 2930.52 | 2215.07 | 2023.49 |

For example, RTX FP32's first request **after loading** fell from 4017.39 to
146.32 ms with both caches; loading itself still took 1916.86 ms. Reporting
only the request time as application startup would overstate the improvement.
The two timing components and their sum have separately computed medians.

## Packed FP32 decoder

The archive preserves original FP32 bits and adds layouts already consumed by
the WGSL kernels. File, borrowed-memory and shared-owned-memory decoding matched
the ordinary decoder bit for bit on both devices. Conversion took 1.03 seconds,
with 437,664 KiB peak process RSS. The archive is 428,854,272 bytes versus the
320,048,128-byte decoder-only source: about 34% larger.

Both cache layers were enabled for this loading comparison. Three fresh measured
processes per entry from the shared-lock confirmation campaign:

| Device/model | Original codec load, ms | Packed codec load, ms | Original ready, ms | Packed ready, ms |
|---|---:|---:|---:|---:|
| RTX FP32 | 228.57 | 156.73 | 1916.86 | 1831.78 |
| RTX FP16 | 238.23 | 172.32 | 1470.49 | 1394.74 |
| Intel FP32 | 270.25 | 163.60 | 2169.24 | 2087.76 |
| Intel FP16 | 294.64 | 163.56 | 1471.41 | 1304.82 |

All raw/packed outputs matched exactly. Packed loading is accepted as an opt-in
tradeoff between file size and load preparation. Its decoder-only archive can
reuse saved voices but cannot encode new reference audio. `wgslpack` remains a
separate, unchanged Sans-I/O format crate.

Median peak **process** RSS for raw → packed was 1081 → 1065 MiB (RTX FP32),
954 → 766 MiB (RTX FP16), 858 → 857 MiB (Intel FP32), and 744 → 587 MiB
(Intel FP16). This is not device VRAM. Combined private cache files occupied
7.6–10.4 MiB per device/precision case after seeding all variants, including
pre-existing tuning entries; those totals do not isolate each cache layer's size.

## Bounded streaming graph reuse

The first evaluation used frozen binary `v2`: MF short (50 frames / 2 seconds)
and long female (250 frames / 10 seconds), both devices and model precisions.
The first window contains 12 output frames and later windows 96, with the same
overlap and final crop as ordinary decoding. Four graph entries permit all
window geometries in these workloads. One-entry caches were also measured for
the two long FP32 cases.

Each variant has one seeding process and three measured processes, with one
cold streamed request followed by five resident streamed requests per process:
**72 successful processes, 54 timed**. Every latent, full PCM, streamed PCM and
trimmed PCM matched; timed processes made no new tuning decisions. The entire
batch held the common lock and the CUDA monitor recorded zero conflicts.
Here “cold” means the first streamed request after ordinary inference/cache
preparation, not a cold boot or an empty compilation cache.

Median milliseconds across the three process medians, ordinary → four entries:

| Device/model | Frames | First audio, warm | Complete streamed request, warm |
|---|---:|---:|---:|
| RTX FP32 | 50 | 62.77 → 63.08 | 87.26 → 87.11 |
| RTX FP32 | 250 | 116.88 → 116.24 | 277.21 → 276.29 |
| RTX FP16 | 50 | 40.56 → 38.11 | 64.97 → 62.30 |
| RTX FP16 | 250 | 71.73 → 66.44 | 231.10 → 223.34 |
| Intel FP32 | 50 | 482.23 → 482.93 | 857.88 → 855.42 |
| Intel FP32 | 250 | 1278.44 → 1276.98 | 3574.74 → 3570.08 |
| Intel FP16 | 50 | 480.12 → 479.21 | 854.86 → 853.77 |
| Intel FP16 | 250 | 1260.92 → 1286.93 | 3559.67 → 3479.06 |

RTX FP16 improved complete streamed latency by 3.4–4.1%; Intel FP32 and RTX
FP32 gains were small. Intel long FP16 improved completion but increased first
audio latency. This does not justify enabling graphs by default on every device.

The original capture path discarded its completed priming output and replayed
the same input. That made the first streamed request expensive: short RTX FP32
86.76 → 178.78 ms and short Intel FP32 859.13 → 1666.48 ms, ordinary → graphs.
With only one graph entry, repeated eviction raised the long FP32 warm request
from 277.21 to 419.09 ms on RTX and 3574.74 to 5183.05 ms on Intel. These values
describe `v2` before the priming-output reuse change, not that change's result.

The graph cache bounds entry count and input geometry rather than bytes. For
RTX FP16 long, total device allocator bytes in use after replay rose from
2563.7 to 2565.1 MiB, while reserved bytes rose from 3435.4 to 3900.9 MiB.
This is whole-device allocator state after replay, **not peak VRAM**. Capture
preparation can reserve substantially more memory than the retained live tensors.

The final implementation reads the completed priming result into owned CPU
memory **before** recording, releases its tensor, and returns the values only
after capture succeeds. Cache hits use the same replay boundary. This avoids
the extra first replay without exposing uninitialized captured output or adding
unsafe code. Existing graph-only consumers do not perform the new readback.

An interleaved `v2` versus final-binary comparison covered five workloads, with
one seed and two measured processes per binary and three warm streamed requests
per process: **30 successful processes, 20 timed**. All outputs were exact;
timed runs had zero fresh tuning decisions. The common lock was held throughout,
with zero foreign CUDA samples. Median milliseconds, previous → priming reuse:

| Device/model | Frames | First audio, cold | Complete stream, cold | Complete stream, warm |
|---|---:|---:|---:|---:|
| RTX FP32 | 50 | 125.19 → 115.70 | 179.21 → 145.42 | 88.33 → 88.51 |
| RTX FP32 | 250 | 262.86 → 231.13 | 527.36 → 393.65 | 294.00 → 276.02 |
| RTX FP16 | 250 | 155.58 → 146.06 | 407.73 → 306.87 | 223.15 → 225.34 |
| Intel FP32 | 50 | 856.00 → 668.91 | 1568.55 → 1067.95 | 839.58 → 839.84 |
| Intel FP32 | 250 | 1458.89 → 1272.78 | 5233.19 → 3693.46 | 3563.54 → 3552.47 |

Cold completion improved 18.9–31.9%. Warm completion changed from a 6.1%
improvement to a 1.0% regression in this small comparison. Warm first-audio
latency was not uniformly better: RTX FP16 long changed from 65.42 to 69.79 ms.
Accept priming reuse for the substantial cold improvement without claiming a
universal resident/first-audio speedup. Initial capture remains more expensive
than ordinary decoding, so graph reuse stays opt-in with explicit bounds.

## RF profile and remaining kernel experiments

The stage profiler now includes autotuned SwiGLU expansion and condition
encoding, which were absent from the first diagnostic capture. The corrected
16 captures preserve all frozen latent/PCM outputs exactly. For RF female,
150 frames / 6 seconds, selected FP32 stage totals are:

| Stage | RTX, ms | Intel, ms |
|---|---:|---:|
| MLP expansion + SwiGLU | 446.46 | 3581.15 |
| QKV projection | 213.75 | 2666.57 |
| MLP contraction | 167.11 | 1824.88 |
| SDPA | 47.34 | 1330.95 |
| Output projection | 97.70 | 783.45 |
| Conditioning | 27.60 | 170.34 |
| Production codec | 77.09 | 1151.65 |

The codec measurement covers the actual cross-block-fused decoder. Standalone
block measurements are not presented as a breakdown of that fused graph.

Kernel and command-buffer experiments are recorded separately from the accepted
cache/loading changes.

The first candidate replaced fused MLP expansion with separate gate/value
projections and the existing pair-SwiGLU epilogue. Five resident requests after
two warmups per variant, same frozen binary except that route. The shared-lock
Intel comparisons rejected it:

| Workload/model | Existing, ms | Split, ms | Change |
|---|---:|---:|---:|
| RF female FP32 | 11830.00 | 21056.81 | +78.0% |
| RF female FP16 | 12044.62 | 15846.15 | +31.6% |
| MF short FP16 | 674.88 | 726.27 | +7.6% |

The earlier exploratory RTX/Intel-short-FP32 pairs also got slower, but were
collected before the shared-lock protocol. No route change was adopted. The
aggregate CUDA-process monitor exits with status 2 because it includes the
intentional SIGSTOP interval when sam3.cpp was allowed to run. All benchmark
children exited successfully; `split-isolation.json` records zero foreign CUDA
clients during the accepted lock interval (20:11:04–20:20:27 JST). The monitor
does not enumerate every Vulkan client, so the cooperative lock is essential.

A second candidate exposed the existing 64-column / K16 tile to the validated
projection tuner. Eight workload/device/precision pairs, two variants and five
resident requests per variant completed under the common lock, with zero
foreign CUDA samples. All **32** new-tile local checks passed numerical
validation, but none won its timing comparison. Its local median was 1.25–5.84×
the fastest existing candidate. The additional tile and tuning-schema change
were removed.

| Workload/model | Existing, ms | With c64 candidate, ms |
|---|---:|---:|
| MF short / RTX FP32 | 76.46 | 82.97 |
| RF female / RTX FP32 | 1167.64 | 1000.19 |
| MF short / RTX FP16 | 53.37 | 52.84 |
| RF female / RTX FP16 | 445.00 | 435.03 |
| MF short / Intel FP32 | 681.56 | 683.32 |
| RF female / Intel FP32 | 11899.06 | 11924.45 |
| MF short / Intel FP16 | 683.96 | 676.27 |
| RF female / Intel FP16 | 12132.14 | 12106.91 |

The RTX FP32 RF difference is **not evidence for c64**: that tile was never
selected. The experimental schema change also forced existing candidates to be
measured again, changing their selection and the final FP32 rounding. A control
using the unchanged binary and fresh projection decisions isolates this effect:

| RTX FP32 control | Saved projection decisions, ms | Re-measured existing candidates, ms |
|---|---:|---:|
| MF short | 75.58 | 75.08 |
| RF female | 1154.54 | 1004.15 |

Only projection entries in a closed, private copy of the cache were cleared;
the binary, candidate set, schema, other tuning decisions and compilation caches
were unchanged. RF improved 13.0%, with latent max error 0.000246 and audio SNR
94.300 dB against the official FP32 fixture, within the existing gates. MF output
was bit-identical; RF output changed rounding as expected from selecting a
different existing kernel. The faster RF decisions remain persisted in the
`retune-control/rf-female-rtx-fp32-fresh-projection-cache/` artifact. This does not
justify forcing a particular kernel or invalidating every user's cache. Normal
users can evaluate a separate fresh cache with the existing `--cache-dir` option.

The unchanged binary was also run with `CUBECL_WGPU_MAX_TASKS` set to 8, 32,
64 and 128. The variant order rotates between workloads. Five resident requests
after two warmups, milliseconds, common lock and zero foreign CUDA samples:

| Workload/device/model | 8 | 32 (default) | 64 | 128 |
|---|---:|---:|---:|---:|
| RF / Intel FP32 | 11975.67 | 11907.34 | 11796.73 | 11777.91 |
| RF / Intel FP16 | 12189.68 | 12052.51 | 12095.37 | 12081.69 |
| RF / RTX FP32 | 1179.37 | 1158.32 | 1152.99 | 1151.44 |
| MF short / Intel FP32 | 697.20 | 683.02 | 682.26 | 682.06 |

All sixteen processes preserved frozen latent and PCM bits. The largest
improvement over 32 was 1.1%, and FP16 favored 32. These results do not justify
another runtime selection layer or a default change; retain 32. This sweep does
not establish that the existing kernels saturate either GPU.

## Long text and FP16 arithmetic

Stage capture schema 2 saves the **actual first-forward text condition**,
including compacted tokens and CFG rows. The initial implementation encoded a
separate unconditioned input, which was unsuitable for diagnosing the actual
request. The reader rejects the obsolete schema. Payload checksums, shapes,
fixture/model/codec identity and sampler contracts are checked before comparison.
The first divergent saved stage is an observation, not proof of its cause.

Sixteen offline comparisons of those captures distinguish precision changes
from device changes. MF FP32 versus FP16 first differs at input noise because
the selected model-precision sampler state rounds that input. RF with explicit
FP32 state retains identical noise and first differs at text state. Comparing
RTX against Intel at the same precision also first differs at text state for
all four workloads. These observations do not isolate a single faulty kernel.

Audio SNR against the frozen official CUDA FP32 fixture, in dB:

| Workload | RTX FP32 | RTX FP16 | Intel FP32 | Intel FP16 |
|---|---:|---:|---:|---:|
| MF short, 50 frames | 93.738 | 31.188 | 99.190 | 39.136 |
| MF long female, 250 frames | 74.265 | 25.665 | 66.957 | 21.435 |
| MF male, 150 frames | 78.380 | 25.376 | 76.327 | 12.282 |
| RF female, 150 frames | 94.686 | 36.154 | 88.694 | 40.144 |

RF FP16 explicitly retains FP32 sampler state. Long-MF FP32 cases already miss
the existing 85 dB audio gate; it has not been relaxed. Cache/packing/graph
changes must preserve the existing outputs, not claim that these pre-existing
cross-backend precision gaps have disappeared.

A candidate computed ModernBERT rotary products and addition in FP32, with
model-rounded coefficients and one final cast, matching the inspected PyTorch
arithmetic boundary. All four FP32 controls stayed bit-identical. FP16 outcomes
were mixed: RTX male improved 25.376 → 35.514 dB, while RTX long female fell
25.665 → 20.367 dB and Intel male fell 12.282 → 10.009 dB. Matched CUDA/XPU FP16
references also gave mixed results. The global arithmetic change was **rejected**
and removed; its patch and all twelve results remain in the artifact directory.
No device/shape-specific precision fallback was introduced.

## Final verification

The final release binary passed five graph-owner tests, real-codec acceptance
on RTX and Intel (including changed values, crop, eviction, cancellation and
environment generation), 16 concurrent-cache processes, and 12 real CLI
processes. CLI WAV files were bit-identical across ordinary, packed and captured
decoders for each device/precision pair. The CLI uses actual text, two warmups
and five measured requests; it is a consumer check rather than an additional
fixed-order loading-performance claim.

Formatting, all-feature/all-target Clippy with warnings denied, relevant feature
checks and Python checks passed. Earlier unchanged CPU/schema/vendor tests and
dependency-graph evidence are reused with their recorded source identities.
The [review record](../review/optimization-acceptance-2026-10-14.md) lists the
scope, fixes and validation boundaries. FP32 remains the default; all new cache,
packing and graph behavior is optional. The measured kernels do not establish
complete hardware utilization or a new across-framework speed advantage.
