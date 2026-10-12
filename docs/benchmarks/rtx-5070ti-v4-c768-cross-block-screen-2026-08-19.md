# RTX 5070 Ti Laptop: C768 cross-block fusion screen (2026-08-19)

## Decision

Do not extend the production cross-block fusion to C768 with any of the three
tested projection routes.

The adopted C384/C192 path removes the final pointwise raw tensor and the next
block's standalone Snake. C768 was the remaining equivalent boundary. A
profile-only, fail-closed implementation tested:

1. the existing direct WGSL pointwise core extended only for the C768
   activated-only store;
2. CubeK CMMA with multi-row partitioning;
3. CubeK CMMA with single-row partitioning.

All three remove the intermediate allocation/write/read and one dispatch. None
meets the production improvement gate. The direct kernel is clearly slower;
CubeK multi-row is slightly slower; CubeK single-row is statistically neutral
at `-0.003472 ms` median paired block delta and wins exactly 10/20 blocks.
Production remains the released C768 pointwise/finalizer plus standalone Snake.

## Isolated design

The experiment does not add C768 to the general production direct-kernel shape
set. Only the final block0 pointwise producer can select the candidate. All
other C768 pointwise operations and all C384/C192/C96 paths retain their
released implementations.

The candidate graph preserves the F16 storage boundary:

```text
pointwise accumulator
  -> bias + F16 shortcut
  -> F16 round
  -> promote to F32
  -> Snake
  -> F16 activated-only output
```

The direct and CubeK implementations use different accumulation orders from
the production projection, so their final hashes differ. This is not a missing
post-cast boundary: all variants pass the F16 waveform gate, are deterministic
within route, and report zero uncaptured WGPU errors.

The runner uses an ADT for CubeK row partition (`Single` or `Multi`) and an
ABBA/BAAB same-process comparison. A contract miss returns an error before a
sample can be recorded; it never falls back under the candidate label.

## Fresh paired measurements

Each condition uses the same fixed 50-latent-step F16 fixture, five alternating
candidate/control warmups, 20 ABBA/BAAB blocks, two samples per route per
block, and the same CPU-owned contiguous F32 waveform readback boundary. Each
run has a separate fresh CubeCL environment. Old `/tmp` and historical timing
samples were not pooled.

| Candidate | candidate device median | paired production median | median block delta | winning blocks | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| direct WGSL C768 | 14.350247 ms | 14.210940 ms | +0.093533 ms | 5/20 | reject |
| CubeK multi-row | 14.075471 ms | 14.026274 ms | +0.041666 ms | 8/20 | reject |
| CubeK single-row | 13.942673 ms | 13.913893 ms | -0.003472 ms | 10/20 | reject / neutral |

Readback-complete medians were respectively:

| Candidate | candidate | paired production |
| --- | ---: | ---: |
| direct WGSL C768 | 15.160553 ms | 14.956700 ms |
| CubeK multi-row | 14.936950 ms | 14.766108 ms |
| CubeK single-row | 14.751587 ms | 14.734413 ms |

The laptop alternated between roughly 14 ms and 18 ms clock/power regimes.
Separate-process screening therefore produced misleading medians; only the
paired block deltas above are used for the decision.

## Accuracy

| Route | max absolute error | RMSE | SNR | cosine | output hash |
| --- | ---: | ---: | ---: | ---: | --- |
| production | 3.417969e-3 | 2.008752e-4 | 56.622776 dB | 0.999998916470 | `04daa965…49cc38` |
| direct C768 | 3.417969e-3 | 2.029668e-4 | 56.532803 dB | 0.999998893793 | `b03bbc06…dd8abf` |
| CubeK single/multi | 4.150391e-3 | 2.084174e-4 | 56.302623 dB | 0.999998837626 | `b907d8e8…7849ad` |

The same route hash remained stable throughout every 40-sample paired run.

## Artifact and environment

```text
/home/sanzentyo/benchmark-artifacts/irodori-v4-c768-cross-block-screen-20260819-attempt1
```

`SHA256SUMS` covers all raw logs, separate cache databases, source/binary pins,
and GPU metadata.

- source base: `0e672f56dd441d9f9073148fcda4e3cb8e6bc015`
- final source patch SHA-256:
  `acafdee3c5e751a524c30dc9efa43b055a82212b09f47b67bb9861dbcca32de4`
- final profiler binary SHA-256:
  `1f680a820ba78cdc5d710c0f6e7984a288b7cb063576409b84d4fcd16196246f`
- oracle SHA-256:
  `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- converted codec SHA-256:
  `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- Vulkan adapter / CUDA-NVML index: `0` / `0`
- PCI bus ID: `00000000:01:00.0`
- total/free VRAM before campaign: `12227 / 11774 MiB`

## Interpretation

The result closes the remaining simple pointwise-to-Snake block-boundary
opportunity. Removing an intermediate is beneficial only when the producer
core and physical output store remain at least as efficient as the released
operators. Here, the direct projection loses too much compute throughput; the
CubeK NCL-oriented transformed store consumes the saved dispatch/memory margin.

Single-row versus multi-row is now represented explicitly and measured in
whole-graph context instead of hidden behind a GPU-name rule. Neither is
accuracy-approved for production because neither provides a repeatable speed
gain. Subsequent autotuning must retain this no-change outcome when the best
candidate fails its minimum-improvement threshold.
