# RTX 5070 Ti Laptop: k7 selector autotune (2026-08-19)

## Decision

Retain the released geometry selector for all twelve decoder k7 problems.

The accuracy-gated tuner evaluated five CubeK selector choices for every
combination of decoder block channel width and dilation. Several candidates
were slightly faster in isolated paired stage measurements, but none reached
the 2% per-operator adoption gate. The assembled selector then improved the
whole-graph paired block median by only `0.001077 ms` (`0.0070%`) and won 5 of
10 blocks, far below the 0.2% whole-graph gate. The emitted manifest therefore
contains the existing geometry choices, not the nominal stage winners.

This is an intentional no-change result. It prevents normal laptop clock and
power variation from becoming a device-name-specific production policy.

## Tuning protocol

The tuner covered all k7 convolution + post-cast Snake operators in the fixed
50-latent-frame decoder graph:

- output channels: `768`, `384`, `192`, and `96`;
- dilation: `1`, `3`, and `9` at every channel width;
- candidates: multi-row, single-row, no-swizzle, automatic stage
  partitioning, doubled stage partitioning, and the valid combinations exposed
  by the selector ADT;
- candidate validation: fixed F16 waveform oracle before timing acceptance;
- isolated decision gate: at least 2.0% median paired operator improvement;
- final decision gate: at least 0.2% median paired whole-graph improvement;
- final comparison: ten alternating ABBA/BAAB blocks with an owned contiguous
  CPU F32 waveform at the readback-complete boundary.

The run used one new named CubeCL environment and a newly created artifact
directory. Historical `/tmp`, prior autotune databases, and previous timing
samples were not imported or pooled.

## Per-operator outcome

The best measured alternative for each problem is shown below. Negative delta
means that the candidate was faster than the released selector.

| output length | channels | dilation | released | nominal best alternative | paired delta | improvement | decision |
| ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| 600 | 768 | 1 | single-row | double partition | -0.001152 ms | 0.3894% | retain |
| 600 | 768 | 3 | single-row | no-swizzle | -0.004480 ms | 1.4079% | retain |
| 600 | 768 | 9 | single-row | no-swizzle | -0.000512 ms | 0.1594% | retain |
| 6,000 | 384 | 1 | multi-row | single/auto partition | +0.122496 ms | 0% | retain |
| 6,000 | 384 | 3 | multi-row | single-row | +0.126336 ms | 0% | retain |
| 6,000 | 384 | 9 | multi-row | single-row | +0.125184 ms | 0% | retain |
| 48,000 | 192 | 1 | single-row | single/auto partition | -0.003584 ms | 0.3348% | retain |
| 48,000 | 192 | 3 | single-row | no-swizzle/auto partition | -0.003456 ms | 0.3153% | retain |
| 48,000 | 192 | 9 | single-row | no-swizzle | -0.001024 ms | 0.0902% | retain |
| 96,000 | 96 | 1 | single-row | single/auto partition | -0.007424 ms | 1.2680% | retain |
| 96,000 | 96 | 3 | single-row | single/auto partition | +0.001664 ms | 0% | retain |
| 96,000 | 96 | 9 | single-row | no-swizzle | +0.001408 ms | 0% | retain |

The output lengths are physical decoder-stage lengths for the 50-frame input,
not twelve independent audio fixtures.

## Final whole-graph comparison

| Boundary | candidate selector | geometry control |
| --- | ---: | ---: |
| device-complete median | 15.147374 ms | 15.328989 ms |
| readback-complete median | 15.904066 ms | 16.107675 ms |

The unpaired medians reflect the two clock/power regimes seen on this laptop
and are not used as the decision statistic. The paired device block delta was
`-0.001077 ms`, with a range of `[-0.227581, 0.248216] ms`; the candidate won
5/10 blocks. The two routes produced the same SHA-256 waveform hash:
`04daa96513fe33c680bc0ca475b2182936074a4578312a76f3dfab821f49cc38`.

All candidates and both final routes passed the F16 waveform gate:

- max absolute error: `3.417968750e-3`;
- RMSE: `2.008751940e-4`;
- SNR: `56.622776 dB`;
- cosine: `0.999998916470`;
- uncaptured WGPU errors: `0`.

## Artifact and pins

```text
/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-autotune-20260819-attempt1
```

`SHA256SUMS` covers the raw log, autotune event log, CubeCL database, input and
binary pins, GPU records, and emitted selector manifest.

- source: `109990256273f6c83eb5dd993d96f1e78c5d81b0`
- profiler binary SHA-256:
  `1f680a820ba78cdc5d710c0f6e7984a288b7cb063576409b84d4fcd16196246f`
- emitted manifest SHA-256:
  `274c5f7a9a371a22634b8fe953ecf10f0138fe604f9f8a854bd4c01dd04f7e04`
- F16 oracle SHA-256:
  `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- converted codec SHA-256:
  `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- Vulkan adapter / CUDA-NVML index: `0` / `0`
- PCI bus ID: `00000000:01:00.0`
- total/free VRAM before campaign: `12227 / 11774 MiB`

## Implication for portable autotuning

The selector is expressed in terms of convolution geometry and typed choices,
not a hard-coded GPU model string. A production restore path must additionally
bind an approved manifest to the runtime/compiler, precision, driver/device
identity, allocator policy, model and codec hashes, kernel source hash, and
shape manifest. A mismatch must fail closed to the released geometry policy.

The next implementation step is therefore a sealed application-level manifest
set keyed by output shape. It should reuse the existing accuracy-approved
runtime identity rather than treating this profile-only schema-1 JSON file as
a portable production cache. Candidate timing remains process-local; only an
accuracy-approved selection receipt may cross processes.
