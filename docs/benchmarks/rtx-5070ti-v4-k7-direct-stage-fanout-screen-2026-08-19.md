# RTX 5070 Ti Laptop: direct-stage k7 fan-out screen (2026-08-19)

## Decision

Do not route production through the direct-stage k7 fan-out loader.

The experiment removes the request-time OIK-to-OKI weight-layout copy and
reads each physical NHWC halo vector once before distributing it to all
logical im2col consumers. It is numerically valid and deterministic, but the
50-latent-step whole-decoder median is 60.49% slower at the device-complete
boundary than the current repack control. The scalar-owned shared-memory
scatter and strided MMA consumption cost much more than the removed global
reads and layout-copy dispatch.

The generic direct-stage partition support and the loader remain available
only through the `profile` feature as negative knowledge and as a correctness
probe. The default `ErasedStagePartition` path is unchanged, so this result
does not alter production routing.

## Structural question

The previous production route is not literally one dispatch per complete k7
operator: it first materializes the checkpoint-native OIK weight into an OKI
contiguous layout, then launches CubeK convolution + bias + Snake. This screen
tested whether the copy could be removed without retaining a duplicate 32 MiB
prepared weight.

The candidate uses a different division of work:

```text
checkpoint-native OIK weight (bound directly)
NHWC input halo vector read once
  -> fan out to every consuming (output-time, channel, kernel) stage cell
  -> direct concrete Stage::tile consumption by CubeK MMA
  -> bias + post-cast Snake
  -> F16 output
```

The fan-out stage is scalar-owned to make each destination write race-free.
It exposes row-major strided tiles directly to the MMA instead of converting
the stage to CubeK's closed `StageTile` enum. Ordinary affine CubeK routes
continue to use their original erased-stage implementation; an earlier smoke
test showed that globally replacing it would regress the existing path.

## Fresh campaign

Artifact root:

```text
/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-fanout-recorded-20260819-attempt1
```

`SHA256SUMS` covers the raw logs, source/binary/input pins, GPU record and the
two independent CubeCL environments. A mistyped expected fixture SHA was
rejected before GPU measurement and is retained as
`raw/control-invalid-fixture-pin.log`; it is not a sample.

Pins:

- source base: `9e4eade070fcf6d4412677d2b6e99cf8a6791589`
- dirty source patch SHA-256:
  `a48f97fc9cf49a4151a85828c6eb6c36ea48400e2b04b4c4e0cd87ac68c8ac48`
- profiler binary SHA-256:
  `4081a0f861fc55bb41c4dab4b4fdefd38a3f6a0c7504dd994c48e0a66efcd590`
- F16 oracle SHA-256:
  `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- converted codec SHA-256:
  `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- Vulkan adapter / CUDA-NVML index: `0` / `0`
- PCI bus ID: `00000000:01:00.0`
- total/free VRAM before campaign: `12227 / 11774 MiB`

Both conditions used strict F16 execution with the same F16 oracle, 50 latent
steps (96,000 output samples), standalone block boundaries, 3 warmups and 10
measured requests. Control and candidate used separate fresh CubeCL cache
roots; no historical timing sample was pooled.

## Result

| Boundary | repack control median | direct-stage fan-out median | candidate delta |
| --- | ---: | ---: | ---: |
| enqueue-complete | 0.586928 ms | 0.485936 ms | -17.21% |
| device-complete | 14.154055 ms | 22.716044 ms | +60.49% |
| readback-complete | 15.034406 ms | 23.520720 ms | +56.45% |

The lower enqueue time is not useful: the device work is substantially
larger. The comparison includes CPU-owned contiguous F32 waveform readback in
both readback-complete values.

Accuracy and stability:

| Metric | repack control | direct-stage fan-out |
| --- | ---: | ---: |
| waveform max absolute error | 3.417969e-3 | 3.906250e-3 |
| waveform RMSE | 2.008752e-4 | 2.018384e-4 |
| waveform SNR | 56.622776 dB | 56.581227 dB |
| waveform cosine | 0.999998916470 | 0.999998907075 |
| repeated output hash | stable | stable |
| uncaptured WGPU errors | 0 | 0 |

The hashes differ because the load/reduction ordering differs, but every
repeat inside each condition is identical and both pass the F16 waveform
gate. The focused partial-M/multi-K-stage Vulkan test also compares the old
halo route and this fan-out route against the repack reference.

## Interpretation and next step

The experiment establishes that avoiding copies is not sufficient by itself.
The current copy pays once to create a layout that gives CubeK vectorized,
affine stage access for the much larger convolution. Replacing that copy with
scalar fan-out turns a small, coalesced preprocessing cost into repeated shared
memory stores and strided MMA reads.

Production therefore retains the copy. The next structural experiment must
remove a global producer/consumer round trip without degrading the hot MMA
layout, or reuse an already-optimal prepared layout without adding duplicate
persistent storage. Per-GPU selector tuning is deliberately deferred until
those structural routes have been exhausted.
