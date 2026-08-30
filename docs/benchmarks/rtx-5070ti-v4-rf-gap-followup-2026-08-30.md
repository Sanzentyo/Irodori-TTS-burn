# RTX 5070 Ti / strict-FP32 40-step RF gap follow-up (2026-08-30)

## Scope and pins

This campaign profiles the remaining strict-FP32 RF gap for the fixed
489-frame Voice Design request. It uses 40 Euler evaluations, physical forward
batches `B3 x 20 + B1 x 20`, 12 layers, and 480 block calls. PyTorch and WGPU
perform the same semantic work but do not use the same operator graph. RF time
is device-complete, from the pre-stage device synchronization through RF device
completion; codec and final readback are excluded.

- fresh campaign root:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-rf-gap-profile-20260830-attempt1`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, Vulkan, driver 595.71.05
- CUDA/NVML index: 0; PCI bus ID: `00000000:01:00.0`
- physical VRAM: 12,227 MiB
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- converted decoder SHA-256:
  `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- candidate binary SHA-256:
  `1f6926ddf394b445f25a3c3fab58f0880bc212bfea1dccde90f58bada2c1ea60`

No timing from an older `/tmp` artifact or an earlier campaign is pooled into
these results. A copied cache is used only in explicitly labelled
`restored-cache` diagnostics; every reported request is a new external process.

## Root cause: coarse inner-matmul autotune key

The adopted SDPA graph is:

```text
Q @ K^T -> in-place scale/mask/softmax -> probabilities @ V
```

The top-level route did not change, but two fresh CubeCL environments selected
different algorithms for the probability-value product. The exact product is
approximately `[...,489,511] @ [...,511,64]`; CubeCL maps it to the coarse key
`(m=512,n=64,k=512)`.

| inner cache | selected PV algorithm | B3 SDPA / 240 calls | B1 SDPA / 240 calls |
|---|---|---:|---:|
| earlier fresh cache | SimpleUnit / MinTile | 469.7--492.0 ms | 150.8--151.3 ms |
| later fresh cache | GEMM | 644.9 ms | 201.3 ms |

The isolated autotune record claimed that GEMM was faster, but the exact
40-step graph showed the opposite. Copying the earlier cache into a fresh
process reduced RF from 4.5701 s to 4.3723 s. This establishes that the
regression is an inner-selection problem, not a change to the WGPU SDPA graph.

`SdpaRoute::MatmulFusedSoftmaxUnitMinPv` now makes the stable strict-F32
SimpleUnit/MinTile PV algorithm an exact-device route candidate. It never
relayouts or converts an operand: unsupported dtype, layout, device, client, or
batch geometry fails before launch. With the later, otherwise-slow cache still
restored, this route produced 4.3785 / 4.3932 s around a 4.5717 s same-binary
control and restored B3/B1 SDPA to 491.7/151.3 ms.

## Vector input staging

The previously added vector-staging candidates were also applied to the B1
half of the schedule instead of only B3. On the same binary and restored cache:

| B1 stage / 240 calls | prior route | vector-input route | saving |
|---|---:|---:|---:|
| MLP expand | 418.34 ms | 342.08 ms | 76.26 ms |
| MLP contract | 219.29 ms | 166.79 ms | 52.50 ms |
| QKV projection | 248.82 ms | 234.29 ms | 14.52 ms |

The combined B1+B3 vector-input profile plus typed PV route measured 4.222859 s
RF. Persistent all-resident in-use memory after the consumer was
3,556,110,976 bytes (3,391.37 MiB), and sampled NVML peak was 6,448 MiB. These
routes change input staging and F32 reduction order, not precision or semantic
work.

## Fresh PyTorch comparison and numerical disposition

A new Python run, not the profiler run, used the same source-noise fixture,
strict FP32, TF32 off, autocast off, and 40-step schedule. Three measured RF
samples were 4.0397--4.0532 s with a 4.043099 s median. The current WGPU screen
is therefore 179.76 ms (4.45%) slower.

The candidate waveform differs from the same-binary previous WGPU route by
only max-abs `2.44e-6`, RMSE `1.24e-7`, SNR 122.23 dB, cosine
0.999999999999701. Both the prior and candidate WGPU graphs are about 72.19 dB
from the fresh Python waveform, so this campaign did not introduce the existing
Python/WGPU trajectory divergence. Per project policy, bounded operation-order
differences are allowed; finite checks and direct candidate/incumbent metrics
remain mandatory.

## Next measurements

Before changing a built-in default, run five fresh sessions with at least two
warmups and ten measured requests, seal the exact route/cache receipt, and
repeat final latent and waveform comparison. The remaining approximately 180
ms is concentrated in B3/B1 projection and MLP arithmetic; inner QK/PV timing
and shape-exact matmul candidates should be evaluated before GPU-name-specific
tile constants.
