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

## Formal five-session comparison

The typed PV route and B1/B3 vector-input projection routes were then measured
without substage profiling. Each runtime used five fresh external processes,
two warmups, and ten measured requests per process. The CubeCL environment was
restored rather than retuned; process-local WGPU pipelines were rebuilt in
every process.

| runtime | samples | session medians (s) | all-sample median (s) | range (s) |
|---|---:|---|---:|---:|
| WGPU candidate | 50 | 4.0904 / 4.1083 / 4.1160 / 4.1217 / 4.1247 | 4.114615 | 4.0648--4.1283 |
| PyTorch CUDA | 50 | 4.0781 / 4.1160 / 4.1273 / 4.1355 / 4.1449 | 4.127328 | 4.0530--4.1605 |

WGPU was 12.712 ms (`0.308%`) faster by the all-sample median. The distributions
overlap, so this is a narrow lead rather than evidence of a large runtime
advantage. Every runtime was deterministic within its 50 requests. WGPU
persistent all-resident in-use memory was 3,556,110,976 bytes in every session;
sampled NVML peak was 6,394 MiB versus 6,370 MiB for Python. PyTorch reported
5,086.16 MiB peak allocated and 6,154 MiB peak reserved.

## Remaining stage deficits

The candidate device-timestamp profile still shows slower dense arithmetic,
partly offset by compact-K/V SDPA and lower unprofiled overhead:

| stage / 240 calls | WGPU B1 / B3 (ms) | PyTorch B1 / B3 (ms) | WGPU minus PyTorch (ms) |
|---|---:|---:|---:|
| MLP expand + SwiGLU | 342.08 / 951.44 | 280.62 / 843.22 | +61.46 / +108.23 |
| MLP contract | 166.79 / 550.01 | 130.96 / 428.21 | +35.83 / +121.79 |
| QKV materialization + output projection | 329.89 / 890.99 | 296.79 / 771.54 | +33.10 / +119.45 |
| SDPA | 157.43 / 491.16 | 194.05 / 595.44 | -36.62 / -104.28 |

The remaining optimization target is therefore projection/MLP throughput, not
SDPA. The SDPA advantage is real but currently compensates for approximately
341 ms of dense-stage deficit.

## Direct output vector staging

The head-major SDPA output projection previously loaded and gated every input
component as a scalar. `PostSdpaRoute::DirectOutputResidualVectorInput` loads
four adjacent head components and their gates per storage/workgroup
transaction, then performs the same ordered scalar FMA sequence and fused
block residual epilogue.

The profiled 480-call output-projection total fell from 305.04 ms to 260.86 ms.
In a same-binary scalar/vector/scalar sequence with 2 warmups and 10 measured
requests per process, the two scalar controls pooled to a 4.103845 s median and
the vector route measured 4.065398 s, a 38.45 ms reduction. All 30 outputs had
the same WGPU waveform SHA-256. This route is still an exact profile candidate;
it is not generalized to other shapes or adapters by a marketing-name check.

Five additional fresh sessions of the composed vector-output route produced
session medians 4.0531 / 4.0676 / 4.0769 / 4.0829 / 4.0885 s. Across all 50
measured requests the median was 4.075993 s (range 4.0414--4.0926 s), which is
51.335 ms (`1.244%`) faster than the formal PyTorch median. Persistent in-use
memory remained 3,556,110,976 bytes and sampled NVML peak remained 6,394 MiB.
The exact B1/B3 S489 cells are therefore adopted in the NVIDIA RTX built-in
profile; other shapes and adapter families remain unchanged candidates or
portable fallbacks.

## Fresh PyTorch comparison and numerical disposition

A new Python run, not the profiler run, used the same source-noise fixture,
strict FP32, TF32 off, autocast off, and 40-step schedule. The initial
three-sample screen was superseded by the 50-sample formal comparison above;
it is retained only as a diagnostic artifact and is not pooled into the formal
distribution.

The candidate waveform differs from the same-binary previous WGPU route by
only max-abs `2.44e-6`, RMSE `1.24e-7`, SNR 122.23 dB, cosine
0.999999999999701. Both the prior and candidate WGPU graphs are about 72.19 dB
from the fresh Python waveform, so this campaign did not introduce the existing
Python/WGPU trajectory divergence. Per project policy, bounded operation-order
differences are allowed; finite checks and direct candidate/incumbent metrics
remain mandatory.

## Next measurements

Continue with shape-exact MLP/projection algorithms and memory-transaction
improvements; do not replace the measured route with GPU-name-specific tile
constants. Repeat final latent and waveform comparison if the selected route
vector changes again.

## Typed CubeK MLP contraction epilogue

The next structural experiment generalized CubeK's output writer so a matmul
can apply `residual + gate * accumulator` before its one primary store. The
writer owns tail masking and typed auxiliary views; no projected branch or
second residual dispatch is created. Because CubeK's strict-F32 unit reader
requires a column-major RHS, `MlpContractCubeKColumn` is prepared once during
model load. Exact residency can keep that one representation and release both
the source and handwritten row-major cache, so no request-time relayout is
hidden in the result.

The exact B1/B3 S489 routes were screened with the same restored cache and
40-step designed-voice request:

| CubeK writer | B1 contract / 240 calls | B3 contract / 240 calls | measured RF | waveform disposition |
|---|---:|---:|---:|---|
| current handwritten vector input | 166.79 ms | 550.01 ms | formal median 4.075993 s | incumbent |
| Unit / MinTile | 284.97 ms | 993.99 ms | 4.588659 s | bit-identical |
| Unit / MaxTile | 366.66 ms | 998.52 ms | 4.614218 s | bit-identical |
| PlaneVec | 4,856.82 ms | 14,625.79 ms | 22.095162 s | reject: 76.36 dB vs incumbent |

The candidate kept persistent in-use memory at 3,556,110,976 bytes, within
6,144 bytes of the incumbent screen, because it replaced rather than
co-retained the contraction layout. It therefore proves the reusable
one-dispatch and profile-locked residency design, but none of these generic
algorithms is an RTX performance winner. The two unit variants are
accuracy-valid candidates for other adapters; PlaneVec is rejected on both
speed and the 80 dB waveform hard gate. The RTX default remains the
handwritten vector-input contraction.

Raw evidence is under `wgpu/cubek-unit-min-contract-v16-screen1`,
`wgpu/cubek-unit-max-contract-v16-screen1`, and
`wgpu/cubek-plane-contract-v16-column-screen1` in this campaign root. Two
earlier fail-closed attempts record the row/column layout mismatch and are not
pooled with these completed screens.

## Exact-shape dense shared-staging reduction

The remaining handwritten dense kernels used a `64 x 128 x 32` cooperative
tile. Its input and weight staging consumes 24 KiB per workgroup. The new K16
family preserves the 64-row/128-column output tile, 256-invocation workgroup,
vec4 global transactions, ordered F32 FMAs, and output layout, while reducing
the cooperative K tile to 16. Workgroup storage is therefore 12 KiB. This is a
memory-residency change rather than a device-name tile constant; both K32 and
K16 remain explicit route candidates and the exact route table owns admission.

`bench_v4_dense_routes` measures the two routes in one process with pre-start
and post-kernel device synchronization. It warms both pipelines, alternates
ABBA/BAAB blocks, performs an owned readback only after timing, and writes raw
samples, adapter identity, allocator state, WGPU errors, and direct accuracy
metrics to JSON. On the exact B1/B3 S489 shapes, 20 blocks produced 40 samples
per route:

| exact operator | K32 median | K16 median | delta | disposition |
|---|---:|---:|---:|---|
| QKV B3 | 2.13676 ms | 2.00232 ms | -6.29% | adopt |
| QKV B1 | 0.82289 ms | 0.77231 ms | -6.15% | adopt |
| MLP contract B3 | 1.84317 ms | 1.62290 ms | -11.95% | adopt |
| MLP contract B1 | 0.79127 ms | 0.79923 ms | +1.01% | keep K32 |
| direct output B3 | 0.59395 ms | 0.49863 ms | -16.05% | adopt |
| direct output B1 | 0.38790 ms | 0.48113 ms | +24.03% | keep K32 |

Every paired output was bitwise equal: QKV compared 7,511,040 B3 and 2,503,680
B1 elements; each contract/output comparison also reported max-abs and RMSE
zero. No uncaptured WGPU error occurred. The B1 reversals show why a broad
`NVIDIA` or sequence-only heuristic is insufficient: the extra K barriers are
amortized at B3 but can dominate the smaller B1 output drain.

The QKV route was also checked with fresh external processes. With only B3
changed, control medians were 4.02495 / 4.05064 s and candidate medians were
3.98478 / 3.98628 s. With B1+B3 selected together, the clock-noisy ABBA series
still retained the same waveform hash and persistent bytes. The direct
same-process result predicts about 44.4 ms saved over the 240 B3 plus 240 B1
calls and is the primary adoption evidence.

After adopting QKV K16 as the common control, the B3 contract plus direct
output routes were composed in a second fresh-process ABBA series:

| route | fresh-process medians |
|---|---|
| K32 contract/output control | 3.97330 / 3.99733 s |
| B3 K16 contract/output | 3.89288 / 3.90123 s |

The composed change saved 80--106 ms despite the first candidate process
running at a mean active SM clock about 51 MHz below its preceding control.
All 12 measured outputs retained SHA-256
`4bf6a50d2805dcd5ea7343229899e21f27c6e32c558b1f9e9858e1719b5278a2`,
and all runs retained 3,556,110,976 persistent in-use bytes. A detailed
timestamp run measured B3 contract at 428--430 ms per 240 calls and direct
output at 153--155 ms, down from approximately 560 ms and 206 ms respectively.

Several superficially plausible alternatives were rejected rather than folded
into the default:

- Burn's ordinary expansion plus separate pitched SwiGLU was about 1,060 ms
  at B3 and 420 ms at B1, versus about 951/342 ms for the compressed
  handwritten route.
- mapping a complete 32-lane subgroup across expansion columns increased B3
  time to about 1,159 ms; its 128-row form reached about 1,259 ms.
- the subgroup-aligned contract was at best within the large mobile-clock
  noise band, while its 128-row/512-invocation form regressed to about 630 ms.
- a 128-row/K16 QKV tile regressed to about 745 ms per 240 B3 calls. The
  512-invocation residency/scheduling cost outweighed reduced repeated weight
  traffic.

Raw evidence is under `dense-route-abba-v17`,
`dense-route-abba-k16-followup-v17`, `paired-qkv-k16-v17-attempt2`,
`paired-qkv-k16-b1-b3-v17`, `paired-b3-dense-k16-v17`, and the corresponding
`wgpu/*-v17-screen*` directories in the same fresh campaign root. Failed
fail-closed launches are retained but excluded from timing summaries.

### Revalidation of the B1 expansion crossover

The full route test still encoded an older receipt which classified B1/S489
as a Burn-default expansion win. Temporarily restoring that route on the
current binary split the profiled expansion into 418.05 ms of Burn projection
plus 6.10 ms of pitched SwiGLU for the 240 B1 calls; the B3 handwritten half
was 992.02 ms. The combined expansion total was 1,416.18 ms, versus 1,337.56
ms when both halves use the one-dispatch vector route.

This was then checked directly in the same process, with four warmups per
route and 20 ABBA/BAAB blocks (40 device-complete samples per route):

| shape | Burn projection + pitched SwiGLU | one-dispatch vector route | delta |
|---|---:|---:|---:|
| B1/S489 | 2.02943 ms | 1.47823 ms | -0.55121 ms (-27.16%) |
| B3/S489 | 3.68874 ms | 3.26372 ms | -0.42502 ms (-11.52%) |

Both synthetic exact-shape comparisons were bitwise equal and reported no
uncaptured WGPU errors. The older S489 crossover is therefore stale for this
binary; B1/S489 remains on `HandwrittenT64VectorInput`, while B1/S333 remains
the independently measured default-graph cell. Raw paired receipts are
`dense-route-abba-expand-current-v17-b1.json` and
`dense-route-abba-expand-current-v17-b3.json` in the campaign root. The
temporary full-request restore is retained at
`wgpu/b1-default-restored-v17-screen1` and is not pooled into an accepted
latency summary.

### K16 MLP expansion staging

The same shared-memory reduction was then applied to the one-dispatch MLP
projection-plus-SwiGLU route. The output tile, workgroup size, vec4 global
loads, ordered F32 accumulation, and compressed `[M, 3072]` output stay
unchanged; only the cooperative K slice falls from 32 to 16. The F32 and F16
shader variants are selected through the existing precision-typed launcher,
and K32 remains an explicit route candidate rather than being removed.

Same-process ABBA/BAAB timing used 40 device-complete samples per route and
performed owned readback only after timing:

| shape | K32 median | K16 median | delta |
|---|---:|---:|---:|
| B1/S489 | 1.50791 ms | 1.35313 ms | -10.26% |
| B3/S489 | 3.39352 ms | 2.85519 ms | -15.86% |

Every synthetic output comparison was bitwise equal and WGPU reported no
uncaptured error. A full 40-step timestamp screen reduced the 480-call MLP
expansion total from 1,337.56 ms to 1,142.28 ms, a 195.28 ms (14.60%) stage
reduction. The final waveform hash and persistent allocator bytes were
unchanged.

Two fresh external-process pairs then alternated the K32 control and K16
candidate. Each process used two warmups and three measured 40-step requests:

| route | pooled RF samples (s) | pooled median |
|---|---|---:|
| K32 control | 3.8769 / 3.8801 / 3.8831 / 3.8880 / 3.9139 / 3.9151 | 3.885528 s |
| K16 candidate | 3.7189 / 3.7358 / 3.7508 / 3.7609 / 3.7838 / 3.7946 | 3.755883 s |

The paired campaign therefore saves 129.65 ms (3.34%) at the RF boundary.
Consumer-complete pooled medians fell from 4.276626 s to 4.158898 s. All 12
measured requests retained waveform SHA-256
`4bf6a50d2805dcd5ea7343229899e21f27c6e32c558b1f9e9858e1719b5278a2`;
persistent in-use memory remained 3,556,110,976 bytes and every process sampled
the same 6,361 MiB NVML peak. Against the separately frozen 4.127328 s Python
RF median, this candidate is 371.45 ms (9.00%) faster. The exact B1/B3 S489
cells are adopted; other shapes and adapters retain independently resolved
routes.

Raw receipts are under `dense-route-abba-expand-k16-v18-b1.json`,
`dense-route-abba-expand-k16-v18-b3.json`,
`wgpu/all-dense-k16-v18-screen1`, and `paired-mlp-expand-k16-v18` in the fresh
campaign root. The unsealed K32 comparison profile is
`profiles/rtx-f489-mlp-expand-k32-control-v18.json`.
