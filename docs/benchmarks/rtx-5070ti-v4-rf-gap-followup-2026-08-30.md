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

### Remaining dense-gap causal screens

After the K16 expansion adoption, the exact device-timestamp totals for the
480 RF block calls were 1,142.28 ms expansion, 614.01 ms contraction,
837.39 ms QKV projection, 46.48 ms Q/K/V materialization, 683.82 ms SDPA,
and 225.34 ms direct output projection. Against the frozen Python operator
profile, expansion is effectively at parity for B3 and 18.42 ms slower for
B1. The remaining positive dense deficits are approximately 54.83 ms in MLP
contraction and 47.63 ms in B3 attention projections; WGPU's compact-K/V SDPA
is approximately 105.68 ms faster. Thus the current full RF lead is not hiding
a remaining expansion bottleneck.

Four exact, bitwise-equal MLP contraction hypotheses were then measured in the
same process. Each route had 40 alternating device-complete samples per B1 and
B3 shape, with an owned readback after timing:

| candidate versus its exact control | B1 median delta | B3 median delta | conclusion |
|---|---:|---:|---|
| 32-row occupancy tile vs 64-row K32 | +22.7% | +9.6% | extra repeated weight traffic dominates |
| 64-column tile vs 128-column K32 | +18.1% | approximately equal | duplicated input traffic offsets lower register/shared state |
| 32-lane mapping K16 vs 16x16 K16 | +0.8% | +6.4% | subgroup-aligned lane ownership is not the missing throughput source |
| component-major shared-weight K16 vs vec4 K16 | +9.5% | +14.5% | scalar transpose/reconstruction costs more than any avoided bank conflict |

The first rows32 B1/B3 attempt was accidentally concurrent on the same GPU and
is explicitly excluded. The valid sequential receipts are
`dense-route-abba-contract-rows32-v19-*-sequential.json`,
`dense-route-abba-contract-c64-v19-*.json`,
`dense-route-abba-contract-warp32-k16-v19-*.json`, and
`dense-route-abba-contract-swizzled-k16-v19-*.json`. These routes remain typed
autotune candidates for other adapters, but the NVIDIA default is unchanged.
Together the screens reject workgroup count, shared-memory capacity, lane
mapping, and shared-bank layout as primary RTX causes. Closing the remaining
contract gap now requires a better generated matmul core or sealed CubeK
algorithm, not another unconditional model-side tile heuristic.

The QKV-to-packed-Q/K/V fusion was also rebuilt with vec4 global input loads
and a K16 tile. It removes all 480 separate materialization dispatches and
reduces the monolithic workgroup footprint from 32 KiB to 20 KiB. Nevertheless,
combined B1 projection/materialization rose from 225.46 to 278.44 ms and B3
from 658.41 to 815.54 ms. Keeping eight projection accumulators live while
performing Q/K normalization, RoPE, sigmoid, and scattered consumer stores
reduces occupancy and store efficiency more than the eliminated temporary
helps. Persistent in-use bytes stayed at 3,556,110,976. The first screen is a
retained fail-closed artifact: the new route initially omitted `QkNormPacked`
from its residency requirements. A route-derived dependency test now covers
the new variant, and only the corrected second screen is used above.

The existing CubeK accumulator scatter was re-screened as a second fusion
boundary. It was slower still (B1 505.25 ms, B3 1,160.11 ms) and required both
row- and column-major QKV layouts, raising persistent in-use memory to
3,870,683,776 bytes. The accepted boundary therefore remains the K16
contiguous projection followed by the small direct packed-K/V transform.
Raw full-request evidence is under `wgpu/projection-direct-k16-v19-screen1`
(failed dependency), `wgpu/projection-direct-k16-v19-screen2`, and
`wgpu/cubek-qkv-scatter-v19-screen1`.

### MLP contract decomposition and generated-core follow-up

The remaining contraction deficit was decomposed with a new same-process
benchmark rather than inferred from the full RF stage total. Both operands use
the exact contiguous `[B * 489, 3680] @ [3680, 1280]` geometry of the adopted
compressed expansion. The control is the exact NVIDIA incumbent (K32 at B1,
K16 at B3); the candidate is either Burn's tuned raw-WGPU matmul alone or that
matmul followed by the existing WGSL residual/gate finalizer. Each route was
warmed six times and measured in 30 alternating ABBA/BAAB blocks (60 samples
per route):

| exact shape | incumbent fused contract | Burn matmul only | Burn matmul + finalizer |
|---|---:|---:|---:|
| B1/S489 | 0.72346 ms | 1.19591 ms | 1.16399 ms |
| B3/S489 | 1.60282 ms | 2.55238 ms | 2.43685 ms |

The two Burn candidates vary slightly with mobile clocks, but their ordering
relative to the interleaved control is unambiguous: even matmul without the
extra finalizer is 65.3% slower at B1 and 59.2% slower at B3. Adding the
residual/gate dispatch does not explain the gap; it is below the between-run
clock band. All four comparisons were bitwise equal on the constant fixture
and reported zero uncaptured WGPU errors. The missing performance is therefore
inside the generated FP32 matrix core/layout selection, not the fused store.

A reusable CubeK extension was then added so the typed accumulator transform
can run on its double-buffered unit routine. This is a generic writer/routine
composition rather than an Irodori tile constant. On this adapter it remained
slower than the handwritten control: 1.30691 vs 0.85408 ms at B1 and 3.42415
vs 1.72440 ms at B3. It remains an exact-autotune candidate for other devices,
but is rejected by the NVIDIA default.

The small-M occupancy hypothesis was also tested between the existing 32- and
64-row extremes. A new 48x128 route retains six vec4 accumulators per thread
and has independent K32/K16 forms. Against the matching incumbent, K32 was
7.6% slower at B1, K16 was 1.9% slower at B1 and 0.5% slower at B3. The one
apparently favorable K32/B3 comparison disappeared when directly paired with
the adopted K16 route: 1.73071 vs 1.55976 ms across 100 samples. This closes
the intermediate occupancy point without changing production selection.

Nsight Systems 2025.1.3 captured the Vulkan API stream and confirmed distinct
pipeline binds/submits, but wgpu/CubeCL did not emit Vulkan debug-utils GPU
ranges, so the report contained no shader-level GPU marker table. It is kept
as diagnostic evidence, not used as timing authority. The pre-sync through
device-completion timestamps in the paired JSON remain the authoritative
boundary.

These experiments extend route ABI v20 with the 48-row K32/K16 and CubeK
double-unit choices. They are available to exact-device tuning on AMD, Intel,
older Apple, and other NVIDIA generations; no device-name heuristic selects
them. Raw evidence is in `dense-burn-decomposition-v20` under the fresh
campaign root.

### Contract load-latency overlap

The paired dense-route harness now records two independent boundaries. Its
existing wall measurement remains pre-sync through owned readback. In
addition, every exact route is enclosed in CubeCL's device profiler and the
deferred `ProfileDuration` is resolved only after the device sync. Receipts
record both the timing source and the GPU elapsed duration. On this Vulkan
adapter the source was `device_timestamp`, and the latter remained stable even
when host scheduling produced isolated wall-time spikes. The timestamp is used
to identify a kernel change; the full RF boundary remains the adoption gate.

Two generic K16 contraction schedules were implemented without changing the
ordered FP32 accumulation or output layout:

- a two-page shared-memory pipeline overlaps cooperative loads with compute,
  but consumes 24 KiB per workgroup. It reduced a K16 B1 comparison by about
  4.3%, yet was 4.13% slower than the production K32 B1 route and 21.2% slower
  than the production K16 B3 route. Reduced shared-memory residency dominates
  at B3, so this route remains an exact-device candidate only;
- a single-page schedule has each invocation prefetch its next input vec4 and
  two weight vec4s into registers before the overwrite barrier. It overlaps
  global-load latency while retaining the 12 KiB shared footprint and the same
  two workgroup barriers per K slice.

The single-page register-prefetch route won at both production shapes in the
same-process exact comparison:

| shape | incumbent GPU timestamp | prefetched GPU timestamp | delta |
|---|---:|---:|---:|
| B1/S489 | 0.457600 ms | 0.435200 ms | -4.90% |
| B3/S489 | 1.331328 ms | 1.181440 ms | -11.26% |

Every F32 and F16 focused comparison was bitwise equal for both contiguous and
pitched inputs. The full 40-step timestamp screen reduced the 480 contract
calls from 613.85 ms to 574.14 ms (6.47%). An instrumentation-free ABBA pair
then measured RF medians of 3.746537 s for the incumbent and 3.697018 s for the
candidate, a 49.52 ms (1.32%) saving. The preceding timestamped pair measured
3.751210 s versus 3.705280 s. All requests retained waveform SHA-256
`325870a564a251a88695b8701af6b24c1dc04dcf46abf35ef8df20f76055742e`
and 3,556,110,976 persistent in-use bytes. The clean pair also had the same
6,387 MiB NVML peak; the timestamped pair differed by 44 MiB of allocator
reservation noise and is not used to claim a memory change.

Route ABI v22 exposes both schedules to exact-device tuning. The register-
prefetch route is adopted only for NVIDIA B1/B3 S489; all other cells and
adapter families retain their independently resolved route. Raw evidence is
under `contract-double-buffer-v21` and `contract-prefetch-v22` in the fresh
campaign root. The unsealed profiles
`profiles/rtx-f489-mlp-contract-prefetch-v22.json` and
`profiles/rtx-f489-mlp-contract-control-v22.json` preserve the same-binary
candidate and prior-control selections.

### Expansion load-latency overlap

The same single-page register-prefetch schedule was then generalized to the
one-dispatch MLP expansion-plus-SwiGLU kernel. Each invocation loads the next
K16 input vec4 and its gate/value weight vec4s before the shared-page overwrite
barrier. The 12 KiB shared footprint, 64x128 output tile, ordered F32 FMA
sequence, compressed `[M, 3680]` output, and F16 storage variant are otherwise
unchanged. The launcher now represents its mutually exclusive tile schedules
with `MlpExpandTileLayout`; invalid combinations of boolean specialization
flags are no longer constructible.

Fifty alternating blocks (100 device-timestamp samples per route) produced:

| shape | incumbent K16 | register-prefetched K16 | delta |
|---|---:|---:|---:|
| B1/S489 | 0.923648 ms | 0.852864 ms | -7.66% |
| B3/S489 | 2.610176 ms | 2.407936 ms | -7.75% |

Both exact comparisons were bitwise equal: 1,799,520 elements at B1 and
5,398,560 elements at B3. The first measured 40-step request reduced the 480
`expand_swiglu` calls from 1,153.436 ms to 1,059.024 ms (8.19%). Its summed
profiled RF substages fell from 3,609.133 to 3,499.320 ms.

Fresh-process adoption used two warmups and three measured requests per
process in candidate/control/control/candidate order. The timestamped pair
reduced the RF median from 3.758096 to 3.706226 seconds (1.38%). The
instrumentation-free reversed pair reduced it from 3.767926 to 3.687178
seconds (2.14%); consumer-complete medians fell from 4.163173 to 4.083772
seconds. Every request retained waveform SHA-256
`325870a564a251a88695b8701af6b24c1dc04dcf46abf35ef8df20f76055742e`.
Persistent in-use allocation remained 3,556,110,976 bytes. A final run through
the built-in NVIDIA table measured a 3.699390-second RF median and a 6,389 MiB
NVML peak, within 2 MiB of the preceding accepted campaign rather than a
material memory change.

Route ABI v23 therefore adopts
`HandwrittenK16PrefetchedVectorInput` only for NVIDIA B1/B3 S489. Other exact
cells and adapter families remain independently selected. The unsealed
same-binary profiles are
`profiles/rtx-f489-mlp-expand-prefetch-v23.json` and
`profiles/rtx-f489-mlp-expand-control-v23.json`. Raw receipts, logs, and NVML
samples are under `expand-prefetch-v23` in the fresh campaign root.

### QKV projection load-latency overlap

The same reusable schedule was next applied to the exact K16 QKV/gate
projection. Its 256 invocations already cooperatively load exactly 256 input
vec4s and 512 weight vec4s for each reduction slice, so every invocation can
own one input and two weight prefetch registers. The change does not increase
the 12 KiB shared page, alter the 64x128 output tile, change the ordered F32
FMA sequence, or fuse the downstream Q/K/V materializer. `ProjectionTileLayout`
also replaces the previous three booleans, making invalid kernel-specialization
combinations unrepresentable.

Fifty alternating blocks (100 device-timestamp samples per route) measured:

| shape | incumbent K16 | register-prefetched K16 | delta |
|---|---:|---:|---:|
| B1/S489 | 0.607104 ms | 0.556160 ms | -8.39% |
| B3/S489 | 1.726848 ms | 1.608320 ms | -6.86% |

The B1 comparison was bitwise equal across 2,503,680 values and B3 across
7,511,040 values; both reported zero uncaptured WGPU errors. In the profiled
40-step request, 480 `qkv_gate` calls fell from 851.656 to 783.957 ms (7.95%),
while `materialize_qkv` remained approximately 46 ms as expected.

Fresh-process candidate/control/control/candidate runs used two warmups and
three measured requests per process. The timestamped pair reduced RF median
from 3.735642 to 3.653706 seconds (2.19%). The reversed instrumentation-free
pair reduced it from 3.680490 to 3.632478 seconds (1.30%). Every candidate and
control output in this campaign had waveform SHA-256
`88a01ca8bb82e6b6e41aef9d31016efe767c4a2a44fae389f9db9ea6f044cf43`;
persistent in-use allocation remained 3,556,110,976 bytes. The hash differs
from the preceding fresh CubeCL environment for both routes, so it is recorded
as a shared autotune/arithmetic-order effect rather than attributed to the
bitwise-equal QKV change. A final built-in-table run measured a 3.656393-second
RF median and 6,389 MiB NVML peak.

Route ABI v24 adopts `HandwrittenC128K16Prefetched` only for NVIDIA B1/B3
S489. The old K16 route remains an exact-tuning candidate and all other cells
retain their separately approved choices. The unsealed profiles are
`profiles/rtx-f489-qkv-prefetch-v24.json` and
`profiles/rtx-f489-qkv-control-v24.json`; raw receipts are in
`qkv-prefetch-v24` under the fresh campaign root.

### Direct attention-tail load-latency overlap

The direct head-major SDPA-to-output-projection kernel has the same exact K16
cooperative-load geometry as the contraction route, but its input load also
gathers head-major attention and multiplies the learned gate. A typed
`DirectOutputTileLayout` now separates scalar K32, vector K32, vector K16, and
register-prefetched K16 forms. The prefetched form loads the next attention/gate
product and two weight vec4s before the overwrite barrier; it preserves the
12 KiB shared page, ordered F32 reduction, and fused block residual store.

Same-process device timestamps (100 samples per route) measured:

| shape and control | control | prefetched K16 | delta |
|---|---:|---:|---:|
| B1/S489, production K32 | 0.161152 ms | 0.146816 ms | -8.90% |
| B3/S489, production K16 | 0.431104 ms | 0.394240 ms | -8.55% |

Both comparisons were bitwise equal (625,920 B1 values and 1,877,760 B3
values) with zero uncaptured WGPU errors. The profiled 40-step pair reduced
the 480-call direct-output total from 231.669 to 210.535 ms (9.12%). Initial
three-sample external-process pairs landed within the RF noise floor, so they
were not used alone to widen the production route.

A formal five-session comparison then ran two warmups and ten measured
requests in every fresh process, alternating candidate/control process order:

| session | candidate RF median | control RF median | candidate minus control |
|---:|---:|---:|---:|
| 1 | 3.618389 s | 3.643031 s | -24.642 ms |
| 2 | 3.643275 s | 3.659325 s | -16.050 ms |
| 3 | 3.651337 s | 3.660574 s | -9.237 ms |
| 4 | 3.649424 s | 3.660538 s | -11.114 ms |
| 5 | 3.655604 s | 3.671791 s | -16.187 ms |

The candidate won all five sessions. Across all 50 samples per route, RF
median fell from 3.661177 to 3.647043 seconds (14.134 ms, 0.386%) and
consumer-complete median fell from 4.057425 to 4.045649 seconds. All 100
outputs retained waveform SHA-256
`325870a564a251a88695b8701af6b24c1dc04dcf46abf35ef8df20f76055742e`
and persistent in-use allocation of 3,556,110,976 bytes. The built-in-table
confirmation measured 3.653629 seconds RF and a 6,389 MiB NVML peak.

Route ABI v25 adopts
`DirectOutputResidualK16PrefetchedVectorInput` only for NVIDIA B1/B3 S489.
`PostSdpaRoute` now owns typed direct/vector/K16 predicates, and residency
derivation retains the packed output weight for every direct variant rather
than recognizing only the original scalar route. Unsealed same-binary profiles
are `profiles/rtx-f489-direct-output-prefetch-v25.json` and
`profiles/rtx-f489-direct-output-control-v25.json`; fresh raw evidence is under
`direct-output-prefetch-v25`.

### B1 MLP contract: working-set-aware split-K screen

The remaining Python/WGPU stage comparison appeared to leave about 37 ms in
the B1 MLP contract over 240 calls. The original dense-route harness repeated
one 18.8 MiB weight, however, while the RF stack cycles twelve independent
layer weights (about 226 MiB). A single-weight microbenchmark therefore
reported an L2-hot 0.447232 ms median that did not represent the model. The
harness now records and rotates an explicit `weight_working_set`; twelve
buffers increased the incumbent median to 0.491136 ms.

A portable global split-K2 candidate was added to test whether extra
workgroups could hide that weight-working-set latency. Two Z partitions each
reduce one disjoint 1,840-element K interval into F32 partials, followed by a
small gated-residual finalizer. The route doubles B1 workgroups from 80 to 160,
retains the 12 KiB shared page and register-prefetch schedule, and adds one
temporary of 5,007,360 bytes plus one dispatch. Its typed launcher validates
shape, pitch, common F32/F16 storage precision, device, binding size/alignment,
and all workgroup/cube limits before either allocation.

With twelve distinct non-integer weight buffers, the same-process GPU median
fell from 0.491136 to 0.470016 ms (4.30%). The changed reduction association
produced only `9.69e-8` max-abs/RMSE at the operator boundary. That local win
did not survive the complete RF graph: separate device profiles measured B1
contract totals of 168.106 ms for the incumbent and 168.652 ms for split-K2,
which are effectively equal at this clock-noise level.

A formal five-session comparison used two warmups and ten measured requests
per fresh process, alternating process order:

| session | split-K2 RF median | incumbent RF median | split-K2 minus incumbent |
|---:|---:|---:|---:|
| 1 | 3.628962 s | 3.631535 s | -2.573 ms |
| 2 | 3.655430 s | 3.640568 s | +14.863 ms |
| 3 | 3.661250 s | 3.665779 s | -4.528 ms |
| 4 | 3.669928 s | 3.650969 s | +18.959 ms |
| 5 | 3.657950 s | 3.656905 s | +1.045 ms |

Split-K2 won only two sessions. Across all 50 requests per route, its RF
median was 3.656161 seconds versus 3.651391 seconds for the incumbent
(+4.770 ms, +0.131%); consumer-complete median regressed by 3.603 ms.
Persistent RF allocation was identical at 3,417,207,424 bytes. Direct waveform
comparison over 938,880 samples reported max abs `3.05e-5`, RMSE `1.55e-6`,
SNR 100.32 dB, and cosine 0.999999999954, so rejection is performance-only.

Route ABI v26 exposes
`HandwrittenSplitK2PrefetchedPitchedVectorInput` to the exact-device tuner but
does not select it in the built-in RTX profile. This distinction is important:
the algorithm is a legitimate candidate for GPUs with fewer compute units or
different cache behavior, while the measured RTX 5070 Ti Laptop default stays
on `HandwrittenK16PrefetchedPitchedVectorInput`. Raw evidence is in the fresh
`irodori-v4-rf-gap-profile-20260831-split-k2-v26` campaign.

### Expanded RF leaf accounting and measurement-boundary correction

The profiler now covers the previously unaccounted backbone input/output
projections, output norm, CFG input materialization/combine, and Euler update.
It records deferred device timestamps and resolves them only after the public
RF device-complete boundary. The residency runner previously resolved and
serialized the profile receipt before capturing that boundary, so an
instrumented request incorrectly included diagnostic host work in its RF
latency. The boundary is now captured immediately after the post-RF device
sync and before any profile receipt work.

One 40-step B3/B1 request produced the following exact-shape leaf totals. Each
block row has 240 B1 and 240 B3 calls; each backbone row has 20 of each:

| stage | B1/S489 | B3/S489 |
|---|---:|---:|
| QKV/gate projection | 201.744 ms | 585.030 ms |
| packed Q/K/V materialization | 9.755 ms | 36.836 ms |
| SDPA | 166.340 ms | 525.211 ms |
| direct attention output | 61.591 ms | 147.246 ms |
| MLP expand + SwiGLU | 281.222 ms | 793.974 ms |
| MLP contract + gate + residual | 168.254 ms | 408.648 ms |
| attention/MLP AdaLN | 7.317 ms | 17.795 ms |
| backbone input/norm/output | 1.915 ms | 5.111 ms |

CFG materialization, CFG combine, and all Euler updates together were below
one millisecond. Against the frozen strict-FP32 Python operator profile, WGPU
now wins B3 expand by 49.25 ms, B3 contract by 19.56 ms, both SDPA phases by
27.71/70.23 ms, and combined QKV/materialization/output by 23.70/2.42 ms.
The only material local deficit is B1 MLP contract: 168.25 ms versus the
Python profile's 130.96 ms, a 37.29 ms deficit over 240 calls.

This is not an additive decomposition of an uninstrumented request. CubeCL's
profile scope flushes the queue when starting and ending each timestamp scope,
so the summed 3,419.348 ms leaf time includes a more serialized schedule. It is
valid for exact-stage attribution and paired route screening, but it must not
be subtracted from an unprofiled RF wall time. A rejected nested whole-forward
scope also suppressed the thread-local leaf scopes and is not used as evidence.

Nsight Systems independently observed the measured Euler host interval. It
contained 301 `vkQueueSubmit` calls whose CPU API time summed to only 1.963 ms,
while 18 completion waits accounted for 3.258 seconds. This rules out raw
`vkQueueSubmit` call overhead as the 37 ms B1 contract cause; the waits measure
queued GPU completion and are not a new CPU optimization target. The same
interval did show 60 `vkAllocateMemory` and 33 `vkFreeMemory` calls, which
motivates a later prepared-workspace/allocator-lifetime experiment, but this
trace does not attribute those allocations to a particular tensor stage.
Raw receipts are in `irodori-v4-rf-gap-profile-20260831-leaf-v2` and
`irodori-v4-rf-gap-profile-20260831-nsys-v4`.

### B1 contract: 128-row schedule screen

The existing 128-row/32-column candidate used a 512-invocation workgroup but
advanced its cooperative input and weight loads by a hard-coded 256. Half the
workgroup therefore repeated the other half's shared-memory loads. The kernel
template now derives the stride from `workgroup_x * workgroup_y`, so every tile
schedule owns the correct cooperative-load partition.

After that correctness fix, a twelve-weight working-set comparison used ten
warmups and 100 alternating device-timestamp samples per route:

| shape | current 64x128 K16 | corrected rows128 | delta |
|---|---:|---:|---:|
| B1/S489 | 0.477056 ms | 0.504832 ms | +5.82% |
| B3/S489 | 1.094400 ms | 1.675904 ms | +53.13% |

Both comparisons were bitwise equal with no WGPU errors. The larger tile
reduces workgroup count and potential weight rereads, but its 512-thread
workgroup, larger shared page, and accumulator/register footprint dominate on
this adapter. It remains a valid exact-tuning candidate and is not selected by
the built-in NVIDIA profile. Raw evidence is in
`irodori-v4-rf-gap-profile-20260831-rows128-v5`.
