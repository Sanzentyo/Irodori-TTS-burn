# RTX 3060 Ti v4 WGPU status — 2026-08-12

This document is the restart point for `codex/v4-wgsl-fusion`. It records the
current implementation before the next optimization cycle; older exploratory
measurements must not be pooled with this campaign.

## Pinned state and protocol

- Measured source commit: `d9997d433a056986d7a5b85446ec14bfee2dbacb`
- GPU: NVIDIA GeForce RTX 3060 Ti, NVML index 1, PCI
  `00000000:07:00.0`
- Precision: strict FP32; TF32 and autocast disabled
- WGPU policy: Vulkan adapter 0, `CUDA_VISIBLE_DEVICES` unset,
  `CUBECL_WGPU_MAX_TASKS=32`
- PyTorch policy: CUDA device 1
- Generation process design: five fresh processes per runtime and length,
  two warmups plus ten measured repetitions per process
- Duration process design: three fresh processes per runtime and text, five
  warmups plus ten measured repetitions per process
- Primary stage interval: pre-start device synchronization through device
  completion, excluding the final result readback
- Secondary stage interval: the same start through completion of an owned,
  contiguous FP32 CPU readback
- RF semantic work: four Euler evaluations, forward batches `[2,2,1,1]`, six
  effective rows, 12 layers and 48 block calls

The RF comparison is a same-request and same-semantic-work comparison, not a
same-operator-graph comparison. PyTorch retains the full conditioning context;
the production WGPU path compacts inactive conditioning and uses derived KV and
fixed timestep-condition caches.

## Duration predictor

The duration predictor completed all six cases. Python and WGPU resolved the
same frame and sample counts in every process. Every WGPU sample was below the
global PyTorch minimum for the head-only device interval, head readback
interval, full duration path device interval, and full duration path readback
interval.

| Audio | Frames | PyTorch head median | WGPU head median | WGPU head + readback | PyTorch full median | WGPU full median |
|---:|---:|---:|---:|---:|---:|---:|
| 1.80 s | 45 | 1.587 ms | 1.067 ms | 1.131 ms | 61.939 ms | 19.528 ms |
| 4.48 s | 112 | 1.599 ms | 1.126 ms | 1.191 ms | 62.255 ms | 24.768 ms |
| 10.20 s | 255 | 1.602 ms | 1.144 ms | 1.207 ms | 62.362 ms | 32.074 ms |
| 13.32 s | 333 | 1.601 ms | 1.222 ms | 1.286 ms | 62.437 ms | 34.212 ms |
| 19.56 s | 489 | 1.610 ms | 1.309 ms | 1.372 ms | 62.700 ms | 37.719 ms |
| 27.40 s | 685 | 1.608 ms | 1.383 ms | 1.444 ms | 62.677 ms | 38.134 ms |

Artifact:
`/tmp/irodori-v4-current-six-lengths-attempt1-20260812/duration`.
Its `SHA256SUMS` file has SHA-256
`6a8ccae07df08b4f43c213ff4844f946e975c0184fa8d1c5593027ccce4a539b`
and verifies in full.

## RF and codec generation stages

The first four lengths completed all ten fresh sessions and all accuracy,
determinism, work-manifest, timing-schema, and source-pin gates. Values below
are medians of 50 measured repetitions. The readback columns stop only after an
owned contiguous FP32 CPU result exists.

| Audio | Frames | PyTorch RF | WGPU RF | RF speedup | WGPU RF + readback | PyTorch codec | WGPU codec | Codec speedup | WGPU codec + readback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.80 s | 45 | 136.867 ms | 120.266 ms | 1.138x | 120.395 ms | 42.782 ms | 39.121 ms | 1.094x | 39.250 ms |
| 4.48 s | 112 | 168.439 ms | 138.085 ms | 1.220x | 138.198 ms | 101.721 ms | 85.781 ms | 1.186x | 85.992 ms |
| 10.20 s | 255 | 237.603 ms | 227.530 ms | 1.044x | 227.649 ms | 240.467 ms | 191.647 ms | 1.255x | 191.961 ms |
| 13.32 s | 333 | 285.650 ms | 270.708 ms | 1.055x | 270.825 ms | 264.948 ms | 251.369 ms | 1.054x | 251.756 ms |

For every row above, all 50 WGPU samples were below the corresponding global
PyTorch minimum for both the device-complete and readback-inclusive intervals.

Completed campaign manifest SHA-256 values:

- 1.80 s: `3ae581d35e3ad12a0a9cdb0bbb1534b5c79ab39abe0751aff998b09c0de7017e`
- 4.48 s: `7b505b5cbd718a3195c0a780aec049c28fec89b058425446c9a1e663a5c7fa34`
- 10.20 s: `b2d20d8372314193a57db523ad0c13365e839c1068861e521d7860daa15febcf`
- 13.32 s: `b5a8f90977a854b602718063282e332abf8b6afa8d5c6e4a6838c8c1952641d8`

All four manifests verify in full under
`/tmp/irodori-v4-current-six-lengths-attempt1-20260812/generation/campaigns`.

## Terminal long-length result

The one-shot campaign stopped permanently in the first WGPU process for the
19.56 s / 489-frame case. It was not retried, resumed, or used to select a
sample.

- PyTorch process 1 completed: RF median 367.239 ms and codec median
  384.822 ms over its ten measured repetitions.
- WGPU RF completed all 12 repetitions with one stable latent hash. Its ten
  measured device-complete values were 415.920–419.861 ms, so the current
  long-sequence RF path is materially slower than PyTorch.
- RF numerical agreement passed: max absolute error `1.139044762e-4`, RMSE
  `8.043011319e-6`, SNR `97.582739 dB`, cosine `0.999999999914`.
- The first codec output was deterministic and finite but failed the waveform
  threshold: max absolute error `1.359656453e-4`, RMSE `4.647087298e-6`, SNR
  `82.655813 dB` against the required `85 dB`, cosine `0.999999997288`.
- The failed case is frozen at
  `/tmp/irodori-v4-current-six-lengths-attempt1-20260812/generation/campaigns/extended`.
  Its `SHA256SUMS` SHA-256 is
  `9143ceb3225c391f9ef95427117325ad3d724042e7320b6d1f8fab71cbfc181a`
  and verifies in full.
- The 27.40 s generation stage was not started. Duration inference for that
  text did complete as reported above.

## Restart order

1. Diagnose the 489-frame codec numerical drift before any performance claim.
   Bisect the production-only codec kernels against the strict FP32 oracle at
   the same exact latent, preserving full-output metrics after every candidate.
2. Optimize long-sequence RF. The first planned candidate is a production-only
   QKV+gate projection that writes normalized/RoPE Q and packed K/V directly,
   eliminating the long QKV intermediate and the following materialization
   dispatch. Preserve the four-forward/six-row/48-block work contract and the
   existing fallback until full accuracy passes.
3. Re-run 489 and 685 frames first as fresh diagnostics. Only after both pass
   accuracy and show a real gain, run a new six-length formal campaign. Never
   pool this document's partial 489-frame values with that future campaign.
4. Keep two performance boundaries for every stage: device-complete with
   readback excluded, and owned contiguous FP32 CPU readback complete. Add a
   separate end-to-end wall interval only as a third product metric.
5. Preserve GPU residency in production: duration may read one scalar to choose
   dynamic output geometry; RF output must remain on GPU for codec input; only
   the final audio required by the caller should cross to CPU. Audit and remove
   diagnostic latent readbacks from the release path.
6. After measurements settle, delete rejected benchmark scripts and kernels,
   brittle WGSL string-literal tests, and non-WGPU backend paths. Keep only
   source-level tests tied to an actual allocator or safety invariant; express
   geometry, resource, and coverage contracts numerically and validate WGSL
   structurally.
7. BF16 is explicitly deferred. Do not add an emulated WGSL BF16 path in this
   optimization cycle.

Each accepted optimization should be committed separately with its focused CPU
tests and its immutable GPU evidence identified in the commit message or this
status document.
