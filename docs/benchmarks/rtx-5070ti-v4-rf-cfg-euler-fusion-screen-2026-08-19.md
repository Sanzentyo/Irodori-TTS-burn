# RTX 5070 Ti Laptop: RF CFG + Euler fusion screen (2026-08-19)

## Decision

Do not enable the fused route in production. It is numerically exact but did
not reduce either aligned timing boundary under the production task-aggregation
policy. The implementation remains profile-only as a differential oracle and
as negative knowledge: reducing eight logical elementwise dispatches in this
50-frame fixture is not itself a useful RF optimization on this runtime.

## Candidate and semantic contract

For each Independent text-CFG step, the ordinary raw-WGPU graph stores the
result of five elementwise operations:

```text
delta   = v_cond - v_uncond
scaled  = delta * cfg_scale
guided  = v_cond + scaled
step    = guided * dt
x_next  = x_t + step
```

The candidate performs this work in one portable WGSL dispatch. Its F16 shader
explicitly rounds both scalar operands and all four intermediate storage
boundaries to F16. An ignored adapter test directly compares the custom kernel
with the five ordinary Burn operations and requires bitwise equality. The
profile fixture has CFG active for two of four Euler evaluations, so the
candidate removes eight logical dispatches per request, not sixteen.

The type-level sampler policy keeps the production default on `Reference`.
Unsupported dtype, layout, device, shape, guidance, solver, temporal-rescale,
or multi-signal topology returns `None` before allocation/dispatch and uses the
ordinary graph.

## Protocol

- source: `ca10533e79848ac93da6816f9d6f990d03019fb4`
- binary SHA-256: `f537eb603b983c512a03ac3c7200a761f6a8d1038ddab11aec41c2b20ba95515`
- adapter: NVIDIA GeForce RTX 5070 Ti Laptop GPU, Vulkan
- driver: 595.71.05
- PCI bus: `00000000:01:00.0`
- reported VRAM: 12,227 MiB
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- fixture SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- converted codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- precision: F16; no TF32/autocast mode exists in this WGPU path
- runtime work: 4 Euler evaluations, effective rows 6, 12 layers, 48 block calls
- cache: new campaign-local CubeCL environment; no `/tmp` measurements or
  tune results were pooled
- warmup: one separate cache-population process, then five fresh measured
  processes
- within each measured process: two discarded paired warmups followed by ten
  measured pairs
- order: ABBA-balanced control/candidate ordering in four-repeat blocks
- boundaries: pre-start device sync to device completion, and the same boundary
  plus owned contiguous F32 CPU latent readback
- telemetry: 100 ms NVML for every measured process

Raw artifact:

```text
/home/sanzentyo/benchmark-artifacts/
  irodori-v4-rf-cfg-euler-fusion-20260819-attempt1
```

`SHA256SUMS` covers the copied binary, environment pins, cache database, warmup
log, all raw session logs, and all NVML streams.

## Results

| Session | measured pairs | paired device delta median | paired readback delta median |
|---:|---:|---:|---:|
| 1 | 10 | +0.124843 ms | +0.013325 ms |
| 2 | 10 | -0.564692 ms | -0.724138 ms |
| 3 | 10 | -0.415407 ms | -0.692134 ms |
| 4 | 10 | -0.097517 ms | -0.139734 ms |
| 5 | 10 | +0.845521 ms | +0.711129 ms |
| aggregate | 50 | **+0.000902 ms** | **-0.028956 ms** |

The candidate improved device-complete in 25/50 pairs and readback-complete in
27/50. Marginal medians were 32.555/32.585 ms for control/candidate device
completion and 33.1025/32.9565 ms for readback completion; paired deltas are the
decision metric.

All 60 candidate/control repetitions were bit-exact with final latent hash
`aaa97505a73ee8b5c9816ecf62b6f1dd4cae60388e0e5167b36359ef2b1f449d`.
Every measured process peaked at 3,093 MiB NVML and reported zero WGPU errors.

## Interpretation and next action

CubeCL's `tasks_max=32` already aggregates the small elementwise work into a
small number of command submissions. The custom dispatch removes intermediate
global stores and logical dispatches, but this fixture has only 1,600 output
elements and two CFG-active steps. The eliminated work is below run-to-run GPU
variation and does not justify another specialized production route.

The next structural work should target operations with materially more bytes or
arithmetic per request: RF projection/epilogue allocation boundaries and codec
prepared-plan ownership/workspace reuse. Parameter/tile tuning remains deferred
until these architecture-level candidates are exhausted.
