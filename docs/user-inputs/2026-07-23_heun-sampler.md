# Heun Sampler Implementation

**Date**: 2026-07-23

## User Instructions

- Implement Heun's 2nd-order RF sampler as an alternative to Euler
- Add `SamplerMethod` enum (`Euler | Heun`) to `SamplingConfig` and `SamplerParams`
- Wire `--sampler` CLI flag into `bench_realmodel.rs` with NFE-aware reporting
- Run tests, clippy, fmt; benchmark Heun 20-step vs Euler 40-step at same NFE=40

## Result

- `SamplerMethod { Euler, Heun }` in `src/config/sampling.rs` with serde/Display/FromStr
- Heun corrector block in `src/rf/euler_sampler.rs`: predictor + trapezoidal avg for all CFG modes
- 14 new tests added (8 Heun sampler + 6 SamplerMethod serde/parse); 334 total pass
- `--sampler euler|heun` flag in `bench_realmodel.rs` with NFE calculation
- Justfile: `bench-tch-mps-f16-heun20`, `bench-wgpu-raw-f16-heun20`

## Benchmark: M4 Pro, LibTorch MPS f16

| Sampler | Steps | NFE | RTF |
|---|---|---|---|
| Euler | 40 | 40 | 0.379 |
| Heun | 20 | 40 | **0.369** |

Heun 20-step is ~2.7% faster at equal NFE (20 loop iterations vs 40).
