# Irodori-TTS-burn — Copilot Instructions

## Project Overview

Full-scratch Rust/burn reimplementation of [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
(Python/PyTorch TTS model, ~500M params). Goal: numerical parity AND performance parity with Python.

## Required Skills (reload after context compression)

Always load these skills at session start and after context compression:
- `rust-best-practices` — Rust idioms, error handling, API design
- `justfile` — just task runner conventions  
- `python-uv-enforcer` — Python env via uv

## Code Conventions

### Error Handling
- **Library code** (`src/lib.rs`, `src/model/`, `src/weights.rs`, etc.): use `thiserror` — `IrodoriError` enum in `src/error.rs`
- **Binaries** (`src/bin/`): use `anyhow` — `anyhow::Result<()>` on `main()`
- **Benches** (`benches/`): use `.expect()` / `anyhow::Result` as appropriate

### Rust Idioms Enforced
- ADT/enum for mutually exclusive variants (not multiple `Option` fields)
- Type state pattern for multi-phase builders (e.g., `InferenceBuilder`)
- New-type pattern for domain values
- No `unwrap()` in library code; use `?` with typed errors
- `cargo add` for all new deps (never edit `Cargo.toml` versions by hand)
- `cargo clippy --all-targets` before every commit; `cargo fmt` before commit
- No `#[allow(...)]` without justification; no `unsafe` without explicit user sign-off

### Feature Flags
- `backend_cuda` — Rust/burn CUDA f32 backend (default for perf benchmarks)
- `backend_cuda_bf16` — CUDA bf16 (available but 17% slower than f32 due to CubeCL)
- `backend_wgpu` — WGPU backend
- `profile` — enables NVTX range annotations for nsys profiling

Features are mutually exclusive for backends; enforced with `compile_error!` in bins.

### Module Layout
- `src/lib.rs` — public API surface; re-exports from submodules
- `src/model/` — burn modules: `dit.rs`, `diffusion.rs`, `condition.rs`, `attention.rs`, `mlp.rs`, `norm.rs`, `text_encoder.rs`, `ref_encoder.rs`
- `src/rf.rs` — Euler RF sampler (`sample_euler_rf_cfg`)
- `src/weights.rs` — safetensors weight loading
- `src/error.rs` — `IrodoriError` (thiserror)
- `src/profiling.rs` — NVTX RAII guard + `nvtx_range!` macro (no-op without `profile` feature)
- `src/bin/` — `bench_realmodel.rs`, `infer.rs`, `validate.rs`, `e2e_compare.rs`
- `benches/inference.rs` — Criterion benchmarks (uses small synthetic model)

## Architecture

### Model: DiT (Diffusion Transformer)
- Text encoder → text hidden states
- Aux encoder: either speaker (ref audio latent) OR caption (text tokens) — ADT enum
- DiT blocks: AdaLN + JointAttention + SwiGLU MLP × 12
- Output: patched latent (seq × latent_dim)

### Key Types
- `AuxConditioner<B>`: `enum { Speaker(SpeakerConditioner<B>), Caption(CaptionConditioner<B>) }`
- `AuxConditionState<B>`: `enum { Speaker(SpeakerConditionState<B>), Caption(CaptionConditionState<B>) }`
- `AuxConditionInput<B>`: `enum { Speaker{ref_latent, ref_mask}, Caption{ids, mask}, None }`
- `EncodedCondition<B>`: text_state, text_mask, aux: `Option<AuxConditionState<B>>`
- `InferenceBuilder<S>`: type-state builder (`Unconfigured → Configured → Ready`)
- `SamplingRequest<B>`: full input bundle
- `GuidanceConfig`: `CfgGuidanceMode::Independent | Joint`

### Sampling: Euler RF-CFG
- `sample_euler_rf_cfg` in `src/rf.rs`
- 40 steps, linear schedule, t ∈ [1.0, 0.0]
- CFG: cond forward + uncond forward; speaker KV cache for efficiency
- KV cache: `Option<KvCache<B>>` avoids re-encoding speaker context each step

## Performance Status

| Backend | Mean (ms) | vs Python |
|---|---|---|
| Python/PyTorch CUDA bf16 | 2,636 | 1.0× |
| Rust/burn CUDA f32 | 5,113 | 1.94× |
| Rust/burn CUDA bf16 | 5,776 | 2.19× |

### GPU Kernel Breakdown (nsys, seq=750, steps=40)
- `matmul_entry`: 59.5% — CubeCL tiled GEMM (bottleneck vs cuBLAS)
- `reduce_kernel`: 19.4% — softmax, layer norm
- `elemwise_fuse`: 16.3% — fused elementwise (fusion is already on)
- `into_contiguous_kernel`: 2.1% — attention transpose layout copies

### Performance Gap Root Causes
1. CubeCL matmul vs cuBLAS: JIT kernels ~2× slower for large GEMM
2. Burn attention: multi-step (Q@K^T → softmax → @V) vs Flash Attention (tiled, no N×N materialization)
3. `burn::backend::Cuda` = `Fusion<CubeBackend<...>>` — fusion already enabled

### Path Forward (decided: burn 0.21 + Flash Attention)
- **burn 0.21 pre-release**: significant attention improvements in pre.2+
  - `attention()` API expanded: scale, attn_bias, softcap, is_causal
  - Causal Flash Attention enabled
  - Attention autotune added
  - CAVEAT: Flash Attention "doesn't work on most backends" issue still open (CUDA assertion failure)
- **Strategy**: explore via git worktree; benchmark before merging to main
- **NOT using**: cuBLAS via `cudarc` (requires `unsafe` — needs explicit user sign-off)

## Numerical Parity Status
- E2E 4-step CFG sampling: max_abs_diff = 0.0 (exact match) ✅
- Single-step forward: max_abs_diff < 1e-7 ✅

## Task Runner (just)
Key recipes:
- `just bench-cuda` — CUDA f32 benchmark (seq=750, steps=40)
- `just bench-cuda-profile` — nsys + NVTX profile run
- `just e2e` — full E2E Python fixture + Rust comparison
- `just e2e-rust` — Rust-only comparison (fixtures must exist)
- `just validate` — tiny synthetic model validation
- `just test` — `cargo test`
- `just check` — `cargo clippy --all-targets`

## Review Criteria
1. **Correctness**: E2E numerical parity with Python (max_abs_diff < 1e-3)
2. **Performance**: target ≤ 2× Python (current 1.94×); ideal parity
3. **Rust idiomaticity**: ADT enums, type state, trait abstractions
4. **Maintainability**: clear module boundaries, typed errors, no magic numbers
5. **API cleanliness**: lib public API intentional; no over-exposure

## Important Files
- `docs/benchmarks.md` — performance results + analysis
- `docs/planning/` — implementation plans
- `docs/review/` — quality reviews and advice
- `docs/user-inputs/` — all user messages (archive)
- `target/profile_warm.nsys-rep` — nsys profile (gitignored)
- `target/model_converted.safetensors` — real 500M checkpoint (gitignored)

## Notes on burn 0.21 Upgrade
- burn 0.21 is pre-release (0.21.0-pre.3 latest as of 2026-04-14)
- Breaking changes likely; use git worktree for safe exploration
- Flash Attention API changes: `tensor::module::attention()` gains scale/attn_bias/is_causal
- Issue #4325 ("FlashAttention doesn't work on most backends") still open — CUDA has assertion failure
- Worktree branch: `feature/burn-0.21`
