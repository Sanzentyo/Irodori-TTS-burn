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
- `cli` — CLI binaries (clap, anyhow, hf-hub, hound)
- `train` — LoRA fine-tuning infrastructure
- `lora` — LoRA adapter loading/merging
- `codec` — DACVAE encoder/decoder (on by default)
- `text-normalization` — Japanese text normaliser (on by default)
- `profile` — enables NVTX range annotations for nsys profiling

Backend selection is at runtime via `--backend` flag (libtorch, libtorch-bf16, cuda, cuda-bf16, wgpu, wgpu-f16, wgpu-raw, wgpu-raw-f16, ndarray).
- **Mac/Metal recommended**: `libtorch-mps-f16` — fastest on M-series Mac (RTF 0.377 with torch 2.10.0); `wgpu-raw-f16` as no-dep fallback (RTF 0.628, avoids burn-fusion DACVAE crash)

### Module Layout
- `src/lib.rs` — public API surface; re-exports from submodules
- `src/model/` — burn modules: `dit.rs`, `diffusion.rs`, `condition.rs`, `attention.rs`, `mlp.rs`, `norm.rs`, `text_encoder.rs`, `ref_encoder.rs`
- `src/rf.rs` — Euler RF sampler (`sample_euler_rf_cfg`)
- `src/weights.rs` — safetensors weight loading
- `src/error.rs` — `IrodoriError` (thiserror)
- `src/profiling.rs` — NVTX RAII guard + `nvtx_range!` macro (no-op without `profile` feature)
- `src/bin/` — `bench_realmodel.rs`, `infer.rs`, `validate.rs`, `e2e_compare.rs`, `codec_e2e.rs`
- `benches/inference.rs` — Criterion benchmarks (uses small synthetic model)
- `src/codec.rs` — DACVAE codec module declaration + re-exports
- `src/codec/` — `model.rs`, `encoder.rs`, `decoder.rs`, `bottleneck.rs`, `layers.rs`, `weights.rs`
- `src/text_normalization.rs` — Japanese text normalization (NFKC + char substitutions + regex passes; 10 unit tests, Python parity verified)
- `src/lora.rs` — LoRA weight merging (merge_lora, inference loader; 3 tests)
- `src/inference.rs` — `InferenceBuilder` type-state pipeline (Unconfigured → Configured → Ready)
- `src/kernels/` — Custom WGSL kernels (reference/experimental; not in hot path): `fused_adaln`, `fused_sdpa`, `fused_sdpa_native`, `fused_sdpa_native_f16`, `subgroup_diagnostic`, `f16_diagnostic`
- `src/train/` — LoRA fine-tuning: `dataset/`, `trainer/`, `lora_layer.rs`, `lora_model.rs`, `lora_weights.rs`, `checkpoint.rs`, `loss.rs`, `lr_schedule.rs`

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

### CUDA / LibTorch (RTX 5070 Ti)

| Backend | Mean (ms) | vs Python |
|---|---|---|
| **Rust/burn LibTorch bf16** | **1,939** | **0.74× (26% FASTER)** ✓ |
| Python/PyTorch CUDA bf16 | 2,636 | 1.0× |
| Rust/burn LibTorch f32 | 3,150 | 1.20× |
| Rust/burn 0.21 CUDA f32 + FA | 4,497 | 1.71× |
| Rust/burn 0.20.1 CUDA f32 | 5,113 | 1.94× |
| Rust/burn CUDA bf16 (CubeCL) | 5,776 | 2.19× |

**Goal met**: Rust/burn LibTorch bf16 is 26% faster than Python baseline.

### WGPU Metal + LibTorch MPS (M4 Pro Mac Mini)

| Backend | Mean (ms) | RTF | Notes |
|---|---|---|---|
| **Rust/burn LibTorch MPS f16** | **11,320** | **0.377** | ✅ recommended on M-series Mac |
| Rust/burn LibTorch MPS f32 | 11,926 | 0.398 | ✅ MPS f32 fallback |
| **Rust/burn WgpuRaw f16** | **18,850** | **0.628** | ✅ recommended no-dep fallback |
| Rust/burn Wgpu f16 | 18,155 | 0.61 | ❌ burn-fusion DACVAE crash |
| Rust/burn WgpuRaw f32 | 36,451 | 1.22 | ✅ f32 no-dep fallback |
| Rust/burn Wgpu f32 | 35,745 | 1.19 | ❌ burn-fusion DACVAE crash |

**LibTorch MPS f16 recommended on Mac**: 1.66× faster than WgpuRaw f16 (with torch 2.10.0).
**WgpuRaw f16 recommended as no-dep fallback**: avoids burn-fusion crash.

### GPU Kernel Breakdown (nsys, seq=750, steps=40)
- `matmul_entry`: 59.5% — CubeCL tiled GEMM (bottleneck vs cuBLAS)
- `reduce_kernel`: 19.4% — softmax, layer norm
- `elemwise_fuse`: 16.3% — fused elementwise (fusion is already on)
- `into_contiguous_kernel`: 2.1% — attention transpose layout copies

### Performance Gap Root Causes
1. CubeCL matmul vs cuBLAS: JIT kernels ~2× slower for large GEMM
2. Burn attention: multi-step (Q@K^T → softmax → @V) vs Flash Attention (tiled, no N×N materialization)
3. `burn::backend::Cuda` = `Fusion<CubeBackend<...>>` — fusion already enabled

### Path Forward (current state: **performance goal met**)
- **burn 0.21 + FA DONE**: 4,497ms (1.71× Python) via CubeCL + autotune Flash Attention
- **LibTorch f32 DONE**: 3,150ms (1.20×) — cuBLAS GEMM + FA3 via PyTorch's SDPA
- **LibTorch bf16 DONE**: 1,939ms (0.74×) — **26% faster than Python baseline** ✅
  - Requires `Irodori-TTS` venv; uses `LIBTORCH_USE_PYTORCH=1`, `LIBTORCH_BYPASS_VERSION_CHECK=1`
  - Benchmark: `just bench-tch-bf16`; E2E check: `just e2e-tch-bf16`
- **LibTorch MPS f16 DONE**: 11,320ms, RTF 0.377 on M4 Pro Mac ✅ — **fastest on Mac**
  - Benchmark: `just bench-tch-mps-f16`; requires torch 2.10.0 venv
- **WgpuRawF16 DONE**: 18,850ms, RTF 0.628 on M4 Pro Metal ✅ — **no-dep fallback on Mac**
  - Full pipeline test passed (text→WAV, no burn-fusion crash)
  - Benchmark: `just bench-wgpu-raw-f16`; Pipeline: `just pipeline-real-raw-f16`
- **Standalone CubeCL path** still useful (no external dep): 4,497ms via `just bench-cuda`
- **NOT using**: cuBLAS via `cudarc` (requires `unsafe` — needs explicit user sign-off)

## DACVAE Codec

- `DacVaeCodec<B>` in `src/codec/model.rs` — encode/decode audio ↔ latent
- `load_codec<B>(path, device)` — loads from pre-converted safetensors
- `pad_to_hop_length` uses `PadMode::Reflect` (matches Python `F.pad(..., "reflect")`)
- E2E parity: mean abs err ~4e-6 vs Python (f32 precision limit) ✅
- Weight conversion: `scripts/convert_dacvae_weights.py` (resolves weight_norm)
- Task runner: `just codec-e2e` — generate Python reference + run Rust parity check

## Numerical Parity Status
- E2E 4-step CFG sampling (NdArray f32): max_abs_diff = 0.0 (exact match) ✅
- E2E 4-step CFG sampling (LibTorch f32): max_abs_diff = 0.0 (exact match) ✅
- E2E 4-step CFG sampling (LibTorch bf16): max_abs_diff = 5.84e-3 (tol=5e-2) ✅
- E2E 4-step CFG sampling (WgpuRaw f16): max_abs_diff = 5.29e-4 (tol=5e-2) ✅
- Single-step forward: max_abs_diff < 1e-7 ✅
- **DACVAE codec encode**: mean_abs_err ~4e-6, max_abs_err ~3.4e-5 ✅

## Task Runner (just)
Key recipes:
- `just bench-tch-bf16` — **fastest on CUDA**: LibTorch bf16 benchmark (0.74× Python)
- `just bench-tch` — LibTorch f32 benchmark (1.20× Python)
- `just bench-cuda` — CUDA f32 (CubeCL, no external dep, 1.71× Python)
- `just bench-cuda-profile` — nsys + NVTX profile run
- `just bench-tch-mps-f16` — **fastest on Mac**: LibTorch MPS f16 benchmark (RTF 0.377)
- `just bench-tch-mps` — LibTorch MPS f32 benchmark (RTF 0.398)
- `just bench-wgpu-raw-f16` — **no-dep fallback on Mac**: WgpuRawF16 benchmark (RTF 0.628)
- `just bench-wgpu` / `just bench-wgpu-f16` — Wgpu Fusion backends (crash during full pipeline)
- `just bench-wgpu-raw` — WgpuRaw no-fusion f32 (RTF 1.22)
- `just pipeline-real-raw-f16` — full text→WAV pipeline test (wgpu-raw-f16, recommended on Mac)
- `just pipeline-real-raw` — full text→WAV pipeline test (wgpu-raw, f32 fallback)
- `just e2e` — full E2E Python fixture + Rust comparison (NdArray)
- `just e2e-tch` — E2E with LibTorch f32 backend
- `just e2e-tch-bf16` — E2E with LibTorch bf16 backend
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

## Notes on burn Version
- Current: `0.21.0-pre.3` with `autotune` feature (CubeCL Flash Attention enabled)
- `tensor::module::attention()` used in `src/model/attention.rs` for all SDPA calls
- `EmptyRecord` (was `ConstantRecord`) used in `src/weights.rs`
- `clamp_min(1.0_f32)` used in `src/model/dit.rs` (avoids CubeCL vectorization bug)
- Worktree `feature/burn-0.21` merged to master; worktree at `../Irodori-TTS-burn-burn21`
