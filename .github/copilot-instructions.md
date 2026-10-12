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
- `lora` — LoRA adapter loading/merging
- `codec` — DACVAE encoder/decoder (on by default)
- `text-normalization` — Japanese text normaliser (on by default)
- `profile` — enables NVTX range annotations for nsys profiling

Production backend selection is intentionally restricted to `wgpu-wgsl`.

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
- `src/kernels/` — Production-connected custom WGSL kernels and focused profiling tools

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

The production path is strict-FP32 raw WGPU plus production-connected WGSL
kernels. CUDA, LibTorch, reduced-precision, and training backends are outside
this branch. Performance acceptance covers 0.5, 1, 2, 4, and 8 second output
lengths and reports both device-complete and CPU-readback-complete timings.

Current RTX 3060 Ti evidence and remaining bottlenecks live in
`docs/benchmarks/rtx-3060ti-v4-length.md` and
`docs/benchmarks/rtx-3060ti-v4-duration.md`.

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
- `just pipeline-real` — production WGPU text-to-WAV inference
- `just validate-stages` — matched FP32 stage timing campaign
- `just validate-lengths` — 0.5/1/2/4/8-second inference sweep
- `just validate-duration` — duration predictor sweep
- `just profile-codec` — production codec stage profiler
- `just ci` — format, clippy, and library tests

## Review Criteria
1. **Correctness**: E2E numerical parity with Python (max_abs_diff < 1e-3)
2. **Performance**: every WGPU sample below the matching PyTorch global minimum
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
- Current: `0.21.0-pre.3` with WGPU, template kernels, and autotune
- `tensor::module::attention()` used in `src/model/attention.rs` for all SDPA calls
- `EmptyRecord` (was `ConstantRecord`) used in `src/weights.rs`
- `clamp_min(1.0_f32)` used in `src/model/dit.rs` (avoids CubeCL vectorization bug)
- Worktree `feature/burn-0.21` merged to master; worktree at `../Irodori-TTS-burn-burn21`
