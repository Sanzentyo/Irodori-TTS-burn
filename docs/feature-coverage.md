# Feature Coverage: Rust/burn vs Python Irodori-TTS

This document tracks the coverage of the original Python
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) features in the Rust/burn
reimplementation.

## Summary

| Category | Status | Notes |
|---|---|---|
| Core model (DiT forward pass) | ✅ Full | All layers ported and numerically validated |
| RF sampler (Euler CFG) | ✅ Full | `sample_euler_rf_cfg` with all CFG modes |
| Weight loading (safetensors) | ✅ Full | Native-dtype (f32/f16/bf16) with byte validation |
| Inference pipeline | ✅ Full | `InferenceBuilder` type-state pattern |
| Caption / VoiceDesign mode | ✅ Full | Caption encoder architecture fully implemented |
| Config serialization | ✅ Full | ModelConfig, SamplingConfig via serde |
| Multi-backend support | ✅ Full | NdArray, Wgpu (f32/f16), Cuda (f32/bf16), LibTorch (f32/bf16) |
| DACVAE codec | ✅ Full | Encoder + Decoder + VAE bottleneck; parity verified (mean err ~4e-6) |
| Text normalization | ✅ Full | `src/text_normalization.rs`, 10 unit tests, Python parity verified |
| LoRA weight merging | ✅ Full | `src/lora.rs` + `InferenceBuilder::load_weights_with_adapter` |
| E2E pipeline (text → WAV) | ✅ Full | `src/bin/pipeline.rs`; RF sampler + DACVAE decode + tail trimming |
| Training loop | ✅ Partial | `src/train/trainer.rs` — LoRA fine-tuning with grad accumulation, validation, warm restart; missing: DDP, wandb, gradient clipping, condition dropout |
| Dataset / manifest | ✅ Full | `src/train/dataset.rs` — JSONL manifest, batched iterator with epoch shuffle, padding/masking |
| Gradio Web UI | ❌ Out of scope | `gradio_app.py`, `gradio_app_voicedesign.py` |

## Ported Components

### Core Model (`src/model/`)

| Python file/class | Rust file | Status |
|---|---|---|
| `model.py: TextToLatentRFDiT` | `src/model/dit.rs` | ✅ |
| `model.py: RMSNorm` | `src/model/norm.rs: RmsNorm` | ✅ |
| `model.py: LowRankAdaLN` | `src/model/norm.rs: LowRankAdaLN` | ✅ |
| `model.py: SelfAttention` | `src/model/attention.rs: SelfAttention` | ✅ |
| `model.py: JointAttention` | `src/model/attention.rs: JointAttention` | ✅ |
| `model.py: SwiGLUMLP` | `src/model/mlp.rs: SwiGluMlp` | ✅ |
| `model.py: DiTBlock` | `src/model/dit.rs: DiTBlock` | ✅ |
| `model.py: TextEncoder` | `src/model/text_encoder.rs` | ✅ |
| `model.py: RoPE` | `src/model/rope.rs` | ✅ |
| `model.py: patchify/unpatchify` | `src/model/dit.rs` (inline) | ✅ |
| Caption conditioning (`wk_caption`, `wv_caption`) | `src/model/attention.rs` | ✅ |

### RF Sampler (`src/rf.rs`)

| Python construct | Rust equivalent | Status |
|---|---|---|
| `sample_euler_rf_cfg` | `rf::sample_euler_rf_cfg` | ✅ |
| `GuidanceConfig` (text/speaker/caption scales) | `rf::GuidanceConfig` | ✅ |
| CFG modes: independent / joint / alternating | `CfgGuidanceMode` enum | ✅ |
| `SamplerParams` | `rf::SamplerParams` | ✅ |

### Inference Pipeline (`src/inference.rs`)

| Python construct | Rust equivalent | Status |
|---|---|---|
| `InferenceRuntime` | `InferenceBuilder` / `InferenceEngine` type-state | ✅ |
| HF Hub checkpoint download | `InferenceBuilder::from_hub()` | ✅ |
| Local checkpoint loading | `InferenceBuilder::load_weights()` | ✅ |

### DACVAE Codec (`src/codec/`)

Fully ported architecture with numerical parity validated against Python:

| Python class | Rust file | Status |
|---|---|---|
| `DACVAE` (`dacvae.py`) | `src/codec/model.rs: DacVaeCodec` | ✅ |
| `Encoder` | `src/codec/encoder.rs: Encoder, EncoderBlock` | ✅ |
| `Decoder` | `src/codec/decoder.rs: Decoder, DecoderBlock` | ✅ |
| `VQBottleneck` (VAE path) | `src/codec/bottleneck.rs: VaeBottleneck` | ✅ |
| `Snake1d`, `NormConv1d`, `ResidualUnit` | `src/codec/layers.rs` | ✅ |
| Weight loading (`weight_norm` resolution) | `src/codec/weights.rs`, `scripts/convert_dacvae_weights.py` | ✅ |

**Parity result** (NdArray f32 backend, 0.5 s sine tone, 48 kHz):
- Mean absolute error vs Python: **~4e-6** (f32 precision limit)
- Max absolute error: **~3.4e-5**
- All 13 latent frames match (E2E parity check: `just codec-e2e`)

**Key design choices**:
- `pad_to_hop_length` uses `PadMode::Reflect` matching Python `F.pad(..., "reflect")`
- Weights resolved from `weight_g`/`weight_v` at conversion time (not at inference)
- Watermark head is loaded but disabled (no-conv path only)

### Text Normalization (`src/text_normalization.rs`)

Fully implemented. All Python normalization rules ported: SIMPLE_REPLACE,
REGEX_REPLACE, strip_outer_brackets, NFKC normalization, dot collapse. 10 unit
tests all pass. Python parity verified.

### LoRA Weight Merging (`src/lora.rs`)

Fully implemented. Supports PEFT-format adapters (keys with `base_model.model.`
prefix). Adapter is merged at load time via `TensorStore::load_with_lora`.
Exposed via `InferenceBuilder::load_weights_with_adapter(checkpoint, adapter_dir)`
and the `--adapter <dir>` flag on both `infer` and `pipeline` binaries.

### E2E Pipeline (`src/bin/pipeline.rs`)

`just pipeline-real --text "..." --output out.wav` for a full text → WAV run.

**E2E Parity Results** (40 steps, seed=42, text="こんにちは、テストです。", no reference audio):

| Backend | Duration | RMS | Peak | Notes |
|---|---|---|---|---|
| Python PyTorch | 3.160s | 0.111 | 0.938 | Reference |
| Rust LibTorch | 3.080s | 0.115 | 0.940 | **≈identical** (2 latent frames diff) |
| Rust NdArray | 6.360s | 0.109 | 1.000 | Expected FP divergence over 120 fwd passes |

**Key finding — safe softmax fix**: The no-reference pipeline previously produced
30s of white noise due to NaN propagation from `softmax(all_neg_inf)`. Fixed by
adding `is_nan().mask_fill(0.0)` after softmax to match PyTorch SDPA behavior.
See `docs/analysis/nan-softmax-fix.md` for full details.

## Codec Performance Benchmarks

Measured on CPU (Intel) using the same 1s/5s sine-tone input.
Rust uses the LibTorch backend (wraps the same PyTorch kernels as Python).

| Benchmark | Python (ms) | Rust/LibTorch (ms) | Ratio |
|-----------|-------------|---------------------|-------|
| encode 1s | 94.9 | 80.8 | **0.85× (Rust faster)** |
| decode 1s | 196.1 | 194.6 | **1.00× (parity)** |
| roundtrip 1s | 300.5 | 239.3 | **0.80× (Rust faster)** |
| encode 5s | 1051.6 | 828.7 | **0.79× (Rust faster)** |
| decode 5s | 1953.6 | 1502.0 | **0.77× (Rust faster)** |
| roundtrip 5s | 3083.5 | 2364.9 | **0.77× (Rust faster)** |

Reproduce with:
```sh
just bench-codec-tch    # Rust/LibTorch
just bench-codec-py     # Python/PyTorch
```

> **NdArray backend note**: NdArray performs pure-CPU matrix ops without
> BLAS tuning. Codec inference on NdArray is ~100× slower than LibTorch and
> is impractical for production use; it is supported only as a fallback for
> environments without a PyTorch installation.

## Multi-Backend Quality Comparison (2026-07-15, updated)

5 test prompts (40 steps, seeds 42–46). All 25/25 runs successful.

### RF Sampler Time (ms, excludes model load)

| Backend | Avg RF time | vs LibTorch f32 |
|---------|------------|-----------------|
| Rust LibTorch f32 | 3923 ms | 1.00× (baseline) |
| Rust LibTorch bf16 | 2216 ms | **0.56× (44% faster)** |
| Rust WGPU f32 | 9972 ms | 2.5× slower |
| Rust CUDA f32 | 20089 ms | 5.1× slower* |
| Rust CUDA bf16 | 25596 ms | 6.5× slower* |

*CubeCL JIT autotuning per-process; first-run overhead. Subsequent runs use cached autotune.

### Wall-Clock Time (seconds, includes model load)

| Backend | Avg | Notes |
|---------|-----|-------|
| Rust LibTorch f32 | 24.9 s | |
| Rust LibTorch bf16 | 10.1 s | **fastest overall** |
| Rust WGPU f32 | 34.9 s | |
| Rust CUDA f32 | 66.1 s | CubeCL JIT |
| Rust CUDA bf16 | 63.5 s | CubeCL JIT |

### Audio Quality Notes

- All 25/25 WAVs are transcribable with correct prosody/timbre
- Not bit-identical across backends (expected: different RNG implementations → different initial noise)
- LibTorch f32/bf16 produce identical durations (same libtorch RNG path)
- CUDA f32/bf16/WGPU produce identical durations to each other (same CubeCL RNG path)
- Duration and RMS are within acceptable variation

### Python bf16

**Not supported on this GPU** (`NVIDIA RTX A6000`, driver 575.57.08): cuBLAS does not
support `CUDA_R_16BF` GEMM. All 5 prompts fail with cuBLAS error. Python f32 remains the reference.

### WGPU SIGSEGV Fix

WGPU f32 previously reported as FAIL (exit code 139) despite producing valid WAVs.
Root cause: NVIDIA Vulkan driver (575.57.08) registers an atexit handler that segfaults
during process teardown. Fixed by calling POSIX `_exit(0)` (via `unsafe extern "C"`)
which bypasses all atexit handlers entirely. See `src/bin/pipeline.rs`.

Reproduce: `just quality-compare`

## Performance Optimizations

### RoPE Caching (`src/model/rope.rs`, `src/model/dit.rs`, `src/rf.rs`)

`RopeFreqs<B>` struct caches the `(cos, sin)` tables for the latent sequence.
`TextToLatentRfDiT::precompute_latent_rope()` computes the table once per inference run.
`forward_with_cond_cached(&RopeFreqs<B>)` is the hot path used by `rf.rs`.
Eliminates 120× redundant recomputation (40 steps × 3 CFG passes per step).

### Numerical Validation Coverage

`just validate` now runs both speaker-conditioned and caption-conditioned paths.
Both paths verified against Python fixtures with `max_abs_diff < 1e-3`.
Layer-by-layer per-DiT-block comparison: see `docs/analysis/layer-comparison.md`.



### DACVAE Codec (`irodori_tts/codec.py`)

**Status: PORTED** — See `src/codec/` and `scripts/convert_dacvae_weights.py`.

The Rust codec achieves full numerical parity with Python (mean error ~4e-6).
Run `just codec-e2e` to reproduce.

### Tail Trimming / `find_flattening_point`

**Status: PORTED** — Implemented as `find_flattening_point` in `src/bin/pipeline.rs`.

The Python `InferenceRuntime` uses a sliding-window heuristic to detect when
the generated latent becomes flat and near-zero (silence), then trims the tail.
This is now ported to Rust and enabled by default (`--trim-tail`).  The same
default parameters are used: `window=20`, `std_threshold=0.05`, `mean_threshold=0.1`.

### Text Normalization (`irodori_tts/text_normalization.py`)

**Status: PORTED** — See `src/text_normalization.rs`.

All rules ported; 10 unit tests pass; Python parity verified.

### LoRA (`irodori_tts/lora.py`)

**Status: PORTED** — See `src/lora.rs`.

PEFT-format adapters merged at weight-load time. Exposed via
`InferenceBuilder::load_weights_with_adapter` and `--adapter` CLI flag.

### Training / Dataset (`src/train/`)

LoRA fine-tuning infrastructure has been implemented:

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| JSONL manifest dataset | `src/train/dataset.rs` | ✅ | `ManifestDataset` + `BatchIterator` with epoch shuffle, padding/masking |
| Training loop | `src/train/trainer.rs` | ✅ | Gradient accumulation, validation, warm restart resume |
| LoRA layers | `src/train/lora_layer.rs` | ✅ | `LoraLinear` adapter (rank/alpha parameterised) |
| LoRA model | `src/train/lora_model.rs` | ✅ | `LoraTextToLatentRfDiT` with frozen base + trainable LoRA |
| LoRA weight I/O | `src/train/lora_weights.rs` | ✅ | PEFT-compatible adapter save/load/restore |
| Checkpointing | `src/train/checkpoint.rs` | ✅ | Per-step adapter + config snapshots |
| LR schedule | `src/train/lr_schedule.rs` | ✅ | Linear warmup + cosine decay |
| Loss | `src/train/loss.rs` | ✅ | Echo-style masked MSE, RF interpolation/velocity |
| Config validation | `src/config.rs` | ✅ | `LoraTrainConfig::validate()` |
| CLI + TOML config | `src/bin/train_lora.rs` | ✅ | `--config` file or individual CLI flags |
| Throughput optimizations | `src/train/trainer.rs` | ✅ | Detached conditioning, safe_softmax bypass |

**Performance** (50 steps, batch=4, RTX A6000, f32):
- Python PyTorch: ~18.9 steps/sec (~53ms/step)
- Rust burn+LibTorch: ~6.4 steps/sec (~155ms/step)
- Gap: ~3× — caused by burn's Autodiff per-op overhead (framework characteristic)
- See `docs/benchmarks/training-performance.md` for detailed analysis

**Not yet implemented** (compared to Python `train.py`):
- DDP / multi-GPU training
- W&B logging
- Muon optimizer (Python has AdamW + Muon; Rust has AdamW only)

**Implemented training features**:
- ✅ Gradient norm clipping (`grad_clip_norm`) — global L2 norm clipping matching PyTorch's `clip_grad_norm_` semantics (not per-parameter). Default 1.0.
- ✅ Condition dropout (`text_condition_dropout`, `speaker_condition_dropout`) — per-sample mask zeroing for CFG regularization. Default 0.1 each.
- ✅ Stratified timestep sampling (`timestep_stratified`) — stratified logit-normal sampling for variance reduction. Default enabled.

## Implementation Quality Fixes

Rubber-duck code review identified four correctness/performance issues, all resolved:

### 1. GPU readback stall in RF sampler (perf)

`src/rf.rs` was performing `Tensor::into_data()` (GPU→CPU sync) every step to log
x_t and v_pred statistics. These stats blocks are now gated behind
`tracing::enabled!(tracing::Level::INFO)`, eliminating the stall in production
(INFO logging disabled).

**Impact**: eliminates 80 GPU sync stalls per 40-step inference run.

### 2. Silent LoRA merge failure (correctness)

`src/weights.rs: apply_lora()` used `.unwrap_or_default()` when decoding base
weights, silently returning an empty `HashMap` on dtype decode failure and zeroing
the weight. Replaced with `.collect::<Result<HashMap<_,_>>>()?` for proper
error propagation.

**Additional**: `src/lora.rs: merge_lora()` now returns `Vec<String>` (affected base
key names) instead of `usize` count. `apply_lora()` pre-scans adapter keys with
`pre_scan_lora_keys()` so only the ~N affected weights are decoded and re-encoded,
not all 600+ base tensors.

### 3. Latent mask ignored in attention (training correctness)

`forward_with_cond_cached()` accepted `latent_mask: Option<Tensor<B, 2, Bool>>` but
the parameter was prefixed `_` and discarded. This caused padded latent positions to
pollute attention during batch training.

The mask is now threaded through the full call chain:
`trainer` → `forward_train` → `forward_backbone` → `DiffusionBlock::forward` →
`JointAttention::forward` → `build_joint_mask`.

**Impact**: training-only; inference parity is unchanged (latent_mask is always None
at inference).

### 5. Independent CFG mode missing alt-pass KV caches (perf)

`Independent` CFG (the default mode) runs one conditioned pass plus N per-signal
unconditioned passes per step. `kv_alt_text/speaker/caption` were only pre-built for
`Alternating` mode, so every unconditional pass in `Independent` mode paid a full
KV projection cost on every step.

With 12 blocks × 40 steps × N-1 alt passes, this was dozens of redundant per-step
KV projections for the most common usage path.

Fix: the `Alternating`-mode guard on `kv_alt_*` construction was replaced with a
`!Joint`-mode guard, so both `Independent` and `Alternating` pre-build their
per-signal caches. `kv_uncond` (the fully-unconditioned cache) is now built only for
`Joint` mode where it is actually used.

### 6. `SpeakerKvConfig` validation (correctness)

`scale_speaker_kv_cache` computes `inv_scale = 1.0 / skv.scale` at revert time.
A `scale=0` would produce `inf`, and negative values produce nonsensical amplification.
`SamplerParams::validate()` now checks:

- `speaker_kv.scale.is_finite() && scale > 0`
- `speaker_kv.min_t` (if set): `finite && in [0, 1]`

`JointAttention::forward` was concatenating `[text_k | aux_k?]`, `[text_v | aux_v?]`
and the context mask on every forward call even when a cache existed. With 12 blocks
× 40 steps × 3 CFG passes = 1440 forward calls, this was 4320 unnecessary
`Tensor::cat` operations per inference run.

`CondKvCache<B>` now stores three pre-concatenated fields: `ctx_k`, `ctx_v`,
`ctx_mask`. These are computed once in `build_kv_cache()` (which now takes `text_mask`
and `aux_mask` parameters). `scale_speaker_kv_cache()` recomputes them after
amplitude scaling.

**Unit tests added**: `kv_cache_matches_non_cached_forward` and
`kv_cache_with_aux_matches_non_cached_forward` in `model::attention::tests` verify
bit-for-bit numerical parity between cached and non-cached paths on the NdArray
backend, both with and without speaker-conditioning aux state.

### 7. Deduplicated ctx KV concatenation helper (code quality)

`JointAttention::forward` and `LoraJointAttention::forward` both had identical
12-line blocks that concatenate projected text+aux K/V tensors along the sequence
dimension. Extracted into `pub(crate) concat_ctx_kv()` in `src/model/attention.rs`;
both call sites now use this shared helper.

**Impact**: eliminates one latent bug class — any future change to the concatenation
logic only needs to be made in one place.

## Unit Test Coverage

| Module | Tests | Coverage |
|--------|-------|---------|
| `src/config.rs` | 17 tests | `ModelConfig::validate()` edge cases; `LoraTrainConfig::validate()` — zero batch_size/max_steps/grad_accum/lora_r, warmup ≥ max_steps, negative lr |
| `src/model/attention.rs` | 7 tests | `sdpa` all-masked→zero, partial mask non-zero; `build_joint_mask` both-None, ctx-only shape, latent mask propagation; KV cache equivalence (no-aux + with-aux) |
| `src/lora.rs` | 3 tests | Prefix stripping, scale computation, 2×2 matmul |
| `src/text_normalization.rs` | 10 tests | Full normalization pipeline coverage |
| `src/rf.rs` | 8 tests | `SamplerParams::validate` — zero steps, zero/negative/inf speaker scale, out-of-range min_t, valid config; `scale_speaker_kv_cache` — doubles aux + rebuilds ctx, respects max_layers |
| `src/train/dataset.rs` | 7 tests | Manifest loading, blank-line handling, shuffle determinism, batch padding/masking, mixed speaker refs, exhaustion |
| `src/train/loss.rs` | 5 tests | `erfinv` known values/boundary, logit-normal range, stratified range/variance |
| `src/train/lora_layer.rs` | 4 tests | Forward shape, initial LoRA=base identity, nonzero delta changes output, scale=alpha/r |
| `src/train/checkpoint.rs` | 4 tests | f32 roundtrip, directory structure, adapter_config fields, safetensors keys+shapes |
| `src/train/lora_model.rs` | 4 tests | Speaker/caption construction, forward backbone shape, encode+backbone consistency |
| `src/train/trainer.rs` | 8 tests | `parse_step` ×4, condition dropout (noop/all/caption/none) |

**Total: 87 tests**, all passing, clippy clean.

### 8. Linear weight zero-init layout fix (correctness)

burn 0.21.0-pre.3 defaults to `LinearLayout::Row` where weight shape is
`[d_input, d_output]` (PyTorch uses `[d_output, d_input]`). Three zero-initialization
sites used the old Col-layout convention, creating transposed weight tensors:

- `LowRankAdaLn` shift/scale/gate_up: `[model_dim, rank]` → `[rank, model_dim]`
- `TextToLatentRfDiT` out_proj: `[patched_latent_dim, model_dim]` → `[model_dim, patched_latent_dim]`
- `LoraTextToLatentRfDiT` out_proj: same fix

**Impact**: Forward passes through freshly-constructed models (before weight loading)
would fail with matmul dimension mismatches. Production inference was unaffected because
`load_record()` replaced the zero-init weights with correctly-shaped checkpoint data.
Training was also unaffected for the same reason. Caught by new unit tests.

## Precision Investigation (Complete)

A three-tier numerical comparison confirmed the Rust implementation is numerically
correct at every layer. See `docs/analysis/precision-investigation.md` for full details.

| Tier | Comparison | max_abs_diff | Result |
|------|-----------|-------------|--------|
| Layer-by-layer (tiny model) | Per-DiT-block vs Python | 9.54e-7 | ✓ All PASS |
| E2E tiny model (NdArray) | 10-step sampler output | 0.00e+0 | ✓ Exact match |
| E2E full model (NdArray f32) | 10-step full 500M model | 2.65e-5 | ✓ PASS |
| E2E full model (LibTorch CUDA f32) | 10-step full 500M model | 3.75e-5 | ✓ PASS |
| E2E full model (LibTorch CUDA bf16) | Rust bf16 vs Python f32 | 1.97e-1 | ✓ PASS (tol=3e-1) |

Audio sounds "similar but not identical" across backends — this is expected:
PyTorch and Burn use different PRNG implementations, producing different initial
noise even with the same integer seed. Use `--noise-file` to get identical audio.

## Backend Availability

| Feature flag | Backend type | Status |
|---|---|---|
| (default) | `NdArray<f32>` (CPU) | ✅ Works — slow (~41min estimated for seq=750/steps=40) |
| `backend_wgpu` | `Wgpu<f32>` | ✅ Works — RF avg 9,972ms (SIGSEGV fix: `_exit(0)` in pipeline.rs) |
| `backend_wgpu_f16` | `Wgpu<f16>` | ✅ Works — faster than f32 (requires shader-f16 GPU extension) |
| `backend_wgpu_bf16` | `Wgpu<bf16>` | ❌ Runtime panic — WGSL has no native bf16 |
| `backend_cuda` | `Cuda<f32>` | ✅ Works — RF avg 20,089ms (CubeCL JIT; improves after first run) |
| `backend_cuda_bf16` | `Cuda<bf16>` | ✅ Works — RF avg 25,596ms (CubeCL JIT; slower than f32 due to autotuning) |
| `backend_tch` | `LibTorch<f32>` | ✅ Works — RF avg 3,923ms |
| `backend_tch_bf16` | `LibTorch<bf16>` | ✅ Works — RF avg 2,216ms (**fastest**, 44% faster than f32) |
