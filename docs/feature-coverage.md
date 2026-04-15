# Feature Coverage: Rust/burn vs Python Irodori-TTS

This document tracks the coverage of the original Python
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) features in the Rust/burn
reimplementation.

## Summary

| Category | Status | Notes |
|---|---|---|
| Core model (DiT forward pass) | ✅ Full | All layers ported and numerically validated |
| RF sampler (Euler CFG) | ✅ Full | `sample_euler_rf_cfg` with all CFG modes — split into `src/rf/` submodules |
| Weight loading (safetensors) | ✅ Full | Native-dtype (f32/f16/bf16) with byte validation — split into `src/weights/` submodules |
| Inference pipeline | ✅ Full | `InferenceBuilder` type-state pattern |
| Caption / VoiceDesign mode | ✅ Full | Caption encoder architecture fully implemented |
| Config serialization | ✅ Full | ModelConfig, SamplingConfig via serde — split into `src/config/` submodules |
| Multi-backend support | ✅ Full | NdArray, Wgpu (f32/f16), Cuda (f32/bf16), LibTorch (f32/bf16) |
| DACVAE codec | ✅ Full | Encoder + Decoder + VAE bottleneck; parity verified (mean err ~4e-6) |
| Text normalization | ✅ Full | `src/text_normalization.rs`, 10 unit tests, Python parity verified |
| LoRA weight merging | ✅ Full | `src/lora.rs` + `InferenceBuilder::load_weights_with_adapter` |
| E2E pipeline (text → WAV) | ✅ Full | `src/bin/pipeline.rs`; RF sampler + DACVAE decode + tail trimming |
| Training loop | ✅ LoRA only | `src/train/trainer/` — LoRA fine-tuning with grad accumulation, validation, warm restart, gradient clipping, condition dropout (text/speaker/caption), stratified timestep sampling. Caption-conditioned training supported with post-encoding dropout. Full-model training not yet ported. |
| Training throughput | ✅ Parity | Rust ~5.8 steps/sec vs Python ~5.6 steps/sec on RTX A6000 (f32, batch=4, LoRA r=8) |
| Dataset / manifest | ✅ Full | `src/train/dataset/` — JSONL manifest, batched iterator with epoch shuffle, padding/masking |
| Gradio Web UI | ❌ Out of scope | `gradio_app.py`, `gradio_app_voicedesign.py` |

## Ported Components

### Core Model (`src/model/`)

| Python file/class | Rust file | Status |
|---|---|---|
| `model.py: TextToLatentRFDiT` | `src/model/dit/model.rs` | ✅ |
| `model.py: RMSNorm` | `src/model/norm.rs: RmsNorm` | ✅ |
| `model.py: LowRankAdaLN` | `src/model/norm.rs: LowRankAdaLN` | ✅ |
| `model.py: SelfAttention` | `src/model/attention.rs: SelfAttention` | ✅ |
| `model.py: JointAttention` | `src/model/attention.rs: JointAttention` | ✅ |
| `model.py: SwiGLUMLP` | `src/model/mlp.rs: SwiGluMlp` | ✅ |
| `model.py: DiTBlock` | `src/model/dit/model.rs: DiTBlock` | ✅ |
| `model.py: TextEncoder` | `src/model/text_encoder.rs` | ✅ |
| `model.py: RoPE` | `src/model/rope.rs` | ✅ |
| `model.py: patchify/unpatchify` | `src/model/dit/model.rs` (inline) | ✅ |
| Caption conditioning (`wk_caption`, `wv_caption`) | `src/model/attention.rs` | ✅ |

### RF Sampler (`src/rf/`)

Split into submodules: `euler_sampler.rs` (Euler CFG loop), `math.rs` (numerical helpers),
`params.rs` (SamplerParams, GuidanceConfig, CfgGuidanceMode).

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
| HF Hub checkpoint download | Manual download + `InferenceBuilder::load_weights()` | ✅ (no built-in Hub API) |
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

### RoPE Caching (`src/model/rope.rs`, `src/model/dit/`, `src/rf/`)

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
| JSONL manifest dataset | `src/train/dataset/` | ✅ | `ManifestDataset` + `BatchIterator` with epoch shuffle, padding/masking |
| Training loop | `src/train/trainer/` | ✅ | Split into submodules: condition_dropout.rs, detached_encoding.rs, gradient_clipping.rs, resume.rs, validation.rs |
| LoRA layers | `src/train/lora_layer.rs` | ✅ | `LoraLinear` adapter (rank/alpha parameterised) |
| LoRA model | `src/train/lora_model.rs` | ✅ | `LoraTextToLatentRfDiT` with frozen base + trainable LoRA |
| LoRA weight I/O | `src/train/lora_weights.rs` | ✅ | PEFT-compatible adapter save/load/restore |
| Checkpointing | `src/train/checkpoint.rs` | ✅ | Per-step adapter + config snapshots |
| LR schedule | `src/train/lr_schedule.rs` | ✅ | Linear warmup + cosine decay |
| Loss | `src/train/loss.rs` | ✅ | Echo-style masked MSE, RF interpolation/velocity |
| Config validation | `src/config/training.rs` | ✅ | `LoraTrainConfig::validate()` |
| CLI + TOML config | `src/bin/train_lora.rs` | ✅ | `--config` file or individual CLI flags |
| Throughput optimizations | `src/train/trainer/` | ✅ | Detached conditioning, safe_softmax bypass |

**Performance** (50 steps, batch=4, RTX A6000, f32, strict parity config):
- Python PyTorch: **5.55** steps/sec (180.3 ms/step)
- Rust burn+LibTorch: **6.06** steps/sec (164.9 ms/step) — **9% faster**
- Rust breakdown: fwd=81ms, bwd=45ms, optim=36ms, data=1.8ms
- See `docs/benchmarks/training-performance.md` for detailed analysis

**Not yet implemented** (compared to Python `train.py`):
- Full-model (non-LoRA) training — only LoRA fine-tuning is supported
- ~~Caption-conditioned training~~ ✅ Implemented (post-encoding dropout, tokenizer fallback)
- DDP / multi-GPU training
- W&B logging
- Muon optimizer (Python has AdamW + Muon; Rust has AdamW only)

**Implemented training features**:
- ✅ Gradient norm clipping (`grad_clip_norm`) — global L2 norm clipping matching PyTorch's `clip_grad_norm_` semantics (not per-parameter). Default 1.0.
- ✅ Condition dropout (`text_condition_dropout`, `speaker_condition_dropout`, `caption_condition_dropout`) — per-sample mask zeroing for CFG regularization. Caption dropout applied post-encoding (NaN-safe). Default 0.1 text/speaker, 0.0 caption.
- ✅ Stratified timestep sampling (`timestep_stratified`) — stratified logit-normal sampling for variance reduction. Default enabled.
- ✅ Reproducible training (`training_seed`) — seeded `StdRng` threaded through timestep sampling and condition dropout; `B::seed()` for backend RNG. Default seed 42. 2 determinism tests in loss.rs.

## Implementation Quality Fixes

Rubber-duck code review identified four correctness/performance issues, all resolved:

### 1. GPU readback stall in RF sampler (perf)

`src/rf/euler_sampler.rs` was performing `Tensor::into_data()` (GPU→CPU sync) every step to log
x_t and v_pred statistics. These stats blocks are now gated behind
`tracing::enabled!(tracing::Level::DEBUG)`, eliminating the stall in production
(DEBUG logging disabled by default).

**Impact**: eliminates 80 GPU sync stalls per 40-step inference run.

### 2. Silent LoRA merge failure (correctness)

`src/weights/tensor_store.rs: apply_lora()` used `.unwrap_or_default()` when decoding base
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

### 9. Deduplicated DiT/LoRA construction helpers (code quality)

`TextToLatentRfDiT::new()` and `LoraTextToLatentRfDiT::new()` had ~43 lines of
100%-identical code for building the auxiliary conditioner and zero-initializing
the output projection. Extracted into two `pub(crate)` helper functions in
`src/model/dit.rs`:

- `build_aux_conditioner()` — speaker XOR caption module construction
- `init_zero_out_proj()` — zero-initialized Linear with correct Row layout

Both constructors now call these shared helpers. 5 regression tests added to
verify speaker/caption/default modes, zero initialization, and weight shape.

**Impact**: eliminates maintenance burden of keeping two identical construction
blocks synchronized. The block-level forward loop was intentionally NOT deduplicated
because inference and training have fundamental differences (KV caching, NVTX profiling,
debug capture).

## Unit Test Coverage

| Module | Tests | Coverage |
|--------|-------|---------|
| `src/config/model.rs` | 10 tests | `ModelConfig::validate()` edge cases — zero heads, non-divisible dims, odd head_dim for RoPE, missing speaker fields |
| `src/config/training.rs` | 16 tests | `LoraTrainConfig::validate()` — zero batch_size/max_steps/grad_accum/lora_r, warmup ≥ max_steps, negative lr, dropout out-of-range, grad_clip, t_std/t_mean, TOML deserialization |
| `src/config/sampling.rs` | 8 tests | `SamplingConfig` defaults, serde roundtrip, empty-JSON→default, partial JSON deserialization; `CfgGuidanceMode` FromStr valid/invalid, Display roundtrip, serde rename |
| `src/model/attention.rs` | 8 tests | `sdpa` all-masked→zero, partial mask non-zero; `build_joint_mask` both-None, ctx-only shape, latent mask propagation; KV cache equivalence (no-aux, with-aux, caption-mode cached-vs-uncached) |
| `src/model/feed_forward.rs` | 7 tests | Default hidden_dim computation, custom hidden_dim, shape preservation, SwiGLU semantics, zero input→zero output, no-bias verification, round_up helper |
| `src/model/text_encoder.rs` | 5 tests | `bool_mask_to_float` shape+values, TextBlock forward shape, `from_cfg` forward shape, masked positions remain zero |
| `src/model/speaker_encoder.rs` | 9 tests | `patch_sequence_with_mask` noop/halving/mask propagation/error, `unpatchify_latent` noop/reshape, `bool_mask_to_int` values, encoder forward shape, masked positions zero |
| `src/model/condition.rs` | 10 tests | AuxConditionState variant identification, state_and_mask shapes, zeros_like preservation, clone values; AuxConditionInput::from_request priority/fallback/none; EncodedCondition::zeros_like with/without aux |
| `src/model/dit/` | 19 tests | CondModule output shape & SiLU activation; model construction (speaker/caption); out_proj zero-init layout; forward output shape; forward_with_cond_cached equivalence; prepend_masked_mean_token shape/values/all-masked edge case; build_aux_conditioner (speaker/caption/default); encode mismatch tests (speaker+caption, caption+speaker, None input); init_zero_out_proj (zero-init + row layout) |
| `src/model/norm.rs` | 5 tests | RmsNorm forward shape, LowRankAdaLN forward shapes, zero-init gate |
| `src/model/rope.rs` | 8 tests | Frequencies, rotation, identity at θ=0, equivariance |
| `src/lora.rs` | 3 tests | Prefix stripping, scale computation, 2×2 matmul |
| `src/text_normalization.rs` | 10 tests | Full normalization pipeline coverage |
| `src/rf/math.rs` | 12 tests | `rf_interpolate` at t=0/t=1/t=0.5/batched, `rf_velocity_target` correctness, `rf_predict_x0` inverts interpolation, `temporal_score_rescale` noop at t=1/t>1, identity at σ=0/k=1+σ>0, exact value, finite output |
| `src/rf/euler_sampler.rs` | 6 tests | `cfg_scale_for` dispatch, timestep schedule shape/endpoints/uniform spacing, alternating CFG cycle/single-signal, use_cfg t-range check |
| `src/rf/` (other) | 8 tests | `SamplerParams::validate` — zero steps, zero/negative/inf speaker scale, out-of-range min_t, valid config; `scale_speaker_kv_cache` — doubles aux + rebuilds ctx, respects max_layers |
| `src/weights/` | 21 tests | TensorEntry validation, f32/bf16/f16 decode, roundtrip encode/decode, TensorStore load, linear transpose, linear with/without bias, linear_dims, embedding, rms_norm, missing weight errors |
| `src/train/dataset/` | 9 tests | Manifest loading, blank-line handling, shuffle determinism, batch padding/masking, mixed speaker refs, exhaustion |
| `src/train/loss.rs` | 9 tests | `erfinv` known values/boundary, logit-normal range, stratified range/variance, seeded RNG reproducibility, loss pipeline determinism, seed divergence |
| `src/train/lr_schedule.rs` | 8 tests | Warmup linear ramp, cosine decay, min_lr floor, edge cases |
| `src/train/lora_layer.rs` | 4 tests | Forward shape, initial LoRA=base identity, nonzero delta changes output, scale=alpha/r |
| `src/train/lora_weights.rs` | 4 tests | Save+restore roundtrip, missing file error, incomplete checkpoint detection, shape mismatch detection |
| `src/train/checkpoint.rs` | 7 tests | f32 roundtrip, directory structure, adapter_config fields, safetensors keys+shapes, stale tmp cleanup, overwrite existing checkpoint, no tmp dir remains |
| `src/train/lora_model.rs` | 7 tests | Speaker/caption construction, forward backbone shape, encode+backbone consistency, caption forward_train, speaker/caption shape parity, freeze_base_weights preservation |
| `src/train/trainer/` | 11 tests | `parse_step` ×4, condition dropout (noop/all/caption/none), caption dropout post-encode (prob1/prob0/none) |
| `src/error.rs` | 6 tests | Display messages, From conversions (io::Error, SafetensorError), Debug, Result alias |
| `src/inference.rs` | 7 tests | InferenceBuilder type-state transitions, weight loading |
| `src/backend_config.rs` | 12 tests | Backend enum dispatch, variant counts, reduced precision detection |
| `src/codec/layers.rs` | 7 tests | Snake1d shape/nonlinearity, conv_pad/conv_transpose_pad sizes, ResidualUnit shape/residual/determinism |
| `src/codec/bottleneck.rs` | 3 tests | Encode returns codebook_dim channels, decode restores latent_dim, time dimension preserved |
| `src/codec/encoder.rs` | 4 tests | EncoderBlock channel doubling, time downsampling by stride, batch preservation; full Encoder channel progression (1→4→8→16→32→64, time 256→16) |
| `src/codec/decoder.rs` | 2 tests | WmHead tanh output bounded [-1,1], single output channel |
| `src/model/diffusion.rs` | 4 tests | DiffusionBlock shape (speaker), hidden_dim accessor, residual finite outputs, caption-conditioned shape |

**Total: 269 tests** (158 core + 33 default features + 78 train/lora), all passing, clippy clean.

### Error handling improvements

- `patch_sequence_with_mask` converted from `assert!` panic to `Result<..., IrodoriError::Shape>`
- Result propagated through: `AuxConditioner::encode`, `encode_conditions`, `forward`,
  `forward_train`, `encode_conditions_detached`, `sample_euler_rf_cfg`
- `trainer.rs` fully migrated from anyhow to thiserror (IrodoriError::Training, Config, Dataset)
- `lora_weights.rs` migrated from anyhow to thiserror
- `lora_weights.rs` strict resume validation: expected key completeness + shape matching

### Checkpoint robustness

- Atomic checkpoint saving: temp-dir + rename pattern for crash-safe writes
- Duplicate final save guard: skips redundant save when last step already checkpointed
- Resume validation: all required LoRA A/B tensors must be present with correct shapes

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

| `--backend` flag | Backend type | Status |
|---|---|---|
| `ndarray` | `NdArray<f32>` (CPU) | ⚠️ Works but impractically slow for real workloads |
| `wgpu` | `Wgpu<f32>` | ✅ Works — RF avg 9,972ms (SIGSEGV fix: `_exit(0)` in pipeline.rs) |
| `wgpu-f16` | `Wgpu<f16>` | ✅ Works — faster than f32 (requires shader-f16 GPU extension) |
| — | `Wgpu<bf16>` | ❌ Runtime panic — WGSL has no native bf16 (excluded from enum) |
| `cuda` | `Cuda<f32>` | ✅ Works — RF avg 20,089ms (CubeCL JIT; improves after first run) |
| `cuda-bf16` | `Cuda<bf16>` | ✅ Works — RF avg 25,596ms (CubeCL JIT; slower than f32 due to autotuning) |
| `libtorch` | `LibTorch<f32>` | ✅ Works — RF avg 3,923ms |
| `libtorch-bf16` | `LibTorch<bf16>` | ✅ Works — RF avg 2,216ms (**fastest**, 44% faster than f32) |

## Runtime Backend Dispatch (enum-based)

For binaries that need runtime backend selection (e.g., `--backend cuda-bf16`), the library
provides `InferenceBackendKind` and `TrainingBackendKind` enums with dispatch macros.

**Key design choices:**
- Fully monomorphised — no `dyn` or dynamic dispatch
- NdArray included in `InferenceBackendKind` for E2E validation (CPU-only, not for production)
- WGPU bf16 excluded from `InferenceBackendKind` (known runtime panics)
- `clap::ValueEnum` derived behind `#[cfg(feature = "cli")]`
- `is_reduced_precision()` method on `InferenceBackendKind` for runtime tolerance selection

### `InferenceBackendKind` (7 variants)

| Variant | CLI name | Backend type |
|---------|----------|-------------|
| `NdArray` | `ndarray` | `NdArray` |
| `Wgpu` | `wgpu` | `Wgpu<f32>` |
| `WgpuF16` | `wgpu-f16` | `Wgpu<f16>` |
| `CudaF32` | `cuda` | `Cuda<f32>` |
| `CudaBf16` | `cuda-bf16` | `Cuda<bf16>` |
| `LibTorchF32` | `libtorch` | `LibTorch<f32>` |
| `LibTorchBf16` | `libtorch-bf16` | `LibTorch<bf16>` |

### `TrainingBackendKind` (4 variants)

| Variant | CLI name | Backend type |
|---------|----------|-------------|
| `CudaF32` | `cuda` | `Autodiff<Cuda<f32>>` |
| `CudaBf16` | `cuda-bf16` | `Autodiff<Cuda<bf16>>` |
| `LibTorchF32` | `libtorch` | `Autodiff<LibTorch<f32>>` |
| `LibTorchBf16` | `libtorch-bf16` | `Autodiff<LibTorch<bf16>>` |

### Dispatch macros

```rust
// With device binding (recommended for CLI entrypoints)
dispatch_inference!(backend_kind, gpu_id, |B, device| {
    let model = TextToLatentRfDiT::<B>::load(&device)?;
    model.forward(&input)
});

// Type-only (custom device setup)
dispatch_inference!(backend_kind, |B| {
    B::backend_label()
});

// Training dispatch (binds Autodiff<Base>)
dispatch_training!(backend_kind, gpu_id, |B, device| {
    let trainer = LoraTrainer::<B>::new(config, &device)?;
    trainer.train(dataset)
});
```

## Cargo Feature Flags

The library is gated behind feature flags. Default features provide inference capability out of the box.

| Feature | Default | Description |
|---------|---------|-------------|
| `inference` | ✅ | `InferenceBuilder` / `InferenceEngine` type-state API |
| `codec` | ✅ | DACVAE encoder/decoder |
| `text-normalization` | ✅ | Japanese text normaliser (regex, unicode-normalization, once_cell) |
| `lora` | — | LoRA adapter loading/merging (additive to inference) |
| `train` | — | LoRA fine-tuning infrastructure (dataset, trainer, loss) |
| `cli` | — | Binary-only deps (clap, anyhow, hf-hub, hound, tracing-subscriber) |

### Subsystem dependencies

Feature flags now activate their specific optional dependencies:

| Feature | Optional deps pulled in |
|---------|------------------------|
| `text-normalization` | `regex`, `unicode-normalization`, `once_cell` |
| `train` | `rand`, `toml`, `tokenizers`, `libm` |
| `cli` | `clap`, `anyhow`, `hf-hub`, `hound`, `tokenizers`, `tracing-subscriber` |

### Test counts by feature configuration

| Configuration | Test count |
|---------------|-----------|
| `--no-default-features` | 138 |
| Default (`inference` + `codec` + `text-normalization`) | 171 |
| All library features (`inference,codec,text-normalization,lora,train`) | **232** |

### Binary required-features

| Binary | Required features |
|--------|-------------------|
| `train_lora` | `train`, `cli` |
| `pipeline` | `inference`, `codec`, `text-normalization`, `cli` |
| `infer` | `inference`, `cli` |
| `bench_codec` | `codec`, `cli` |
| `codec_e2e` | `codec`, `cli` |
| `bench_realmodel` | `inference`, `cli` |
| `validate` | `cli` |
| `e2e_compare` | `inference`, `cli` |
| `full_model_e2e` | `inference`, `cli` |

### Usage

```bash
# Default (inference + codec + text-normalization, no binaries)
cargo build --release

# Build a binary (backend selected at runtime via --backend)
cargo build --release --features cli --bin pipeline

# With LoRA support
cargo build --release --features "lora,cli" --bin pipeline

# Training
cargo build --release --features "train,cli" --bin train_lora

# Run with a specific backend
cargo run --release --features cli --bin pipeline -- --backend libtorch-bf16 ...
cargo run --release --features "train,cli" --bin train_lora -- --backend libtorch --config train.toml
```

### Binary dispatch migration status

7 of 9 binaries have been migrated to runtime dispatch macros:

| Binary | Dispatch method | Notes |
|--------|----------------|-------|
| `bench_realmodel` | ✅ `dispatch_inference!` | `--backend` required arg |
| `bench_codec` | ✅ `dispatch_inference!` | `--backend` required arg; device threaded through |
| `infer` | ✅ `dispatch_inference!` | `--backend` required; `_exit(0)` preserved |
| `pipeline` | ✅ `dispatch_inference!` | `--backend` required; `_exit(0)` preserved |
| `train_lora` | ✅ `dispatch_training!` | `--backend` required |
| `e2e_compare` | ✅ `dispatch_inference!` | `--backend ndarray` default; runtime tolerance via `is_reduced_precision()` |
| `full_model_e2e` | ✅ `dispatch_inference!` | `--backend ndarray` default; runtime tolerance via `is_reduced_precision()` |
| `validate` | Hardcoded `type B = NdArray` | CPU fixture validation — intentionally not migrated |
| `codec_e2e` | Hardcoded `type B = NdArray<f32>` | Codec parity test — intentionally not migrated |

`select_inference_backend!()` and `select_train_backend!()` have been **removed** along with
all `backend_*` feature flags. Backend selection is now exclusively via runtime enum dispatch
(`dispatch_inference!` / `dispatch_training!` with `--backend` CLI flag).
