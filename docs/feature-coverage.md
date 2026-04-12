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
| Training loop | ❌ Out of scope | `train.py`, optimizer scheduling, W&B logging |
| Dataset / manifest | ❌ Out of scope | `dataset.py`, `prepare_manifest.py` |
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

## Performance Optimizations

### RoPE Caching (`src/model/rope.rs`, `src/model/dit.rs`, `src/rf.rs`)

`RopeFreqs<B>` struct caches the `(cos, sin)` tables for the latent sequence.
`TextToLatentRfDiT::precompute_latent_rope()` computes the table once per inference run.
`forward_with_cond_cached(&RopeFreqs<B>)` is the hot path used by `rf.rs`.
Eliminates 120× redundant recomputation (40 steps × 3 CFG passes per step).

### Numerical Validation Coverage

`just validate` now runs both speaker-conditioned and caption-conditioned paths.
Both paths verified against Python fixtures with `max_abs_diff < 1e-3`.



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

### Training / Dataset (`train.py`, `lora.py`, `dataset.py`)

Training infrastructure is out of scope for this port (inference-focused).

## Backend Availability

| Feature flag | Backend type | Status |
|---|---|---|
| (default) | `NdArray<f32>` (CPU) | ✅ Works — slow (~41min estimated for seq=750/steps=40) |
| `backend_wgpu` | `Wgpu<f32>` | ✅ Works — 7,033ms |
| `backend_wgpu_f16` | `Wgpu<f16>` | ✅ Works — 4,486ms (requires shader-f16 GPU extension) |
| `backend_wgpu_bf16` | `Wgpu<bf16>` | ❌ Runtime panic — WGSL has no native bf16 |
| `backend_cuda` | `Cuda<f32>` | ✅ Works — 4,497ms |
| `backend_cuda_bf16` | `Cuda<bf16>` | ✅ Works — 5,776ms (CubeCL not Tensor-Core tuned) |
| `backend_tch` | `LibTorch<f32>` | ✅ Works — 3,150ms |
| `backend_tch_bf16` | `LibTorch<bf16>` | ✅ Works — 1,899ms (28% faster than Python baseline) |
