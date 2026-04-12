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
| DACVAE codec | ❌ Not ported | Audio → latent and latent → audio reconstruction |
| Text normalization | ❌ Not ported | Japanese text normalization (`text_normalization.py`) |
| Training loop | ❌ Out of scope | `train.py`, optimizer scheduling, W&B logging |
| Dataset / manifest | ❌ Out of scope | `dataset.py`, `prepare_manifest.py` |
| LoRA fine-tuning | ❌ Out of scope | `lora.py` PEFT/LoRA adapter support |
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

## Not Ported (and why)

### DACVAE Codec (`irodori_tts/codec.py`)

The Python code uses `DACVAECodec` to encode reference audio into latent
sequences and decode output latents back to audio waveforms.

**Current state**: The Rust CLI accepts pre-encoded latent safetensors as input
and writes output latents as safetensors. Encoding/decoding requires the DACVAE
model weights and a CUDA-capable DACVAE inference runtime.

**Path to porting**: Requires porting the DACVAE model architecture separately
(it is not part of the Irodori-TTS checkpoint), or wrapping the Python codec
via `tch` foreign function calls.

### Text Normalization (`irodori_tts/text_normalization.py`)

Simple Unicode and Japanese text normalization (convert full-width numbers to
half-width, expand abbreviations, etc.). Not required for inference correctness
as long as the tokenizer handles raw text; the normalization is purely a
preprocessing convenience.

**Path to porting**: Straightforward regex/Unicode rules. Could be ported as a
small `text_normalization.rs` module using the `unicode-normalization` crate.

### Training / LoRA (`train.py`, `lora.py`, `dataset.py`)

Training infrastructure is outside the current scope of this port, which targets
inference performance parity. LoRA requires injecting adapter modules into the
model weights at runtime.

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
