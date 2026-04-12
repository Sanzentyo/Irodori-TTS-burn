# Irodori-TTS-burn Architecture

## Overview

A full-scratch Rust reimplementation of [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
using the [`burn`](https://burn.dev) ML framework (v0.20.1, latest stable).

**Goal**: Functionally equivalent implementation with:
- Identical model architecture
- Equivalent numerical results (within floating-point tolerance)
- Competitive or superior performance vs the PyTorch reference

---

## Model Architecture

### Top-Level: `TextToLatentRFDiT`

A **Rectified Flow Diffusion Transformer** (RF-DiT) conditioned on:
1. **Text** — phoneme/character token IDs → `TextEncoder` → cross-attention keys/values
2. **Speaker** (optional) — reference audio latents → `ReferenceLatentEncoder` → cross-attention K/V
3. **Caption** (optional, voice-design mode) — caption text → `TextEncoder` → cross-attention K/V
4. **Timestep** — sinusoidal embedding → MLP → `LowRankAdaLN` modulation

Input/output:
- Input `x_t`: `[B, S, latent_dim × patch_size]` — noisy latent at timestep t
- Output `v_pred`: same shape — predicted velocity (noise direction)

---

## Module Hierarchy

```
TextToLatentRFDiT<B>
├── text_encoder: TextEncoder<B>
│   ├── embedding: Embedding<B>
│   └── blocks: Vec<TextBlock<B>>
│       ├── attention_norm: RmsNorm<B>
│       ├── attention: SelfAttention<B>
│       │   ├── wq, wk, wv, wo, gate: Linear<B>
│       │   ├── q_norm, k_norm: HeadRmsNorm<B>
│       │   └── RoPE (stateless, computed on-the-fly)
│       ├── mlp_norm: RmsNorm<B>
│       └── mlp: SwiGlu<B>
├── caption_encoder: Option<TextEncoder<B>>     ← voice-design mode only
├── speaker_encoder: Option<ReferenceLatentEncoder<B>>
│   ├── in_proj: Linear<B>
│   └── blocks: Vec<TextBlock<B>>
├── text_norm: RmsNorm<B>
├── speaker_norm: Option<RmsNorm<B>>
├── caption_norm: Option<RmsNorm<B>>
├── cond_module: CondModule<B>                  ← timestep → conditioning
│   └── [Linear, SiLU, Linear, SiLU, Linear]
├── in_proj: Linear<B>
├── blocks: Vec<DiffusionBlock<B>>
│   ├── attention_adaln: LowRankAdaLn<B>
│   ├── attention: JointAttention<B>
│   │   ├── wq, wk, wv, wo, gate: Linear<B>
│   │   ├── wk_text, wv_text: Linear<B>
│   │   ├── wk_speaker?, wv_speaker?: Optional<Linear<B>>
│   │   ├── wk_caption?, wv_caption?: Optional<Linear<B>>
│   │   └── q_norm, k_norm: HeadRmsNorm<B>
│   ├── mlp_adaln: LowRankAdaLn<B>
│   └── mlp: SwiGlu<B>
├── out_norm: RmsNorm<B>
└── out_proj: Linear<B>
```

---

## Key Components

### RmsNorm
Standard RMS Layer Norm without mean subtraction:
```
y = x / rms(x) * weight    where rms(x) = sqrt(mean(x²) + ε)
```
- `RmsNorm<B>`: weight shape `[D]`
- `HeadRmsNorm<B>`: weight shape `[H, D_h]` (for QK normalization)

### LowRankAdaLn (Adaptive Layer Norm)
Conditions the diffusion blocks on timestep embedding:
```
shift, scale, gate = chunk(cond_embed, 3, dim=-1)
shift = shift_up(shift_down(silu(shift))) + shift   ← low-rank residual
scale = scale_up(scale_down(silu(scale))) + scale
gate  = gate_up(gate_down(silu(gate)))  + gate

h = rms_norm(x) * (1 + scale) + shift
gate_out = tanh(gate)
return h, gate_out
```

### SwiGLU
```
y = W2(silu(W1(x)) ⊗ W3(x))
```

### Rotary Position Embedding (RoPE)
- Standard RoPE for `SelfAttention` (applied to full `head_dim`)
- Half-RoPE for `JointAttention` (applied to first `heads/2`, rest passthrough)
- Frequencies precomputed as `(cos, sin)` tables: `[seq_len, head_dim/2]`

### JointAttention
Cross-attends latent sequence to text + speaker + caption conditioning:
```
q = wq(x)                   [B, S, H, D_h]
k = cat([k_self, k_text, k_speaker?, k_caption?], dim=1)
v = cat([v_self, v_text, v_speaker?, v_caption?], dim=1)
mask = cat([self_mask, text_mask, speaker_mask?, caption_mask?], dim=1)

y = scaled_dot_product_attention(q, k, v, mask)
y = y ⊗ sigmoid(gate(x))
return wo(y)
```

---

## Inference: Euler Rectified Flow

```
x_T ~ N(0, I)   (initial noise)

for t in linspace(1, 0, num_steps):
    v = model(x_t, t, conditions)    ← optionally with CFG
    x_{t-dt} = x_t + v * dt
```

### Classifier-Free Guidance (CFG) Modes
- **independent**: separate forward passes for each condition dropout
- **joint**: single combined unconditional pass
- **alternating**: alternate which condition is dropped per step

---

## Rust-Specific Design Decisions

### Type State Pattern

#### Backend phantom type
The burn backend type provides compile-time training vs inference distinction:
```rust
// Training: gradients enabled
type TrainModel = TextToLatentRFDiT<Autodiff<NdArray>>;

// Inference: no overhead
type InferModel  = TextToLatentRFDiT<NdArray>;
```

#### `InferenceBuilder` — sealed type-state builder
`src/inference.rs` implements a type-state builder pattern with a sealed marker trait
so construction order is enforced at compile time:

```rust
// States (sealed, cannot be implemented externally)
pub struct Unconfigured;  // no weights
pub struct Loaded;        // weights loaded, no sampling config
pub struct Ready;         // weights + sampling config → can build

let engine = InferenceBuilder::<NdArray, _>::new(device)
    .load_weights("weights.safetensors")?   // Unconfigured → Loaded
    .with_default_sampling()                 // Loaded → Ready
    .build();                                // Ready → InferenceEngine

let latent = engine.sample(request);
```

`InferenceEngine<B>` exposes:
- `.sample(request)` — run full CFG Euler sampler
- `.with_sampling(params)` — clone engine with new params
- `.model_config()` — the `ModelConfig` embedded in the checkpoint
- `.sampling_params()` / `.device()` / `.model()` — accessors

### Enum for CFG Mode
```rust
pub enum CfgGuidanceMode { Independent, Joint, Alternating }
```

### Conditioning Bundle
Rather than Python's loose tuples, use typed structs:
```rust
pub struct ConditionBundle<B: Backend> {
    pub text_state: Tensor<B, 3>,
    pub text_mask: Tensor<B, 2, Bool>,
    pub speaker: Option<SpeakerBundle<B>>,
    pub caption: Option<CaptionBundle<B>>,
}
```

### Context KV Cache
```rust
pub struct CondKvCache<B: Backend> {
    pub k_text: Tensor<B, 4>,
    pub v_text: Tensor<B, 4>,
    pub k_speaker: Option<Tensor<B, 4>>,
    pub v_speaker: Option<Tensor<B, 4>>,
    pub k_caption: Option<Tensor<B, 4>>,
    pub v_caption: Option<Tensor<B, 4>>,
}
pub type ContextKvCache<B> = Vec<CondKvCache<B>>;
```

---

## Source Layout

```
src/
├── lib.rs              — Public API re-exports
├── main.rs             — Thin main (routes to bin/)
├── bin/
│   ├── infer.rs        — Inference CLI (latent only, no codec)
│   ├── pipeline.rs     — Full TTS pipeline (text → WAV via RF + DACVAE)
│   ├── validate.rs     — Numerical validation vs Python fixtures
│   ├── e2e_compare.rs  — E2E comparison of latent outputs
│   ├── codec_e2e.rs    — DACVAE codec parity test
│   ├── bench_realmodel.rs — Real-model wall-clock benchmark
│   └── bench_codec.rs  — DACVAE codec wall-clock benchmark
├── config.rs           — ModelConfig, SamplingConfig (serde)
├── error.rs            — Error types (thiserror)
├── model.rs            — Module declarations (no mod.rs pattern)
├── model/
│   ├── rope.rs         — RoPE frequency tables + application
│   ├── norm.rs         — RmsNorm, HeadRmsNorm, LowRankAdaLn
│   ├── feed_forward.rs — SwiGlu
│   ├── attention.rs    — SelfAttention, JointAttention (+ safe softmax)
│   ├── text_encoder.rs — TextBlock, TextEncoder
│   ├── speaker_encoder.rs — ReferenceLatentEncoder
│   ├── diffusion.rs    — DiffusionBlock
│   └── dit.rs          — TextToLatentRFDiT (main model)
├── rf.rs               — Rectified Flow: sampling, loss, CFG
├── inference.rs        — InferenceBuilder (type-state) + InferenceEngine
├── text_normalization.rs — Unicode/regex text cleaning (Python parity)
├── lora.rs             — PEFT LoRA adapter loading + weight merging
├── weights.rs          — Safetensors weight loading utilities
├── backend_config.rs   — Multi-backend feature-flag configuration
├── profiling.rs        — Optional NVTX profiling support
├── codec.rs            — Codec module declarations
└── codec/
    ├── layers.rs       — Snake1d, NormConv1d, ResidualUnit, padding helpers
    ├── encoder.rs      — Encoder, EncoderBlock
    ├── decoder.rs      — Decoder, DecoderBlock
    ├── bottleneck.rs   — VaeBottleneck
    ├── model.rs        — DacVaeCodec (top-level codec model)
    └── weights.rs      — Weight loading (weight_norm resolution)
```

---

## Numerical Validation

### Infrastructure

- **`scripts/validate_numerics.py`** — Python fixture generator (uv inline script)
  - Creates `target/validate_weights.safetensors`: small model weights (125 tensors + embedded `config_json`)
  - Creates `target/validate_tensors.safetensors`: reference outputs (10 tensors) from PyTorch
  - Small model: `model_dim=64, layers=2, heads=8, latent_dim=8`
  - Deterministic: `torch.manual_seed(42)`, sequential float inputs

- **`src/bin/validate.rs`** — Rust validation binary
  - Loads weights via `load_model::<NdArray>`
  - Parses reference tensors from safetensors bytes (f32 LE)
  - Compares `text_state`, `speaker_state`, `v_pred` vs Python reference
  - Threshold: `max_abs_diff < 1e-3`

### Results (NdArray backend, float32)

| Tensor | max_abs_diff | Status |
|---|---|---|
| `text_state` | 2.98e-7 | PASS |
| `speaker_state` | 9.54e-7 | PASS |
| `v_pred` | 0.00e0 | PASS |

### Bug History

8 correctness bugs were found and fixed during implementation:

| # | Module | Bug | Fix |
|---|---|---|---|
| 1 | `SelfAttention` | Missing `gate` field | Add `gate: Linear`, apply gating |
| 2 | `JointAttention` | Half-RoPE not applied to `k_self` | Apply `apply_rotary_half` to `k_self` |
| 3 | `JointAttention` | Gate applied to attention output instead of `x` | Gate input: `x`, not attention result |
| 4 | `JointAttention` | Gate linear had `with_bias(true)` | Change to `with_bias(false)` |
| 5 | `JointAttention` | `k_norm` not applied to `k_text`/`k_aux` | Apply `k_norm` to all key projections |
| 6 | `HeadRmsNorm` | `eps` hardcoded to `1e-6` instead of `cfg.norm_eps` | Use `cfg.norm_eps` |
| 7 | `SelfAttention` + `JointAttention` | Extra `swap_dims(1,2)` before reshape corrupted layout | Remove the erroneous `swap_dims` |
| 8 | `scaled_dot_product_attention` | `softmax(all_neg_inf)` = NaN (IEEE 754); PyTorch SDPA returns 0 | `is_nan().mask_fill(0.0)` after softmax — see `docs/analysis/nan-softmax-fix.md` |

### just Commands

```sh
just validate-fixtures   # Run Python fixture generator
just validate-rust       # Run Rust validation binary
just validate            # Run both in sequence
```
