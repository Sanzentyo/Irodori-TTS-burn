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
The burn backend type provides compile-time training vs inference distinction:
```rust
// Training: gradients enabled
type TrainModel = TextToLatentRFDiT<Autodiff<NdArray>>;

// Inference: no overhead
type InferModel  = TextToLatentRFDiT<NdArray>;
```

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
├── main.rs             — Routes to src/bin/ or single binary
├── bin/
│   └── infer.rs        — Inference CLI
├── config.rs           — ModelConfig, SamplingConfig (serde)
├── error.rs            — Error types (thiserror)
├── model.rs            — Module declarations (no mod.rs pattern)
├── model/
│   ├── rope.rs         — RoPE frequency tables + application
│   ├── norm.rs         — RmsNorm, HeadRmsNorm, LowRankAdaLn
│   ├── feed_forward.rs — SwiGlu
│   ├── attention.rs    — SelfAttention, JointAttention
│   ├── text_encoder.rs — TextBlock, TextEncoder
│   ├── speaker_encoder.rs — ReferenceLatentEncoder
│   ├── diffusion.rs    — DiffusionBlock
│   └── dit.rs          — TextToLatentRFDiT (main model)
├── rf.rs               — Rectified Flow: sampling, loss, CFG
└── inference.rs        — InferenceRuntime (high-level API)
```
