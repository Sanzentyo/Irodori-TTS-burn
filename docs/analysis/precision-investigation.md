# Precision Investigation: Rust vs Python Numerical Parity

## Motivation

After completing the Rust/burn reimplementation of Irodori-TTS and achieving working
E2E audio generation, the user observed that audio from Rust and Python backends was
"similar vibe but not completely identical." This document records the exhaustive
three-tier numerical comparison used to determine the source of divergence.

**Conclusion (upfront)**: The Rust implementation is numerically correct at every
tested level. Audio differences stem entirely from different RNG implementations
producing different initial noise, not from implementation bugs.

---

## Methodology

Three tiers of comparison, increasing in scope:

| Tier | Model | Comparison |
|------|-------|------------|
| 1. Layer-by-layer | Tiny (dim=64, 2 layers) | Per-DiT-block outputs |
| 2. E2E tiny model | Tiny (dim=64, 2 layers) | Full 10-step sampler output |
| 3. E2E full model | Full 500M (dim=1280, 12 layers) | 10-step sampler output |

All comparisons use **same initial noise, same weights** to isolate implementation
differences from RNG differences.

---

## Tier 1: Layer-by-Layer (Tiny Model)

### Setup

- `scripts/validate_numerics.py` — fixture generator with per-block output hooks
- `src/bin/validate.rs` — Rust comparison binary

Captured tensors:
- `text_state` — output of TextEncoder (speaker and caption modes)
- `speaker_state` / `caption_state` — output of speaker/caption encoder
- `after_in_proj` — output of DiT `in_proj` before first block
- `block_0_out`, `block_1_out` — output after each DiT block
- `v_pred` — final velocity prediction

### Results

| Check | max_abs_diff | Tolerance | Result |
|-------|-------------|-----------|--------|
| text_state (speaker mode) | 2.98e-7 | 1e-3 | ✓ PASS |
| speaker_state | 9.54e-7 | 1e-3 | ✓ PASS |
| after_in_proj | 0.00e+0 | 1e-3 | ✓ PASS |
| block_0_out | 5.96e-8 | 1e-3 | ✓ PASS |
| block_1_out | 5.96e-8 | 1e-3 | ✓ PASS |
| v_pred (speaker mode) | 0.00e+0 | 1e-3 | ✓ PASS |
| text_state (caption mode) | 4.77e-7 | 1e-3 | ✓ PASS |
| caption_state | 4.77e-7 | 1e-3 | ✓ PASS |
| v_pred (caption mode) | 0.00e+0 | 1e-3 | ✓ PASS |
| cached_vs_uncached | 0.00e+0 | 1e-5 | ✓ PASS |

Maximum observed: **9.54e-7** (well within floating-point rounding for f32).

---

## Tier 2: E2E Tiny Model

### Setup

- `scripts/e2e_compare.py` — full 10-step RF sampler with per-step output dumps
- `src/bin/e2e_compare.rs` — Rust comparison binary using same initial noise

The Python fixture generator uses `initial_noise_seed42.safetensors` as input to
the sampler, injected via `sample_euler_rf_cfg(initial_noise=...)`.

### Results

| Backend | max_abs_diff | mean_abs_diff | Result |
|---------|-------------|---------------|--------|
| NdArray f32 | 0.00e+0 | 0.00e+0 | ✓ PASS |

**Exact match** for the tiny model — confirms sampler logic is bit-for-bit identical
when using the same initial noise on the NdArray CPU backend.

---

## Tier 3: E2E Full Model (500M Parameters)

### Setup

- `scripts/full_model_e2e.py` — 10-step Independent CFG with:
  - Input: `target/initial_noise_seed42.safetensors` (shape 1×750×32)
  - Text: "テスト" tokenized as `[1, 43417]` (BOS + one subword)
  - Speaker: `torch.randn(1, 16, 32) * 0.01` (seed=42 non-zero reference latent)
  - CFG scales: text=3.0, speaker=5.0, cfg_min_t=0.0 (all steps use CFG)
- `src/bin/full_model_e2e.rs` — Rust binary using same fixtures

### Config

Derived from safetensors metadata (`config_json` key):

| Parameter | Value |
|-----------|-------|
| model_dim | 1280 |
| num_layers | 12 |
| num_heads | 20 |
| latent_dim | 32 |
| text_dim | 512 |
| text_layers | 10 |
| speaker_dim | 768 |
| speaker_layers | 8 |

### Results

| Backend | max_abs_diff | mean_abs_diff | Tolerance | Result |
|---------|-------------|---------------|-----------|--------|
| NdArray f32 | 2.65e-5 | 5.56e-7 | 1e-2 | ✓ PASS |
| LibTorch CUDA f32 | 3.75e-5 | 7.67e-7 | 1e-2 | ✓ PASS |

The small differences (≈1e-5) are caused by floating-point operation ordering
variations between PyTorch and burn's computation graphs, and between GPU kernel
implementations. These are expected for f32 arithmetic over 10 transformer steps
with 12 layers each.

---

## Root Cause of Audible Difference (Not a Bug)

When running the E2E pipeline with the **same seed but different backends**, the audio
sounds "similar but not identical." This is explained by RNG differences:

| Runtime | RNG Implementation |
|---------|--------------------|
| Python PyTorch | `torch.Generator(device).manual_seed(seed)` → MT19937 |
| Rust NdArray | `B::seed(&device, seed)` → backend-specific PRNG |
| Rust LibTorch | LibTorch's global RNG (different internal state) |

All produce valid Gaussian noise, but the specific tensor values differ — leading to
different diffusion trajectories (and thus different audio realizations) even for the
same text and speaker.

**This is not a bug.** The implementation is numerically verified to be correct.
To get identical audio, use `--noise-file` to inject the same initial noise file into
both Python and Rust.

---

## Reproducing

```sh
# Tier 1 + 2: tiny model layer-by-layer + E2E
just validate

# Tier 3: full-model E2E (NdArray backend)
just full-e2e

# Tier 3: full-model E2E (LibTorch CUDA backend)
just full-e2e-tch
```

All comparisons require the full 500M model at `target/model_converted.safetensors`
and Python environment at `/home/sanzentyo/Irodori-TTS/.venv`.
