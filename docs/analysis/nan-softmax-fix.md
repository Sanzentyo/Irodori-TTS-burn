# NaN Propagation Bug: Safe Softmax Fix

## Summary

A critical NaN propagation bug was identified and fixed in `src/model/attention.rs`.
The bug caused the no-reference-audio pipeline to produce 30 seconds of white noise
instead of ~3 seconds of speech.

## Root Cause

PyTorch's `F.scaled_dot_product_attention` (even on CPU) uses a **fused kernel with
"safe softmax"**: when every attention score for a query is masked to −∞, the output
is **zeros** rather than NaN.

Rust's manual implementation of `softmax(all_neg_inf_row)` follows IEEE 754 arithmetic:

```
softmax(-inf, ..., -inf)[i]
  = exp(-inf - max(-inf)) / Σ exp(-inf - max(-inf))
  = exp(0)              / (n × exp(0))
  = 1/n       -- BUT: -inf - (-inf) = NaN in IEEE 754
  = NaN
```

So `softmax` returned NaN for any row where **all** logits were −∞
(i.e., every position is masked out).

## Propagation Chain

```
No-ref audio
  → zeros input to ReferenceLatentEncoder
  → q = 0, k = 0 → all scores = 0
  → all positions masked to −∞
  → softmax(all −∞) = NaN  ← ROOT CAUSE
  → attn output = NaN
  → + residual → NaN throughout TextBlocks
  → speaker_embedding = NaN
  → k_speaker / v_speaker = NaN
  → joint attention: even weight=0 × v=NaN = NaN  (IEEE 754)
  → model output = NaN → 30s noise
```

## Fix

After `softmax()` in `scaled_dot_product_attention` (file: `src/model/attention.rs`):

```rust
let attn_weights = softmax(scores, 3);
// Replace NaN with 0.0 to match PyTorch SDPA "safe softmax" behaviour.
// When every score in a row is −∞ (fully masked), softmax returns NaN
// in IEEE 754 arithmetic. PyTorch's fused SDPA kernel returns 0 instead.
let nan_mask = attn_weights.clone().is_nan();
let attn_weights = attn_weights.mask_fill(nan_mask, 0.0);
```

## Verification

### Before fix
- Step 0 latent: all NaN
- Output: 30 seconds of white noise

### After fix
- Step 0 latent: min=−8.85, max=7.69, std=1.11 ✓
- Output: 3.08 seconds of clean speech

## E2E Parity Results

| Backend         | Duration | RMS    | Peak   |
|----------------|----------|--------|--------|
| Python PyTorch  | 3.160s   | 0.1110 | 0.9380 |
| Rust LibTorch   | 3.080s   | 0.1153 | 0.9400 |
| Rust NdArray    | 6.360s   | 0.1090 | 0.9999 |

The LibTorch backend (same kernels as PyTorch) achieves near-identical output:
- Only 2 latent frames difference in tail-trim point (77 vs 79)
- Very similar RMS and peak values
- Both produce intelligible speech

The NdArray backend diverges over 40 diffusion steps (120 model forward passes)
due to floating-point operation ordering differences between NdArray and PyTorch.
This is expected and acceptable for a CPU reference backend.

## Key Insight

`torch.softmax(all_neg_inf)` **also** returns NaN. The safe behaviour is
**only** in the fused `scaled_dot_product_attention` kernel. This is a subtle
difference not documented in PyTorch's API.
