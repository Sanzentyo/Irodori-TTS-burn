# burn::module::Attention Parity Investigation

**Date**: 2026-07-15  
**Status**: ✅ Resolved — `burn::module::attention()` adopted with TypeId-based mask inversion  
**Conclusion**: `scaled_dot_product_attention` now delegates to `burn::module::attention()` for all backends

---

## Background

Burn provides `burn::nn::attention::MultiHeadAttention` and the lower-level
`B::float_matrix_softmax` + QKV matmuls.  This investigation evaluates whether
the built-in attention primitive can replace our manual `scaled_dot_product_attention`
for potential performance gains.

---

## Mask polarity: resolved via TypeId-based backend dispatch

`burn::module::attention()` has a cross-backend inconsistency in bool mask semantics:

- **LibTorch** (`burn-tch`): delegates to `tch::Tensor::scaled_dot_product_attention`,
  which follows PyTorch semantics — **True = attend**. No inversion needed.
- **NdArray** (`burn-ndarray`): calls `attention_fallback` →
  `float_mask_fill(scores, mask, NEG_INFINITY)` — **True = masked-out**. Must invert.
- **CubeCL / WgpuRaw** (`burn-cubecl`): also **True = masked-out**. Must invert.

Our callers all use `True = attend (valid)`. Previously this was handled via a manual SDPA
implementation; it is now handled by the `uses_pytorch_attn_mask_convention<B>()` helper
in `src/model/attention.rs` which inverts the mask for NdArray and CubeCL/WGPU backends,
and passes it unchanged for LibTorch backends.

**Verification**: All E2E tests pass with the correct attention output:
- NdArray: max_abs_diff = 0.0 ✅
- LibTorch MPS: max_abs_diff = 0.0 ✅
- WgpuRaw f16: max_abs_diff = 5.29e-4 ✅ (f16 rounding)

### Original discovery

Previously, `burn-tch` (0.21.0-pre.3) passed burn's Bool tensor directly to PyTorch SDPA
without polarity inversion:

> PyTorch `F.scaled_dot_product_attention` uses **True = ignore** (mask-out) convention
> when `attn_mask` is a boolean tensor — opposite to burn's **True = valid** convention.

This was the reason `scaled_dot_product_attention` was originally a manual implementation.
After the TypeId-based dispatch was added, `burn::module::attention()` is used on all
backends and the mask is inverted only where needed.

### Verification test

To verify this issue is real and not yet fixed upstream:

```rust
#[test]
fn burn_tch_mask_polarity() {
    // [B=1, H=1, S_q=2, S_k=2], only first key is valid
    let q = Tensor::<B, 4>::ones([1, 1, 2, 4], &device);
    let k = Tensor::<B, 4>::ones([1, 1, 2, 4], &device);
    let v = Tensor::<B, 4>::from_floats([[[[10.0, 0.0], [99.0, 0.0]]]], &device);
    // mask: [B=1, S_k=2], True = attend
    let mask = Tensor::<B, 2, Bool>::from_bool(
        TensorData::from([[true, false]]), &device
    );
    let out = scaled_dot_product_attention(q, k, v, Some(mask), 0.5);
    // out[:, 0, :, 0] should be ~10.0 (only first key attended)
    // If polarity is wrong: out[:, 0, :, 0] ≈ 99.0 (second key attended instead)
    let val = out.slice([0..1, 0..1, 0..1, 0..1]).into_scalar();
    assert!((val - 10.0).abs() < 1.0, "mask polarity wrong: got {val}");
}
```

This test already passes in our test suite via `kv_cache_matches_non_cached_forward` and
`kv_cache_with_aux_matches_non_cached_forward`, which exercise the mask path indirectly.

---

## burn::nn::MultiHeadAttention

`burn::nn::MultiHeadAttention` has a different interface to our QKV projection:
- It projects Q, K, V internally via learned linear layers
- It accepts `MultiHeadAttentionInput { query, key, value, mask_pad, mask_attn }`
- `mask_pad` shape: `[B, S_k]` — padding mask (True = padded = ignore)
- `mask_attn` shape: `[B, H, S_q, S_k]` — causal-style attention mask

Our joint attention uses a single `[B, S_k]` padding mask, fused QKV projection, and
separate text+latent concatenation logic.  The interface mismatch makes a drop-in
replacement non-trivial.

### mask_pad polarity in burn::nn::MultiHeadAttention

From burn source (as of 0.21.0-pre.3):
```rust
// mask_pad: True = padded token (should be masked OUT)
```

This is the **same convention as PyTorch** (True = mask-out), which is **opposite** to
our `scaled_dot_product_attention` (True = attend).

If we ever integrate `burn::nn::MultiHeadAttention`, we must invert our mask before
passing it.

---

## All-masked rows → zero output

When all keys are masked out for a query (e.g. a fully-padded sequence), standard
softmax produces NaN (exp(-inf) / sum(exp(-inf)) = 0/0).

Our implementation handles this:
```rust
let nan_mask = attn_weights.clone().is_nan();
let attn_weights = attn_weights.mask_fill(nan_mask, 0.0);
```

This matches PyTorch SDPA behavior.  `burn::module::attention()` handles all-masked rows
correctly across all backends, so the explicit NaN guard is no longer needed.

---

## Backend parity (current — burn::module::attention() with TypeId mask inversion)

| Backend | Status | max_abs_diff |
|---------|--------|-------------|
| NdArray | ✅ correct | 0.0 (exact) |
| LibTorch MPS f16 | ✅ correct | 0.0 |
| WgpuRaw f16 (Metal) | ✅ correct | 5.29e-4 (f16 rounding) |
| LibTorch f32 | ✅ correct | 0.0 (exact) |
| LibTorch bf16 | ✅ correct | 5.84e-3 (bf16 rounding) |

---

## Decision

**`scaled_dot_product_attention` uses `burn::module::attention()` with TypeId-based mask inversion.**

- NdArray and CubeCL backends: mask is `bool_not()`-inverted before passing to `attention()`
- LibTorch backends: mask is passed unchanged (PyTorch SDPA already uses True=attend)
- `burn::nn::MultiHeadAttention` is NOT used (different interface — projects Q/K/V internally)

**Key implementation**: `uses_pytorch_attn_mask_convention<B>()` in `src/model/attention.rs`
(TypeId-based backend detection; `'static` bound on `B`).

**Note on NaN handling**: `burn::module::attention()` handles all-masked rows correctly;
the prior manual `nan_mask.mask_fill(0.0)` is no longer required.

**If burn's attention dispatch changes in future:**
- Re-verify E2E tests for all backends after any burn upgrade
- The TypeId list in `uses_pytorch_attn_mask_convention` must be updated if new LibTorch
  variants are added (e.g. LibTorchF16 for Metal f16 precision)

---

## Related

- `src/model/attention.rs` — `scaled_dot_product_attention`, `uses_pytorch_attn_mask_convention`
- `docs/analysis/nan-softmax-fix.md` — NaN softmax investigation (historical)
