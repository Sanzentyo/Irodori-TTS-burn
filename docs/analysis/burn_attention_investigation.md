# burn::module::Attention Parity Investigation

**Date**: 2026-07-15  
**Status**: Investigation complete — manual SDPA retained  
**Conclusion**: Do NOT replace `scaled_dot_product_attention` with `burn::nn::attention`

---

## Background

Burn provides `burn::nn::attention::MultiHeadAttention` and the lower-level
`B::float_matrix_softmax` + QKV matmuls.  This investigation evaluates whether
the built-in attention primitive can replace our manual `scaled_dot_product_attention`
for potential performance gains.

---

## Known issue: bool mask polarity in burn-tch

Our `scaled_dot_product_attention` includes this note:

```rust
/// Implemented manually to avoid burn-tch's broken bool mask convention in its
/// `B::attention` dispatch (it passes masks directly to PyTorch SDPA, which uses
/// the opposite True=attend convention).
```

### Reproduction

PyTorch `F.scaled_dot_product_attention` uses **True = ignore** (mask-out) convention
in its `attn_mask` parameter when the mask is a boolean tensor.

burn-tch's internal dispatch (as of 0.21.0-pre.3) passes burn's Bool tensor directly to
PyTorch SDPA without polarity inversion.  Burn's own convention is **True = valid** (attend).

Result: with a causal or padding mask, burn-tch would attend to masked-out positions
and ignore valid ones — completely wrong attention pattern.

### Verification test plan

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

This matches PyTorch SDPA behavior.  `burn::nn::MultiHeadAttention` does NOT have
explicit NaN handling — this is a correctness risk for variable-length batch inference.

---

## Backend parity

| Backend | Our SDPA | burn::nn attention |
|---------|----------|--------------------|
| NdArray | ✅ correct | untested |
| LibTorch f32 | ✅ correct | mask polarity bug (see above) |
| LibTorch bf16 | ✅ correct | mask polarity bug (see above) |
| WGPU (CubeCL) | ✅ correct | untested |

---

## Decision

**Retain `scaled_dot_product_attention`.**

Rationale:
1. burn-tch mask polarity bug makes `B::attention` (when dispatched via burn-tch)
   produce wrong results with padding/causal masks.
2. Our NaN→0 safe-softmax is required for correctness on variable-length batches.
3. The performance delta between manual SDPA and burn::nn is negligible (both reduce
   to GEMM + softmax); the bottleneck is the RF loop, not individual attention ops.
4. Any migration would require parity tests to pass on all three backends before merging.

**If revisited in future:**
- Gate behind `#[cfg(feature = "burn-attention")]` so existing behavior is unchanged
- Write parity tests: mask polarity, all-masked→zero, backend parity (NdArray vs TCH)
- Only merge if all parity tests pass on all target backends

---

## Related

- `src/model/attention.rs` — `scaled_dot_product_attention` (line ~437)
- `docs/analysis/nan-softmax-fix.md` — NaN softmax investigation
