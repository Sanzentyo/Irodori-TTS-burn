# Backend Generics & Abstraction Analysis

> **Historical document.** Written when `backend_*` feature flags and `select_*_backend!()` macros
> were still in use. Those have since been removed in favour of runtime enum dispatch
> (`dispatch_inference!`/`dispatch_training!`). The generic architecture analysis below
> remains accurate; only the compile-time feature selection mechanism is obsolete.

## Overview

This document analyzes the current state of generics usage, backend-specific code, and
abstraction opportunities in Irodori-TTS-burn.

## Summary

| Aspect | Status | Notes |
|---|---|---|
| Library model code (`src/model/`) | ✅ Excellent | Fully `<B: Backend>` generic |
| Sampling / RF code (`src/rf.rs`) | ✅ Good | Generic |
| Weight loading (`src/weights.rs`) | ⚠️ Partial | Loads as `f32`, then casts — bf16 double-conversion |
| Norm epsilon scalars | ⚠️ Medium | `add_scalar(eps as f32)` — ignores backend float type |
| RoPE precompute (`src/model/rope.rs`) | ⚠️ Acceptable | CPU-side `Vec<f32>` — cast via `from_floats` is OK |
| Backend selection in binaries | ⚠️ Boilerplate | `#[cfg]` blocks repeated in every binary |
| Device init in binaries | ⚠️ Hardcoded | `LibTorchDevice::Cuda(0)` duplicated |
| Backend label | ⚠️ Inconsistent | Only `bench_realmodel.rs` has the label fn |
| Compile-time mutual-exclusion guards | ⚠️ Duplicated | Same `compile_error!` pattern in all binaries |

---

## 1. Library Code — Generics (Good)

The library (`src/lib.rs`, `src/model/`, `src/rf.rs`, `src/inference.rs`) is entirely
`<B: Backend>` generic. There are **zero `#[cfg(feature = "backend_*")]`** blocks in the
library. This is the correct architecture.

Example:
```rust
// src/model/attention.rs
pub struct SelfAttention<B: Backend> {
    pub(crate) wq: Linear<B>,
    pub(crate) wk: Linear<B>,
    ...
}
```

Config uses `f64` for hyperparameters (high precision), only cast to tensor ops when needed.

---

## 2. Issues Found in Library Code

### 2a. Norm epsilon — hardcoded `f32` scalar

**File:** `src/model/norm.rs`, lines ~28, 61, 157

```rust
// Current:
.add_scalar(self.eps as f32)

// Should be (for backend float-type correctness):
.add_scalar(burn::tensor::ElementConversion::elem::<B::FloatElem>(self.eps))
// or equivalently: cast to backend type
```

**Impact:** On bf16 backends, `add_scalar(f32_val)` may involve a type promotion/demotion
depending on burn's internal handling. The correct approach is to use the backend's
native float type as the scalar.

**Severity:** Medium — may slightly affect numerical precision on bf16 (burn may handle
this transparently, but correctness depends on burn's scalar broadcasting rules).

### 2b. RoPE table — CPU `f32` precompute

**File:** `src/model/rope.rs`, lines 20–28

```rust
let freqs: Vec<f32> = (0..half).map(...).collect();
let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
let mut cos_data = vec![0.0f32; ...];
let mut sin_data = vec![0.0f32; ...];
// ...
Tensor::from_floats(TensorData::new(cos_data, shape), device)
```

**`Tensor::from_floats`** internally calls `.convert::<B::FloatElem>()`, so the cast to
backend type is handled correctly. The CPU-side `Vec<f32>` is fine for setup/precompute.

**Severity:** Low — `from_floats` handles the conversion. Only risk is extra CPU memory
during precompute on first call, which is negligible.

### 2c. Weight loading — always via `f32`

**File:** `src/weights.rs`, lines 63–78

```rust
fn to_f32_vec(key: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(...)).collect()),
        Dtype::BF16 => Ok(bytes.chunks_exact(2).map(|b| half::bf16::from_le_bytes(b).to_f32()).collect()),
        ...
    }
}
```

All weights are loaded as `Vec<f32>`, then stored in `TensorData::new(floats, shape)`.
When loading into a bf16 backend, burn converts `f32 → bf16` during `Tensor::from_data()`.

**Impact:**
- Peak memory during load: 2× what the model actually needs on GPU for bf16 (f32 intermediate)
- For a 500M-param bf16 model: ~2 GB CPU RAM intermediate vs ~1 GB needed
- No impact on inference correctness or speed

**Optimization (not yet done):**
```rust
fn to_native_vec<E: burn::tensor::Element>(
    dtype: Dtype, bytes: &[u8]
) -> Result<Vec<E>> {
    // When dtype matches E, skip conversion entirely
    // When dtype == BF16 and E == bf16: direct memcopy
}
```

This would halve peak memory during model load for bf16.

---

## 3. Backend-Specific Code in Binaries

All backend selection is confined to `src/bin/`. The library has none.

### 3a. Repeated boilerplate across binaries

Every binary independently declares the same `type B = ...` block:

| Binary | Backend features supported |
|---|---|
| `bench_realmodel.rs` | All 6 (wgpu, cuda, cuda_bf16, tch, tch_bf16, NdArray fallback) |
| `e2e_compare.rs` | 3 (tch, tch_bf16, NdArray fallback) |
| `validate.rs` | Hardcoded `type B = NdArray` — no feature selection |
| `infer.rs` | Needs inspection |

The device initialization pattern is duplicated verbatim:
```rust
// In bench_realmodel.rs AND e2e_compare.rs:
#[cfg(any(feature = "backend_tch", feature = "backend_tch_bf16"))]
let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
#[cfg(not(any(feature = "backend_tch", feature = "backend_tch_bf16")))]
let device = Default::default();
```

The `compile_error!` mutual-exclusion guard is also duplicated.

### 3b. Hardcoded GPU device index

`LibTorchDevice::Cuda(0)` is hardcoded. No `--gpu-id` CLI argument in any binary.
This is fine for single-GPU use but would need changing for multi-GPU.

---

## 4. Abstraction Opportunities

### Option A: Shared binary helper module (`src/bin/backend_select.rs`)

Move the `#[cfg]` boilerplate into a shared file that all binaries `include!()` or
`mod` into. Limitation: Rust doesn't easily allow a shared `type alias` to be
`include!`-d (type aliases can't be re-exported across files trivially).

**Practicality:** Low. Rust doesn't support `mod` across binary targets easily.
The cleanest solution is a macro.

### Option B: `BackendConfig` trait on the backend type

```rust
// Could live in src/lib.rs (conditionally compiled)
pub trait BackendConfig: Backend {
    fn default_device() -> Self::Device;
    fn label() -> &'static str;
}

#[cfg(feature = "backend_cuda")]
impl BackendConfig for burn::backend::Cuda {
    fn default_device() -> Self::Device { Default::default() }
    fn label() -> &'static str { "Cuda (f32)" }
}

#[cfg(feature = "backend_tch")]
impl BackendConfig for burn::backend::LibTorch {
    fn default_device() -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    }
    fn label() -> &'static str { "LibTorch (cuBLAS/FA3 f32)" }
}
// ... etc.
```

Then binaries do just:
```rust
let device = B::default_device();
let label = B::label();
```

**Tradeoffs:**
- ✅ Removes duplicated `#[cfg]` device init from all binaries
- ✅ Centralizes backend metadata (label, device)
- ⚠️ Requires `BackendConfig` trait to be in `src/lib.rs` behind feature flags
  — this introduces feature-gated code into the library, which is currently clean
- ⚠️ The `type B = ...` alias still must be in each binary (Rust limitation)

### Option C: `select_backend!` macro

```rust
macro_rules! select_backend {
    () => {
        // type B declaration + compile_error guards + device init
        // all generated by macro
    }
}
```

**Tradeoffs:**
- ✅ Maximum DRY
- ⚠️ Macros reduce readability/discoverability
- ⚠️ Rust macros can't define type aliases in caller scope cleanly

### Option D: Status quo with minor improvements (recommended)

Rather than full abstraction (which has Rust limitations), targeted improvements:
1. Fix `add_scalar(eps as f32)` → backend-native scalar in `norm.rs`
2. Add `backend_label()` to `e2e_compare.rs` for consistent output
3. Extend `validate.rs` to support feature-based backend selection (consistent with bench)
4. Consider `--gpu-id` CLI arg in `bench_realmodel.rs` and `infer.rs`
5. (Optional) Optimize bf16 weight loading in `weights.rs`

These changes have clear benefit without over-engineering.

---

## 5. Performance Impact Assessment

| Issue | Performance impact | Effort |
|---|---|---|
| `add_scalar(eps as f32)` in norm.rs | None to negligible | Low |
| RoPE `Vec<f32>` precompute | None (one-time setup) | Low |
| f32 weight loading intermediate | Peak RAM only, not runtime | Medium |
| Backend boilerplate in bins | None | Low–Medium |
| Hardcoded `Cuda(0)` | None (single GPU) | Low |

All identified issues are **non-performance-critical**. The library's generic core is
correct and the only optimization with measurable effect would be the bf16 weight
loading (reduces peak memory load from ~4 GB → ~2 GB for 500M bf16 model).

---

## 6. Proposed Work Items (if approved)

| ID | Description | Files affected | Priority |
|---|---|---|---|
| A | Fix `add_scalar(eps as f32)` → backend-native in `norm.rs` | `src/model/norm.rs` | Medium |
| B | Add `BackendConfig` trait or shared `backend_label()` to lib | `src/lib.rs`, `src/bin/*.rs` | Low |
| C | `validate.rs`: add feature-based backend selection | `src/bin/validate.rs` | Low |
| D | `infer.rs`: verify and align with bench backend selection | `src/bin/infer.rs` | Low |
| E | Optimize bf16 weight loading (skip f32 intermediate) | `src/weights.rs` | Medium |
| F | Add `--gpu-id` CLI arg to `bench_realmodel.rs`, `infer.rs` | `src/bin/*.rs` | Low |

**Recommendation:** Items A and E have the most technical merit. B–D are DRY improvements.
F is a usability improvement.
