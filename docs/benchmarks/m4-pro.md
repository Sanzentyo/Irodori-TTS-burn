# Benchmark Results: Apple M4 Pro (Mac Mini)

## System

| Component | Details |
|---|---|
| CPU | Apple M4 Pro |
| GPU | Apple M4 Pro integrated GPU (Metal, ~14 TFLOPS) |
| RAM | 24 GB unified memory |
| OS | macOS (arm64) |
| burn version | 0.21.0-pre.3 |
| WGPU backend | Metal (via wgpu 29.0.1) |
| Rust edition | 2024 |

**Note**: This device has no CUDA GPU. Only WGPU (Metal) and NdArray backends are available.
LibTorch f32/bf16 is CPU-only here and not representative for comparison.

## Model

| Field | Value |
|---|---|
| Checkpoint | `Aratako/Irodori-TTS-500M-v2` |
| model_dim | 1280 |
| num_layers | 12 |
| num_heads | 20 |
| Parameters | ~500M |

---

## Full-Model Benchmark (seq=750, steps=40)

Model: `Aratako/Irodori-TTS-500M-v2` (500M params, model_dim=1280, layers=12, heads=20).

| Backend | Mean (ms) | RTF | vs Wgpu f32 |
|---|---|---|---|
| Rust/burn **Wgpu f16** (Metal) | **18,155** | **0.61** | 0.51× |
| Rust/burn Wgpu f32 (Metal) | 35,745 | 1.19 | 1.00× |
| Rust/burn WgpuRaw f32 (Metal) | 36,451 | 1.22 | 1.02× |

> No CUDA or LibTorch GPU available on this device — Python/CUDA baseline N/A.
> For reference: RTX 5070 Ti LibTorch bf16=1,309ms; Wgpu f16=4,538ms; Wgpu f32=6,720ms.

**Key findings:**
- **Wgpu f16 is 1.97× faster than f32** on Metal (vs 1.48× on DX12) — Metal excels at f16
- **WgpuRaw vs Wgpu fusion overhead: 2%** (same as DX12's 5%) — fusion has minimal benefit
- **Wgpu f16 achieves RTF < 1** (1.65× real-time) — viable for real-time synthesis on Mac
- Metal WGPU is ~4× slower than RTX 5070 Ti WGPU f32 (M4 Pro integrated GPU vs discrete)

---

## Micro-Benchmarks: Custom WGSL Kernels vs burn Generic (Metal)

All tests use `WgpuRaw` backend (Metal). Results compared to DX12/Vulkan on RTX 5070 Ti.

### RMSNorm (custom WGSL vs burn generic)

Config: warmup=10, bench_iters=50

| Scenario | burn (µs) | custom WGSL (µs) | speedup | DX12 speedup |
|---|---|---|---|---|
| DiT RmsNorm (batch=1, seq=750, dim=1024) | 277 | 175 | **1.58×** | 3.08× |
| TextEncoder RmsNorm (batch=1, seq=200, dim=1024) | 114 | 77 | **1.48×** | 4.11× |
| HeadRmsNorm (batch=1, seq=750×16, dim=64) | 268 | 299 | **0.90× (slower!)** | 1.22× |

**Key finding**: Custom RMSNorm is faster for dim=1024 on Metal (1.5×) but slower for small dim (64).
The speedup is smaller than DX12 — Metal's generic kernels are already better-optimized by the driver.

### Fused AdaLN (custom WGSL vs burn generic)

Config: warmup=10, bench_iters=50

| Scenario | burn (µs) | custom fused (µs) | speedup | DX12 speedup |
|---|---|---|---|---|
| DiT AdaLN (1×750×1024) | 312 | 176 | **1.77×** | 3.95× |
| DiT AdaLN short (1×100×1024) | 125 | 53 | **2.37×** | 6.02× |
| DiT AdaLN batch2 (2×750×1024) | 373 | 129 | **2.90×** | 3.38× |
| Small dim (1×750×256) | 158 | 127 | **1.25×** | 3.87× |

Estimated per-inference savings: **~130ms** (960 calls × 135µs Δ at DiT dims × 40 steps).

**Key finding**: Fused AdaLN shows 1.77-2.90× speedup on Metal. Savings are larger in absolute
terms vs DX12 (~130ms vs ~44ms) but model integration is still complex for <2% total-time impact.

### Fused SDPA — Row-Streaming Online Softmax

Config: warmup=10, bench_iters=50

| Scenario | burn (µs) | custom (µs) | ratio | DX12 ratio |
|---|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 7,468 | 43,728 | **0.17×** | 0.21× |
| Short seq (1×16×100×150×128) | 313 | 1,127 | **0.28×** | 0.07× |
| Square (1×16×256×256×128) | 1,167 | 4,614 | **0.25×** | 0.06× |
| Small head (1×16×750×850×64) | 5,248 | 34,493 | **0.15×** | 0.17× |

**Conclusion**: Row-streaming SDPA is 4-7× SLOWER than burn on Metal. Same fundamental
problem as DX12 — no K/V reuse across query rows. Not viable.

### FlashAttention (Tiled, Native f32, and Native f16 variants)

Config: warmup=5, bench_iters=20

| Scenario | burn (µs) | T16×8 | T8×16 | N32×8 f32 | N16×16 f32 | N32×16 | N32×8 f16 | N16×16 f16 |
|---|---|---|---|---|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 7,522 | 36,721 | 31,870 | 31,130 | **23,136** | 29,743 | 30,825 | 23,711 |
| Short seq (1×16×100×150×128) | 323 | 1,262 | 1,023 | 1,084 | **716** | 905 | 1,032 | 732 |
| Square (1×16×256×256×128) | 1,220 | 3,952 | 3,357 | 3,538 | **2,454** | 2,992 | 3,197 | 2,588 |
| Large seq (1×16×1024×1200×128) | 10,070 | 68,303 | 58,310 | 58,489 | **42,724** | 53,492 | 58,286 | 44,096 |

Best f32 config: **N16×16** at DiT dims → **3.08× burn** (DX12 N32×8 was **1.46× burn**).

**f16 storage variant results:**
- N32×8 f16: ~1% faster than f32 on Metal — negligible benefit
- N16×16 f16: ~2% SLOWER than f32 on Metal — no benefit
- **Conclusion**: f16 storage optimization does NOT help on Metal

**Key findings on Metal:**
- All custom FA variants are 3-8× SLOWER than burn on Metal
- f16 storage brings no benefit on Metal (unified memory, Metal driver already f16-optimized)
- f16 kernel parity confirmed: max_diff ~1e-7 (sub-f16-precision) ✅
- Expected f16 benefit: DX12/RTX 5070 Ti where global memory bandwidth is the bottleneck

### f16 Storage Kernel — Parity Test Results (Metal)

`enable f16;` is confirmed safe on Metal (unlike `enable subgroups;`).

All parity tests pass with max_diff ~1e-7 (expected for f16 input quantization + f32 accumulation):

| Test | Config | max_diff | Status |
|---|---|---|---|
| Small (B=1 H=2 SQ=4 SKV=6) | Q32×8 | 5.96e-8 | ✅ |
| Masked (B=1 H=2 SQ=4 SKV=8) | Q32×8 | 8.94e-8 | ✅ |
| Model dims (B=1 H=16 SQ=32 SKV=48) | Q32×8 | 1.12e-7 | ✅ |
| Edge (B=1 H=1 SQ=17 SKV=9) | Q32×8 | 1.27e-7 | ✅ |
| Small (B=1 H=2 SQ=4 SKV=6) | Q16×16 | 5.96e-8 | ✅ |
| Model dims (B=1 H=16 SQ=32 SKV=48) | Q16×16 | 1.12e-7 | ✅ |

---

## WGSL Extension Status on Metal

### f16 Storage Extension (`enable f16;`)

**Confirmed on Metal**: `enable f16;` works correctly — **NO naga bug**.

Diagnostic: `cargo test --lib kernels::f16_diagnostic -- --ignored --nocapture`

- Small input (32 elements): max_diff=0.00e0 (all_zero=false) ✅
- Large input (1024 elements): max_diff=1.87e-3 (f16 quantization error only) ✅

**Conclusion**: `enable f16;` is safe on Metal (unlike `enable subgroups;`).

### Subgroup Operations (`enable subgroups;`)

**Confirmed on Metal**: `enable subgroups;` causes **silent all-zero output** — same naga bug
as DX12 and Vulkan. Workaround: use subgroup builtins without the directive (they work).

Diagnostic: `cargo test --release --lib kernels::subgroup_diagnostic -- --ignored --nocapture`

```
WITH    `enable subgroups;` (ran first): max_diff=2.05e2 (all_zero=true)
WITHOUT `enable subgroups;` (ran second): max_diff=0.00e0 (correct)
>>> BUG CONFIRMED: `enable subgroups;` breaks kernel output on Metal too <<<
```

**Conclusion**: The naga `enable subgroups;` bug is universal (DX12, Vulkan, Metal).
Deferred until upstream wgpu/naga fix.

---

## Comparison vs RTX 5070 Ti (DX12)

| Metric | RTX 5070 Ti DX12 | M4 Pro Metal | Notes |
|---|---|---|---|
| burn SDPA (DiT dims) | 2,149 µs | 7,522 µs | Metal 3.5× slower |
| best FA f32 / burn ratio | 1.46× | 3.08× | Metal ratio 2× worse |
| best FA f16 / burn ratio | est. ~1.0× | 3.16× | f16 helps DX12, not Metal |
| Fused AdaLN speedup | 3.95× | 1.77× | Metal less dramatic |
| Custom RMSNorm speedup (dim=1024) | 3.08× | 1.58× | Metal less dramatic |
| `enable subgroups;` bug | DX12 ✗ | Metal ✗ | Universal naga issue |
| `enable f16;` | DX12 ✅ | Metal ✅ | Safe on all backends |

**Strategic insight**: Custom WGSL FA kernels are further from competitive on Metal
than on DX12. Metal's GPU scheduler and driver optimizations benefit burn's
generic CubeCL path more than WGSL source kernels.

---

## Conclusions for M4 Pro

1. **Custom FA kernels: NOT viable on Metal** (3.08× burn at best, worse than DX12's 1.46×)
2. **f16 storage FA kernel: implemented and correct** (max_diff ~1e-7), but no Metal speedup
3. **f16 storage expected to help DX12** (N32×8 estimate ~1.0× burn — bandwidth limited there)
4. **Fused AdaLN: marginally useful** (1.77-2.90× speedup; ~130ms savings on Metal)
5. **Subgroup bug confirmed on Metal** — same as DX12/Vulkan (naga issue)
6. **`enable f16;` is safe on Metal** — no naga bug
7. **WGPU (Metal) is the only GPU backend on Mac** — no CUDA/LibTorch bf16
8. **Full inference time measured**: Wgpu f16=18,155ms (RTF 0.61), f32=35,745ms

The WGPU Metal backend is the correct choice for Mac deployment.
Performance will be limited by Metal's matmul efficiency relative to NVIDIA CUDA.
