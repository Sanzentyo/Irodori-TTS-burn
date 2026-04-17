# WGPU Performance Optimization Plan

## Goal

Reduce WGPU f32 inference time. Current: **6,720ms (1.75× Python)** on RTX 5070 Ti.
Best WGPU result: **4,538ms (f16, 1.18× Python)**.
Reference fastest: LibTorch bf16 at **1,309ms (0.34× Python)**.

## Current State (2025-07-24)

### Completed

- ✅ `WgpuRaw` backend variant (no Fusion wrapper, enables `client.launch()`)
- ✅ Custom WGSL RMSNorm kernel — 3-5× per-call speedup, ~0.4% total impact
- ✅ Fused AdaLN kernel (RMSNorm + modulate) — 3.9-6× per-call, ~0.7% total impact
- ✅ WGPU-specific operator profiling (isolated timing with GPU sync)
- ✅ WgpuRaw viability confirmed: only 5% slower than fusion Wgpu (7,049ms vs 6,720ms)
- ✅ Wgpu-f16: 32% faster than f32 (4,538ms vs 6,720ms)
- ✅ Vulkan ≈ DX12: <1% difference
- ✅ Fused SDPA WGSL kernel (row-streaming online softmax) — **0.21× burn generic** ❌
- ✅ Tiled FlashAttention WGSL kernel (score-parallel 2D tiling) — **0.41× burn generic** ❌

### Key Decision: Custom Norm Kernels NOT Worth Model Integration

Fused AdaLN (best kernel) saves ~44ms/inference. Model integration would add
backend-specific code paths for <1% gain. **Skip integration.**

### Key Decision: Fused SDPA Kernel NOT Competitive

Row-streaming online softmax WGSL kernel benchmarked at **0.06-0.21× burn generic**:

| Scenario | burn (µs) | custom (µs) | ratio |
|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 2,029 | 9,764 | 0.21× |
| Short seq (1×16×100×150×128) | 121 | 1,670 | 0.07× |
| Square (1×16×256×256×128) | 188 | 3,405 | 0.06× |
| Small head (1×16×750×850×64) | 1,471 | 8,450 | 0.17× |

**Root cause: untiled row streaming** — each workgroup processes one query row and
reloads ALL K/V positions sequentially. No cross-query K/V reuse via shared memory.
burn's auto-tuned tiled GEMM amortises K/V loads across many output elements.

### Key Decision: Tiled FlashAttention Also NOT Competitive

Tiled FA with score-parallel 2D tiling (shared memory K/V tiles, online softmax):

| Scenario | burn (µs) | FA 16×8 (µs) | FA 8×16 (µs) | 16×8 ratio | 8×16 ratio |
|---|---|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 2,012 | 4,945 | 8,329 | 0.41× | 0.24× |
| Short seq (1×16×100×150×128) | 152 | 1,271 | 1,354 | 0.12× | 0.11× |
| Square (1×16×256×256×128) | 189 | 2,325 | 3,332 | 0.08× | 0.06× |
| Large seq (1×16×1024×1200×128) | 3,554 | 6,455 | 11,876 | 0.55× | 0.30× |

**Improvement vs row-streaming**: 16×8 tiled FA is ~2× faster than row-streaming
(4,945µs vs 9,764µs at DiT dims) — shared memory K/V tiling works as expected.

**Still 2.5× slower than burn generic**: burn's CubeCL auto-tuned tiled matmul with
fusion pipeline (fused softmax + GEMM) is fundamentally harder to beat via WGSL.
The 16×8 config is consistently better than 8×16 (as predicted — fewer Q tiles means
less total K/V traffic).

**Conclusion**: Both SDPA kernel approaches have been exhaustively explored.
The gap is structural: burn's fused CubeCL pipeline vs hand-written WGSL source kernels.

**Conclusion**: SDPA is still a meaningful cost center (22.6%), but attainable gains
require a much more sophisticated tiled kernel than is justified for this portable backend.

## WGPU Operator Profile (WgpuRaw f32, DX12, RTX 5070 Ti)

| Category | µs/forward | % |
|---|---|---|
| Linear (single, ×48) | 18,130 | 19.9% |
| Linear (fused QKV, ×24) | 17,859 | 19.6% |
| **Linear (SwiGLU proj, ×12)** | **29,262** | **32.2%** |
| **SDPA (joint attention, ×12)** | **20,558** | **22.6%** |
| AdaLN (RMSNorm + modulate, ×24) | 1,613 | 1.8% |
| RMSNorm (standalone, ×14) | 713 | 0.8% |
| SwiGLU gate (silu*gate, ×12) | 2,243 | 2.5% |
| Add (residuals, ×24) | 575 | 0.6% |
| **TOTAL** | **90,953** | 100% |

**Takeaway**: matmul (71.7%) + SDPA (22.6%) = **94.3%** of compute.
Custom norm/elementwise kernels address the remaining ~5.7%.

## Strategic Assessment

### What WGSL custom kernels CAN'T do
- Close the 5× gap to LibTorch bf16 (vendor-tuned cuBLAS + FlashAttention)
- Replace burn's matmul implementation (too complex, too deep in the framework)

### What WGSL custom kernels CAN do
- Fuse attention (Q@K^T → softmax → @V) to avoid N×N materialization
- Target f16 path for higher-ROI improvements

### Viable next targets (in priority order)
1. ~~**Fused SDPA kernel (row-streaming)**~~ — 0.21× burn generic (untiled, no K/V reuse)
2. ~~**Tiled FlashAttention SDPA**~~ — 0.41× burn generic (2D tiling, still 2.5× slower)
3. **Accept WGPU as portable backend** — ~4.5s f16 is "good enough portable" ✅
   - LibTorch bf16 (1.3s) remains the performance backend

### NOT worth pursuing
- More norm/elementwise kernels (addressed only ~5.7%)
- Custom matmul in WGSL (too complex for marginal gain over burn's autotune)
- Model integration of existing custom kernels (<1% impact)
- Untiled fused SDPA (0.21× burn generic — wrong GPU decomposition)
- Tiled FlashAttention SDPA (0.41× burn generic — still can't beat CubeCL fusion)

## New Device Protocol

On a new device, run these steps before any optimization work:

```sh
# 1. Verify GPU is idle
nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used --format=csv,noheader

# 2. Get device spec for benchmark filename
# e.g. GPU name → "rtx-4090" or "m3-max" etc.

# 3. Create device benchmark file
cp docs/benchmarks/rtx-a6000.md docs/benchmarks/<device>.md
# Edit System section

# 4. Run all benchmarks (with GPU idle verification first)
just bench-wgpu          # Wgpu fusion f32
just bench-wgpu-f16      # Wgpu fusion f16
just bench-wgpu-raw      # WgpuRaw no-fusion f32
just bench-cuda          # CubeCL CUDA f32
just bench-tch           # LibTorch f32
just bench-tch-bf16      # LibTorch bf16

# 5. Python baseline
cd ~/Irodori-TTS && uv run python bench.py

# 6. Fill in device benchmark file, commit
```

## Key Technical Details

### WGPU buffer aliasing rule (critical)

All WGSL bindings MUST use `var<storage, read_write>`, even for read-only inputs:
- WGPU's `SlicedPool` memory manager packs multiple tensors into shared physical `wgpu::Buffer` pages
- If bindings on the same physical page have different usage flags (STORAGE_READ_ONLY vs
  STORAGE_READ_WRITE), WGPU validation rejects the dispatch
- Using `read_write` for all bindings gives them the same flag, preventing conflicts
- Source: `src/kernels/rms_norm.wgsl` lines 15-17

### WgpuRaw type

```rust
pub type WgpuRaw = burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u32>;
// Defined in: src/backend_config.rs
// Re-exported from: src/lib.rs
```

`burn::backend::Wgpu` = `Fusion<CubeBackend<WgpuRuntime, ...>>` when compiled with `fusion` feature.
The fusion layer provides automatic kernel fusion (~30% speedup expected) but prevents raw `client.launch()` access.

### Kernel dispatch chain

```
SourceKernel<RmsNormKernel: KernelSource>
  → CubeTask<AutoCompiler>  (via Box::new(...))
  → client.launch(kernel_box, cube_count, bindings)
```

`KernelArguments::new().with_buffer(handle.binding())` — each binding from `tensor.handle.binding()`.

### Workgroup size

Capped at 256 for WebGPU portability (`next_power_of_two(dim).min(256)`).
Do not exceed 256 without querying device limits — not guaranteed on all WebGPU implementations.

## Files

| File | Purpose |
|------|---------|
| `src/kernels.rs` | Module root — `rms_norm`, `fused_adaln`, `fused_sdpa`, `fused_sdpa_tiled` |
| `src/kernels/rms_norm.rs` | RMSNorm kernel launcher + test |
| `src/kernels/rms_norm.wgsl` | WGSL compute shader (RMSNorm) |
| `src/kernels/fused_adaln.rs` | Fused AdaLN kernel launcher + tests |
| `src/kernels/fused_adaln.wgsl` | WGSL compute shader (fused RMSNorm + modulate) |
| `src/kernels/fused_sdpa.rs` | Fused SDPA kernel launcher + parity tests |
| `src/kernels/fused_sdpa.wgsl` | WGSL compute shader (row-streaming online softmax) |
| `src/kernels/fused_sdpa_tiled.rs` | Tiled FlashAttention launcher + parity tests |
| `src/kernels/fused_sdpa_tiled.wgsl` | WGSL compute shader (score-parallel 2D tiled FA) |
| `src/backend_config.rs` | `WgpuRaw` type alias + `WgpuRawF32` variant |
| `src/lib.rs` | Re-exports `WgpuRaw` |
| `src/bin/bench_rmsnorm.rs` | RMSNorm micro-benchmark |
| `src/bin/bench_fused_adaln.rs` | Fused AdaLN micro-benchmark |
| `src/bin/bench_fused_sdpa.rs` | Fused SDPA micro-benchmark |
| `src/bin/profile_wgpu_ops.rs` | WGPU operator-level profiling |
| `docs/benchmarks/rtx-5070ti-laptop.md` | Full benchmark results |
| `docs/planning/wgpu-optimization.md` | This plan |
