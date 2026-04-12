# Benchmark Results: Rust/burn vs Python/PyTorch

## System

| Component | Details |
|---|---|
| GPU | NVIDIA RTX A6000 (49140 MiB VRAM) |
| CUDA Driver | 575.57.08 (CUDA 12.9) |
| CPU | Linux host (burn NdArray) |
| burn version | 0.21.0-pre.3 |
| PyTorch version | 2.10+ (cu128 build) |
| Rust edition | 2024 |

## Model

| Field | Value |
|---|---|
| Checkpoint | `Aratako/Irodori-TTS-500M-v2` |
| model_dim | 1280 |
| num_layers | 12 |
| num_heads | 20 |
| Parameters | ~500M |

## Benchmark Configuration

| Parameter | Value |
|---|---|
| seq_len | 750 (canonical `fixed_target_latent_steps`) |
| num_steps | 40 (default) |
| cfg_guidance_mode | Independent |
| cfg_scale_text | 3.0 |
| cfg_scale_speaker | 5.0 |
| cfg_min_t | 0.5 |
| Batch size | 1 |
| Input | Synthetic (batch=1, text_len=4, ref_frames=8) |
| Warmup runs | 1 |
| Timed runs | 3 |

## Results: Full Inference (seq=750, steps=40)

| Backend | Mean (ms) | Min (ms) | p50 (ms) | p95 (ms) | vs Python |
|---|---|---|---|---|---|
| **Python PyTorch CUDA (bf16)** | **2,636** | 2,632 | 2,637 | 2,640 | 1.0× (baseline) |
| **Rust/burn LibTorch CUDA bf16** | **1,939** | 1,938 | 1,939 | 1,941 | **0.74×** ✓ |
| Rust/burn LibTorch CUDA f32 | 3,150 | 3,146 | 3,148 | 3,157 | 1.20× |
| Rust/burn 0.21 CUDA f32 + FA | 4,497 | 4,481 | 4,494 | — | 1.71× |
| Rust/burn 0.21 CUDA f32 (no autotune) | 4,561 | 4,533 | 4,561 | — | 1.73× |
| Rust/burn 0.20.1 CUDA f32 | 5,113 | 5,101 | 5,116 | 5,123 | 1.94× |
| Rust/burn CUDA bf16 (CubeCL) | 5,776 | — | — | — | 2.19× |
| Rust/burn WGPU | 7,396 | 7,354 | 7,394 | 7,439 | 2.81× |
| Rust/burn NdArray (CPU) | ~250,000+ | — | — | — | ~95× |

Notes:
- CPU (NdArray) was not fully benchmarked at seq=750/steps=40; extrapolated from smoke test (19.5s for seq=64/steps=4)
- WGPU produces a segfault on process exit (known WGPU cleanup issue); results are correct
- CUDA first run is ~250–500s (JIT kernel compilation); post-warmup results shown above
- **LibTorch bf16** is 26% **faster than Python** (1,939ms vs 2,636ms) — same cuBLAS+FA3 ops without Python overhead
- **LibTorch f32** is 1.20× Python — cuBLAS/FA3 gains, but float32 compute cost  
- **LibTorch backend** uses PyTorch's cuBLAS GEMM + SDPA (FA3) via `tch 0.22.0` / PyTorch 2.10
- LibTorch uses PyTorch 2.10 with `LIBTORCH_BYPASS_VERSION_CHECK=1` (tch targets 2.9, ABI-compatible)
- CubeCL bf16 slower than f32 — CubeCL GEMM not Tensor Core-tuned; LibTorch bf16 uses CUTLASS WMMA kernels

## Results: Smoke Test (seq=64, steps=4)

| Backend | Mean (ms) | Notes |
|---|---|---|
| Python PyTorch CUDA | — | Not run at this size |
| Rust/burn CUDA (post-warmup) | 613 | 3 runs |
| Rust/burn WGPU (post-warmup) | 768 | 3 runs |
| Rust/burn NdArray (CPU) | 19,540 | 1 run, no warmup |

## Numerical Accuracy

| Test | Backend | Max Abs Diff | Status |
|---|---|---|---|
| Single-step forward pass | NdArray f32 | < 1e-7 | ✓ PASS |
| 4-step E2E CFG sampling | NdArray f32 | 0.0 (exact) | ✓ PASS |
| 4-step E2E CFG sampling | LibTorch f32 | 0.0 (exact) | ✓ PASS |
| 4-step E2E CFG sampling | LibTorch bf16 | 5.84e-3 | ✓ PASS (tol=5e-2) |

Numerical accuracy was measured with small synthetic weights (validate binary), not the real 500M model checkpoint.

## GPU Kernel Profile (nsys, warm run, seq=750, steps=40)

Captured with `just bench-cuda-profile` (warmup=1, runs=1). The `into_contiguous_kernel`
calls represent attention transpose/reshape layout copies.

| Kernel | GPU Time (ms) | % Total | Instances | Avg (µs) | Notes |
|---|---|---|---|---|---|
| `matmul_entry` | 1,895 | 59.5% | 23,669 | 80 | CubeCL tiled GEMM |
| `reduce_kernel` | 617 | 19.4% | 6,937 | 89 | softmax, layer norm |
| `elemwise_fuse` | 518 | 16.3% | 29,919 | 17 | fused elementwise ops |
| `slice_assign_kernel` | 73 | 2.3% | 19,436 | 4 | attention mask writes |
| `into_contiguous_kernel` | 66 | 2.1% | 7,824 | 17 | layout copies (transpose) |
| Other | 16 | 0.5% | — | — | — |
| **Total GPU kernel time** | **3,185** | | | | |
| **Wall clock (warmup=1)** | **~5,100** | | | | ~1,915ms CPU/sync overhead |

Key observation: matmul accounts for 59.5% of GPU time. CubeCL's `matmul_entry` kernel
achieves 80µs avg vs cuBLAS which would be significantly faster on this GPU (RTX A6000).

### NVTX timing (CPU-side, warmup=0 run showing JIT cost)

| Range | Instances | Avg (ms) | Notes |
|---|---|---|---|
| `euler_step_0` | 1 | 11,837 | **JIT compilation + 1st step** |
| `euler_step_1..39` | 39 | ~245 | Steady-state per-step cost |
| `forward_cond` | 20 | 650 | Per-step cond forward (CFG) |
| `joint_attention` | 960 | 6.9 | 40 steps × 2 × 12 blocks |
| `swiglu_mlp` | 960 | 3.7 | |
| `adaln_mlp` | 960 | 3.2 | |

## Performance Gap Analysis

The Rust/burn implementations currently run at 1.20×–1.71× the Python/PyTorch baseline (2,636ms).
The **LibTorch backend (cuBLAS + SDPA)** closes the gap to 1.20× — just 514ms behind Python.

### Why LibTorch is faster than CubeCL

The LibTorch (`burn-tch`) backend delegates all computation to PyTorch's C++ runtime:
- **GEMM**: cuBLAS with Tensor Core acceleration (CUTLASS WMMA kernels)
- **Attention**: `torch.nn.functional.scaled_dot_product_attention` → Flash Attention 3 (FA3) on Ampere+
- **Kernel fusion**: PyTorch's eager-mode fusion for elementwise ops

CubeCL generates JIT kernels that are functionally correct but not Tensor Core-tuned for GEMM.

### Remaining gap (LibTorch at 1.20×)

The remaining 514ms gap (vs Python at 2,636ms) likely comes from:
1. **dtype**: Python runs bf16 (Tensor Core 2× throughput); LibTorch here runs f32
2. **Overhead**: Rust → tch FFI call overhead per op
3. **Kernel warmup**: PyTorch's CUDA graph caching is not used by burn-tch

### bf16 regresses vs f32 in CubeCL (+17% slower)
CubeCL's JIT-compiled bf16 CUDA kernels are **less optimized** than its f32 kernels.
Python's bf16 speedup comes from PyTorch dispatching to cuBLAS Tensor Core paths (CUTLASS
GEMM with WMMA), which CubeCL does not replicate at this level of tuning.

### Root causes of the CubeCL ~1.71× gap

1. **GEMM implementation**: PyTorch uses cuBLAS with Tensor Core acceleration for all
   matmuls (Q/K/V projections, attention, MLP). CubeCL generates its own tiled GEMM kernels
   via JIT — functionally correct but not Tensor Core-tuned.
2. **Flash Attention**: Python uses `F.scaled_dot_product_attention` which in PyTorch 2.x
   dispatches to Flash Attention (fused QKV + softmax + dropout in one kernel). Burn's
   CubeCL attention is a multi-step sequence of individual ops.
3. **Kernel fusion**: PyTorch's CUDA graphs + operator fusion reduce kernel launch overhead.
   CubeCL's fusion compiler is improving but not at parity.
4. **Memory bandwidth**: f32 weights are 2× larger than bf16; Python's bf16 weights fit in
   L2/HBM better, reducing bandwidth pressure in GEMM.

### Path forward

| Approach | Backend | Expected wall-clock | Status |
|---|---|---|---|
| **LibTorch (cuBLAS/FA3) bf16** | burn-tch | **1,939ms (0.74×)** | ✓ Done — **FASTER than Python** |
| LibTorch (cuBLAS/FA3) f32 | burn-tch | 3,150ms (1.20×) | ✓ Done |
| CubeCL f32 + Flash Attention | burn-cuda | 4,497ms (1.71×) | ✓ Done |
| CubeCL f32 (no FA) | burn-cuda | 4,561ms (1.73×) | ✓ Done |

## Planned Improvements

- [x] bf16 inference (CubeCL regresses — not Tensor Core tuned; LibTorch bf16 is fastest at 0.74×)
- [x] NVTX profiling (profile in `target/profile_warm.nsys-rep`)
- [x] Upgrade to burn 0.21 pre-release (12% faster CubeCL kernels + FA autotune)
- [x] Flash Attention via `burn::tensor::module::attention()` with autotune
- [x] **LibTorch f32 via `burn-tch`** — cuBLAS + FA3, 3,150ms (1.20× Python)
- [x] **LibTorch bf16 via `burn-tch`** — cuBLAS Tensor Core + FA3, **1,939ms (0.74× Python — 26% faster!)**
- [ ] Use bf16 model weights for the standalone CubeCL path (may require Tensor Core-tuned GEMM from cubecl)
- [ ] Profile burn-tch bf16 to understand remaining overhead vs theoretical TFLOPS peak

## Commands

```sh
# Python baseline
just bench-python

# Rust LibTorch CUDA f32 (fastest, 1.20× Python) — requires Irodori-TTS venv
just bench-tch

# Rust CUDA f32 + Flash Attention (1.71× Python, no external dep)
just bench-cuda

# Rust CUDA bf16 (slower due to CubeCL not Tensor Core tuned)
just bench-cuda-bf16

# Rust WGPU
just bench-wgpu

# Rust CPU (slow — only use with small seq/steps)
just bench-cpu-smoke

# Smoke tests (seq=64, steps=4)
just bench-cuda-smoke
just bench-cuda-bf16-smoke
just bench-wgpu-smoke
just bench-cpu-smoke

# E2E correctness (NdArray)
just e2e

# E2E correctness on LibTorch backend
just e2e-tch
```
