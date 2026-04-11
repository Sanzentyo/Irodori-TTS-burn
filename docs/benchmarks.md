# Benchmark Results: Rust/burn vs Python/PyTorch

## System

| Component | Details |
|---|---|
| GPU | NVIDIA RTX A6000 (49140 MiB VRAM) |
| CUDA Driver | 575.57.08 (CUDA 12.9) |
| CPU | Linux host (burn NdArray) |
| burn version | 0.20.1 |
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
| Rust/burn CUDA f32 | 4,957 | — | — | — | 1.88× slower |
| Rust/burn CUDA bf16 | 5,776 | — | — | — | 2.19× slower |
| Rust/burn WGPU | 7,396 | 7,354 | 7,394 | 7,439 | 2.81× slower |
| Rust/burn NdArray (CPU) | ~250,000+ | — | — | — | ~95× slower |

Notes:
- CPU (NdArray) was not fully benchmarked at seq=750/steps=40; extrapolated from smoke test (19.5s for seq=64/steps=4)
- WGPU produces a segfault on process exit (known WGPU cleanup issue); results are correct
- CUDA first run is ~250–500s (JIT kernel compilation); post-warmup results shown above
- **bf16 is 17% slower than f32** on this backend — see analysis below

## Results: Smoke Test (seq=64, steps=4)

| Backend | Mean (ms) | Notes |
|---|---|---|
| Python PyTorch CUDA | — | Not run at this size |
| Rust/burn CUDA (post-warmup) | 613 | 3 runs |
| Rust/burn WGPU (post-warmup) | 768 | 3 runs |
| Rust/burn NdArray (CPU) | 19,540 | 1 run, no warmup |

## Numerical Accuracy

| Test | Max Abs Diff | Status |
|---|---|---|
| Single-step forward pass | < 1e-7 | ✓ PASS |
| 4-step E2E CFG sampling | 0.0 (exact) | ✓ PASS |

Numerical accuracy was measured with small synthetic weights (validate binary), not the real 500M model checkpoint.

## Performance Gap Analysis

The Rust/burn CUDA implementation runs at ~1.88–2.19× the Python/PyTorch time. Key findings:

### bf16 regresses vs f32 in burn (+17% slower)
CubeCL's JIT-compiled bf16 CUDA kernels are **less optimized** than its f32 kernels.
Python's bf16 speedup comes from PyTorch dispatching to cuBLAS Tensor Core paths (CUTLASS
GEMM with WMMA), which CubeCL does not replicate at this level of tuning.

### Root causes of the ~2× gap

1. **GEMM implementation**: PyTorch uses cuBLAS with Tensor Core acceleration for all
   matmuls (Q/K/V projections, attention, MLP). CubeCL generates its own tiled GEMM kernels
   via JIT — functionally correct but not Tensor Core-tuned.
2. **Flash Attention**: Python uses `F.scaled_dot_product_attention` which in PyTorch 2.x
   dispatches to Flash Attention (fused QKV + softmax + dropout in one kernel). Burn's
   attention is a multi-step sequence of individual ops.
3. **Kernel fusion**: PyTorch's CUDA graphs + operator fusion reduce kernel launch overhead.
   CubeCL's fusion compiler is improving but not at parity.
4. **Memory bandwidth**: f32 weights are 2× larger than bf16; Python's bf16 weights fit in
   L2/HBM better, reducing bandwidth pressure in GEMM.

### Path forward to close the gap

In priority order (highest impact first):

| Approach | Expected gain | Complexity |
|---|---|---|
| Custom Flash Attention (CUDA kernel via burn-jit) | 1.5–2× | High |
| cuBLAS GEMM integration (if burn exposes it) | 1.3–1.5× | Medium |
| Weight/activation quantization (INT8) | variable | High |
| Operator graph-level fusion (custom) | 1.1–1.2× | Medium |

## Planned Improvements

- [x] bf16 inference (implemented; regresses vs f32 — CubeCL not Tensor Core tuned)
- [ ] Profile with `nsys` / `nvtx` to quantify time per kernel type (GEMM vs attention vs other)
- [ ] Investigate Flash Attention integration for attention layers
- [ ] Investigate cuBLAS dispatch path in burn or custom CUDA kernel for GEMM

## Commands

```sh
# Python baseline
just bench-python

# Rust CUDA f32 (seq=750, steps=40, warmup=1, runs=3)
just bench-cuda

# Rust CUDA bf16 (same settings — slower due to CubeCL kernel quality)
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
```
