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
| **Python PyTorch CUDA** | **2,636** | 2,632 | 2,637 | 2,640 | 1.0× (baseline) |
| Rust/burn CUDA | 5,143 | 5,133 | 5,142 | 5,154 | 1.95× slower |
| Rust/burn WGPU | 7,396 | 7,354 | 7,394 | 7,439 | 2.81× slower |
| Rust/burn NdArray (CPU) | ~250,000+ | — | — | — | ~95× slower |

Notes:
- CPU (NdArray) was not fully benchmarked at seq=750/steps=40; extrapolated from smoke test (19.5s for seq=64/steps=4)
- WGPU produces a segfault on process exit (known WGPU cleanup issue); results are correct
- CUDA first run is ~250s (JIT kernel compilation); post-warmup is stable at ~5.1s

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

The Rust/burn CUDA implementation runs at ~1.95× the Python/PyTorch time. Key factors:

1. **Precision**: Python runs bf16 (from checkpoint); Rust runs f32. bf16 GEMM is typically 2× faster on Tensor Cores.
2. **Attention kernels**: Python may use Flash Attention via torch.nn.functional.scaled_dot_product_attention; burn uses its own CUDA kernel.
3. **GEMM dispatch**: PyTorch uses cuBLAS with Tensor Core paths; burn 0.20.1 CUDA backend uses cublas but may not route all matmuls optimally.
4. **Memory layout**: PyTorch applies contiguous layout + fusion; burn kernel dispatch is per-op.

## Planned Improvements

- [ ] bf16 inference in Rust (burn supports `Tensor<B, N, burn::Float>` with bf16 element)
- [ ] Profile with `nsys` / `nvtx` to find hotspots
- [ ] Investigate Flash Attention integration for attention layers
- [ ] Consider kernel fusion for AdaLN modulation (heavy in DiT)

## Commands

```sh
# Python baseline
just bench-python

# Rust CUDA (seq=750, steps=40, warmup=1, runs=3)
just bench-cuda

# Rust WGPU
just bench-wgpu

# Rust CPU (slow — only use with small seq/steps)
just bench-cpu-smoke

# Smoke tests (seq=64, steps=4)
just bench-cuda-smoke
just bench-wgpu-smoke
just bench-cpu-smoke
```
