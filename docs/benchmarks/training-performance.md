# LoRA Training Performance Benchmark

**Date**: 2026-04-14 (updated)  
**Hardware**: NVIDIA RTX A6000 (48 GB VRAM), AMD EPYC host CPU  
**Model**: Irodori-TTS-500M-v2 with LoRA (r=8, alpha=16)

## Python vs Rust Throughput Comparison

**Workload**: 100-sample synthetic dataset, batch_size=4, 50 training steps, f32.

| Framework           | Steps/sec | ms/step | Breakdown (ms)                        |
|---------------------|-----------|---------|---------------------------------------|
| Python (PyTorch)    | ~18.9     | ~53     | —                                     |
| Rust (burn+LibTorch)| ~6.4      | ~155    | data=1.6, fwd=80, bwd=47, optim=27   |
| **Gap**             | **3.0×**  |         |                                       |

### Root Cause Analysis

The ~3× gap is a **framework characteristic** of burn's Autodiff layer, not an
application-level issue:

1. **Per-op AD dispatch (~40-70ms overhead)**: burn creates a separate computation
   graph on top of LibTorch. Each tensor op incurs ~0.1ms for graph node creation +
   parent tracking (Tracked ops) vs PyTorch's ~0.01ms C++ autograd. With ~700+ ops
   per forward pass, this accumulates to the majority of the gap.

2. **Non-fused optimizer (~22ms overhead)**: burn's AdamW iterates per-parameter
   (~168 LoRA params × ~10 tensor ops = ~1680 kernel launches). PyTorch uses a
   single fused CUDA kernel.

3. **Manual SDPA (minor)**: burn uses 6 individual kernels vs PyTorch's 1-2 fused
   Flash Attention kernels.

### Optimizations Applied

| Optimization                     | Savings  | Description                                     |
|----------------------------------|----------|-------------------------------------------------|
| NaN check removal (safe_softmax) | ~4ms     | Skip NaN→0 in softmax for training SDPA calls   |
| Forward/backward timing split    | 0 (diag) | Better profiling diagnostics                    |
| Inner-backend conditioning       | ~2ms     | Run frozen text encoder on non-AD backend        |
| **Total improvement**            | **~6ms** | 161ms → 155ms/step                              |

Further optimization is limited by burn's architecture. The remaining gap requires
upstream changes to burn (fused optimizers, reduced graph construction overhead).

## Multi-Backend Training Results

**Workload**: 40-sample dataset, 20 steps, LoRA r=16.

| Backend              | Batch | Sec/step  | Samples/sec | vs NdArray CPU |
|----------------------|-------|-----------|-------------|----------------|
| NdArray CPU f32      | 1     | ~11.0 s   | ~0.09       | 1× (baseline)  |
| LibTorch CUDA f32    | 2     | ~0.20 s   | ~9.9        | ~108×          |
| LibTorch CUDA bf16   | 2     | ~0.13 s   | ~16.0       | ~176×          |
| CubeCL CUDA f32      | 2     | ~2.0 s†   | ~1.0        | ~11×           |
| CubeCL CUDA bf16     | 2     | N/A       | N/A         | BROKEN         |

† CubeCL f32 JIT warmup: ~773 s total for 20 steps (first run). Post-warmup
  (autotune cache populated) steady-state is ~1–3 s/step.

## Key Findings

### LibTorch CUDA is the clear winner for training
- **bf16 beats f32 by ~1.6×** on A6000 tensor cores.
- **~176× faster than NdArray CPU** on a per-sample basis.
- Recommended backend for production LoRA fine-tuning.

### CubeCL CUDA f32: usable after JIT cache warmup
- First run incurs heavy JIT autotune (~10–15 min on A6000 for training shapes).
- Warm runs: ~1–3 s/step at batch=2, roughly 10× slower than LibTorch.
- CubeCL targets portability (no CUDA-only dependency); trade-off is performance.
- Autotune cache is persistent across runs so warmup cost is one-time.

### CubeCL CUDA bf16: upstream bug
- Panics in background device thread at `burn-ir-0.21.0-pre.3/src/handle.rs:88`
  and `burn-fusion-0.21.0-pre.3/src/stream/execution/ordering.rs:49`.
- Root cause: tensor handle teardown in the fusion streaming executor.
- **Workaround**: use LibTorch CUDA bf16 which is fully stable.

## Environment

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv
export PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:$PATH
export LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

PyTorch version: 2.x (from Irodori-TTS `.venv`)  
burn version: 0.21.0-pre.3 (git, all backends)

## Binaries

Build command:
```bash
cargo build --release --features <feature> --bin train_lora
```

## Recommended Usage

```bash
# Fastest: LibTorch CUDA bf16
./target/release/train_lora \
  --config train_config.toml

# CPU-only fallback (no GPU required, very slow)
cargo build --release --bin train_lora
./target/release/train_lora --config train_config.toml
```
