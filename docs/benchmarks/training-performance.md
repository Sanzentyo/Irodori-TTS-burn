# LoRA Training Performance Benchmark

**Date**: 2026-04-14 (updated with fair comparison)
**Hardware**: NVIDIA RTX A6000 (48 GB VRAM), AMD EPYC host CPU
**Model**: Irodori-TTS-500M-v2 with LoRA (r=8, alpha=16, targets=wq/wk/wv/wo/gate)

## Python vs Rust Throughput Comparison

**Workload**: 100-sample synthetic dataset, batch_size=4, 50 training steps, f32.

### Fair comparison (strict parity config)

Both sides: identical LoRA targets, no grad clipping, no condition dropout,
no stratified timesteps, same optimizer (AdamW, lr=1e-4, weight_decay=0.01).

| Framework            | Steps/sec | ms/step | Breakdown (ms)                        |
|----------------------|-----------|---------|---------------------------------------|
| Python (PyTorch)     | **5.55**  | 180.3   | — (end-to-end timing only)            |
| Rust (burn+LibTorch) | **6.06**  | 164.9   | data=1.8, fwd=81, bwd=45, optim=36   |
| **Rust advantage**   | **+9%**   |         |                                       |

### Previous (unfair) comparison

Earlier measurements showed Python at ~18 steps/sec, but the Python benchmark
had a bug: LoRA target modules were `q_proj/k_proj/v_proj/out_proj/gate_proj`
instead of the correct `wq/wk/wv/wo/gate`. This resulted in only 10K trainable
params (0.00%) instead of 2.13M (0.43%), making the Python benchmark artificially
fast. After fixing the target module names, Python throughput dropped to 5.55 steps/sec.

### Loss scale difference

Rust loss ~1.9 vs Python loss ~60 is expected and **not a bug**: Rust uses
`mean` over feature dimension D, Python uses `sum`. With D=32: 60/1.9 ≈ 31.6 ≈ D.
Gradients scale accordingly; training behavior is equivalent with appropriate LR.

### Optimizations Applied

| Optimization                     | Savings  | Description                                     |
|----------------------------------|----------|-------------------------------------------------|
| NaN check removal (safe_softmax) | ~4ms     | Skip NaN→0 in softmax for training SDPA calls   |
| Inner-backend conditioning       | ~2ms     | Run frozen text encoder on non-AD backend        |
| Forward/backward timing split    | 0 (diag) | Better profiling diagnostics                    |

### Benchmark config

Rust: `target/bench_data/bench_train.toml` — strict parity config with all
Rust-only features disabled (grad_clip_norm=0.0→None, dropout=0.0, stratified=false).

Python: `scripts/bench_train_py.py` with `--steps 50 --batch-size 4 --device cuda`.
Run from Irodori-TTS venv with `PYTHONPATH=.`.

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

### Rust burn+LibTorch is 9% faster than PyTorch for LoRA training
- Same backend (LibTorch/cuBLAS) underneath — burn adds minimal overhead.
- Optimizer step (36ms) accounts for 22% of per-step time (497M params).
- Forward + backward (126ms) is the dominant cost (78% of step time).

### LibTorch CUDA bf16 is fastest for production
- **bf16 beats f32 by ~1.6×** on A6000 tensor cores.
- **~176× faster than NdArray CPU** on a per-sample basis.
- Recommended backend for production LoRA fine-tuning.

### CubeCL CUDA f32: usable after JIT cache warmup
- First run incurs heavy JIT autotune (~10–15 min on A6000 for training shapes).
- Warm runs: ~1–3 s/step at batch=2, roughly 10× slower than LibTorch.
- Autotune cache is persistent across runs so warmup cost is one-time.

### CubeCL CUDA bf16: upstream bug
- Panics in background device thread at `burn-ir-0.21.0-pre.3/src/handle.rs:88`.
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

## Reproduce

```bash
# Generate synthetic data (100 samples)
uv run scripts/gen_synthetic_train_data.py --output-dir target/bench_data --num-samples 100 --apply

# Rust benchmark
cargo run --release --features backend_tch --bin train_lora -- --config target/bench_data/bench_train.toml

# Python benchmark
cd ../Irodori-TTS && PYTHONPATH=. python ../Irodori-TTS-burn/scripts/bench_train_py.py \
    --manifest ../Irodori-TTS-burn/target/bench_data/train_py.jsonl \
    --model ~/.cache/huggingface/.../model.safetensors \
    --steps 50 --batch-size 4 --device cuda --warmup-steps 3
```
