# LoRA Training Performance Benchmark

**Date**: 2026-04-13  
**Hardware**: NVIDIA RTX A6000 (48 GB VRAM), AMD EPYC host CPU  
**Workload**: Synthetic 40-sample dataset, latent shape `[1, T, 32]` (T=200–750), 20 training steps  
**Model**: Irodori-TTS-500M-v2 with LoRA (r=16, alpha=32)  
**Command common args**: `--lora-r 16 --lora-alpha 32.0 --warmup-steps 3 --max-steps 20 --log-every 1`

## Results

| Backend              | Batch | Sec/step  | Samples/sec | vs NdArray CPU |
|----------------------|-------|-----------|-------------|----------------|
| NdArray CPU f32      | 1     | ~11.0 s   | ~0.09       | 1× (baseline)  |
| LibTorch CUDA f32    | 2     | ~0.20 s   | ~9.9        | ~108×          |
| LibTorch CUDA bf16   | 2     | ~0.13 s   | ~16.0       | ~176×          |
| CubeCL CUDA f32      | 2     | ~2.0 s†   | ~1.0        | ~11×           |
| CubeCL CUDA bf16     | 2     | N/A       | N/A         | BROKEN         |

† CubeCL f32 JIT warmup: ~773 s total for 20 steps (first run). Post-warmup
  (autotune cache populated) steady-state is ~1–3 s/step.

## Wall-clock Totals

| Backend              | 20-step total | Notes                                      |
|----------------------|---------------|--------------------------------------------|
| NdArray CPU f32      | 3 m 40 s      | Pure Rust, no GPU                          |
| LibTorch CUDA f32    | ~9 s          | Uses cuBLAS via PyTorch C++ API            |
| LibTorch CUDA bf16   | ~8 s          | bf16 tensor cores                          |
| CubeCL CUDA f32      | 12 m 54 s     | Autotune heavy on first run, ~10 s warm    |
| CubeCL CUDA bf16     | 7 m 46 s      | Hung after step 1 (burn-ir panic, see §Bug)|

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
- Same panic observed in inference binary (separate session).
- Root cause: tensor handle teardown in the fusion streaming executor; occurs during
  bf16 training backward pass.
- Tracked upstream: burn-ir fusion stream ordering issue with bf16 on CubeCL.
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

Pre-built release binaries (each compiled with a single backend feature):

| Binary                       | Feature flag           |
|------------------------------|------------------------|
| `train_lora_ndarray`         | (default, NdArray)     |
| `train_lora_tch_f32`         | `backend_tch`          |
| `train_lora_tch_bf16`        | `backend_tch_bf16`     |
| `train_lora_cuda_f32`        | `backend_cuda`         |
| `train_lora_cuda_bf16`       | `backend_cuda_bf16`    |

Build command:
```bash
cargo build --release --features <feature> --bin train_lora
cp target/release/train_lora target/release/train_lora_<label>
```

## Recommended Usage

```bash
# Fastest: LibTorch CUDA bf16
./train_lora_tch_bf16 \
  --model model_converted.safetensors \
  --manifest manifest.jsonl \
  --tokenizer tokenizer.json \
  --lora-r 16 --lora-alpha 32.0 \
  --batch-size 4 \
  --gpu-id 0 \
  --output-dir ./lora_out

# CPU-only fallback (no GPU required)
./train_lora_ndarray \
  --model model_converted.safetensors \
  --manifest manifest.jsonl \
  --tokenizer tokenizer.json \
  --lora-r 16 --lora-alpha 32.0 \
  --batch-size 1 \
  --output-dir ./lora_out
```
