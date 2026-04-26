# Irodori-TTS-burn

A full-scratch Rust reimplementation of [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
using the [Burn](https://burn.dev) machine learning framework.

Irodori-TTS is a Japanese text-to-speech model built around a Diffusion Transformer (DiT)
with RoPE positional encoding, Joint attention with KV caching, and a DACVAE neural codec.
This project ports the entire inference **and training** pipeline from PyTorch to Burn,
preserving numerical parity with the Python reference.

## Features

- **Full-model port** — DiT blocks, RoPE, Joint attention, DACVAE codec (15.3K LOC, 46 source files)
- **Multi-backend** — LibTorch f32/bf16, CUDA f32 (CubeCL), WGPU f32; enum-based runtime dispatch
- **KV cache** — cached forward pass for the RF sampler loop
- **LoRA** — adapter merging for inference + LoRA fine-tuning with flow-matching loss
- **Training** — dataset loading, LR schedules, gradient clipping, atomic checkpointing, resume
- **Numerical parity** — layer-by-layer validation against Python fixtures (all PASS)
- **E2E pipeline** — `pipeline` binary: text → WAV
- **Cargo feature flags** — `inference`, `codec`, `text-normalization`, `lora`, `train`, `cli`
- **306 unit tests**, clippy clean

## Performance (RTX A6000)

| Backend | RF latency (40 steps) | vs Python f32 |
|---------|----------------------|---------------|
| Python f32 (reference) | 2,761 ms | 1.00× |
| Python bf16 | N/A | ❌ cuBLAS crash |
| **Rust LibTorch bf16** | **987 ms** | **0.37×** ✓ |
| Rust LibTorch f32 | 2,817 ms | 1.02× |
| Rust CUDA f32 (CubeCL) | 4,623 ms | 1.67× |
| Rust WGPU f32 | 7,315 ms | 2.65× |

> Rust LibTorch bf16 is **63% faster** than Python — and Python can't even run bf16.
> Rust LibTorch f32 is within **2.4%** of Python — near-parity (FFI abstraction overhead).

## Performance (M4 Pro Mac Mini)

| Backend | CFG Mode | RF latency (40 steps) | RTF |
|---------|----------|----------------------|-----|
| **LibTorch MPS f16** | **Joint (scale=3.0)** | **7,495 ms** | **0.250** ✓ |
| LibTorch MPS f16 | Independent (default) | 10,204 ms | 0.340 |
| LibTorch MPS f32 | Independent (default) | 11,660 ms | 0.389 |
| **WgpuRaw f16 (no-dep)** | **Joint (scale=3.0)** | **14,287 ms** | **0.476** ✓ |
| WgpuRaw f16 (no-dep) | Independent (default) | 18,887 ms | 0.630 |

RTF < 1 = **faster than real-time**. Joint CFG requires equal guidance scales (quality tradeoff).
MPS benchmarks use LibTorch from `Aratako/Irodori-TTS` PyTorch venv.

Training throughput (LoRA fine-tuning, RTX A6000):

| Implementation | Steps/sec |
|---------------|-----------|
| Python (reference) | ~1.15 |
| Rust (LibTorch) | ~1.25 (+9%) |

## Cargo features

| Feature | Default | Description |
|---------|---------|-------------|
| `inference` | ✓ | `InferenceBuilder` / `InferenceEngine` type-state API |
| `codec` | ✓ | DACVAE encoder/decoder |
| `text-normalization` | ✓ | Japanese text normaliser |
| `lora` | | LoRA adapter loading/merging (additive to inference) |
| `train` | | LoRA fine-tuning infrastructure |
| `cli` | | CLI binaries (clap, anyhow, hf-hub, hound) |

Backend selection is handled at runtime via `--backend` CLI flag. See
`InferenceBackendKind` and `TrainingBackendKind` enums in `src/backend_config.rs`.

## Quick start

See **[docs/setup.md](docs/setup.md)** for the full setup guide including:
- Setting up the Python reference environment
- Downloading and converting model weights
- Configuring the LibTorch backend
- Running validation and benchmarks

```bash
# Inference (default features + cli)
cargo run --release --features cli --bin pipeline -- \
    --checkpoint target/model_converted.safetensors \
    --codec-weights target/dacvae_weights.safetensors \
    --text "こんにちは、世界！" \
    --output /tmp/out.wav

# With LoRA adapter
cargo run --release --features "lora,cli" --bin pipeline -- \
    --checkpoint target/model_converted.safetensors \
    --codec-weights target/dacvae_weights.safetensors \
    --adapter path/to/adapter_dir \
    --text "こんにちは、世界！" \
    --output /tmp/out.wav

# LoRA training
cargo run --release --features "train,cli" --bin train_lora -- \
    --backend libtorch \
    --checkpoint target/model_converted.safetensors \
    --manifest path/to/manifest.jsonl
```

## Repository structure

```
src/
├── lib.rs                  — public API (feature-gated modules)
├── model/                  — DiT, attention, RoPE, positional encoding
├── codec/                  — DACVAE encoder/decoder
├── rf.rs                   — RF (rectified-flow) sampler + KV cache
├── inference.rs            — InferenceBuilder type-state API
├── weights.rs              — weight loading (safetensors)
├── lora.rs                 — LoRA adapter merging
├── train/                  — LoRA training (dataset, trainer, loss, LR schedule)
├── text_normalization.rs   — Japanese text normaliser
├── config.rs               — model/training/sampling configuration
├── error.rs                — thiserror-based error types
└── bin/
    ├── pipeline.rs         — E2E inference binary
    ├── train_lora.rs       — LoRA fine-tuning binary
    ├── infer.rs            — inference-only binary
    ├── bench_realmodel.rs  — multi-backend latency benchmark
    └── ...
docs/
├── setup.md                — installation guide (start here)
├── feature-coverage.md     — implementation status + test tracking
├── benchmarks.md           — benchmark results
└── planning/               — architecture notes
```

## Development

```bash
just ci           # fmt-check + clippy -D warnings + test
just validate     # numerical parity (requires ../Irodori-TTS venv)
just bench-tch    # LibTorch f32 benchmark
just quality-compare   # multi-backend WAV comparison
```

## License

MIT — see [LICENSE](LICENSE).
