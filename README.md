# Irodori-TTS-burn

A full-scratch Rust reimplementation of [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
using the [Burn](https://burn.dev) machine learning framework.

Irodori-TTS is a Japanese text-to-speech model built around a Diffusion Transformer (DiT)
with RoPE positional encoding, Joint attention with KV caching, and a DACVAE neural codec.
This project ports the entire inference pipeline from PyTorch to Burn, preserving numerical
parity with the Python reference.

## Features

- **Full-model port** — DiT blocks, RoPE, Joint attention (with LoRA support), DACVAE codec
- **Multi-backend** — NdArray (CPU), LibTorch f32/bf16, CUDA f32 (CubeCL), WGPU
- **KV cache** — cached forward pass for the RF sampler loop
- **LoRA inference** — load PEFT adapters and merge into base weights
- **Numerical parity** — layer-by-layer validation against Python fixtures (all PASS)
- **E2E pipeline** — `pipeline` binary: text → WAV
- **Benchmarks** — RTF, xRT, tokens/sec across all backends

## Performance (LibTorch, single A100)

| Backend | RF latency | vs Python | RTF |
|---------|-----------|-----------|-----|
| Python fp32 (reference) | ~3200 ms | 1.0× | ~0.11 |
| Rust LibTorch f32 | ~4000 ms | +25% | ~0.13 |
| Rust LibTorch bf16 | ~2230 ms | **−30%** | ~0.074 |

RF latency measured for 40-step, seq_len=750 (~30 s audio).

## Quick start

See **[docs/setup.md](docs/setup.md)** for the full setup guide including:
- Cloning the Python reference repo
- Downloading and converting model weights
- Setting up the LibTorch environment
- Running validation and benchmarks

## Repository structure

```
src/
├── lib.rs                  — public API
├── model/                  — DiT, attention, RoPE, positional encoding
├── codec/                  — DACVAE encoder/decoder
├── rf.rs                   — RF (rectified-flow) sampler + KV cache
├── weights.rs              — weight loading (safetensors)
├── lora.rs                 — LoRA adapter merging
├── text_normalization.rs   — Japanese text normaliser
└── bin/
    ├── pipeline.rs         — E2E inference binary
    ├── validate.rs         — numerical parity checker
    ├── bench_realmodel.rs  — multi-backend latency benchmark
    └── ...
scripts/
├── run_quality_comparison.py  — multi-backend quality comparison
├── validate_numerics.py       — Python fixture generator
└── ...
docs/
├── setup.md                   — installation guide (start here)
├── feature-coverage.md        — implementation status + fixes
├── benchmarks.md              — benchmark results
└── planning/                  — architecture notes
```

## Development

```bash
just ci           # fmt-check + clippy -D warnings + test
just validate     # numerical parity (requires ../Irodori-TTS venv)
just bench-tch    # LibTorch f32 benchmark
just quality-compare   # multi-backend WAV comparison
```

## License

Research / personal use only — see upstream [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) license.
