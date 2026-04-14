# Setup Guide

This document describes how to set up Irodori-TTS-burn for development and inference,
including the Python reference environment required for the LibTorch backend and
numerical validation.

## Directory layout

The project assumes the Python reference repo lives as a **sibling directory**:

```
~/
├── Irodori-TTS/                   ← original Python repository
│   ├── .venv/                     ← uv-managed Python 3.10 env
│   └── infer.py
└── Irodori-TTS-burn/              ← this Rust project
    ├── target/
    │   ├── model_converted.safetensors   ← converted TTS weights
    │   └── dacvae_weights.safetensors    ← converted codec weights
    └── ...
```

> **Important**: Python 3.10 is required.  The LibTorch backend (`burn-tch`) loads
> PyTorch shared libraries directly from the uv virtualenv.  The library path is
> currently hardcoded to `…/python3.10/site-packages/torch/lib`.

---

## Step 1 — Clone and set up the Python reference repo

```bash
git clone https://github.com/Aratako/Irodori-TTS ../Irodori-TTS
cd ../Irodori-TTS
uv sync          # installs Python 3.10 + all dependencies
cd -
```

Verify the Python environment works:

```bash
just py-sync     # same as above, runs from this repo root
```

---

## Step 2 — Clone this repo (if you haven't already)

```bash
git clone https://github.com/sanzentyo/Irodori-TTS-burn
cd Irodori-TTS-burn
```

---

## Step 3 — Download and convert model weights

### 3a. TTS model (Irodori-TTS-500M-v2)

```bash
just download-model    # downloads to target/hf_model/
just convert-model     # → target/model_converted.safetensors
```

Both commands use `uv run` internally so no Python activation is needed.

### 3b. DACVAE codec weights

The codec weights are downloaded separately from Hugging Face and must be converted
from PyTorch `.pth` format to clean safetensors:

```bash
just codec-convert
```

This runs `scripts/convert_dacvae_weights.py` inside the Irodori-TTS venv (which has
PyTorch available) and outputs `target/dacvae_weights.safetensors`.

---

## Step 4 — Build environment for the LibTorch backend

The default NdArray backend requires nothing extra.  For the LibTorch (CUDA) backend:

### Environment variables

Add these to your shell profile or `.env` file in the project root.
Adjust paths to match your Python environment:

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export VIRTUAL_ENV=../Irodori-TTS/.venv
export LD_LIBRARY_PATH=../Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}
```

> `justfile` uses a `.env` file loaded automatically via `set dotenv-load`.
> You can create `.env` in the project root with the above variables (without `export`).
>
> Example `.env`:
> ```
> LIBTORCH_USE_PYTORCH=1
> LIBTORCH_BYPASS_VERSION_CHECK=1
> VIRTUAL_ENV=../Irodori-TTS/.venv
> LD_LIBRARY_PATH=../Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib
> ```

### Build (LibTorch release)

```bash
cargo build --release    # default: NdArray + LibTorch features
```

---

## Step 5 — Verify the installation

Run numerical validation to confirm the Rust model matches Python outputs:

```bash
just validate
```

This:
1. Generates Python reference fixtures with the toy 2-layer model
2. Runs the Rust `validate` binary against those fixtures
3. Prints PASS/FAIL for each check (conditions, DiT blocks, v_pred, KV cache)

All checks should print **PASS**.

---

## Quick-start inference

```bash
# Single prompt, LibTorch bf16 (fastest)
just py-infer --text "こんにちは、世界！" --no-ref --output-wav /tmp/out.wav

# Rust pipeline, LibTorch f32
cargo run --release --bin pipeline -- \
    --checkpoint target/model_converted.safetensors \
    --codec-weights target/dacvae_weights.safetensors \
    --text "こんにちは、世界！" \
    --output /tmp/out_rust.wav
```

---

## Multi-backend quality comparison

```bash
just quality-compare                        # all backends (rebuilds first)
just quality-compare-fast                   # skip rebuild
just quality-compare-backend rust_libtorch  # single backend
```

Results are written to `target/quality_comparison/performance_report.md` and zipped
to `target/quality_comparison.zip`.

---

## Benchmarks

```bash
just bench-tch           # LibTorch f32 — 5 runs, seq_len=750
just bench-tch-bf16      # LibTorch bf16
just bench-all           # all three Rust backends
just bench-python        # Python reference
```

Benchmark output includes RTF, xRT, and tokens/sec throughput metrics.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `libtorch_cpu.so: cannot open shared object file` | Set `LD_LIBRARY_PATH` as shown in Step 4 |
| `error: could not find Python 3.10` | Irodori-TTS venv requires Python 3.10; use `uv python pin 3.10` in `../Irodori-TTS` |
| `validate` FAIL on any check | Re-run `just validate-fixtures` then `just validate-rust` to regenerate fixtures |
| CUDA OOM | Reduce `--seq-len` in the bench command |
| burn-ir panic on CUDA bf16 teardown | Known `burn-tch 0.21.0-pre.3` regression; use f32 or WGPU |

---

## Development workflow

```bash
just ci          # fmt-check + clippy + test (run before every commit)
just test        # unit tests only
just lint        # clippy --all-targets -- -D warnings
just fmt         # cargo fmt --all
```
