set shell := ["bash", "-cu"]
set dotenv-load

# Default: list all recipes
default:
    @just --list

# ── Rust ───────────────────────────────────────────────────────────────────────

# Build debug
build:
    cargo build

# Build release
build-release:
    cargo build --release

# Run all tests
test:
    cargo test --all-targets

# Run tests with output
test-verbose:
    cargo test --all-targets -- --nocapture

# Lint with clippy
lint:
    cargo clippy --all-targets -- -D warnings

# Format code
fmt:
    cargo fmt --all

# Format check (CI)
fmt-check:
    cargo fmt --all -- --check

# Full CI check
ci: fmt-check lint test

# ── Python (reference Irodori-TTS) ────────────────────────────────────────────

# Sync Python environment for reference Irodori-TTS
py-sync:
    cd ../Irodori-TTS && uv sync

# Run inference with reference Python implementation
py-infer *args:
    cd ../Irodori-TTS && uv run python infer.py {{args}}

# Run Python linter
py-lint:
    cd ../Irodori-TTS && uv run ruff check .

# ── Validation ────────────────────────────────────────────────────────────────

# Generate Python reference fixtures then check Rust outputs match
# Checks: encode_conditions, per-DiT-block outputs, v_pred, KV-cache consistency
validate:
    uv run scripts/validate_numerics.py
    cargo run --bin validate

# Only regenerate Python fixtures (no Rust run)
validate-fixtures:
    uv run scripts/validate_numerics.py

# Only run Rust comparison (assumes fixtures already exist)
validate-rust:
    cargo run --bin validate

# ── E2E Comparison ───────────────────────────────────────────────────────────

# Generate Python E2E fixtures (requires validate-fixtures to have run first)
e2e-fixtures:
    uv run scripts/e2e_compare.py

# Run Rust E2E comparison (assumes e2e-fixtures already run)
e2e-rust:
    cargo run --bin e2e_compare

# Full E2E: generate Python fixtures then run Rust comparison
e2e: validate-fixtures e2e-fixtures e2e-rust

# Run Rust E2E comparison on the LibTorch (CUDA) backend
e2e-tch-rust:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --features backend_tch --bin e2e_compare

# Full E2E with LibTorch backend
e2e-tch: validate-fixtures e2e-fixtures e2e-tch-rust

# Run Rust E2E comparison on the LibTorch bf16 backend
e2e-tch-bf16-rust:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --features backend_tch_bf16 --bin e2e_compare

# Full E2E with LibTorch bf16 backend
e2e-tch-bf16: validate-fixtures e2e-fixtures e2e-tch-bf16-rust



# Convert Python safetensors checkpoint to Burn-compatible key names
# Usage: just convert <src.safetensors> <dst.safetensors> [--apply]
convert input output *args:
    uv run scripts/convert_for_burn.py {{input}} {{output}} {{args}}

# Run Rust inference CLI
infer *args:
    cargo run --release --bin infer -- {{args}}

# Run the full TTS pipeline (text → WAV) using RF model + DACVAE codec
pipeline *args:
    cargo run --release --bin pipeline -- {{args}}

# Full pipeline against the real converted model + DACVAE codec
pipeline-real *args:
    cargo run --release --bin pipeline -- \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with LibTorch backend (CPU, same kernels as Python)
pipeline-real-tch *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/local/bin:/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
    cargo run --release --features backend_tch --bin pipeline -- \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with LibTorch bf16 backend (fastest CPU option)
pipeline-real-tch-bf16 *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/local/bin:/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
    cargo run --release --features backend_tch_bf16 --bin pipeline -- \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# ── LoRA Training ────────────────────────────────────────────────────────────

# Run LoRA fine-tuning (NdArray CPU by default; add --features backend_tch for LibTorch)
train-lora *args:
    cargo run --release --bin train_lora -- {{args}}

# Encode a dataset of WAV files into latent safetensors + JSONL manifest
encode-dataset *args:
    uv run scripts/encode_dataset.py {{args}}

# ── Docs ──────────────────────────────────────────────────────────────────────

# Show current progress
progress:
    @cat docs/planning/progress.md 2>/dev/null || echo "No progress doc yet."

# ── Real model ───────────────────────────────────────────────────────────────

# Download the Aratako/Irodori-TTS-500M-v2 checkpoint from HuggingFace
download-model:
    uv run scripts/download_model.py

# Convert the downloaded HF model to Burn-compatible safetensors
convert-model:
    uv run scripts/convert_for_burn.py \
        target/hf_model/model.safetensors \
        target/model_converted.safetensors \
        --apply

# Run inference against the real converted model
infer-real *args:
    cargo run --release --bin infer -- \
        --checkpoint target/model_converted.safetensors {{args}}

# Inference with CUDA bf16 backend
infer-real-bf16 *args:
    cargo run --release --features backend_cuda_bf16 --bin infer -- \
        --checkpoint target/model_converted.safetensors {{args}}

# ── Benchmarks ───────────────────────────────────────────────────────────────

# Run criterion benchmarks (requires validate fixtures: just validate-fixtures)
bench *args:
    cargo bench --bench inference -- {{args}}

# Run criterion codec benchmarks (requires: just codec-convert)
bench-codec *args:
    cargo bench --bench codec -- {{args}}

# Fast wall-clock codec timing benchmark — NdArray backend (WARNING: very slow)
bench-codec-ndarray *args:
    cargo run --release --bin bench_codec -- --weights {{DACVAE_WEIGHTS}} {{args}}

# Fast wall-clock codec timing benchmark — LibTorch CPU (comparable to Python/PyTorch)
bench-codec-tch *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/home/sanzentyo/.cargo/bin:/usr/local/bin:/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features backend_tch --bin bench_codec -- --weights {{DACVAE_WEIGHTS}} {{args}}

# Benchmark Python DACVAE codec for comparison
bench-codec-py *args:
    cd /home/sanzentyo/Irodori-TTS && uv run --extra dev python \
        /home/sanzentyo/Irodori-TTS-burn/scripts/bench_codec_py.py \
        --model-path {{DACVAE_WEIGHTS}} {{args}}

# Run both Rust (LibTorch) and Python codec benchmarks for comparison
bench-codec-compare: (bench-codec-tch) (bench-codec-py)

# Open last HTML benchmark report
bench-report:
    xdg-open target/criterion/report/index.html 2>/dev/null || open target/criterion/report/index.html

# Smoke test real-model inference on NdArray CPU (small seq_len for quick check)
bench-cpu-smoke:
    cargo run --release --features backend_cpu --bin bench_realmodel -- \
        --seq-len 64 --num-steps 4 --warmup 0 --runs 1

# Smoke test real-model inference on WGPU
bench-wgpu-smoke:
    cargo run --release --features backend_wgpu --bin bench_realmodel -- \
        --seq-len 64 --num-steps 4 --warmup 0 --runs 1

# Smoke test real-model inference on CUDA (f32)
bench-cuda-smoke:
    cargo run --release --features backend_cuda --bin bench_realmodel -- \
        --seq-len 64 --num-steps 4 --warmup 0 --runs 1

# Smoke test real-model inference on CUDA (bf16)
bench-cuda-bf16-smoke:
    cargo run --release --features backend_cuda_bf16 --bin bench_realmodel -- \
        --seq-len 64 --num-steps 4 --warmup 0 --runs 1

# Full benchmark — NdArray CPU (seq=750, steps=40)
bench-cpu *args:
    cargo run --release --features backend_cpu --bin bench_realmodel -- {{args}}

# Full benchmark — WGPU f32 (seq=750, steps=40)
bench-wgpu *args:
    cargo run --release --features backend_wgpu --bin bench_realmodel -- {{args}}

# Full benchmark — WGPU f16 (requires shader-f16 GPU support)
bench-wgpu-f16 *args:
    cargo run --release --features backend_wgpu_f16 --bin bench_realmodel -- {{args}}

# Full benchmark — WGPU bf16 (⚠ NOT supported: WGSL has no native bf16; panics at runtime)
bench-wgpu-bf16 *args:
    cargo run --release --features backend_wgpu_bf16 --bin bench_realmodel -- {{args}}

# Full benchmark — Burn CUDA f32 (seq=750, steps=40)
bench-cuda *args:
    cargo run --release --features backend_cuda --bin bench_realmodel -- {{args}}

# Full benchmark — Burn CUDA bf16 (seq=750, steps=40; faster on Tensor Core GPUs)
bench-cuda-bf16 *args:
    cargo run --release --features backend_cuda_bf16 --bin bench_realmodel -- {{args}}

# Run all CUDA benchmarks (f32 and bf16)
bench-cuda-all:
    just bench-cuda
    just bench-cuda-bf16

# NVTX profiling run — generates nsys report at target/profile.nsys-rep
bench-cuda-profile *args:
    nsys profile --output target/profile.nsys-rep --force-overwrite true \
        cargo run --release --features "backend_cuda,profile" --bin bench_realmodel -- \
        --warmup 0 --runs 1 {{args}}

# Full benchmark — LibTorch f32 (cuBLAS / FA3 via PyTorch)
# Requires the Irodori-TTS venv to have PyTorch installed.
bench-tch *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features backend_tch --bin bench_realmodel -- {{args}}

# Full benchmark — LibTorch bf16 (Tensor Core + FA3 via PyTorch)
bench-tch-bf16 *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features backend_tch_bf16 --bin bench_realmodel -- {{args}}

# Run all three backends sequentially
bench-all:
    just bench-cpu
    just bench-wgpu
    just bench-cuda

# Benchmark Python reference implementation (runs in Irodori-TTS uv env)
bench-python:
    cd ../Irodori-TTS && uv run python ../Irodori-TTS-burn/scripts/bench_python.py

# ── Codec E2E ────────────────────────────────────────────────────────────────

TORCH_LIB := env_var_or_default("TORCH_LIB", "/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib")
DACVAE_WEIGHTS := env_var_or_default("DACVAE_WEIGHTS", "target/dacvae_weights.safetensors")
DACVAE_MODEL := env_var_or_default("DACVAE_MODEL", "/home/sanzentyo/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth")

# Convert Python DACVAE weights to safetensors (needed once)
codec-convert:
    LD_LIBRARY_PATH="{{TORCH_LIB}}:${LD_LIBRARY_PATH}" \
        cd ../Irodori-TTS && uv run --extra dev python ../Irodori-TTS-burn/scripts/convert_dacvae_weights.py \
        --model-path {{DACVAE_MODEL}} \
        --output {{DACVAE_WEIGHTS}}

# Generate Python reference latent for DACVAE parity test
codec-ref:
    cd ../Irodori-TTS && uv run --extra dev python ../Irodori-TTS-burn/scripts/codec_e2e_ref.py \
        --output /tmp/py_latent.npy \
        --save-audio target/test_audio.wav \
        --model-dir {{DACVAE_MODEL}}

# Run Rust DACVAE E2E parity check (requires codec-ref first)
codec-e2e-rust:
    LD_LIBRARY_PATH="{{TORCH_LIB}}:${LD_LIBRARY_PATH}" \
        cargo run --release --bin codec_e2e -- \
        --weights {{DACVAE_WEIGHTS}} \
        --ref-latent /tmp/py_latent.npy \
        --audio target/test_audio.wav

# Full codec E2E: generate Python reference + run Rust parity check
codec-e2e: codec-ref codec-e2e-rust

# ── Quality comparison ────────────────────────────────────────────────────

# Run multi-backend quality comparison (Python + all Rust backends, 5 test prompts)
quality-compare *args:
    uv run scripts/run_quality_comparison.py {{args}}

# Quick quality comparison — skip build (assumes binaries already built)
quality-compare-fast *args:
    uv run scripts/run_quality_comparison.py --skip-build {{args}}

# Quality comparison — single backend only (e.g. just quality-compare-backend rust_libtorch_bf16)
quality-compare-backend backend *args:
    uv run scripts/run_quality_comparison.py --skip-build --backends {{backend}} {{args}}



# Push to GitHub
push:
    git push origin master

# Commit and push
commit-push msg:
    git add -A
    git commit -m "{{msg}}" -m "" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    git push origin master

# ── LoRA Training ────────────────────────────────────────────────────────────

# Train a LoRA adapter (NdArray backend, CPU)
train-lora config:
    cargo run --release --bin train_lora -- --config {{config}}

# Train a LoRA adapter using the LibTorch backend (CPU, same kernels as Python)
train-lora-tch config:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/local/bin:/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features backend_tch --bin train_lora -- --config {{config}}

# Train a LoRA adapter using the LibTorch bf16 backend (fastest CPU option)
train-lora-tch-bf16 config:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sanzentyo/Irodori-TTS/.venv \
    PATH=/home/sanzentyo/Irodori-TTS/.venv/bin:{{env_var_or_default("PATH", "/usr/local/bin:/usr/bin:/bin")}} \
    LD_LIBRARY_PATH=/home/sanzentyo/Irodori-TTS/.venv/lib/python3.10/site-packages/torch/lib:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features backend_tch_bf16 --bin train_lora -- --config {{config}}

# Encode audio files → safetensors latents using DACVAE (for training data prep)
encode-dataset *args:
    uv run scripts/encode_dataset.py {{args}}
