set shell := ["bash", "-cu"]
set dotenv-load

# ── Configurable paths (override via environment or .env) ─────────────────────
PYTHON_REF_DIR  := env_var_or_default("PYTHON_REF_DIR", "../Irodori-TTS")
PYTHON_VENV     := PYTHON_REF_DIR / ".venv"
# Cross-platform: "Scripts" on Windows, "bin" on Unix
PYTHON_VENV_BIN := if os() == "windows" { PYTHON_VENV / "Scripts" } else { PYTHON_VENV / "bin" }
# Cross-platform torch lib path (override via TORCH_LIB_DIR env var or .env for non-Python-3.10 venvs)
TORCH_LIB_DIR   := env_var_or_default("TORCH_LIB_DIR", if os() == "windows" { PYTHON_VENV / "Lib/site-packages/torch/lib" } else { PYTHON_VENV / "lib/python3.10/site-packages/torch/lib" })
SYSTEM_PATH     := env_var_or_default("PATH", if os() == "windows" { "" } else { "/usr/local/bin:/usr/bin:/bin" })

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
    cargo test --lib

# Run all tests including train/lora features
test-all:
    cargo test --lib --features "cli,train,lora"

# Run tests with output
test-verbose:
    cargo test --all-targets -- --nocapture

# Run WGPU kernel tests (ignored by default due to WGPU teardown SIGSEGV).
# Must run single-threaded; excludes the intentional subgroup-bug-diagnostic failure.
test-kernels:
    cargo test --lib --features cli kernels:: -- --ignored --test-threads=1 \
        --skip kernels::subgroup_diagnostic::tests::subgroup_diagnostic_with_enable

# Lint with clippy
lint:
    cargo clippy --all-targets -- -D warnings

# Lint all features (library + all binaries)
lint-all:
    cargo clippy --all-targets --features "cli,train,lora" -- -D warnings

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
    {{PYTHON_VENV_BIN}}/python3 scripts/validate_numerics.py
    cargo run --features cli --bin validate

# Only regenerate Python fixtures (no Rust run)
validate-fixtures:
    {{PYTHON_VENV_BIN}}/python3 scripts/validate_numerics.py

# Only run Rust comparison (assumes fixtures already exist)
validate-rust:
    cargo run --features cli --bin validate

# ── E2E Comparison ───────────────────────────────────────────────────────────

# Generate Python E2E fixtures (requires validate-fixtures to have run first)
e2e-fixtures:
    {{PYTHON_VENV_BIN}}/python3 scripts/e2e_compare.py

# Run Rust E2E comparison (assumes e2e-fixtures already run)
e2e-rust:
    cargo run --features cli --bin e2e_compare

# Full E2E: generate Python fixtures then run Rust comparison
e2e: validate-fixtures e2e-fixtures e2e-rust

# Run Rust E2E comparison on the WgpuRaw f16 backend (Metal/Vulkan/DX12)
e2e-wgpu-raw-f16-rust:
    cargo run --features cli --bin e2e_compare -- --backend wgpu-raw-f16

# Full E2E with WgpuRaw f16 backend
e2e-wgpu-raw-f16: validate-fixtures e2e-fixtures e2e-wgpu-raw-f16-rust

# Run Rust E2E comparison on the LibTorch (CUDA) backend
e2e-tch-rust:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --features cli --bin e2e_compare -- --backend libtorch

# Full E2E with LibTorch backend
e2e-tch: validate-fixtures e2e-fixtures e2e-tch-rust

# Run Rust E2E comparison on the LibTorch bf16 backend
e2e-tch-bf16-rust:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --features cli --bin e2e_compare -- --backend libtorch-bf16

# Full E2E with LibTorch bf16 backend
e2e-tch-bf16: validate-fixtures e2e-fixtures e2e-tch-bf16-rust

# Run Rust E2E comparison on the LibTorch MPS backend (Apple Silicon only)
e2e-tch-mps-rust:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
        cargo run --features cli --bin e2e_compare -- --backend libtorch-mps

# Full E2E with LibTorch MPS backend (Apple Silicon only)
e2e-tch-mps: validate-fixtures e2e-fixtures e2e-tch-mps-rust

# ── Full-model E2E comparison ─────────────────────────────────────────────────

# Generate full-model E2E fixtures from hf_model (Python reference)
full-e2e-fixtures:
    {{PYTHON_VENV_BIN}}/python3 scripts/full_model_e2e.py

# Run Rust full-model E2E comparison against Python fixtures (NdArray)
full-e2e-rust:
    cargo run --release --features cli --bin full_model_e2e

# Run Rust full-model E2E comparison (LibTorch CUDA f32)
full-e2e-tch-rust:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin full_model_e2e -- --backend libtorch

# Run Rust full-model E2E comparison (LibTorch CUDA bf16)
full-e2e-tch-bf16-rust:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin full_model_e2e -- --backend libtorch-bf16

# Full-model E2E: generate Python fixtures then run Rust NdArray comparison
full-e2e: full-e2e-fixtures full-e2e-rust

# Full-model E2E with LibTorch CUDA f32
full-e2e-tch: full-e2e-fixtures full-e2e-tch-rust

# Full-model E2E with LibTorch CUDA bf16
full-e2e-tch-bf16: full-e2e-fixtures full-e2e-tch-bf16-rust

# Run Python f32 vs bf16 dtype comparison (shows bf16 audio diff is dtype-induced)
full-e2e-py-dtype-compare:
    {{PYTHON_VENV_BIN}}/python3 scripts/full_model_e2e.py --dtype-compare

# Convert Python safetensors checkpoint to Burn-compatible key names
# Usage: just convert <src.safetensors> <dst.safetensors> [--apply]
convert input output *args:
    uv run scripts/convert_for_burn.py {{input}} {{output}} {{args}}

# Run Rust inference CLI
infer *args:
    cargo run --release --features cli --bin infer -- {{args}}

# Run the full TTS pipeline (text → WAV) using RF model + DACVAE codec
pipeline *args:
    cargo run --release --features cli --bin pipeline -- {{args}}

# Full pipeline against the real converted model + DACVAE codec
# Requires --backend <kind> via {{args}}, e.g.: just pipeline-real --backend wgpu
pipeline-real *args:
    cargo run --release --features cli --bin pipeline -- \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with WgpuRaw backend (no fusion — workaround for burn-fusion DACVAE panic)
# Use when --backend wgpu crashes with "Ordering is bigger than operations" on codec decode
pipeline-real-raw *args:
    cargo run --release --features cli --bin pipeline -- \
        --backend wgpu-raw \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with WgpuRaw f16 backend (no fusion + half precision; requires SHADER_F16)
# Combines burn-fusion crash workaround with f16 speed-up (~2× vs wgpu-raw on Metal/Vulkan)
pipeline-real-raw-f16 *args:
    cargo run --release --features cli --bin pipeline -- \
        --backend wgpu-raw-f16 \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with LibTorch backend (f32, same kernels as Python)
pipeline-real-tch *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
    cargo run --release --features cli --bin pipeline -- \
        --backend libtorch \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with LibTorch bf16 backend (fastest option)
pipeline-real-tch-bf16 *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
    cargo run --release --features cli --bin pipeline -- \
        --backend libtorch-bf16 \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with LibTorch MPS f32 backend (Apple Silicon)
pipeline-real-tch-mps *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
    cargo run --release --features cli --bin pipeline -- \
        --backend libtorch-mps \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# Full pipeline with LibTorch MPS f16 backend (Apple Silicon, fastest)
pipeline-real-tch-mps-f16 *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
    cargo run --release --features cli --bin pipeline -- \
        --backend libtorch-mps-f16 \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        {{args}}

# ── LoRA Training ────────────────────────────────────────────────────────────

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
# Requires --backend <kind> via {{args}}, e.g.: just infer-real --backend wgpu
infer-real *args:
    cargo run --release --features cli --bin infer -- \
        --checkpoint target/model_converted.safetensors {{args}}

# Inference with CUDA bf16 backend
infer-real-bf16 *args:
    cargo run --release --features cli --bin infer -- \
        --backend cuda-bf16 --checkpoint target/model_converted.safetensors {{args}}

# ── LoRA Inference ────────────────────────────────────────────────────────────

# Run full pipeline (text → WAV) with a LoRA adapter
# Usage: just pipeline-lora target/lora/43ch --backend wgpu
pipeline-lora adapter *args:
    cargo run --release --features "cli,lora" --bin pipeline -- \
        --checkpoint target/model_converted.safetensors \
        --codec-weights target/dacvae_weights.safetensors \
        --adapter {{adapter}} \
        {{args}}

# Run inference only (no codec) with a LoRA adapter
# Usage: just infer-lora target/lora/43ch --backend wgpu
infer-lora adapter *args:
    cargo run --release --features "cli,lora" --bin infer -- \
        --checkpoint target/model_converted.safetensors \
        --adapter {{adapter}} \
        {{args}}

# ── Benchmarks ───────────────────────────────────────────────────────────────

# Run criterion benchmarks (requires validate fixtures: just validate-fixtures)
bench *args:
    cargo bench --bench inference -- {{args}}

# Run criterion codec benchmarks (requires: just codec-convert)
bench-codec *args:
    cargo bench --bench codec -- {{args}}

# Fast wall-clock codec timing benchmark — LibTorch CPU (comparable to Python/PyTorch)
bench-codec-tch *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_codec -- --backend libtorch --weights {{DACVAE_WEIGHTS}} {{args}}

# Benchmark Python DACVAE codec for comparison
bench-codec-py *args:
    cd {{PYTHON_REF_DIR}} && uv run --extra dev python \
        {{justfile_directory()}}/scripts/bench_codec_py.py \
        --model-path {{DACVAE_WEIGHTS}} {{args}}

# Run both Rust (LibTorch) and Python codec benchmarks for comparison
bench-codec-compare: (bench-codec-tch) (bench-codec-py)

# Open last HTML benchmark report
bench-report:
    xdg-open target/criterion/report/index.html 2>/dev/null || open target/criterion/report/index.html

# Smoke test real-model inference on WGPU
bench-wgpu-smoke:
    cargo run --release --features cli --bin bench_realmodel -- \
        --backend wgpu --seq-len 64 --num-steps 4 --warmup 0 --runs 1

# Smoke test real-model inference on CUDA (f32)
bench-cuda-smoke:
    cargo run --release --features cli --bin bench_realmodel -- \
        --backend cuda --seq-len 64 --num-steps 4 --warmup 0 --runs 1

# Smoke test real-model inference on CUDA (bf16)
bench-cuda-bf16-smoke:
    cargo run --release --features cli --bin bench_realmodel -- \
        --backend cuda-bf16 --seq-len 64 --num-steps 4 --warmup 0 --runs 1

# Full benchmark — WGPU f32 (seq=750, steps=40)
bench-wgpu *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend wgpu {{args}}

# Full benchmark — WGPU f16 (requires shader-f16 GPU support)
bench-wgpu-f16 *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-f16 {{args}}

# Full benchmark — WGPU raw (no fusion, custom WGSL kernels)
bench-wgpu-raw *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-raw {{args}}

# Full benchmark — WGPU raw f16 (no fusion, half precision; requires SHADER_F16 GPU support)
bench-wgpu-raw-f16 *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-raw-f16 {{args}}

# Tiled FA micro-benchmark (custom WGSL flash-attention vs burn_attention baseline, WgpuRaw backend)
bench-tiled-fa *args:
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
        cargo run --release --bin bench_tiled_fa -- {{args}}

# FlashUnit FA feasibility benchmark (burn_cubecl FlashUnit vs naive SDPA fallback, WgpuRaw backend)
bench-flashunit-fa *args:
    cargo run --release --bin bench_flashunit_fa -- {{args}}

# Full benchmark — Burn CUDA f32 (seq=750, steps=40)
bench-cuda *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend cuda {{args}}

# Full benchmark — Burn CUDA bf16 (seq=750, steps=40; faster on Tensor Core GPUs)
bench-cuda-bf16 *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend cuda-bf16 {{args}}

# Run all CUDA benchmarks (f32 and bf16)
bench-cuda-all:
    just bench-cuda
    just bench-cuda-bf16

# NVTX profiling run — generates nsys report at target/profile.nsys-rep
bench-cuda-profile *args:
    nsys profile --output target/profile.nsys-rep --force-overwrite true \
        cargo run --release --features "profile,cli" --bin bench_realmodel -- \
        --backend cuda --warmup 0 --runs 1 {{args}}

# Full benchmark — LibTorch f32 (cuBLAS / FA3 via PyTorch)
# Requires the Irodori-TTS venv to have PyTorch installed.
bench-tch *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_realmodel -- --backend libtorch {{args}}

# Full benchmark — LibTorch bf16 (Tensor Core + FA3 via PyTorch)
bench-tch-bf16 *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-bf16 {{args}}

# Full benchmark — LibTorch MPS f32 (Metal Performance Shaders, Apple Silicon only)
bench-tch-mps *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-mps {{args}}

# Full benchmark — LibTorch MPS f16 (Metal Performance Shaders + half precision, Apple Silicon only)
bench-tch-mps-f16 *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-mps-f16 {{args}}

# Joint CFG benchmark — LibTorch MPS f16, equal scales (RTF 0.282 on M4 Pro, torch 2.10.0)
# Requires --cfg-speaker <scale> to set equal text+speaker scale (default text=3.0)
bench-tch-mps-f16-joint *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-mps-f16 --cfg-mode joint --cfg-speaker 3.0 {{args}}

# Alternating CFG benchmark — LibTorch MPS f16 (RTF 0.283 on M4 Pro, ≈ Joint/Speaker-only)
bench-tch-mps-f16-alt *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-mps-f16 --cfg-mode alternating {{args}}

# Joint CFG benchmark — WgpuRaw f16, equal scales (RTF 0.476 on M4 Pro, no-dep)
bench-wgpu-raw-f16-joint *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-raw-f16 --cfg-mode joint --cfg-speaker 3.0 {{args}}

# Alternating CFG benchmark — WgpuRaw f16 (RTF 0.477 on M4 Pro, ≈ Joint)
bench-wgpu-raw-f16-alt *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-raw-f16 --cfg-mode alternating {{args}}

# Heun 20-step benchmark — LibTorch MPS f16 (same NFE as Euler 40-step)
# Compare with bench-tch-mps-f16 (40 steps) for quality/speed tradeoff.
bench-tch-mps-f16-heun20 *args:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    DYLD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("DYLD_LIBRARY_PATH", "")}} \
        cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-mps-f16 --sampler heun --num-steps 20 {{args}}

# Heun 20-step benchmark — WgpuRaw f16 (same NFE as Euler 40-step)
bench-wgpu-raw-f16-heun20 *args:
    cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-raw-f16 --sampler heun --num-steps 20 {{args}}

# Run GPU backends sequentially (NdArray excluded — impractically slow)
bench-all:
    just bench-wgpu
    just bench-cuda
    just bench-tch

# Benchmark Python reference implementation (runs in Irodori-TTS uv env)
# Usage: just bench-python [dtype]  where dtype = f32 | bf16 | autocast-bf16
bench-python dtype="f32":
    cd ../Irodori-TTS && uv run python ../Irodori-TTS-burn/scripts/bench_python.py --dtype {{dtype}}

# ── Codec E2E ────────────────────────────────────────────────────────────────

TORCH_LIB := env_var_or_default("TORCH_LIB", TORCH_LIB_DIR)
DACVAE_WEIGHTS := env_var_or_default("DACVAE_WEIGHTS", "target/dacvae_weights.safetensors")
DACVAE_MODEL := env_var_or_default("DACVAE_MODEL", if os() == "windows" { env_var_or_default("USERPROFILE", "") / ".cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth" } else { env_var_or_default("HOME", "") / ".cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth" })

# Convert Python DACVAE weights to safetensors (needed once)
codec-convert:
    LD_LIBRARY_PATH="{{TORCH_LIB}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}}" \
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
    LD_LIBRARY_PATH="{{TORCH_LIB}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}}" \
        cargo run --release --features cli --bin codec_e2e -- \
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

# Train a LoRA adapter (requires --backend argument)
train-lora config *args:
    cargo run --release --features "train,cli" --bin train_lora -- --config {{config}} {{args}}

# Train a LoRA adapter using the LibTorch f32 backend
train-lora-tch config:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features "train,cli" --bin train_lora -- --backend libtorch --config {{config}}

# Train a LoRA adapter using the LibTorch bf16 backend
train-lora-tch-bf16 config:
    LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    VIRTUAL_ENV={{PYTHON_VENV}} \
    PATH={{PYTHON_VENV_BIN}}:{{TORCH_LIB_DIR}}:{{SYSTEM_PATH}} \
    LD_LIBRARY_PATH={{TORCH_LIB_DIR}}:{{env_var_or_default("LD_LIBRARY_PATH", "")}} \
        cargo run --release --features "train,cli" --bin train_lora -- --backend libtorch-bf16 --config {{config}}

