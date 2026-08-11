set shell := ["bash", "-cu"]
set dotenv-load := true

PYTHON_REF_DIR := env_var_or_default("PYTHON_REF_DIR", "../Irodori-TTS")
TORCH_LIB_DIR := env_var_or_default("TORCH_LIB_DIR", PYTHON_REF_DIR / ".venv/lib/python3.10/site-packages/torch/lib")
DACVAE_WEIGHTS := env_var_or_default("DACVAE_WEIGHTS", "target/v4_dacvae_weights.safetensors")
DACVAE_MODEL := env_var_or_default("DACVAE_MODEL", env_var_or_default("HOME", "") / ".cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth")

default:
    @just --list

# ── Rust quality gates ───────────────────────────────────────────────────────

build:
    cargo build --locked

build-release:
    cargo build --locked --release

test:
    cargo test --locked --lib

lint:
    cargo clippy --locked --all-targets -- -D warnings

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

ci: fmt-check lint test

# ── Python reference ─────────────────────────────────────────────────────────

py-sync:
    cd {{ PYTHON_REF_DIR }} && uv sync

py-infer *args:
    cd {{ PYTHON_REF_DIR }} && uv run python infer.py {{ args }}

py-lint:
    cd {{ PYTHON_REF_DIR }} && uv run ruff check .

bench-python dtype="f32":
    cd {{ PYTHON_REF_DIR }} && uv run python ../Irodori-TTS-wgpu/scripts/bench_python.py --dtype {{ dtype }}

# ── Production WGPU inference ────────────────────────────────────────────────

convert input output *args:
    uv run scripts/convert_for_burn.py {{ input }} {{ output }} {{ args }}

codec-convert:
    LD_LIBRARY_PATH="{{ TORCH_LIB_DIR }}:{{ env_var_or_default("LD_LIBRARY_PATH", "") }}" \
        uv run scripts/convert_dacvae_weights.py --pth {{ DACVAE_MODEL }} --output {{ DACVAE_WEIGHTS }}

pipeline *args:
    cargo run --locked --release --features cli --bin pipeline -- \
        --backend wgpu-wgsl {{ args }}

pipeline-real *args:
    cargo run --locked --release --features cli --bin pipeline -- \
        --backend wgpu-wgsl \
        --checkpoint target/model_converted.safetensors \
        --codec-weights {{ DACVAE_WEIGHTS }} \
        {{ args }}

pipeline-lora adapter *args:
    cargo run --locked --release --features "cli,lora" --bin pipeline -- \
        --backend wgpu-wgsl \
        --checkpoint target/model_converted.safetensors \
        --codec-weights {{ DACVAE_WEIGHTS }} \
        --adapter {{ adapter }} \
        {{ args }}

# ── Reproducible performance campaigns ──────────────────────────────────────

validate-stages output="/tmp/irodori-v4-same-precision-stage-ab":
    scripts/run_v4_same_precision_stage_ab.sh --output-dir {{ output }}

validate-lengths output="/tmp/irodori-v4-length-sweep":
    scripts/run_v4_length_sweep.sh --output-dir {{ output }}

validate-duration output="/tmp/irodori-v4-duration-sweep":
    scripts/run_v4_duration_sweep.sh {{ output }}

profile-codec *args:
    cargo run --locked --release --features "codec,cli,profile" \
        --bin profile_codec_decoder -- {{ args }}

# ── Utilities ────────────────────────────────────────────────────────────────

encode-dataset *args:
    uv run scripts/encode_dataset.py {{ args }}

download-model:
    uv run scripts/download_model.py

convert-model:
    uv run scripts/convert_for_burn.py \
        target/hf_model/model.safetensors \
        target/model_converted.safetensors \
        --apply

progress:
    @cat docs/planning/progress.md 2>/dev/null || echo "No progress doc yet."
