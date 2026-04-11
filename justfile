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



# Convert Python safetensors checkpoint to Burn-compatible key names
# Usage: just convert <src.safetensors> <dst.safetensors> [--apply]
convert input output *args:
    uv run scripts/convert_for_burn.py {{input}} {{output}} {{args}}

# Run Rust inference CLI
infer *args:
    cargo run --release --bin infer -- {{args}}

# ── Docs ──────────────────────────────────────────────────────────────────────

# Show current progress
progress:
    @cat docs/planning/progress.md 2>/dev/null || echo "No progress doc yet."

# ── Benchmarks ───────────────────────────────────────────────────────────────

# Run criterion benchmarks (requires validate fixtures: just validate-fixtures)
bench *args:
    cargo bench --bench inference -- {{args}}

# Open last HTML benchmark report
bench-report:
    xdg-open target/criterion/report/index.html 2>/dev/null || open target/criterion/report/index.html

# ── Git ───────────────────────────────────────────────────────────────────────

# Push to GitHub
push:
    git push origin master

# Commit and push
commit-push msg:
    git add -A
    git commit -m "{{msg}}" -m "" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    git push origin master
