# User Input: Continue — Muon optimizer + MetricsSink implementation

**Date:** 2026-07-28

## Input

> please continue

## Context

Session continued from summary where `src/config/training.rs` had been partially modified to add
`AdjustLrPolicy`, `MuonOptimizerConfig`, `OptimizerKind` types, and new fields
`optimizer: OptimizerKind` and `metrics_file: Option<PathBuf>` to `LoraTrainConfig`.

The remaining work was:
- Complete `validate()` with Muon-specific checks
- Add tests for new config types
- Create `src/train/metrics.rs`
- Refactor `src/train/trainer.rs` to be generic over optimizer kind

## Work Done This Session

1. **`src/config/training.rs`** — Added Muon validate checks and 9 new tests (total 353 tests passing)
2. **`src/train/metrics.rs`** (new) — `MetricsSink` trait, `StdoutSink`, `JsonlSink`, `MultiSink<A,B>`, 4 tests
3. **`src/train.rs`** — Exported `metrics` module and public types
4. **`src/config.rs`** — Exported `AdjustLrPolicy`, `MuonOptimizerConfig`, `OptimizerKind`
5. **`src/train/trainer.rs`** — Refactored to generic `train_lora_inner<B,O>`, Muon/AdamW dispatch in `train_lora<B>`, `MetricsSink` wired for `train/loss`, `train/lr`, `val/loss`
6. Fixed serde rename issue (`adamw`→`adamw`, `match_rms_adamw` explicit rename)
