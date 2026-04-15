# Skills Required After Context Compression

When resuming this session after context compression, load the following skills in order:

```
/skill rust-best-practices
/skill justfile
/skill python-uv-enforcer
```

## Why These Skills Are Needed

| Skill | Purpose |
|-------|---------|
| `rust-best-practices` | Rust idioms, API design, module structure, Clippy compliance |
| `justfile` | Task runner syntax for justfile recipes |
| `python-uv-enforcer` | uv workflow for Irodori-TTS Python reference environment |

## Quick Context Summary

This project is **Irodori-TTS-burn**: a full-scratch Rust reimplementation of
[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) using the `burn` ML library.

- **Original**: PyTorch TTS model (Text-to-Speech) in Python
- **Port**: burn 0.20.x (latest stable), Rust 2024 edition
- **Architecture**: Rectified Flow Diffusion Transformer (RFDiT) with text + speaker conditioning
- **Python reference**: `../Irodori-TTS/` (sibling directory, uv-managed)
- **Rust project**: `./` (this repo)
- **GitHub**: `Sanzentyo/Irodori-TTS-burn` (private)

## Key Files

- `docs/planning/architecture.md` — Full architecture doc
- `docs/planning/progress.md` — Current implementation status
- `src/lib.rs` — Library entry point
- `src/config/` — ModelConfig, TrainingConfig, SamplingConfig (split into submodules)
- `src/model/` — Model modules (dit/, attention, diffusion, etc.)
- `src/rf/` — Rectified Flow sampling (euler_sampler, kv_scaling)
