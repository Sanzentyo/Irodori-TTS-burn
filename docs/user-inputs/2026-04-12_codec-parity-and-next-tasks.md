# User Input — 2026-04-12: Codec parity resolution and next tasks

## Context

After DACVAE E2E parity was achieved (mean err ~4e-6, all 13 frames passing),
the user was asked what to work on next.

## User Response

> **next_task**: do from top to bottom
>
> **notes**: please don't forget read and update docs content (include user input,
> your instructions in .github, etc.)

## Interpretation

Work through the remaining tasks in order:
1. Text normalization — port Japanese rules from Python
2. LoRA weight merging — complete the integration
3. Full E2E inference test — TTS model → DACVAE decode → WAV output
4. Performance benchmarking of the codec (Rust vs Python encode speed)
