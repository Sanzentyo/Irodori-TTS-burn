# Deep Quality Review — 2026-04-13

Source: `deep-review` rubber-duck agent (comprehensive pass over all source files).

## Summary

9 findings total: 2 BLOCKING, 7 MEDIUM.
All 2 BLOCKING findings and 5 of 7 MEDIUM findings were fixed.
2 MEDIUM findings deferred (RoPE caching optimisation, validation coverage expansion).

---

## Findings & Resolutions

| # | Severity | Area | Status |
|---|----------|------|--------|
| 1 | BLOCKING | `ModelConfig::validate()` incomplete | ✅ Fixed (commit c8037e5) |
| 2 | MEDIUM | All-masked attention NaN contract | ✅ Fixed — doc clarified |
| 3 | MEDIUM | RoPE tables recomputed per step | ⏳ Deferred (optimisation) |
| 4 | MEDIUM | Speaker CFG active with no ref_latent | ✅ Fixed (commit c8037e5) |
| 5 | MEDIUM | `scale_speaker_kv_cache` applied in caption mode | ✅ Fixed (commit c8037e5) |
| 6 | MEDIUM | CLI ref_latent shape doc wrong + no dim validation | ✅ Fixed (commit c8037e5) |
| 7 | MEDIUM | Sampler panics on bad inputs | ✅ Fixed — `SamplerParams::validate()` + `Result` (commit c8037e5) |
| 8 | MEDIUM | Validation fixtures only cover speaker path | ⏳ Deferred |
| 9 | BLOCKING | `[[bench]]` declared but `benches/inference.rs` missing | ✅ Already fixed in prior commit 174399c |

---

## Detail

### Fix #1 — `ModelConfig::validate()` expansion

`validate()` previously only checked the main diffusion head.  Now covers:

- `model_dim / num_heads` divisibility + even head_dim (for RoPE) — already present
- `text_dim / text_heads` divisibility + even head_dim
- `latent_patch_size`, `timestep_embed_dim`, `adaln_rank` > 0
- When `use_speaker_condition()` (i.e. no caption): all `speaker_*` fields must be `Some` and valid (non-zero, divisible, even head_dim)
- When `use_caption_condition`: `caption_dim / caption_heads` divisibility + even head_dim

### Fix #2 — `encode_conditions` NaN warning

Updated the `encode_conditions` doc comment to clearly state that all-false text
masks are unsafe (NaN via softmax on all-`-inf` rows).  The correct approach for
unconditional text is to zero the *output* via `EncodedCondition::zeros_like`, not
zero the input mask.

### Fix #3 — RoPE caching (deferred)

Precomputing `(cos, sin)` tables once per trajectory is a valid optimisation but
requires plumbing through the model API or caching in a `HashMap<(usize,usize), ...>`.
Left for a dedicated performance pass.

### Fix #4 — Speaker CFG guard

`has_speaker_cfg` was `cfg_scale_speaker > 0.0 && model.use_speaker_condition()`.
With `--cfg-speaker 1.0` (the CLI default) and no `--ref-latent`, this evaluated
to `true`, causing an extra identical unconditional forward pass per step.

Fix: add `&& cond.speaker_state.is_some()`.

### Fix #5 — KV scale caption-mode guard

`scale_speaker_kv_cache` scales `aux_k / aux_v` tensors.  In caption mode, those
slots hold caption context — scaling them would incorrectly amplify captions instead
of speaker.  Both the initial scale and the revert are now gated on
`model.use_speaker_condition()`.

### Fix #6 — CLI ref_latent shape

Doc said `[1, T, latent_dim * patch_size]` but the actual contract is `[1, T, latent_dim]`
because `encode_conditions` patches internally.  Fixed doc + added runtime shape
validation against `cfg.latent_dim`.

### Fix #7 — Sampler panics → `Result`

- Added `SamplerParams::validate() -> Result<()>` checking `num_steps > 0`,
  `truncation_factor > 0` (if set), and `temporal_rescale.sigma != 0`.
- `sample_euler_rf_cfg` now returns `Result<Tensor<B, 3>>`; the `assert!(num_steps > 0)`
  and the Joint-mode `assert!` are replaced with typed `IrodoriError::Config` returns.
- `InferenceEngine::sample` returns `Result<Tensor<B, 3>>`.

### Fix #8 — Validation coverage (deferred)

Current validation only exercises the speaker-conditioned path.  Adding caption and
alternating-CFG fixtures requires changes to both `scripts/validate_numerics.py` and
`src/bin/validate.rs`.  Deferred to a dedicated validation pass.

### Fix #9 — Missing `benches/inference.rs`

Fixed in commit 174399c (immediately before this review) by creating
`benches/inference.rs` with three criterion benchmarks:
`encode_conditions`, `forward_with_cond`, `sample_4steps`.
