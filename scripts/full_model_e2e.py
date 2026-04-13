#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.0",
#   "safetensors>=0.5",
#   "transformers>=4.0",
#   "packaging",
#   "numpy",
# ]
# ///
"""
Full-model E2E fixture generator for the Rust/burn Irodori-TTS port.

Loads the full TextToLatentRFDiT from target/hf_model/model.safetensors,
uses a fixed initial noise (target/initial_noise_seed42.safetensors), then
runs a 10-step Independent-CFG sampling loop that EXACTLY mirrors the Rust
sampler internals:

  - cond forward pass uses pre-built KV cache
  - uncond forward passes use NO KV cache (matching Rust's Independent mode)
  - separate forward passes per condition (not batched)
  - same timestep schedule: init_scale * (1.0 - i / num_steps)

**Scope**: validates Rust sampler parity with an explicitly-unrolled 3-pass
Python CFG loop. The 3-pass approach is functionally equivalent to the batched
Python sampler (no cross-batch interactions, dropout=0, RMSNorm is batch-
independent). This is confirmed by the tiny-model E2E test which passes at
max_abs_diff = 0.0.

Writes to target/:
  full_e2e_inputs.safetensors  – x_t_init, text_ids, text_mask, ref_latent, ref_mask
  full_e2e_output.safetensors  – output + per-step v_cond_i, v_text_unc_i, v_spk_unc_i, x_t_i

Usage:
    uv run scripts/full_model_e2e.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
IRODORI_ROOT = REPO_ROOT.parent / "Irodori-TTS"
sys.path.insert(0, str(IRODORI_ROOT))

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer

# ---------------------------------------------------------------------------
# Sampler settings (must match Rust SamplerParams exactly)
# ---------------------------------------------------------------------------
NUM_STEPS = 10
CFG_SCALE_TEXT = 3.0
CFG_SCALE_SPEAKER = 5.0
CFG_MIN_T = 0.0
CFG_MAX_T = 1.0
INIT_SCALE = 0.999

# Short Japanese text for tokenization
INPUT_TEXT = "テスト"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config_json(path: Path) -> dict:
    with safe_open(str(path), framework="np") as sf:
        meta = sf.metadata() or {}
    if "config_json" not in meta:
        sys.exit(f"ERROR: no 'config_json' in metadata of {path}")
    return json.loads(meta["config_json"])


def _build_model_config(raw: dict) -> ModelConfig:
    """Filter raw metadata JSON to ModelConfig fields and construct it."""
    mc_fields = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in raw.items() if k in mc_fields})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    target = REPO_ROOT / "target"

    hf_path = target / "hf_model" / "model.safetensors"
    converted_path = target / "model_converted.safetensors"
    noise_path = target / "initial_noise_seed42.safetensors"

    for p in [hf_path, converted_path, noise_path]:
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    # ── Load & cross-check config ──────────────────────────────────────────
    raw_hf = _load_config_json(hf_path)
    raw_conv = _load_config_json(converted_path)

    if raw_hf != raw_conv:
        diffs = {k for k in set(raw_hf) | set(raw_conv) if raw_hf.get(k) != raw_conv.get(k)}
        sys.exit(
            f"ERROR: config_json mismatch between hf_model and model_converted "
            f"(differing keys: {diffs})"
        )

    cfg = _build_model_config(raw_hf)
    sequence_length: int = int(raw_hf.get("fixed_target_latent_steps", 750))

    print(
        f"Config: model_dim={cfg.model_dim}, layers={cfg.num_layers}, "
        f"heads={cfg.num_heads}, latent_dim={cfg.latent_dim}, seq_len={sequence_length}"
    )
    print(
        f"Text:   vocab={cfg.text_vocab_size}, "
        f"tokenizer={cfg.text_tokenizer_repo!r}, add_bos={cfg.text_add_bos}"
    )

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading Python model from hf_model/model.safetensors ...")
    # hf_model uses Python-native keys (cond_module.0/2/4.weight) — load directly.
    state_dict = load_file(str(hf_path))
    model = TextToLatentRFDiT(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    # ── Tokenize ──────────────────────────────────────────────────────────
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=cfg.text_tokenizer_repo,
        add_bos=bool(cfg.text_add_bos),
        local_files_only=False,
    )
    text_ids, text_mask = tokenizer.batch_encode([INPUT_TEXT])
    print(f"\nText: {INPUT_TEXT!r}  →  ids={text_ids[0].tolist()}  (seq_len={text_ids.shape[1]})")

    # ── Reference speaker latent (seeded, non-zero to exercise full path) ─
    torch.manual_seed(42)
    ref_latent = torch.randn(1, 16, cfg.latent_dim) * 0.01
    ref_mask = torch.ones(1, 16, dtype=torch.bool)
    print(f"ref_latent shape={tuple(ref_latent.shape)}  ref_mask all-true")

    # ── Initial noise (from fixed file) ───────────────────────────────────
    raw_noise = load_file(str(noise_path))
    x_t_init: torch.Tensor = raw_noise["initial_noise"]
    expected_shape = (1, sequence_length, cfg.latent_dim)
    if tuple(x_t_init.shape) != expected_shape:
        sys.exit(
            f"Noise shape mismatch: got {tuple(x_t_init.shape)}, expected {expected_shape}"
        )
    print(f"Initial noise: shape={tuple(x_t_init.shape)}  (from {noise_path.name})")

    # ── Timestep schedule (same formula as Rust rf.rs) ────────────────────
    t_schedule = [INIT_SCALE * (1.0 - i / NUM_STEPS) for i in range(NUM_STEPS + 1)]

    # ── Sampling loop ─────────────────────────────────────────────────────
    batch = text_ids.shape[0]

    with torch.no_grad():
        print("\nEncoding conditions ...")
        (
            text_state_cond,
            text_mask_cond,
            speaker_state_cond,
            speaker_mask_cond,
            _caption_state,
            _caption_mask,
        ) = model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
        )

        assert speaker_state_cond is not None, "speaker_state_cond is None"
        assert speaker_mask_cond is not None, "speaker_mask_cond is None"
        print(
            f"  text_state={tuple(text_state_cond.shape)}, "
            f"speaker_state={tuple(speaker_state_cond.shape)}"
        )

        # Unconditional states: zeros with all-False masks (matches Rust zeros_like)
        text_state_uncond = torch.zeros_like(text_state_cond)
        text_mask_uncond = torch.zeros_like(text_mask_cond)
        speaker_state_uncond = torch.zeros_like(speaker_state_cond)
        speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)

        # KV cache for conditioned pass only; uncond passes receive None (Rust Independent mode)
        kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
        )

        x_t = x_t_init.clone()
        per_step: dict[str, torch.Tensor] = {}

        print(f"\nSampling {NUM_STEPS} steps  (Independent CFG, scale_text={CFG_SCALE_TEXT}, scale_spk={CFG_SCALE_SPEAKER}) ...")

        for i in range(NUM_STEPS):
            t = t_schedule[i]
            t_next = t_schedule[i + 1]
            tt = torch.full([batch], t, dtype=torch.float32)

            use_cfg = CFG_MIN_T <= t <= CFG_MAX_T

            if use_cfg:
                v_cond = model.forward_with_encoded_conditions(
                    x_t=x_t, t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    context_kv_cache=kv_cond,
                )
                v_text_unc = model.forward_with_encoded_conditions(
                    x_t=x_t, t=tt,
                    text_state=text_state_uncond,
                    text_mask=text_mask_uncond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    context_kv_cache=None,
                )
                v_spk_unc = model.forward_with_encoded_conditions(
                    x_t=x_t, t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_uncond,
                    speaker_mask=speaker_mask_uncond,
                    context_kv_cache=None,
                )
                v = (
                    v_cond
                    + CFG_SCALE_TEXT * (v_cond - v_text_unc)
                    + CFG_SCALE_SPEAKER * (v_cond - v_spk_unc)
                )
                per_step[f"v_cond_{i}"] = v_cond.float()
                per_step[f"v_text_unc_{i}"] = v_text_unc.float()
                per_step[f"v_spk_unc_{i}"] = v_spk_unc.float()
            else:
                v = model.forward_with_encoded_conditions(
                    x_t=x_t, t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    context_kv_cache=kv_cond,
                )
                per_step[f"v_cond_{i}"] = v.float()

            dt = t_next - t
            x_t = x_t + v * dt
            per_step[f"x_t_{i}"] = x_t.float()

            print(
                f"  step {i:2d}: t={t:.4f}→{t_next:.4f}  dt={dt:.4f}  "
                f"cfg={use_cfg}  "
                f"|x_t| min={x_t.min().item():.4f} max={x_t.max().item():.4f}"
            )

    # ── Save fixtures ─────────────────────────────────────────────────────
    target.mkdir(exist_ok=True)

    inputs = {
        "x_t_init": x_t_init.float(),
        "text_ids": text_ids.float(),
        "text_mask": text_mask.float(),
        "ref_latent": ref_latent.float(),
        "ref_mask": ref_mask.float(),
    }
    save_file(inputs, str(target / "full_e2e_inputs.safetensors"))

    outputs = {"output": x_t.float().clone(), **per_step}
    save_file(outputs, str(target / "full_e2e_output.safetensors"))

    final = x_t.float()
    print(
        f"\nFinal output: shape={tuple(final.shape)}  "
        f"min={final.min().item():.6f}  max={final.max().item():.6f}  "
        f"mean={final.mean().item():.6f}"
    )
    print(f"Saved inputs  → target/full_e2e_inputs.safetensors  ({len(inputs)} tensors)")
    print(f"Saved outputs → target/full_e2e_output.safetensors  ({len(outputs)} tensors)")


if __name__ == "__main__":
    main()
