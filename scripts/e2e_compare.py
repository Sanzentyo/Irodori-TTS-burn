#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch>=2.0",
#   "safetensors>=0.5",
#   "packaging",
#   "numpy",
# ]
# ///
"""
E2E comparison fixture generator for the Rust/burn Irodori-TTS port.

Loads the tiny validation model from target/validate_weights.safetensors,
pre-generates a fixed initial noise tensor (seed=0), then runs a 4-step
Independent-CFG sampling loop that EXACTLY mirrors the Rust sampler:

  - cond forward pass uses pre-built KV cache
  - uncond forward passes use NO KV cache (matching Rust's Independent mode)
  - separate forward passes per condition (not batched)

Writes to target/:
  e2e_inputs.safetensors  – x_t_init, text_ids, text_mask, ref_latent, ref_mask
  e2e_output.safetensors  – final latent + per-step x_t (for failure localisation)

Usage:
    uv run scripts/e2e_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
IRODORI_ROOT = REPO_ROOT.parent / "Irodori-TTS"
sys.path.insert(0, str(IRODORI_ROOT))

import torch
from safetensors.torch import load_file, save_file

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT

# ---------------------------------------------------------------------------
# Model config — must match validate_numerics.py exactly
# ---------------------------------------------------------------------------
CFG = ModelConfig(
    latent_dim=8,
    latent_patch_size=1,
    model_dim=64,
    num_layers=2,
    num_heads=8,
    mlp_ratio=2.0,
    text_mlp_ratio=None,
    speaker_mlp_ratio=None,
    dropout=0.0,
    text_vocab_size=256,
    text_tokenizer_repo="test",
    text_add_bos=False,
    text_dim=64,
    text_layers=2,
    text_heads=8,
    use_caption_condition=False,
    speaker_dim=64,
    speaker_layers=2,
    speaker_heads=8,
    speaker_patch_size=1,
    timestep_embed_dim=64,
    adaln_rank=8,
    norm_eps=1e-5,
)

# ---------------------------------------------------------------------------
# Fixed inputs — must match validate_numerics.py exactly
# ---------------------------------------------------------------------------
BATCH = 1
SEQ_TEXT = 4
SEQ_LAT = 8
SEQ_REF = 4

text_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
text_mask = torch.ones(BATCH, SEQ_TEXT, dtype=torch.bool)
ref_latent = (
    torch.arange(BATCH * SEQ_REF * CFG.latent_dim, dtype=torch.float32)
    .reshape(BATCH, SEQ_REF, CFG.latent_dim)
    * 0.01
)
ref_mask = torch.ones(BATCH, SEQ_REF, dtype=torch.bool)

# ---------------------------------------------------------------------------
# Key-rename helpers
# ---------------------------------------------------------------------------

# validate_numerics.py saves weights with renamed cond_module keys.
# We need to reverse-rename before loading into Python's nn.Sequential model.
_FORWARD_RENAMES = {
    "cond_module.0.weight": "cond_module.linear0.weight",
    "cond_module.2.weight": "cond_module.linear1.weight",
    "cond_module.4.weight": "cond_module.linear2.weight",
}
_REVERSE_RENAMES = {v: k for k, v in _FORWARD_RENAMES.items()}


def _restore_cond_module_keys(state_dict: dict) -> dict:
    """Undo the Rust-compatible rename before loading into the Python model."""
    return {_REVERSE_RENAMES.get(k, k): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Sampler settings
# ---------------------------------------------------------------------------
NUM_STEPS = 4
CFG_SCALE_TEXT = 3.0
CFG_SCALE_SPEAKER = 5.0
CFG_MIN_T = 0.0
CFG_MAX_T = 1.0
INIT_SCALE = 0.999


def main() -> None:
    target = REPO_ROOT / "target"
    weights_path = target / "validate_weights.safetensors"

    if not weights_path.exists():
        sys.exit(
            f"ERROR: {weights_path} not found — run `just validate-fixtures` first"
        )

    # ------------------------------------------------------------------ model
    # Load from file (same bytes Rust uses) rather than recreating from seed.
    raw_state_dict = load_file(str(weights_path))
    state_dict = _restore_cond_module_keys(raw_state_dict)

    model = TextToLatentRFDiT(CFG)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # ------------------------------------------------------------------ noise
    # Pre-generate with a different seed so initial noise != model init noise.
    torch.manual_seed(0)
    x_t_init = torch.randn(BATCH, SEQ_LAT, CFG.latent_dim)

    # ---------------------------------------------------------------- t schedule
    t_schedule = [INIT_SCALE * (1.0 - i / NUM_STEPS) for i in range(NUM_STEPS + 1)]

    # ----------------------------------------------------------------- sampling
    with torch.no_grad():
        # Encode conditioned states.
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

        assert speaker_state_cond is not None, "expected speaker_state_cond"
        assert speaker_mask_cond is not None, "expected speaker_mask_cond"

        # Unconditional states — zeros with all-False masks, matching Rust zeros_like().
        text_state_uncond = torch.zeros_like(text_state_cond)
        text_mask_uncond = torch.zeros_like(text_mask_cond)
        speaker_state_uncond = torch.zeros_like(speaker_state_cond)
        speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)

        # Pre-build KV cache for the conditioned pass only.
        # Uncond passes use context_kv_cache=None, exactly as Rust does for
        # Independent mode (kv_alt_text / kv_alt_speaker are None there).
        kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
        )

        x_t = x_t_init.clone()
        per_step: dict[str, torch.Tensor] = {}

        for i in range(NUM_STEPS):
            t = t_schedule[i]
            t_next = t_schedule[i + 1]
            tt = torch.full([BATCH], t, dtype=torch.float32)

            use_cfg = CFG_MIN_T <= t <= CFG_MAX_T

            if use_cfg:
                # Independent CFG — separate forward passes (matches Rust).
                v_cond = model.forward_with_encoded_conditions(
                    x_t=x_t,
                    t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    context_kv_cache=kv_cond,
                )
                # text uncond: drop text signal, keep speaker signal
                v_text_unc = model.forward_with_encoded_conditions(
                    x_t=x_t,
                    t=tt,
                    text_state=text_state_uncond,
                    text_mask=text_mask_uncond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    context_kv_cache=None,
                )
                # speaker uncond: keep text signal, drop speaker signal
                v_spk_unc = model.forward_with_encoded_conditions(
                    x_t=x_t,
                    t=tt,
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
                    x_t=x_t,
                    t=tt,
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
                f"  step {i}: t={t:.4f}→{t_next:.4f}  dt={dt:.4f}  "
                f"cfg={use_cfg}  |x_t| min={x_t.min().item():.4f} max={x_t.max().item():.4f}"
            )

    target.mkdir(exist_ok=True)

    # Save inputs (initial noise + conditioning inputs).
    inputs = {
        "x_t_init": x_t_init.float(),
        "text_ids": text_ids.float(),
        "text_mask": text_mask.float(),
        "ref_latent": ref_latent.float(),
        "ref_mask": ref_mask.float(),
    }
    save_file(inputs, str(target / "e2e_inputs.safetensors"))

    # Save final output + per-step data for localisation.
    outputs = {"output": x_t.float().clone(), **per_step}
    save_file(outputs, str(target / "e2e_output.safetensors"))

    print(f"\nE2E output  shape={tuple(x_t.shape)}  min={x_t.min().item():.6f}  max={x_t.max().item():.6f}")
    print(f"Saved inputs  → {target / 'e2e_inputs.safetensors'}")
    print(f"Saved outputs → {target / 'e2e_output.safetensors'}")


if __name__ == "__main__":
    main()
