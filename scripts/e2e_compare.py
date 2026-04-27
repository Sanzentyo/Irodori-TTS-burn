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
pre-generates a fixed initial noise tensor (seed=0), then runs both an
Euler sampler loop and a Heun (2nd-order) sampler loop that EXACTLY mirror
the Rust sampler implementations:

  - cond forward pass uses pre-built KV cache
  - uncond passes use context_kv_cache=None (Independent CFG)
  - separate forward passes per condition (not batched)
  - CFG gate is checked at t (v1) and at t_next (v2 for Heun), matching Rust

Writes to target/:
  e2e_inputs.safetensors       – x_t_init, text_ids, text_mask, ref_latent, ref_mask
  e2e_output.safetensors       – Euler 4-step final + per-step x_t / v tensors
  e2e_heun_output.safetensors  – Heun 2-step final + per-step intermediates

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
CFG_SCALE_TEXT = 3.0
CFG_SCALE_SPEAKER = 5.0
CFG_MIN_T = 0.0
CFG_MAX_T = 1.0
INIT_SCALE = 0.999


def _cfg_forward(
    model,
    x_t: "torch.Tensor",
    tt: "torch.Tensor",
    *,
    text_state_cond: "torch.Tensor",
    text_mask_cond: "torch.Tensor",
    speaker_state_cond: "torch.Tensor",
    speaker_mask_cond: "torch.Tensor",
    text_state_uncond: "torch.Tensor",
    text_mask_uncond: "torch.Tensor",
    speaker_state_uncond: "torch.Tensor",
    speaker_mask_uncond: "torch.Tensor",
    kv_cond,
    use_cfg: bool,
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    """Single velocity evaluation with Independent CFG (mirrors Rust exactly).

    Returns (v_cfg, debug_dict) where debug_dict contains v_cond and uncond
    velocities for per-step failure localisation.
    """
    debug: dict[str, "torch.Tensor"] = {}
    if use_cfg:
        v_cond = model.forward_with_encoded_conditions(
            x_t=x_t, t=tt,
            text_state=text_state_cond, text_mask=text_mask_cond,
            speaker_state=speaker_state_cond, speaker_mask=speaker_mask_cond,
            context_kv_cache=kv_cond,
        )
        v_text_unc = model.forward_with_encoded_conditions(
            x_t=x_t, t=tt,
            text_state=text_state_uncond, text_mask=text_mask_uncond,
            speaker_state=speaker_state_cond, speaker_mask=speaker_mask_cond,
            context_kv_cache=None,
        )
        v_spk_unc = model.forward_with_encoded_conditions(
            x_t=x_t, t=tt,
            text_state=text_state_cond, text_mask=text_mask_cond,
            speaker_state=speaker_state_uncond, speaker_mask=speaker_mask_uncond,
            context_kv_cache=None,
        )
        v = (
            v_cond
            + CFG_SCALE_TEXT * (v_cond - v_text_unc)
            + CFG_SCALE_SPEAKER * (v_cond - v_spk_unc)
        )
        debug = {
            "v_cond": v_cond.float(),
            "v_text_unc": v_text_unc.float(),
            "v_spk_unc": v_spk_unc.float(),
        }
    else:
        v = model.forward_with_encoded_conditions(
            x_t=x_t, t=tt,
            text_state=text_state_cond, text_mask=text_mask_cond,
            speaker_state=speaker_state_cond, speaker_mask=speaker_mask_cond,
            context_kv_cache=kv_cond,
        )
        debug = {"v_cond": v.float()}
    return v, debug


def run_sampler(
    model,
    x_t_init: "torch.Tensor",
    *,
    method: str,
    num_steps: int,
    text_state_cond: "torch.Tensor",
    text_mask_cond: "torch.Tensor",
    speaker_state_cond: "torch.Tensor",
    speaker_mask_cond: "torch.Tensor",
    text_state_uncond: "torch.Tensor",
    text_mask_uncond: "torch.Tensor",
    speaker_state_uncond: "torch.Tensor",
    speaker_mask_uncond: "torch.Tensor",
    kv_cond,
    output_path: "Path",
) -> "torch.Tensor":
    """Run the sampler (euler or heun) and save the fixture.

    The timestep schedule, CFG gate logic, and forward-pass structure
    exactly mirror the Rust sampler in src/rf/euler_sampler.rs.
    """
    assert method in ("euler", "heun"), f"unknown method: {method}"
    t_schedule = [INIT_SCALE * (1.0 - i / num_steps) for i in range(num_steps + 1)]
    batch = x_t_init.shape[0]

    x_t = x_t_init.clone()
    per_step: dict[str, torch.Tensor] = {}

    kw = dict(
        text_state_cond=text_state_cond,
        text_mask_cond=text_mask_cond,
        speaker_state_cond=speaker_state_cond,
        speaker_mask_cond=speaker_mask_cond,
        text_state_uncond=text_state_uncond,
        text_mask_uncond=text_mask_uncond,
        speaker_state_uncond=speaker_state_uncond,
        speaker_mask_uncond=speaker_mask_uncond,
        kv_cond=kv_cond,
    )

    print(f"\n--- {method.upper()} {num_steps}-step (NFE={num_steps * (2 if method == 'heun' else 1)}) ---")
    for i in range(num_steps):
        t = t_schedule[i]
        t_next = t_schedule[i + 1]
        dt = t_next - t
        tt = torch.full([batch], t, dtype=torch.float32)
        tt_next = torch.full([batch], t_next, dtype=torch.float32)

        use_cfg = CFG_MIN_T <= t <= CFG_MAX_T

        v1, dbg1 = _cfg_forward(model, x_t, tt, use_cfg=use_cfg, **kw)
        for k, val in dbg1.items():
            per_step[f"v1_{k}_{i}"] = val

        if method == "heun":
            x_pred = x_t + v1 * dt
            per_step[f"x_pred_{i}"] = x_pred.float()
            # Mirror Rust: CFG gate for v2 uses t_next, not t.
            use_cfg_v2 = CFG_MIN_T <= t_next <= CFG_MAX_T
            v2, dbg2 = _cfg_forward(model, x_pred, tt_next, use_cfg=use_cfg_v2, **kw)
            for k, val in dbg2.items():
                per_step[f"v2_{k}_{i}"] = val
            v = (v1 + v2) / 2.0
        else:
            v = v1

        x_t = x_t + v * dt
        per_step[f"x_t_{i}"] = x_t.float()

        print(
            f"  step {i}: t={t:.4f}→{t_next:.4f}  dt={dt:.4f}  "
            f"cfg={use_cfg}  |x_t| min={x_t.min().item():.4f} max={x_t.max().item():.4f}"
        )

    outputs = {"output": x_t.float().clone(), **per_step}
    save_file(outputs, str(output_path))
    print(f"  shape={tuple(x_t.shape)}  min={x_t.min().item():.6f}  max={x_t.max().item():.6f}")
    print(f"  Saved → {output_path}")
    return x_t


def main() -> None:
    target = REPO_ROOT / "target"
    weights_path = target / "validate_weights.safetensors"

    if not weights_path.exists():
        sys.exit(
            f"ERROR: {weights_path} not found — run `just validate-fixtures` first"
        )

    # ------------------------------------------------------------------ model
    raw_state_dict = load_file(str(weights_path))
    state_dict = _restore_cond_module_keys(raw_state_dict)

    model = TextToLatentRFDiT(CFG)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # ------------------------------------------------------------------ noise
    torch.manual_seed(0)
    x_t_init = torch.randn(BATCH, SEQ_LAT, CFG.latent_dim)

    target.mkdir(exist_ok=True)

    # Save inputs (shared across Euler and Heun runs).
    inputs = {
        "x_t_init": x_t_init.float(),
        "text_ids": text_ids.float(),
        "text_mask": text_mask.float(),
        "ref_latent": ref_latent.float(),
        "ref_mask": ref_mask.float(),
    }
    save_file(inputs, str(target / "e2e_inputs.safetensors"))
    print(f"Saved inputs → {target / 'e2e_inputs.safetensors'}")

    with torch.no_grad():
        # Encode conditioned states (shared for all runs).
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

        text_state_uncond = torch.zeros_like(text_state_cond)
        text_mask_uncond = torch.zeros_like(text_mask_cond)
        speaker_state_uncond = torch.zeros_like(speaker_state_cond)
        speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)

        kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
        )

        shared_kw = dict(
            text_state_cond=text_state_cond,
            text_mask_cond=text_mask_cond,
            speaker_state_cond=speaker_state_cond,
            speaker_mask_cond=speaker_mask_cond,
            text_state_uncond=text_state_uncond,
            text_mask_uncond=text_mask_uncond,
            speaker_state_uncond=speaker_state_uncond,
            speaker_mask_uncond=speaker_mask_uncond,
            kv_cond=kv_cond,
        )

        # ----------------------------------------------------------- Euler 4-step
        run_sampler(
            model, x_t_init,
            method="euler", num_steps=4,
            output_path=target / "e2e_output.safetensors",
            **shared_kw,
        )

        # ----------------------------------------------------------- Heun 2-step
        run_sampler(
            model, x_t_init,
            method="heun", num_steps=2,
            output_path=target / "e2e_heun_output.safetensors",
            **shared_kw,
        )


if __name__ == "__main__":
    main()
