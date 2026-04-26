#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.13"
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
  full_e2e_inputs.safetensors         – x_t_init, text_ids, text_mask, ref_latent, ref_mask
  full_e2e_output.safetensors         – output + per-step tensors (f32 run)
  full_e2e_output_bf16.safetensors    – output + per-step tensors (bf16 run, --bf16)

Usage:
    uv run scripts/full_model_e2e.py           # f32 (default)
    uv run scripts/full_model_e2e.py --bf16    # bf16 (also prints f32 vs bf16 diff)
    uv run scripts/full_model_e2e.py --dtype-compare  # compare f32 vs bf16 (runs both)
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Full-model E2E fixture generator")
    dtype_group = parser.add_mutually_exclusive_group()
    dtype_group.add_argument("--bf16", action="store_true", help="Run model in bf16")
    dtype_group.add_argument(
        "--dtype-compare",
        action="store_true",
        help="Run both f32 and bf16, compare their outputs",
    )
    args = parser.parse_args()

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

    # ── Run sampling ──────────────────────────────────────────────────────
    def run_sampling(
        run_dtype: torch.dtype, label: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the 10-step Independent-CFG loop with the given dtype.

        Returns (final_x_t_f32, per_step_dict_f32).
        """
        m = model.to(run_dtype)
        # cast all inputs to run_dtype (masks stay bool/int)
        x0 = x_t_init.to(run_dtype)
        ids = text_ids
        tmask = text_mask
        rlat = ref_latent.to(run_dtype)
        rmask = ref_mask
        batch = ids.shape[0]

        print(f"\n[{label}] Encoding conditions ...")
        with torch.no_grad():
            (
                text_state_cond,
                text_mask_cond,
                speaker_state_cond,
                speaker_mask_cond,
                _caption_state,
                _caption_mask,
            ) = m.encode_conditions(
                text_input_ids=ids,
                text_mask=tmask,
                ref_latent=rlat,
                ref_mask=rmask,
            )

            assert speaker_state_cond is not None
            assert speaker_mask_cond is not None
            print(
                f"  text_state={tuple(text_state_cond.shape)}, "
                f"speaker_state={tuple(speaker_state_cond.shape)}"
            )

            text_state_uncond = torch.zeros_like(text_state_cond)
            text_mask_uncond = torch.zeros_like(text_mask_cond)
            speaker_state_uncond = torch.zeros_like(speaker_state_cond)
            speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)

            kv_cond = m.build_context_kv_cache(
                text_state=text_state_cond,
                speaker_state=speaker_state_cond,
            )

            x_t = x0.clone()
            per_step: dict[str, torch.Tensor] = {}

            print(
                f"[{label}] Sampling {NUM_STEPS} steps  "
                f"(Independent CFG, scale_text={CFG_SCALE_TEXT}, scale_spk={CFG_SCALE_SPEAKER}) ..."
            )

            for i in range(NUM_STEPS):
                t = t_schedule[i]
                t_next = t_schedule[i + 1]
                # timestep tensor always f32 (the model converts internally)
                tt = torch.full([batch], t, dtype=torch.float32)

                use_cfg = CFG_MIN_T <= t <= CFG_MAX_T

                if use_cfg:
                    v_cond = m.forward_with_encoded_conditions(
                        x_t=x_t, t=tt,
                        text_state=text_state_cond, text_mask=text_mask_cond,
                        speaker_state=speaker_state_cond, speaker_mask=speaker_mask_cond,
                        context_kv_cache=kv_cond,
                    )
                    v_text_unc = m.forward_with_encoded_conditions(
                        x_t=x_t, t=tt,
                        text_state=text_state_uncond, text_mask=text_mask_uncond,
                        speaker_state=speaker_state_cond, speaker_mask=speaker_mask_cond,
                        context_kv_cache=None,
                    )
                    v_spk_unc = m.forward_with_encoded_conditions(
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
                    per_step[f"v_cond_{i}"] = v_cond.float()
                    per_step[f"v_text_unc_{i}"] = v_text_unc.float()
                    per_step[f"v_spk_unc_{i}"] = v_spk_unc.float()
                else:
                    v = m.forward_with_encoded_conditions(
                        x_t=x_t, t=tt,
                        text_state=text_state_cond, text_mask=text_mask_cond,
                        speaker_state=speaker_state_cond, speaker_mask=speaker_mask_cond,
                        context_kv_cache=kv_cond,
                    )
                    per_step[f"v_cond_{i}"] = v.float()

                dt = t_next - t
                x_t = x_t + v.to(run_dtype) * dt
                per_step[f"x_t_{i}"] = x_t.float()

                print(
                    f"  step {i:2d}: t={t:.4f}→{t_next:.4f}  dt={dt:.4f}  "
                    f"cfg={use_cfg}  "
                    f"|x_t| min={x_t.float().min().item():.4f} max={x_t.float().max().item():.4f}"
                )

        return x_t.float().clone(), per_step

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
    print(f"Saved inputs  → target/full_e2e_inputs.safetensors  ({len(inputs)} tensors)")

    if args.dtype_compare:
        # Run BOTH dtypes and compare
        final_f32, per_f32 = run_sampling(torch.float32, "f32")
        final_bf16, per_bf16 = run_sampling(torch.bfloat16, "bf16")

        save_file({"output": final_f32, **per_f32}, str(target / "full_e2e_output.safetensors"))
        save_file({"output": final_bf16, **per_bf16}, str(target / "full_e2e_output_bf16.safetensors"))

        diff = (final_f32 - final_bf16).abs()
        print("\n=== Python dtype comparison (f32 vs bf16) ===")
        print(f"  f32  final: min={final_f32.min().item():.6f}  max={final_f32.max().item():.6f}  mean={final_f32.mean().item():.6f}")
        print(f"  bf16 final: min={final_bf16.min().item():.6f}  max={final_bf16.max().item():.6f}  mean={final_bf16.mean().item():.6f}")
        print(f"  Python f32 vs bf16:  max_abs={diff.max().item():.3e}  mean_abs={diff.mean().item():.3e}")
        print("  (Context: Rust f32 vs Python f32 = max_abs=3.75e-5)")
        print("  If Python f32 vs bf16 ≫ 3.75e-5, bf16 audio differences are dtype-induced, not bugs.")

        print(f"\nSaved f32  outputs → target/full_e2e_output.safetensors  ({len(per_f32)+1} tensors)")
        print(f"Saved bf16 outputs → target/full_e2e_output_bf16.safetensors  ({len(per_bf16)+1} tensors)")

    elif args.bf16:
        final, per_step = run_sampling(torch.bfloat16, "bf16")
        outputs = {"output": final, **per_step}
        save_file(outputs, str(target / "full_e2e_output_bf16.safetensors"))
        print(
            f"\nFinal output (bf16): shape={tuple(final.shape)}  "
            f"min={final.min().item():.6f}  max={final.max().item():.6f}  "
            f"mean={final.mean().item():.6f}"
        )
        print(f"Saved outputs → target/full_e2e_output_bf16.safetensors  ({len(outputs)} tensors)")
        # Compare against saved f32 if it exists
        f32_path = target / "full_e2e_output.safetensors"
        if f32_path.exists():
            f32_out = load_file(str(f32_path))
            diff = (f32_out["output"] - final).abs()
            print(f"\n  Python f32 vs bf16:  max_abs={diff.max().item():.3e}  mean_abs={diff.mean().item():.3e}")

    else:
        final, per_step = run_sampling(torch.float32, "f32")
        outputs = {"output": final, **per_step}
        save_file(outputs, str(target / "full_e2e_output.safetensors"))
        print(
            f"\nFinal output (f32): shape={tuple(final.shape)}  "
            f"min={final.min().item():.6f}  max={final.max().item():.6f}  "
            f"mean={final.mean().item():.6f}"
        )
        print(f"Saved outputs → target/full_e2e_output.safetensors  ({len(outputs)} tensors)")


if __name__ == "__main__":
    main()
