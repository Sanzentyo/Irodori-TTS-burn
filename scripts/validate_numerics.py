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
Numerical validation fixture generator for the Rust/burn Irodori-TTS port.

Generates two sets of fixtures in target/:

  Speaker-conditioned (use_caption_condition=False):
    validate_weights.safetensors  – model weights + config_json metadata
    validate_tensors.safetensors  – all inputs / intermediate / output tensors

  Caption-conditioned (use_caption_condition=True):
    validate_caption_weights.safetensors  – caption model weights + config_json
    validate_caption_tensors.safetensors  – caption inputs / outputs

The Rust `validate` binary reads these and asserts max-abs-diff < 1e-3.

Usage:
    uv run scripts/validate_numerics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add the repo root to sys.path so we can import the Python reference
REPO_ROOT = Path(__file__).parent.parent
IRODORI_ROOT = REPO_ROOT.parent / "Irodori-TTS"
sys.path.insert(0, str(IRODORI_ROOT))

import torch
from safetensors.torch import save_file

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT

# ---------------------------------------------------------------------------
# Small fixture configuration — speaker-conditioned
# ---------------------------------------------------------------------------
# Constraints:
#   model_dim % num_heads == 0  →  64 / 8 = 8  (head_dim, must be even ✓)
#   text_dim  % text_heads == 0 →  64 / 8 = 8  (head_dim, even ✓)
#   adaln_rank < model_dim
CFG = ModelConfig(
    latent_dim=8,
    latent_patch_size=1,
    model_dim=64,
    num_layers=2,
    num_heads=8,
    mlp_ratio=2.0,
    text_mlp_ratio=None,     # uses mlp_ratio
    speaker_mlp_ratio=None,  # uses mlp_ratio
    dropout=0.0,
    text_vocab_size=256,
    text_tokenizer_repo="test",
    text_add_bos=False,
    text_dim=64,
    text_layers=2,
    text_heads=8,
    use_caption_condition=False,  # → speaker conditioning active
    speaker_dim=64,
    speaker_layers=2,
    speaker_heads=8,
    speaker_patch_size=1,
    timestep_embed_dim=64,
    adaln_rank=8,
    norm_eps=1e-5,
)

# ---------------------------------------------------------------------------
# Small fixture configuration — caption-conditioned (voice-design path)
# ---------------------------------------------------------------------------
CFG_CAPTION = ModelConfig(
    latent_dim=8,
    latent_patch_size=1,
    model_dim=64,
    num_layers=2,
    num_heads=8,
    mlp_ratio=2.0,
    text_mlp_ratio=None,
    dropout=0.0,
    text_vocab_size=256,
    text_tokenizer_repo="test",
    text_add_bos=False,
    text_dim=64,
    text_layers=2,
    text_heads=8,
    use_caption_condition=True,   # → caption conditioning active, speaker disabled
    # caption encoder defaults to text dims when not specified
    timestep_embed_dim=64,
    adaln_rank=8,
    norm_eps=1e-5,
)

# ---------------------------------------------------------------------------
# Fixed inputs (shared where compatible)
# ---------------------------------------------------------------------------
BATCH = 1
SEQ_TEXT = 4
SEQ_LAT = 8
SEQ_REF = 4
SEQ_CAPTION = 5

text_ids   = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)        # [1, 4]
text_mask  = torch.ones(BATCH, SEQ_TEXT, dtype=torch.bool)          # [1, 4]
x_t        = torch.arange(BATCH * SEQ_LAT * CFG.latent_dim,
                           dtype=torch.float32).reshape(BATCH, SEQ_LAT, CFG.latent_dim) * 0.01
t          = torch.tensor([0.5], dtype=torch.float32)               # [1]
ref_latent = torch.arange(BATCH * SEQ_REF * CFG.latent_dim,
                           dtype=torch.float32).reshape(BATCH, SEQ_REF, CFG.latent_dim) * 0.01
ref_mask   = torch.ones(BATCH, SEQ_REF, dtype=torch.bool)           # [1, 4]

caption_ids  = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)  # [1, 5]
caption_mask = torch.ones(BATCH, SEQ_CAPTION, dtype=torch.bool)         # [1, 5]


def _rename_cond_module(state_dict: dict) -> dict:
    """Rename cond_module.0/2/4.weight → cond_module.linear0/1/2.weight."""
    renames = {
        "cond_module.0.weight": "cond_module.linear0.weight",
        "cond_module.2.weight": "cond_module.linear1.weight",
        "cond_module.4.weight": "cond_module.linear2.weight",
    }
    return {renames.get(k, k): v for k, v in state_dict.items()}


def generate_speaker_fixture(target: Path) -> None:
    """Generate speaker-conditioned validation fixtures."""
    print("=== Speaker-conditioned fixture ===")
    torch.manual_seed(42)

    model = TextToLatentRFDiT(CFG)
    model.eval()

    config_json = json.dumps(
        {k: v for k, v in CFG.__dict__.items()},
        default=lambda o: o if not callable(o) else None,
    )

    # --- Register hooks to capture per-block outputs and in_proj output ---
    block_outputs: list[torch.Tensor] = []
    in_proj_output: list[torch.Tensor] = []

    hook_handles = []

    def make_block_hook(idx: int):
        def hook(module, input, output):
            block_outputs.append(output.detach().float().cpu().contiguous())
        return hook

    for i, block in enumerate(model.blocks):
        handle = block.register_forward_hook(make_block_hook(i))
        hook_handles.append(handle)

    def in_proj_hook(module, input, output):
        in_proj_output.append(output.detach().float().cpu().contiguous())

    hook_handles.append(model.in_proj.register_forward_hook(in_proj_hook))

    with torch.no_grad():
        (
            text_state,
            _text_mask_out,
            speaker_state,
            speaker_mask_out,
            _caption_state,
            _caption_mask,
        ) = model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
        )

        assert speaker_state is not None, "expected speaker_state"
        assert speaker_mask_out is not None, "expected speaker_mask_out"

        v_pred = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask_out,
        )

    # Remove hooks to avoid side-effects
    for h in hook_handles:
        h.remove()

    assert len(block_outputs) == CFG.num_layers, (
        f"Expected {CFG.num_layers} block outputs, got {len(block_outputs)}"
    )
    assert len(in_proj_output) == 1, f"Expected 1 in_proj output, got {len(in_proj_output)}"

    print(f"text_state     shape={tuple(text_state.shape)}  "
          f"min={text_state.min().item():.6f}  max={text_state.max().item():.6f}")
    print(f"speaker_state  shape={tuple(speaker_state.shape)}  "
          f"min={speaker_state.min().item():.6f}  max={speaker_state.max().item():.6f}")
    print(f"after_in_proj  shape={tuple(in_proj_output[0].shape)}")
    for i, bo in enumerate(block_outputs):
        print(f"block_{i}_out   shape={tuple(bo.shape)}  "
              f"min={bo.min().item():.6f}  max={bo.max().item():.6f}")
    print(f"v_pred         shape={tuple(v_pred.shape)}  "
          f"min={v_pred.min().item():.6f}  max={v_pred.max().item():.6f}")

    state_dict = _rename_cond_module(
        {k: v.float().contiguous() for k, v in model.state_dict().items()}
    )
    weights_path = target / "validate_weights.safetensors"
    save_file(state_dict, str(weights_path), metadata={"config_json": config_json})
    print(f"Weights  → {weights_path}  ({len(state_dict)} tensors)")

    tensors_path = target / "validate_tensors.safetensors"
    ref_tensors: dict[str, torch.Tensor] = {
        "text_ids":      text_ids.float(),
        "text_mask":     text_mask.float(),
        "x_t":           x_t,
        "t":             t,
        "ref_latent":    ref_latent,
        "ref_mask":      ref_mask.float(),
        "text_state":    text_state.float(),
        "speaker_state": speaker_state.float(),
        "speaker_mask":  speaker_mask_out.float(),
        "v_pred":        v_pred.float(),
        # Per-layer intermediates for layer-by-layer comparison
        "after_in_proj": in_proj_output[0],
        **{f"block_{i}_out": bo for i, bo in enumerate(block_outputs)},
    }
    save_file(ref_tensors, str(tensors_path))
    print(f"Tensors  → {tensors_path}  ({len(ref_tensors)} tensors, "
          f"including {len(block_outputs)} per-block outputs)\n")


def generate_caption_fixture(target: Path) -> None:
    """Generate caption-conditioned validation fixtures (voice-design path)."""
    print("=== Caption-conditioned fixture ===")
    torch.manual_seed(99)

    model = TextToLatentRFDiT(CFG_CAPTION)
    model.eval()

    config_json = json.dumps(
        {k: v for k, v in CFG_CAPTION.__dict__.items()},
        default=lambda o: o if not callable(o) else None,
    )

    with torch.no_grad():
        (
            text_state,
            _text_mask_out,
            _ref_state,
            _ref_mask,
            caption_state_out,
            _caption_mask_out,
        ) = model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
        )

        assert caption_state_out is not None, "expected caption_state"

        v_pred = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=None,
            speaker_mask=None,
            caption_state=caption_state_out,
            caption_mask=caption_mask,
        )

    print(f"text_state     shape={tuple(text_state.shape)}  "
          f"min={text_state.min().item():.6f}  max={text_state.max().item():.6f}")
    print(f"caption_state  shape={tuple(caption_state_out.shape)}  "
          f"min={caption_state_out.min().item():.6f}  max={caption_state_out.max().item():.6f}")
    print(f"v_pred         shape={tuple(v_pred.shape)}  "
          f"min={v_pred.min().item():.6f}  max={v_pred.max().item():.6f}")

    state_dict = _rename_cond_module(
        {k: v.float().contiguous() for k, v in model.state_dict().items()}
    )
    weights_path = target / "validate_caption_weights.safetensors"
    save_file(state_dict, str(weights_path), metadata={"config_json": config_json})
    print(f"Weights  → {weights_path}  ({len(state_dict)} tensors)")

    tensors_path = target / "validate_caption_tensors.safetensors"
    ref_tensors: dict[str, torch.Tensor] = {
        "text_ids":       text_ids.float(),
        "text_mask":      text_mask.float(),
        "caption_ids":    caption_ids.float(),
        "caption_mask":   caption_mask.float(),
        "x_t":            x_t,
        "t":              t,
        "text_state":     text_state.float(),
        "caption_state":  caption_state_out.float(),
        "v_pred":         v_pred.float(),
    }
    save_file(ref_tensors, str(tensors_path))
    print(f"Tensors  → {tensors_path}  ({len(ref_tensors)} tensors)\n")


def main() -> None:
    target = REPO_ROOT / "target"
    target.mkdir(exist_ok=True)

    generate_speaker_fixture(target)
    generate_caption_fixture(target)


if __name__ == "__main__":
    main()
