# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.10.0", "torchaudio==2.10.0", "transformers==5.12.1", "safetensors", "numpy", "irodori-tts"]
# [tool.uv.sources]
# irodori-tts = { git = "https://github.com/Aratako/Irodori-TTS", rev = "89f9d8fbd4d51ea019867ee1197725ede1df13c5" }
# ///
"""Compare official ModernBERT boundaries with validate_synthesis --dump-text-layers.

Inputs are the exact compacted Rust IDs/mask. Hooks observe existing operators;
they do not implement a substitute attention or normalization algorithm.
"""

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import (
    _load_checkpoint_for_inference,
    _move_inference_module,
)
from irodori_tts.model import TextToLatentRFDiT
from transformers.models.modernbert import modeling_modernbert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "xpu", "cpu"], required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--sdpa", choices=["auto", "math"], default="auto")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    trace = json.loads(args.trace.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    state, config, _, encoder_config = _load_checkpoint_for_inference(args.checkpoint)
    config = merge_dataclass_overrides(
        ModelConfig(), config, section="checkpoint model_config"
    )
    model = TextToLatentRFDiT(
        config,
        pretrained_backbone_config=encoder_config,
        load_pretrained_backbone_weights=False,
    )
    model.load_state_dict(state)
    del state
    dtype = torch.float32 if args.precision == "fp32" else torch.float16
    model = _move_inference_module(
        model, device=torch.device(args.device), dtype=dtype
    ).eval()
    ids = torch.tensor(
        trace["input_ids"], dtype=torch.long, device=args.device
    ).reshape(trace["input_shape"])
    mask = torch.tensor(
        trace["attention_mask"], dtype=torch.bool, device=args.device
    ).reshape(trace["input_shape"])
    backbone = model.pretrained_text_backbone.backbone
    values = {}
    handles = []

    def record(name, value):
        if name in values:
            raise ValueError(f"duplicate boundary {name}")
        values[name] = value.detach().float().cpu().numpy().copy()

    def hook(module, name, before=False):
        if before:
            handles.append(
                module.register_forward_pre_hook(
                    lambda _, inputs: record(name, inputs[0])
                )
            )
        else:
            handles.append(
                module.register_forward_hook(
                    lambda _, inputs, output: record(name, output)
                )
            )

    hook(backbone.embeddings.tok_embeddings, "embedding")
    hook(backbone.embeddings.norm, "embedding_norm")
    active_layer = [None]
    for index, layer in enumerate(backbone.layers):
        prefix = f"layer_{index:02}_"
        handles.append(
            layer.attn.register_forward_pre_hook(
                lambda _, inputs, index=index: active_layer.__setitem__(0, index)
            )
        )
        hook(layer.attn_norm, prefix + "attention_norm")
        hook(layer.attn.Wqkv, prefix + "qkv")
        hook(layer.attn.Wo, prefix + "sdpa", before=True)
        hook(layer.attn.Wo, prefix + "attention_output")
        hook(layer.mlp_norm, prefix + "attention_residual", before=True)
        hook(layer.mlp_norm, prefix + "mlp_norm")
        hook(layer.mlp.Wi, prefix + "mlp_projection")
        hook(layer.mlp.act, prefix + "mlp_gelu")
        hook(layer.mlp.Wo, prefix + "mlp_gated", before=True)
        hook(layer.mlp.Wo, prefix + "mlp_output")
        hook(layer, prefix + "layer_output")
    hook(backbone.final_norm, "final_norm")
    hook(model.pretrained_text_backbone, "backbone")
    hook(model.text_encoder, "projected")
    hook(model.text_norm, "text_state")
    original_rope = modeling_modernbert.apply_rotary_pos_emb

    def rope(*args, **kwargs):
        q, k = original_rope(*args, **kwargs)
        record(f"layer_{active_layer[0]:02}_rope_q", q.transpose(1, 2))
        record(f"layer_{active_layer[0]:02}_rope_k", k.transpose(1, 2))
        return q, k

    def attention_context():
        return (
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
            if args.sdpa == "math"
            else nullcontext()
        )

    with (
        torch.inference_mode(),
        attention_context(),
        patch.object(modeling_modernbert, "apply_rotary_pos_emb", rope),
    ):
        observed = model.text_norm(
            model.text_encoder(model.pretrained_text_backbone, ids, mask)
        )
    for handle in handles:
        handle.remove()
    with torch.inference_mode(), attention_context():
        plain = model.text_norm(
            model.text_encoder(model.pretrained_text_backbone, ids, mask)
        )
    if not torch.equal(observed, plain):
        raise ValueError("observations changed the official output")
    comparisons = []
    for entry in trace["tensors"]:
        name = entry["name"]
        expected = values.pop(name)
        if list(expected.shape) != entry["shape"]:
            raise ValueError(f"shape mismatch {name}")
        raw = (args.trace.parent / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Rust trace checksum mismatch {name}")
        actual = np.frombuffer(raw, dtype="<f4").reshape(expected.shape)
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"nonfinite {name}")
        rmse = float(np.sqrt(np.mean(difference**2)))
        signal = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
        filename = name + ".f32le"
        expected.astype("<f4").tofile(args.output / filename)
        comparisons.append(
            {
                "name": name,
                "shape": entry["shape"],
                "max_abs": float(np.abs(difference).max()),
                "rmse": rmse,
                "snr_db": float(20 * np.log10(signal / rmse))
                if rmse and signal
                else None,
                "bitwise_equal": bool(
                    np.array_equal(actual.view("u4"), expected.view("u4"))
                ),
                "file": filename,
            }
        )
    if values:
        raise ValueError(f"unmatched official boundaries {list(values)}")
    result = {
        "device": args.device,
        "precision": args.precision,
        "sdpa": args.sdpa,
        "torch_version": torch.__version__,
        "input_shape": trace["input_shape"],
        "observed_output_bitwise_equal": True,
        "comparisons": comparisons,
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "device": args.device,
                "precision": args.precision,
                "final": comparisons[-1],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
