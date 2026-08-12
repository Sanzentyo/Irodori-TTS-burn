#!/usr/bin/env python3
"""
Convert DACVAE weights.pth → dacvae_weights.safetensors.

Resolves PyTorch weight_norm parametrisation (weight_g + weight_v → weight)
so the Rust model can load plain convolution weights without handling the
parametrisation itself.

Usage:
    uv run scripts/convert_dacvae_weights.py
    uv run scripts/convert_dacvae_weights.py --output /path/to/out.safetensors
    uv run scripts/convert_dacvae_weights.py --pth /path/to/weights.pth --output out.safetensors
    uv run scripts/convert_dacvae_weights.py --profile decoder-only --output decoder.safetensors
"""
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "torch==2.10.0",
#   "safetensors==0.7.0",
#   "huggingface-hub==1.23.0",
#   "numpy==2.2.6",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Weight-norm resolution
# ---------------------------------------------------------------------------


def resolve_weight_norm(state_dict: dict) -> dict:
    """Replace every (X.weight_g, X.weight_v) pair with X.weight.

    PyTorch weight_norm stores:
      weight_g : [out, 1, 1]   — per-output-channel magnitude
      weight_v : [out, in, k]  — direction vector
    and computes:
      weight   = g * v / ||v||
    where ||v|| is the L2 norm over all dimensions except dim-0.
    """
    out = {}
    skip = set()

    # First pass: collect all weight_g keys and resolve the pair.
    for key, val in state_dict.items():
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key in state_dict:
                g = val  # [out, 1, 1]
                v = state_dict[v_key]  # [out, in, k]
                # norm over all dims except dim-0
                v_flat = v.reshape(v.shape[0], -1)
                norm = v_flat.norm(dim=1, keepdim=True).reshape(g.shape)
                weight = g * v / (norm + 1e-12)
                out[base + ".weight"] = weight.contiguous()
                skip.add(key)
                skip.add(v_key)

    # Second pass: copy everything else unchanged.
    for key, val in state_dict.items():
        if key not in skip:
            out[key] = val.contiguous()

    return out


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def build_metadata(original_metadata: dict, profile: str) -> dict:
    """Preserve original model kwargs as JSON in safetensors metadata."""
    metadata = {
        "config_json": json.dumps(original_metadata.get("kwargs", {})),
        "source": "convert_dacvae_weights.py",
    }
    # Keep the released full-checkpoint byte representation stable. Absence of
    # this key means `full`; only the reduced artifact needs an explicit guard.
    if profile != "full":
        metadata["irodori_codec_profile"] = profile
    return metadata


def select_profile(resolved: dict, profile: str) -> dict:
    """Select exactly the tensors required by the requested Rust codec API."""
    if profile == "full":
        return resolved
    if profile == "decoder-only":
        selected = {
            key: value
            for key, value in resolved.items()
            if key.startswith(("decoder.", "quantizer.out_proj."))
        }
        required_prefixes = ("decoder.", "quantizer.out_proj.")
        missing = [
            prefix
            for prefix in required_prefixes
            if not any(key.startswith(prefix) for key in selected)
        ]
        if missing:
            raise ValueError(
                f"decoder-only profile is missing required prefixes: {missing}"
            )
        return selected
    raise ValueError(f"unsupported codec profile: {profile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pth",
        default=None,
        help="Path to weights.pth (defaults to HF cache).",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parent.parent / "target" / "dacvae_weights.safetensors"
        ),
        help="Output safetensors path.",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "decoder-only"),
        default="full",
        help="Checkpoint contents. decoder-only excludes encoder-side tensors.",
    )
    return parser.parse_args()


def resolve_pth_path(pth_arg: str | None) -> str:
    if pth_arg is not None:
        return pth_arg

    # Try HF hub cache first.
    try:
        sys.path.insert(
            0,
            str(
                Path(__file__).parent.parent
                / ".venv"
                / "lib"
                / "python3.10"
                / "site-packages"
            ),
        )
        from huggingface_hub import hf_hub_download  # type: ignore

        return hf_hub_download(
            repo_id="Aratako/Semantic-DACVAE-Japanese-32dim", filename="weights.pth"
        )
    except Exception as exc:  # noqa: BLE001 - report optional resolver failures uniformly.
        sys.exit(f"Could not resolve weights.pth; pass --pth explicitly. Error: {exc}")


def main():
    if (
        os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("MKL_NUM_THREADS") != "1"
    ):
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        os.execvpe(
            "uv",
            ["uv", "run", "--script", str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )

    import torch

    # Parallel reductions may choose a different floating-point reduction tree
    # and produce a different byte-level weight-norm result. Keep conversion
    # reproducible across hosts and repeated clean builds.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    args = parse_args()
    pth_path = resolve_pth_path(args.pth)

    print(f"Loading {pth_path} ...", flush=True)
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        sys.exit("Unexpected checkpoint format: missing 'state_dict' key.")

    state_dict: dict = checkpoint["state_dict"]
    metadata: dict = checkpoint.get("metadata", {})

    print(f"  {len(state_dict)} tensors in state_dict", flush=True)

    # Resolve weight_norm.
    resolved = resolve_weight_norm(state_dict)

    wn_pairs_before = sum(1 for k in state_dict if k.endswith(".weight_g"))
    wn_pairs_after = sum(1 for k in resolved if k.endswith(".weight_g"))
    print(
        f"  Resolved {wn_pairs_before} weight_norm pairs → {len(resolved)} tensors total",
        flush=True,
    )
    assert wn_pairs_after == 0, "Some weight_g keys remain after resolution!"

    selected = select_profile(resolved, args.profile)

    # Write safetensors.
    try:
        from safetensors.torch import save_file  # type: ignore
    except ImportError:
        sys.exit("safetensors not installed; run: pip install safetensors")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = build_metadata(metadata, args.profile)
    save_file(selected, str(out_path), metadata=meta)

    size_mb = out_path.stat().st_size / 1_048_576
    print(
        f"Saved {len(selected)} tensors ({args.profile}) → {out_path}  "
        f"({size_mb:.1f} MB)",
        flush=True,
    )

    # Quick sanity check: no weight_g/v keys should remain.
    from safetensors import safe_open  # type: ignore

    with safe_open(str(out_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    bad = [k for k in keys if ".weight_g" in k or ".weight_v" in k]
    if bad:
        sys.exit(
            f"Sanity FAIL: {len(bad)} weight_g/v keys remain in output!\n  {bad[:5]}"
        )
    print(f"Sanity OK: {len(keys)} keys, no weight_g/v remaining.")


if __name__ == "__main__":
    main()
