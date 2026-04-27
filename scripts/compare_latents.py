#!/usr/bin/env python3
"""Compare latent safetensors files produced by different samplers.

Usage (from repo root with Python venv active or uv run):
    python scripts/compare_latents.py --ref <ref.safetensors> <cand1.safetensors> [cand2 ...]

Each file must contain a tensor named "latent" with shape [1, T, D].

Metrics reported vs the reference:
  MAE     — mean absolute error
  RMSE    — root mean squared error
  MaxAE   — max absolute error
  CosSim  — mean cosine similarity across the time axis
  PSNR    — peak signal-to-noise ratio (higher is better, dB)

Rubber-duck guidance: latent-space comparison is more reliable than raw WAV
comparison because WAV comparison is confounded by DACVAE reconstruction noise
and tail-trimming jitter.  All candidates should share the same initial noise
(use --noise-file when generating latents).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import safetensors.torch as st
    import torch
except ImportError as exc:
    print(f"[error] Required import failed: {exc}", file=sys.stderr)
    print("[error] Run inside Irodori-TTS venv or use: uv run python scripts/compare_latents.py",
          file=sys.stderr)
    sys.exit(1)


def load_latent(path: Path) -> torch.Tensor:
    tensors = st.load_file(str(path))
    if "latent" not in tensors:
        raise ValueError(f"'latent' key not found in {path}; keys: {list(tensors.keys())}")
    t = tensors["latent"].float()  # [1, T, D] or [T, D]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t  # [1, T, D]


def cosine_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity along the time axis.  a/b: [1, T, D]."""
    a = a.squeeze(0)  # [T, D]
    b = b.squeeze(0)
    sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [T]
    return sim.mean().item()


def psnr(ref: torch.Tensor, cand: torch.Tensor, max_val: float | None = None) -> float:
    """Peak signal-to-noise ratio in dB."""
    if max_val is None:
        max_val = max(ref.abs().max().item(), 1e-8)
    mse = ((ref - cand) ** 2).mean().item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


def compare(ref: torch.Tensor, cand: torch.Tensor, label: str, ref_maxval: float) -> None:
    diff = (ref - cand).abs()
    mae  = diff.mean().item()
    rmse = ((ref - cand) ** 2).mean().sqrt().item()
    maxae = diff.max().item()
    cos  = cosine_sim_mean(ref, cand)
    psnr_val = psnr(ref, cand, max_val=ref_maxval)

    name = label[-50:] if len(label) > 50 else label
    print(f"  {name:<50}  MAE={mae:.5f}  RMSE={rmse:.5f}  MaxAE={maxae:.5f}  "
          f"CosSim={cos:.6f}  PSNR={psnr_val:6.2f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, type=Path,
                        help="Reference latent safetensors file (e.g. euler40.safetensors)")
    parser.add_argument("candidates", nargs="+", type=Path,
                        help="Candidate latent files to compare against reference")
    args = parser.parse_args()

    ref = load_latent(args.ref)
    ref_maxval = ref.abs().max().item()

    print(f"\nReference : {args.ref.name}  shape={list(ref.shape)}"
          f"  max_abs={ref_maxval:.4f}")
    print(f"{'Candidate':<52}  {'MAE':>8}  {'RMSE':>8}  {'MaxAE':>8}  "
          f"{'CosSim':>9}  {'PSNR':>8}")
    print("-" * 110)

    any_fail = False
    for cand_path in args.candidates:
        cand = load_latent(cand_path)
        if cand.shape != ref.shape:
            print(f"  [SKIP] {cand_path.name}: shape mismatch {list(cand.shape)} vs {list(ref.shape)}")
            any_fail = True
            continue
        compare(ref, cand, cand_path.name, ref_maxval)

    print()
    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
