#!/usr/bin/env python3
"""Export a fixed initial noise tensor from Python (for same-seed Rust comparison).

Usage (from repo root with Python venv active):
    python scripts/export_initial_noise.py --seed 42 --out target/initial_noise.safetensors

The exported tensor has shape [1, 750, 32] (batch=1, seq=750, patched_dim=32),
matching the default pipeline parameters.  Pass `--seq-len` / `--latent-dim` to
override.  The tensor can then be loaded into the Rust pipeline via the
`initial_noise` field of `SamplingRequest`.

Also runs the full Python inference with this seed and saves output.wav so the
results can be compared against the Rust pipeline run with the same noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Guard: must be run from within the Irodori-TTS Python venv
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    import safetensors.torch as st
except ImportError as exc:
    print(f"[error] Required import failed: {exc}", file=sys.stderr)
    print(
        "[error] Run this script from the Irodori-TTS Python venv:\n"
        "  cd ../Irodori-TTS && uv run python "
        "../Irodori-TTS-burn/scripts/export_initial_noise.py [args]",
        file=sys.stderr,
    )
    sys.exit(1)


def make_initial_noise(
    seed: int,
    batch: int,
    seq_len: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reproduce the exact initial noise torch.Generator produces in rf.py."""
    try:
        rng = torch.Generator(device=device).manual_seed(seed)
        return torch.randn((batch, seq_len, latent_dim), device=device, dtype=dtype, generator=rng)
    except RuntimeError:
        # CUDA generators may fail; fall back to CPU
        rng = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn((batch, seq_len, latent_dim), device="cpu", dtype=dtype, generator=rng).to(device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("target/initial_noise.safetensors"),
        help="Output path for the initial noise tensor (default: target/initial_noise.safetensors)",
    )
    parser.add_argument("--seq-len", type=int, default=750, help="Sequence length (default: 750)")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dim (default: 32)")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (default: cpu)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] seed={args.seed}  seq_len={args.seq_len}  latent_dim={args.latent_dim}")
    x_t = make_initial_noise(args.seed, 1, args.seq_len, args.latent_dim, device)
    print(f"[export] tensor shape: {list(x_t.shape)}")
    print(f"[export] tensor stats: min={x_t.min():.4f}  max={x_t.max():.4f}  std={x_t.std():.4f}")

    st.save_file({"initial_noise": x_t.cpu()}, str(out_path))
    print(f"[export] saved → {out_path}")
    print()
    print("Now run Rust with this noise:")
    print(
        f"  just pipeline-real --seed-from-file {out_path} --num-steps 40 "
        f"--text 'こんにちは、テストです。' --output target/rust_output.wav"
    )
    print()
    print("And compare against Python:")
    print(
        f"  cd ../Irodori-TTS && uv run python infer.py "
        f"--text 'こんにちは、テストです。' --seed {args.seed} "
        f"--num-steps 40 --output target/python_output.wav "
        f"--checkpoint ../Irodori-TTS-burn/target/model_converted.safetensors"
    )


if __name__ == "__main__":
    main()
