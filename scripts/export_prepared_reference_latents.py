#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "safetensors==0.7.0",
#   "torch==2.10.0",
# ]
# ///
"""Convert freshly prepared PyTorch reference latents to pinned safetensors.

The input files are outputs from ``bench_python_runtime_scenarios.py``.  This
converter does not encode audio or alter values; it makes the already prepared
latents readable by the Rust/WGPU measurement harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.numpy import save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest: list[dict[str, object]] = []
    for index, source in enumerate(args.input, start=1):
        if not source.is_file():
            raise SystemExit(f"input does not exist: {source}")
        value = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise SystemExit(f"input is not a tensor: {source}")
        value = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if value.ndim != 2 or value.shape[1] != 32:
            raise SystemExit(
                f"expected [frames, 32], got {tuple(value.shape)}: {source}"
            )
        array = value.unsqueeze(0).numpy()
        destination = args.output_dir / f"ref{index}.safetensors"
        save_file(
            {"latent": array},
            destination,
            metadata={
                "format": "irodori-prepared-reference-latent-v1",
                "source_sha256": sha256_file(source),
            },
        )
        manifest.append(
            {
                "source": str(source.resolve()),
                "source_sha256": sha256_file(source),
                "output": str(destination.resolve()),
                "output_sha256": sha256_file(destination),
                "shape": list(array.shape),
                "dtype": "float32",
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"format": "irodori-prepared-reference-export-v1", "items": manifest},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
