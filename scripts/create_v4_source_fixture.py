# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy==2.2.6",
#   "safetensors==0.7.0",
# ]
# ///
"""Create a fresh canonical FP32 noise source for a benchmark campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

UPSTREAM_COMMIT = "9f19d9a9048099a4b978a762d0509228fe624e3f"
MODEL_SHA256 = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593"
CODEC_SHA256 = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"
SHAPE = (1, 50, 32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)

    generator = np.random.Generator(np.random.PCG64(0))
    noise = generator.standard_normal(SHAPE, dtype=np.float32)
    if not noise.flags.c_contiguous or not np.isfinite(noise).all():
        raise RuntimeError("generated source noise is invalid")
    tensor_sha256 = hashlib.sha256(noise.tobytes(order="C")).hexdigest()
    metadata = {
        "format": "irodori-v4-e2e-oracle-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "model_sha256": MODEL_SHA256,
        "codec_sha256": CODEC_SHA256,
        "noise_generation": {
            "algorithm": "numpy.random.PCG64",
            "numpy": np.__version__,
            "seed": 0,
        },
        "tensor_manifest": {
            "initial_noise": {
                "shape": list(SHAPE),
                "dtype": "float32",
                "elements": int(noise.size),
                "bytes": int(noise.nbytes),
                "sha256": tensor_sha256,
            }
        },
    }
    save_file(
        {"initial_noise": noise},
        str(args.output),
        metadata={"oracle_json": json.dumps(metadata, sort_keys=True)},
    )
    with args.output.open("rb") as source:
        os.fsync(source.fileno())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "tensor_sha256": tensor_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
