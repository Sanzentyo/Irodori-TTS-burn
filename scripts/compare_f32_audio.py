#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compare two owned contiguous raw little-endian f32 audio artifacts."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--max-rmse", type=float)
    parser.add_argument("--min-snr-db", type=float)
    parser.add_argument("--min-cosine", type=float)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_f32le(path: Path) -> array.array[float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    byte_len = path.stat().st_size
    if byte_len == 0 or byte_len % 4 != 0:
        raise ValueError(f"{path} must contain a non-empty whole number of f32 values")
    values = array.array("f")
    with path.open("rb") as file:
        values.fromfile(file, byte_len // 4)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path} contains NaN or infinity")
    return values


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    reference = read_f32le(args.reference)
    candidate = read_f32le(args.candidate)
    if len(reference) != len(candidate):
        raise ValueError(
            f"sample count mismatch: reference={len(reference)}, candidate={len(candidate)}"
        )

    max_abs = 0.0
    abs_sum = 0.0
    square_error_sum = 0.0
    reference_square_sum = 0.0
    candidate_square_sum = 0.0
    dot_sum = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        error = float(actual) - float(expected)
        absolute = abs(error)
        max_abs = max(max_abs, absolute)
        abs_sum += absolute
        square_error_sum += error * error
        reference_square_sum += float(expected) * float(expected)
        candidate_square_sum += float(actual) * float(actual)
        dot_sum += float(expected) * float(actual)

    count = len(reference)
    rmse = math.sqrt(square_error_sum / count)
    reference_rms = math.sqrt(reference_square_sum / count)
    snr_db = (
        math.inf
        if rmse == 0.0
        else 20.0 * math.log10(max(reference_rms, sys.float_info.min) / rmse)
    )
    denominator = math.sqrt(reference_square_sum * candidate_square_sum)
    cosine = (
        1.0 if denominator == 0.0 and square_error_sum == 0.0 else dot_sum / denominator
    )
    gates = {
        "max_abs": args.max_abs is None or max_abs <= args.max_abs,
        "rmse": args.max_rmse is None or rmse <= args.max_rmse,
        "snr_db": args.min_snr_db is None or snr_db >= args.min_snr_db,
        "cosine": args.min_cosine is None or cosine >= args.min_cosine,
    }
    payload = {
        "format": "irodori-raw-f32-audio-comparison-v1",
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256_file(args.candidate),
        },
        "samples": count,
        "metrics": {
            "max_abs": max_abs,
            "mean_abs": abs_sum / count,
            "rmse": rmse,
            "reference_rms": reference_rms,
            "snr_db": snr_db,
            "cosine": cosine,
        },
        "thresholds": {
            "max_abs": args.max_abs,
            "max_rmse": args.max_rmse,
            "min_snr_db": args.min_snr_db,
            "min_cosine": args.min_cosine,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
