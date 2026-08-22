# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy==2.2.6",
#   "safetensors==0.7.0",
# ]
# ///
"""Compare semantically compacted Python and production-WGPU conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-artifact", type=Path, required=True)
    parser.add_argument("--wgpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(state: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if state.ndim != 3 or mask.shape != state.shape[:2]:
        raise ValueError(
            f"condition state/mask shape mismatch: {state.shape}/{mask.shape}"
        )
    valid = mask > 0.5
    columns = np.flatnonzero(valid.any(axis=0))
    if columns.size == 0:
        return None
    return state[:, : int(columns[-1]) + 1, :]


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.astype(np.float64, copy=False).ravel()
    got = candidate.astype(np.float64, copy=False).ravel()
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("non-finite condition tensor")
    error = got - ref
    signal_power = float(np.mean(ref * ref))
    noise_power = float(np.mean(error * error))
    dot = float(np.dot(ref, got))
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(got))
    return {
        "shape": list(reference.shape),
        "elements": int(reference.size),
        "max_abs": float(np.max(np.abs(error))),
        "rmse": math.sqrt(noise_power),
        "snr_db": (
            math.inf
            if noise_power == 0.0
            else 10.0 * math.log10(signal_power / noise_power)
        ),
        "cosine": 1.0 if denominator == 0.0 and dot == 0.0 else dot / denominator,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    python_tensors = load_file(args.python_artifact)
    with args.wgpu_report.open(encoding="utf-8") as file:
        report = json.load(file)
    if report.get("latency_results_valid") is not False:
        raise ValueError("WGPU source-condition report is not diagnostic-only")
    artifacts = {
        artifact["name"]: artifact
        for artifact in report["diagnostic_artifacts"]["tensors"]
    }

    comparisons: dict[str, Any] = {}
    for context in ("text", "speaker", "caption"):
        state_name = f"rf_selected_{context}_state"
        mask_name = f"rf_selected_{context}_mask"
        state = python_tensors.get(state_name)
        mask = python_tensors.get(mask_name)
        if (state is None) != (mask is None):
            raise ValueError(f"Python {context} state/mask presence mismatch")
        compacted = None if state is None else compact(state, mask)
        artifact = artifacts.get(state_name)
        if compacted is None:
            if artifact is not None:
                raise ValueError(f"WGPU retained inactive {context} context")
            comparisons[context] = {"semantically_absent_in_both": True}
            continue
        if artifact is None:
            raise ValueError(f"WGPU omitted active {context} context")
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"WGPU condition SHA mismatch: {path}")
        candidate = np.fromfile(path, dtype="<f4").reshape(artifact["shape"])
        comparisons[context] = metrics(compacted, candidate)

    payload = {
        "format": "irodori-v4-compacted-condition-compare-v1",
        "latency_values_used": False,
        "python_artifact": str(args.python_artifact.resolve()),
        "python_artifact_sha256": sha256_file(args.python_artifact),
        "wgpu_report": str(args.wgpu_report.resolve()),
        "wgpu_report_sha256": sha256_file(args.wgpu_report),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


if __name__ == "__main__":
    main()
