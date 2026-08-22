# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy==2.2.6",
#   "safetensors==0.7.0",
# ]
# ///
"""Compare an exact-input Python DiT trace with WGPU block boundaries."""

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


def load_raw(path: Path, shape: list[int]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != math.prod(shape):
        raise ValueError(f"{path}: element count does not match {shape}")
    return values.reshape(shape)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.astype(np.float64, copy=False).ravel()
    got = candidate.astype(np.float64, copy=False).ravel()
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("non-finite boundary tensor")
    error = got - ref
    signal_power = float(np.mean(ref * ref))
    noise_power = float(np.mean(error * error))
    dot = float(np.dot(ref, got))
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(got))
    return {
        "shape": list(reference.shape),
        "elements": int(reference.size),
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
        "rmse": math.sqrt(noise_power),
        "snr_db": (
            math.inf
            if noise_power == 0.0
            else 10.0 * math.log10(signal_power / noise_power)
        ),
        "cosine": 1.0 if denominator == 0.0 and dot == 0.0 else dot / denominator,
        "exact_f32_elements": int(np.count_nonzero(ref == got)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    python_tensors = load_file(args.python_artifact)
    with args.wgpu_report.open(encoding="utf-8") as file:
        report = json.load(file)
    if report.get("diagnostic_only") is not True:
        raise ValueError("WGPU report is not diagnostic-only")
    if report.get("latency_results_valid") is not False:
        raise ValueError("diagnostic WGPU report exposes valid latency")

    boundary_metrics: list[dict[str, Any]] = []
    previous_rmse: float | None = None
    for ordinal, artifact in enumerate(report["tensors"]):
        name = artifact["name"]
        if name not in python_tensors:
            raise KeyError(f"Python artifact is missing {name}")
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"WGPU tensor SHA mismatch: {path}")
        result = metrics(python_tensors[name], load_raw(path, artifact["shape"]))
        rmse = float(result["rmse"])
        result.update(
            {
                "ordinal": ordinal,
                "name": name,
                "rmse_growth_from_previous": (
                    None
                    if previous_rmse is None or previous_rmse == 0.0
                    else rmse / previous_rmse
                ),
            }
        )
        boundary_metrics.append(result)
        previous_rmse = rmse

    worst = min(boundary_metrics, key=lambda item: float(item["snr_db"]))
    largest_growth = max(
        (item for item in boundary_metrics if item["rmse_growth_from_previous"]),
        key=lambda item: float(item["rmse_growth_from_previous"]),
    )
    payload = {
        "format": "irodori-v4-exact-input-block-compare-v1",
        "latency_values_used": False,
        "python_artifact": str(args.python_artifact.resolve()),
        "python_artifact_sha256": sha256_file(args.python_artifact),
        "wgpu_report": str(args.wgpu_report.resolve()),
        "wgpu_report_sha256": sha256_file(args.wgpu_report),
        "boundaries": boundary_metrics,
        "worst_boundary": {
            "name": worst["name"],
            "snr_db": worst["snr_db"],
            "max_abs": worst["max_abs"],
        },
        "largest_rmse_growth": {
            "name": largest_growth["name"],
            "factor": largest_growth["rmse_growth_from_previous"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


if __name__ == "__main__":
    main()
