#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Fail-closed aggregation for the six-length F16 Python/WGPU campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

FRAMES = (45, 112, 255, 333, 489, 685)
VOICES = ("text", "design", "clone")
SCENARIOS = {
    "text": "text_only_fixed",
    "design": "design_fixed",
    "clone": "clone_prepared_fixed",
}
SESSIONS = range(1, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def median(values: list[float]) -> float:
    if len(values) != 5:
        raise ValueError(f"expected five fresh-session values, got {len(values)}")
    return float(statistics.median(values))


def nvml_peak(path: Path) -> float:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) < 2:
                continue
            values.append(float(row[1].strip()))
    if not values:
        raise ValueError(f"NVML log contains no samples: {path}")
    return max(values)


def rust_condition(root: Path, frames: int, voice: str) -> dict[str, Any]:
    session_metrics: list[dict[str, float]] = []
    hashes: set[str] = set()
    for session in SESSIONS:
        directory = root / "rust" / f"f{frames}-{voice}-s{session}"
        report = load_json(directory / "result.json")
        if not isinstance(report, dict):
            raise TypeError(f"expected Rust JSON object: {directory}")
        expected_batches = [2, 2, 1, 1] if voice == "text" else [3, 3, 1, 1]
        expected_rows = sum(expected_batches)
        if not (
            report["schema_version"] == 8
            and report["precision"] == "fp16"
            and report["requests"] == 12
            and report["warmups"] == 2
            and report["measured"] == 10
            and report["forward_batches"] == expected_batches
            and report["effective_rows"] == expected_rows
            and report["euler_evaluations"] == 4
            and report["layers"] == 12
            and report["block_calls"] == 48
        ):
            raise ValueError(f"invalid Rust manifest: {directory}")
        measured = [
            row for row in report["resident_request_timings"] if not row["warmup"]
        ]
        if len(measured) != 10:
            raise ValueError(f"Rust measured request count is not ten: {directory}")
        hashes.update(row["audio_f32_sha256"] for row in measured)
        memory = report["memory"]
        persistent = next(
            row for row in memory if row["stage"] == "rf_duration_codec_resident"
        )
        session_metrics.append(
            {
                "load": float(report["load_wall_seconds"]),
                "first_consumer": float(
                    report["resident_request_timings"][0]["consumer_complete_seconds"]
                ),
                "rf_device": statistics.median(
                    float(row["rf_device_complete_seconds"]) for row in measured
                ),
                "codec_device": statistics.median(
                    float(row["codec_device_complete_seconds"]) for row in measured
                ),
                "codec_readback": statistics.median(
                    float(row["codec_readback_complete_seconds"]) for row in measured
                ),
                "device": statistics.median(
                    float(row["rf_device_complete_seconds"])
                    + float(row["codec_device_complete_seconds"])
                    for row in measured
                ),
                "readback": statistics.median(
                    float(row["rf_device_complete_seconds"])
                    + float(row["codec_readback_complete_seconds"])
                    for row in measured
                ),
                "consumer": statistics.median(
                    float(row["consumer_complete_seconds"]) for row in measured
                ),
                "persistent_in_use_bytes": float(persistent["bytes_in_use"]),
                "persistent_reserved_bytes": float(persistent["bytes_reserved"]),
                "allocator_peak_reserved_bytes": float(
                    max(row["bytes_reserved"] for row in memory)
                ),
                "nvml_peak_mib": nvml_peak(directory / "nvml.csv"),
            }
        )
    fields = tuple(session_metrics[0])
    aggregate = {key: median([row[key] for row in session_metrics]) for key in fields}
    aggregate["nvml_peak_max_mib"] = max(
        row["nvml_peak_mib"] for row in session_metrics
    )
    aggregate["audio_hashes"] = sorted(hashes)
    aggregate["deterministic_across_sessions"] = len(hashes) == 1
    return aggregate


def python_condition(root: Path, frames: int, voice: str) -> dict[str, Any]:
    scenario = SCENARIOS[voice]
    session_metrics: list[dict[str, float]] = []
    hashes: set[str] = set()
    expected_samples = frames * 1_920
    for session in SESSIONS:
        directory = root / "python" / f"f{frames}-s{session}"
        report = load_json(directory / "result.json")
        if not isinstance(report, dict):
            raise TypeError(f"expected Python JSON object: {directory}")
        parameters = report["parameters"]
        if not (
            report["environment"]["precision"] == "fp16"
            and parameters["latent_frames"] == frames
            and parameters["warmups"] == 2
            and parameters["measured"] == 10
            and parameters["num_steps"] == 4
        ):
            raise ValueError(f"invalid Python manifest: {directory}")
        rows = report["scenarios"][scenario]["rows"]
        measured = [row for row in rows if not row["warmup"]]
        if len(measured) != 10:
            raise ValueError(f"Python measured request count is not ten: {directory}")
        if any(
            round(float(row["audio_seconds"]) * 48_000) != expected_samples
            for row in rows
        ):
            raise ValueError(f"Python output is not frame-aligned: {directory}")
        hashes.update(row["audio_sha256_f32"] for row in measured)
        summary = report["scenarios"][scenario]["summary"]
        session_metrics.append(
            {
                "load": float(report["load"]["wall_seconds"]),
                "first_readback": float(rows[0]["cpu_audio_ready_wall_seconds"]),
                "rf_device": float(summary["stages"]["sample_rf"]["median_seconds"]),
                "codec_device": float(
                    summary["stages"]["decode_latent"]["median_seconds"]
                ),
                "device": float(summary["cuda_event"]["median_seconds"]),
                "readback": float(summary["cpu_audio_ready_wall"]["median_seconds"]),
                "idle_allocated_mib": float(report["load"]["idle_allocated_mib"]),
                "idle_reserved_mib": float(report["load"]["idle_reserved_mib"]),
                "request_peak_allocated_mib": float(summary["peak_allocated_mib"]),
                "request_peak_reserved_mib": float(summary["peak_reserved_mib"]),
                "nvml_peak_mib": nvml_peak(directory / "nvml.csv"),
            }
        )
    fields = tuple(session_metrics[0])
    aggregate = {key: median([row[key] for row in session_metrics]) for key in fields}
    aggregate["nvml_peak_max_mib"] = max(
        row["nvml_peak_mib"] for row in session_metrics
    )
    aggregate["audio_hashes"] = sorted(hashes)
    aggregate["deterministic_across_sessions"] = len(hashes) == 1
    return aggregate


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    accuracy_rows = load_json(root / "accuracy" / "summary.json")
    if not isinstance(accuracy_rows, list):
        raise TypeError("accuracy summary must be a JSON array")
    accuracy = {row["condition"]: row for row in accuracy_rows}
    conditions: list[dict[str, Any]] = []
    for frames in FRAMES:
        for voice in VOICES:
            rust = rust_condition(root, frames, voice)
            python = python_condition(root, frames, voice)
            device_delta = (rust["device"] / python["device"] - 1.0) * 100.0
            readback_delta = (rust["readback"] / python["readback"] - 1.0) * 100.0
            conditions.append(
                {
                    "frames": frames,
                    "output_seconds": frames * 1_920 / 48_000,
                    "voice": voice,
                    "forward_batches": [2, 2, 1, 1]
                    if voice == "text"
                    else [3, 3, 1, 1],
                    "rust": rust,
                    "python": python,
                    "comparison": {
                        "device_delta_percent": device_delta,
                        "readback_delta_percent": readback_delta,
                        "rust_faster_device": device_delta < 0.0,
                        "rust_faster_readback": readback_delta < 0.0,
                    },
                    "accuracy": accuracy[f"{frames}-{voice}"],
                }
            )
    payload = {
        "format": "irodori-v4-f16-six-length-e2e-summary-v1",
        "aggregation": "median of five fresh-session medians; ten measured per session",
        "same_semantic_work_not_same_operator_graph": True,
        "conditions": conditions,
        "counts": {
            "conditions": len(conditions),
            "rust_sessions": len(conditions) * 5,
            "python_sessions": len(FRAMES) * 5,
            "accuracy_pass": sum(row["accuracy"]["passed"] for row in conditions),
            "rust_faster_device": sum(
                row["comparison"]["rust_faster_device"] for row in conditions
            ),
            "rust_faster_readback": sum(
                row["comparison"]["rust_faster_readback"] for row in conditions
            ),
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
