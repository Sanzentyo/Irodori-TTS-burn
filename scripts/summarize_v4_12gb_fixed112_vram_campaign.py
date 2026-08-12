#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Summarize a sealed fixed-112 WGPU residency decomposition campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

CONDITIONS = (
    "control",
    "exact-only",
    "rf-one-layout",
    "rf-packed-only",
    "codec-packed-only",
    "combined-packed-only",
)


def distribution(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def nvml_peak(path: Path) -> int:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            try:
                values.append(int(fields[3]))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"no NVML samples: {path}")
    return max(values)


def snapshot(session: dict[str, Any], stage: str) -> dict[str, Any]:
    return next(row for row in session["memory"] if row["stage"] == stage)


def summarize(campaign: Path, condition: str) -> dict[str, Any]:
    paths = [campaign / "sessions" / f"{condition}-{index}" for index in range(1, 6)]
    sessions = [json.loads((path / "result.json").read_text()) for path in paths]
    timing_rows = [session["resident_request_timings"] for session in sessions]
    measured = [
        [row["consumer_complete_seconds"] for row in rows[2:]] for rows in timing_rows
    ]
    measured_rf = [
        [row["rf_device_complete_seconds"] for row in rows[2:]] for rows in timing_rows
    ]
    measured_codec = [
        [row["codec_device_complete_seconds"] for row in rows[2:]]
        for rows in timing_rows
    ]
    resident = [snapshot(session, "rf_duration_codec_resident") for session in sessions]
    completed = [
        snapshot(session, "all_resident_after_consumer") for session in sessions
    ]
    hashes = [row["audio_f32_sha256"] for rows in timing_rows for row in rows]
    session_medians = [statistics.median(values) for values in measured]
    session_rps = [10.0 / sum(values) for values in measured]
    mib = 1_048_576
    return {
        "fresh_sessions": 5,
        "warmups_per_session": 2,
        "measured_per_session": 10,
        "duration_residency": sessions[0]["duration_residency"],
        "rf_weight_residency": sessions[0]["rf_weight_residency"],
        "codec_weight_residency": sessions[0]["codec_weight_residency"],
        "load_wall_seconds": distribution(
            [session["load_wall_seconds"] for session in sessions]
        ),
        "first_request_seconds": distribution(
            [rows[0]["consumer_complete_seconds"] for rows in timing_rows]
        ),
        "second_warmup_seconds": distribution(
            [rows[1]["consumer_complete_seconds"] for rows in timing_rows]
        ),
        "steady_consumer_seconds": distribution(session_medians),
        "steady_rf_device_seconds": distribution(
            [statistics.median(values) for values in measured_rf]
        ),
        "steady_codec_device_seconds": distribution(
            [statistics.median(values) for values in measured_codec]
        ),
        "steady_requests_per_second": distribution(session_rps),
        "steady_audio_seconds_per_wall_second": distribution(
            [4.48 * value for value in session_rps]
        ),
        "load_resident_in_use_mib": distribution(
            [row["bytes_in_use"] / mib for row in resident]
        ),
        "load_resident_reserved_mib": distribution(
            [row["bytes_reserved"] / mib for row in resident]
        ),
        "completed_in_use_mib": distribution(
            [row["bytes_in_use"] / mib for row in completed]
        ),
        "completed_reserved_mib": distribution(
            [row["bytes_reserved"] / mib for row in completed]
        ),
        "request_peak_in_use_mib": distribution(
            [
                max(row["bytes_in_use"] for row in session["memory"]) / mib
                for session in sessions
            ]
        ),
        "request_peak_reserved_mib": distribution(
            [
                max(row["bytes_reserved"] for row in session["memory"]) / mib
                for session in sessions
            ]
        ),
        "external_nvml_peak_mib": distribution(
            [nvml_peak(path / "nvml.csv") for path in paths]
        ),
        "deterministic_across_60_requests": len(set(hashes)) == 1,
        "audio_f32_sha256_values": sorted(set(hashes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    conditions = {
        condition: summarize(args.campaign, condition) for condition in CONDITIONS
    }
    control = conditions["control"]
    for result in conditions.values():
        result["steady_speed_ratio_vs_control"] = (
            control["steady_consumer_seconds"]["median"]
            / result["steady_consumer_seconds"]["median"]
        )
        result["completed_in_use_saved_vs_control_mib"] = (
            control["completed_in_use_mib"]["median"]
            - result["completed_in_use_mib"]["median"]
        )
        result["request_peak_reserved_saved_vs_control_mib"] = (
            control["request_peak_reserved_mib"]["median"]
            - result["request_peak_reserved_mib"]["median"]
        )
    report = {
        "format": "irodori-v4-fixed112-vram-decomposition-v1",
        "campaign": str(args.campaign.resolve()),
        "request_contract": {
            "seconds": 4.48,
            "frames": 112,
            "voice": "unconditioned",
            "strict_fp32": True,
            "euler_evaluations": 4,
            "forward_batches": [2, 2, 1, 1],
            "final_audio_cpu_readback_in_consumer_complete": True,
        },
        "conditions": conditions,
        "old_measurements_pooled": False,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
