#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Summarize a sealed five-session paired all-resident campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def median(values: list[float]) -> float:
    return statistics.median(values)


def nvml_peak(path: Path) -> int:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            try:
                values.append(int(fields[3]))
            except ValueError:
                pass
    if not values:
        raise ValueError(f"no NVML values: {path}")
    return max(values)


def distribution(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": median(values), "max": max(values)}


def summarize_runtime(sessions: list[dict[str, Any]], runtime: str) -> dict[str, Any]:
    if runtime == "python":
        rows = [session["repeat_results"] for session in sessions]
        consumer = [
            [row["synthesize_wall_seconds"] for row in values] for values in rows
        ]
        rf = [
            [row["sample_rf_probe"]["synchronized_wall_seconds"] for row in values]
            for values in rows
        ]
        codec = [
            [row["decode_latent_probe"]["synchronized_wall_seconds"] for row in values]
            for values in rows
        ]
        load = [session["load_wall_seconds"] for session in sessions]
        persistent_in_use = [
            session["load_idle_cuda_allocated_mib"] for session in sessions
        ]
        persistent_reserved = [
            session["load_idle_cuda_reserved_mib"] for session in sessions
        ]
        request_peak_in_use = [
            max(row["peak_cuda_allocated_mib"] for row in values) for values in rows
        ]
        request_peak_reserved = [
            max(row["peak_cuda_reserved_mib"] for row in values) for values in rows
        ]
        hashes = [row["audio_f32_sha256"] for values in rows for row in values]
    else:
        rows = [session["resident_request_timings"] for session in sessions]
        consumer = [
            [row["consumer_complete_seconds"] for row in values] for values in rows
        ]
        rf = [[row["rf_device_complete_seconds"] for row in values] for values in rows]
        codec = [
            [row["codec_device_complete_seconds"] for row in values] for values in rows
        ]
        load = [session["load_wall_seconds"] for session in sessions]
        resident = [
            next(
                snapshot
                for snapshot in session["memory"]
                if snapshot["stage"] == "rf_duration_codec_resident"
            )
            for session in sessions
        ]
        persistent_in_use = [
            snapshot["bytes_in_use"] / 1048576 for snapshot in resident
        ]
        persistent_reserved = [
            snapshot["bytes_reserved"] / 1048576 for snapshot in resident
        ]
        request_peak_in_use = [
            max(snapshot["bytes_in_use"] for snapshot in session["memory"]) / 1048576
            for session in sessions
        ]
        request_peak_reserved = [
            max(snapshot["bytes_reserved"] for snapshot in session["memory"]) / 1048576
            for session in sessions
        ]
        hashes = [row["audio_f32_sha256"] for values in rows for row in values]

    measured_consumer = [values[2:] for values in consumer]
    measured_rf = [values[2:] for values in rf]
    measured_codec = [values[2:] for values in codec]
    session_medians = [median(values) for values in measured_consumer]
    session_rps = [10.0 / sum(values) for values in measured_consumer]
    return {
        "fresh_sessions": 5,
        "warmups_per_session": 2,
        "measured_per_session": 10,
        "load_wall_seconds": distribution(load),
        "first_request_consumer_complete_seconds": distribution(
            [values[0] for values in consumer]
        ),
        "second_warmup_consumer_complete_seconds": distribution(
            [values[1] for values in consumer]
        ),
        "steady_session_median_seconds": distribution(session_medians),
        "steady_pooled_50_median_seconds": median(
            [value for values in measured_consumer for value in values]
        ),
        "steady_rf_device_complete_session_median_seconds": distribution(
            [median(values) for values in measured_rf]
        ),
        "steady_codec_device_complete_session_median_seconds": distribution(
            [median(values) for values in measured_codec]
        ),
        "steady_requests_per_second": distribution(session_rps),
        "steady_audio_seconds_per_wall_second": distribution(
            [4.48 * value for value in session_rps]
        ),
        "persistent_in_use_mib": distribution(persistent_in_use),
        "persistent_reserved_mib": distribution(persistent_reserved),
        "request_peak_in_use_mib": distribution(request_peak_in_use),
        "request_peak_reserved_mib": distribution(request_peak_reserved),
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
    if not (args.campaign / "COMPLETE").is_file():
        raise SystemExit("campaign is not COMPLETE")
    python_sessions = [
        json.loads(
            (args.campaign / "sessions" / f"python-{index}" / "result.json").read_text()
        )
        for index in range(1, 6)
    ]
    wgpu_sessions = [
        json.loads(
            (args.campaign / "sessions" / f"wgpu-{index}" / "result.json").read_text()
        )
        for index in range(1, 6)
    ]
    python = summarize_runtime(python_sessions, "python")
    wgpu = summarize_runtime(wgpu_sessions, "wgpu")
    python_nvml = [
        nvml_peak(args.campaign / "sessions" / f"python-{index}" / "nvml.csv")
        for index in range(1, 6)
    ]
    wgpu_nvml = [
        nvml_peak(args.campaign / "sessions" / f"wgpu-{index}" / "nvml.csv")
        for index in range(1, 6)
    ]
    python["external_nvml_peak_mib"] = distribution(python_nvml)
    wgpu["external_nvml_peak_mib"] = distribution(wgpu_nvml)
    speedup = (
        python["steady_session_median_seconds"]["median"]
        / wgpu["steady_session_median_seconds"]["median"]
    )
    report = {
        "format": "irodori-v4-all-resident-paired-comparison-v1",
        "campaign": str(args.campaign.resolve()),
        "request_contract": {
            "text": "こんにちは。",
            "voice": "unconditioned",
            "seconds": 4.48,
            "frames": 112,
            "strict_fp32": True,
            "tf32": False,
            "autocast": False,
            "euler_evaluations": 4,
            "forward_batches": [2, 2, 1, 1],
            "effective_rows": 6,
            "layers": 12,
            "block_calls": 48,
            "final_audio_cpu_readback_in_consumer_complete": True,
            "intermediate_latent_cpu_readback_in_consumer_complete": False,
        },
        "python": python,
        "wgpu": wgpu,
        "wgpu_speedup_consumer_complete": speedup,
        "attempt1_pooled": False,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
