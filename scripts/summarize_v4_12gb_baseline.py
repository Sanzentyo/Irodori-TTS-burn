#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Create a machine-readable summary from one sealed fresh 12 GiB campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


def median(values: list[float]) -> float:
    return statistics.median(values)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvml_peak(path: Path) -> int:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            try:
                values.append(int(fields[3]))
            except ValueError:
                pass
    if not values:
        raise ValueError(f"no NVML memory values in {path}")
    return max(values)


def elapsed_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"elapsed(?:_seconds)?[=:](\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"no elapsed time in {path}")
    return float(match.group(1))


def cold_summary(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for directory in sorted((root / "cold").iterdir()):
        if not directory.is_dir():
            continue
        log = (directory / "stdout.log").read_text(encoding="utf-8")
        wall_candidates = [directory / "command-time.txt", directory / "wall.txt"]
        wall = next((path for path in wall_candidates if path.is_file()), None)
        if wall is None:
            continue

        def capture(pattern: str, text: str = log) -> float | int | None:
            match = re.search(pattern, text)
            if not match:
                return None
            value = float(match.group(1))
            return int(value) if value.is_integer() else value

        output[directory.name] = {
            "cold_e2e_seconds": elapsed_seconds(wall),
            "duration_prediction_ms": capture(r"duration_time=([\d.]+)ms"),
            "rf_ms": capture(r"rf_time=(\d+)ms"),
            "codec_ms": capture(r"codec=(\d+)ms"),
            "audio_seconds": capture(r"audio_duration=([\d.]+)s"),
            "nvml_peak_mib": nvml_peak(directory / "nvml.csv"),
            "wav_sha256": sha256_file(directory / "output.wav"),
        }
    return output


def online_summary(root: Path) -> dict[str, Any]:
    sessions = [
        json.loads(
            (root / "sessions" / f"python-resident-session-{index}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        for index in range(1, 6)
    ]
    scenarios: dict[str, Any] = {}
    for name in sessions[0]["scenarios"]:
        summaries = [session["scenarios"][name]["summary"] for session in sessions]
        first = [
            session["scenarios"][name]["rows"][0]["cpu_audio_ready_wall_seconds"]
            for session in sessions
        ]
        steady = [summary["cpu_audio_ready_wall"]["median_seconds"] for summary in summaries]
        scenarios[name] = {
            "first_request_median_across_sessions_seconds": median(first),
            "steady_latency_median_across_sessions_seconds": median(steady),
            "steady_session_median_range_seconds": [min(steady), max(steady)],
            "requests_per_second_median_across_sessions": median(
                [summary["throughput"]["requests_per_second_from_wall_median"] for summary in summaries]
            ),
            "audio_seconds_per_wall_second_median_across_sessions": median(
                [summary["throughput"]["audio_seconds_per_wall_second"] for summary in summaries]
            ),
            "maximum_peak_allocated_mib": max(summary["peak_allocated_mib"] for summary in summaries),
            "maximum_peak_reserved_mib": max(summary["peak_reserved_mib"] for summary in summaries),
            "prepare_reference_stage_median_across_sessions_seconds": median(
                [summary["stages"]["prepare_reference"]["median_seconds"] for summary in summaries]
            ),
            "deterministic_per_voice_all_sessions": all(
                summary["deterministic_per_voice"] for summary in summaries
            ),
        }
    return {
        "fresh_sessions": 5,
        "warmups_per_scenario": 2,
        "measured_per_scenario": 10,
        "all_resident": {
            "load_wall_seconds": [session["load"]["wall_seconds"] for session in sessions],
            "load_idle_allocated_mib": [session["load"]["idle_allocated_mib"] for session in sessions],
            "load_idle_reserved_mib": [session["load"]["idle_reserved_mib"] for session in sessions],
            "request_peak_allocated_mib": max(
                scenario["maximum_peak_allocated_mib"] for scenario in scenarios.values()
            ),
            "request_peak_reserved_mib": max(
                scenario["maximum_peak_reserved_mib"] for scenario in scenarios.values()
            ),
            "external_nvml_peak_mib": max(
                nvml_peak(root / "sessions" / f"python-resident-session-{index}" / "nvml.csv")
                for index in range(1, 6)
            ),
        },
        "scenarios": scenarios,
        "prepared_reference_one_time_encode_seconds": [
            session["prepared_reference"]["one_time_encode_wall_seconds"] for session in sessions
        ],
    }


def accuracy_summary(root: Path) -> dict[str, Any]:
    slugs = ["s1p8", "s4p48", "s10p2", "s13p32", "s19p56", "s27p4"]
    metric_pattern = re.compile(
        r"^(final_patched_latent|raw_decoded_waveform)\[1\]: count=(\d+) "
        r"max_abs=([\deE+.-]+) mean_abs=([\deE+.-]+) rmse=([\deE+.-]+) "
        r"snr_db=([\deE+.-]+) cosine=([\deE+.-]+)$"
    )
    cases: dict[str, Any] = {}
    for slug in slugs:
        directory = root / "accuracy-campaign" / "lengths" / slug
        python = json.loads((directory / "python.json").read_text(encoding="utf-8"))
        measured = python["repeat_results"][2:]
        lines = (directory / "wgpu.stdout.log").read_text(encoding="utf-8").splitlines()
        rf = [json.loads(line.split("=", 1)[1]) for line in lines if line.startswith("rf_timing_manifest=")][2:]
        codec = [
            json.loads(line.split("=", 1)[1])
            for line in lines
            if line.startswith("codec_timing_manifest=")
        ][2:]
        metrics: dict[str, dict[str, float]] = {}
        for line in lines:
            match = metric_pattern.match(line)
            if match:
                metrics[match.group(1)] = dict(
                    zip(
                        ["max_abs", "mean_abs", "rmse", "snr_db", "cosine"],
                        map(float, match.groups()[2:]),
                        strict=True,
                    )
                )
        latent = metrics["final_patched_latent"]
        waveform = metrics["raw_decoded_waveform"]
        latent_pass = (
            latent["max_abs"] <= 2e-4
            and latent["mean_abs"] <= 1e-5
            and latent["rmse"] <= 2e-5
            and latent["snr_db"] >= 90
            and latent["cosine"] >= 0.99999999
        )
        waveform_pass = (
            waveform["max_abs"] <= 1.5e-4
            and waveform["mean_abs"] <= 5e-6
            and waveform["rmse"] <= 1e-5
            and waveform["snr_db"] >= 85
            and waveform["cosine"] >= 0.99999999
        )
        cases[slug] = {
            "frames": python["length_contract"]["latent_steps"],
            "seconds": python["length_contract"]["seconds"],
            "python": {
                "rf_device_complete_median_seconds": median(
                    [row["sample_rf_probe"]["synchronized_wall_seconds"] for row in measured]
                ),
                "rf_readback_complete_median_seconds": median(
                    [row["sample_rf_probe"]["synchronized_wall_with_readback_seconds"] for row in measured]
                ),
                "codec_device_complete_median_seconds": median(
                    [row["decode_latent_probe"]["synchronized_wall_seconds"] for row in measured]
                ),
                "codec_readback_complete_median_seconds": median(
                    [row["decode_latent_probe"]["synchronized_wall_with_readback_seconds"] for row in measured]
                ),
            },
            "wgpu": {
                "rf_device_complete_median_seconds": median([row["sample_device_complete_s"] for row in rf]),
                "rf_readback_complete_median_seconds": median([row["sample_and_readback_s"] for row in rf]),
                "codec_device_complete_median_seconds": median(
                    [row["decode_device_complete_s"] for row in codec]
                ),
                "codec_readback_complete_median_seconds": median(
                    [row["decode_and_readback_s"] for row in codec]
                ),
            },
            "accuracy": {
                "latent": latent,
                "waveform": waveform,
                "latent_gate_pass": latent_pass,
                "waveform_gate_pass": waveform_pass,
                "overall_pass": latent_pass and waveform_pass,
            },
        }
    return {
        "warmups": 2,
        "measured": 10,
        "same_boundaries": True,
        "gates": {
            "latent": {"max_abs": 2e-4, "mean_abs": 1e-5, "rmse": 2e-5, "min_snr_db": 90, "min_cosine": 0.99999999},
            "waveform": {"max_abs": 1.5e-4, "mean_abs": 5e-6, "rmse": 1e-5, "min_snr_db": 85, "min_cosine": 0.99999999},
        },
        "cases": cases,
    }


def phase_summary(root: Path) -> dict[str, Any]:
    campaign = root / "phase-batch" / "measurements"
    if not (campaign / "COMPLETE").is_file():
        raise ValueError("phase-batch campaign is not sealed COMPLETE")
    conditions: dict[str, Any] = {}
    for directory in sorted((campaign / "conditions").iterdir()):
        if not directory.is_dir():
            continue
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        result["external_nvml_peak_mib"] = nvml_peak(directory / "nvml.csv")
        conditions[directory.name] = result
    return {"conditions": conditions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    duration = json.loads((args.campaign / "duration-attempt2" / "summary.json").read_text(encoding="utf-8"))
    evidence = {
        "format": "irodori-v4-12gb-baseline-evidence-v1",
        "fresh_campaign": str(args.campaign.resolve()),
        "old_tmp_artifacts_pooled": False,
        "cold": cold_summary(args.campaign),
        "online_pytorch": online_summary(args.campaign),
        "duration_prediction": duration,
        "strict_fp32_accuracy_and_timing": accuracy_summary(args.campaign),
        "wgpu_residency_and_phase_batch": phase_summary(args.campaign),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
