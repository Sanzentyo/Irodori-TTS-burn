# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy==2.2.6",
# ]
# ///
"""Validate and summarize a fresh Irodori v4 40-step formal campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONDITION_PATTERN = re.compile(r"^(f\d+)-(text|design|clone)-s(\d+)$")
SCENARIOS = {
    "text": "text_only_fixed",
    "design": "design_fixed",
    "clone": "clone_prepared_fixed",
}
MIB = 1024.0 * 1024.0


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError(f"invalid metric sample: {values}")
    return float(statistics.median(values))


def nvml_peak_mib(path: Path) -> float:
    values: list[float] = []
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            columns = [column.strip() for column in raw.split(",")]
            if len(columns) < 4:
                raise RuntimeError(f"malformed NVML row in {path}: {raw!r}")
            values.append(float(columns[3]))
    if not values:
        raise RuntimeError(f"empty NVML log: {path}")
    return max(values)


def memory_stage(report: dict[str, Any], stage: str) -> dict[str, Any]:
    matches = [row for row in report["memory"] if row["stage"] == stage]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {stage!r} memory row, got {len(matches)}")
    return matches[0]


@dataclass(frozen=True)
class Accuracy:
    max_abs: float
    rmse: float
    snr_db: float
    cosine: float
    finite: bool
    hard_pass: bool
    target_pass: bool
    status: str
    python_audio_sha256: str
    wgpu_audio_sha256: str


def compare_audio(python_path: Path, wgpu_path: Path) -> Accuracy:
    reference = np.fromfile(python_path, dtype="<f4").astype(np.float64)
    candidate = np.fromfile(wgpu_path, dtype="<f4").astype(np.float64)
    if reference.shape != candidate.shape or reference.size == 0:
        raise RuntimeError(
            f"audio shape mismatch: {python_path} {reference.shape}, "
            f"{wgpu_path} {candidate.shape}"
        )
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return Accuracy(
            max_abs=math.inf,
            rmse=math.inf,
            snr_db=-math.inf,
            cosine=-math.inf,
            finite=False,
            hard_pass=False,
            target_pass=False,
            status="fail",
            python_audio_sha256=sha256_file(python_path),
            wgpu_audio_sha256=sha256_file(wgpu_path),
        )
    delta = candidate - reference
    max_abs = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    signal_rms = float(np.sqrt(np.mean(reference * reference)))
    snr_db = math.inf if rmse == 0.0 else 20.0 * math.log10(signal_rms / rmse)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    cosine = 1.0 if denominator == 0.0 and max_abs == 0.0 else float(
        np.dot(reference, candidate) / denominator
    )
    hard_pass = snr_db >= 80.0 and cosine >= 0.99999999
    target_pass = hard_pass and snr_db >= 85.0 and max_abs <= 2.0e-4
    status = "target_pass" if target_pass else "hard_pass_warning" if hard_pass else "fail"
    return Accuracy(
        max_abs=max_abs,
        rmse=rmse,
        snr_db=snr_db,
        cosine=cosine,
        finite=True,
        hard_pass=hard_pass,
        target_pass=target_pass,
        status=status,
        python_audio_sha256=sha256_file(python_path),
        wgpu_audio_sha256=sha256_file(wgpu_path),
    )


@dataclass(frozen=True)
class SessionMetrics:
    session: int
    python_device_ms: float
    python_readback_ms: float
    python_first_device_ms: float
    python_first_readback_ms: float
    python_load_s: float
    python_persistent_allocated_mib: float
    python_persistent_reserved_mib: float
    python_nvml_peak_mib: float
    wgpu_device_ms: float
    wgpu_readback_ms: float
    wgpu_consumer_ms: float
    wgpu_first_device_ms: float
    wgpu_first_readback_ms: float
    wgpu_first_consumer_ms: float
    wgpu_load_s: float
    wgpu_persistent_in_use_mib: float
    wgpu_persistent_reserved_mib: float
    wgpu_nvml_peak_mib: float
    python_audio_f32_sha256: str
    wgpu_audio_f32_sha256: str
    python_audio_path: str
    wgpu_audio_path: str


def session_metrics(base: Path, voice: str, session: int) -> SessionMetrics:
    scenario = SCENARIOS[voice]
    py_dir = base / "python"
    wg_dir = base / "wgpu"
    py = load_json(py_dir / "result.json")
    wg = load_json(wg_dir / "result.json")

    py_rows = py["scenarios"][scenario]["rows"]
    py_measured = [row for row in py_rows if not row["warmup"]]
    wg_rows = wg["resident_request_timings"]
    wg_measured = [row for row in wg_rows if not row["warmup"]]
    if not py_measured or not wg_measured:
        raise RuntimeError(f"missing measured rows: {base}")

    py_hashes = {row["audio_sha256_f32"] for row in py_rows}
    wg_hashes = {row["audio_f32_sha256"] for row in wg_rows}
    if len(py_hashes) != 1 or len(wg_hashes) != 1:
        raise RuntimeError(f"runtime was not deterministic within session: {base}")

    py_artifact = py["audio_artifacts"][scenario]
    wg_artifacts = wg["audio_artifacts"]
    if len(wg_artifacts) != 1:
        raise RuntimeError(f"expected one WGPU audio artifact: {base}")
    py_audio = Path(py_artifact["path"])
    wg_audio = Path(wg_artifacts[0]["path"])
    if not py_audio.is_file() or not wg_audio.is_file():
        raise RuntimeError(f"missing audio artifact: {base}")

    persistent = memory_stage(wg, "rf_duration_codec_resident")
    py_first = py_rows[0]
    wg_first = wg_rows[0]
    return SessionMetrics(
        session=session,
        python_device_ms=1000.0
        * median([float(row["cuda_event_seconds"]) for row in py_measured]),
        python_readback_ms=1000.0
        * median([float(row["cpu_audio_ready_wall_seconds"]) for row in py_measured]),
        python_first_device_ms=1000.0 * float(py_first["cuda_event_seconds"]),
        python_first_readback_ms=1000.0 * float(py_first["cpu_audio_ready_wall_seconds"]),
        python_load_s=float(py["load"]["wall_seconds"]),
        python_persistent_allocated_mib=float(py["load"]["idle_allocated_mib"]),
        python_persistent_reserved_mib=float(py["load"]["idle_reserved_mib"]),
        python_nvml_peak_mib=nvml_peak_mib(py_dir / "nvml.csv"),
        wgpu_device_ms=1000.0
        * median(
            [
                float(row["rf_device_complete_seconds"])
                + float(row["codec_device_complete_seconds"])
                for row in wg_measured
            ]
        ),
        wgpu_readback_ms=1000.0
        * median(
            [
                float(row["rf_device_complete_seconds"])
                + float(row["codec_readback_complete_seconds"])
                for row in wg_measured
            ]
        ),
        wgpu_consumer_ms=1000.0
        * median([float(row["consumer_complete_seconds"]) for row in wg_measured]),
        wgpu_first_device_ms=1000.0
        * (
            float(wg_first["rf_device_complete_seconds"])
            + float(wg_first["codec_device_complete_seconds"])
        ),
        wgpu_first_readback_ms=1000.0
        * (
            float(wg_first["rf_device_complete_seconds"])
            + float(wg_first["codec_readback_complete_seconds"])
        ),
        wgpu_first_consumer_ms=1000.0 * float(wg_first["consumer_complete_seconds"]),
        wgpu_load_s=float(wg["load_wall_seconds"]),
        wgpu_persistent_in_use_mib=float(persistent["bytes_in_use"]) / MIB,
        wgpu_persistent_reserved_mib=float(persistent["bytes_reserved"]) / MIB,
        wgpu_nvml_peak_mib=nvml_peak_mib(wg_dir / "nvml.csv"),
        python_audio_f32_sha256=next(iter(py_hashes)),
        wgpu_audio_f32_sha256=next(iter(wg_hashes)),
        python_audio_path=str(py_audio),
        wgpu_audio_path=str(wg_audio),
    )


def pct(candidate: float, reference: float) -> float:
    return 100.0 * (candidate / reference - 1.0)


def summarize_condition(
    slug: str,
    voice: str,
    sessions: list[SessionMetrics],
    frames: int,
) -> dict[str, Any]:
    python_hashes = {row.python_audio_f32_sha256 for row in sessions}
    wgpu_hashes = {row.wgpu_audio_f32_sha256 for row in sessions}
    if len(python_hashes) != 1 or len(wgpu_hashes) != 1:
        raise RuntimeError(f"fresh-session nondeterminism: {slug}/{voice}")

    accuracy = compare_audio(
        Path(sessions[0].python_audio_path), Path(sessions[0].wgpu_audio_path)
    )
    py_device = median([row.python_device_ms for row in sessions])
    py_readback = median([row.python_readback_ms for row in sessions])
    wg_device = median([row.wgpu_device_ms for row in sessions])
    wg_readback = median([row.wgpu_readback_ms for row in sessions])
    output_seconds = frames * 1920 / 48_000
    return {
        "slug": slug,
        "frames": frames,
        "output_seconds": output_seconds,
        "voice": voice,
        "fresh_sessions": len(sessions),
        "measured_requests_per_runtime": len(sessions)
        * int(load_json(Path(sessions[0].python_audio_path).parents[1] / "result.json")["parameters"]["measured"]),
        "latency_ms": {
            "python_device": py_device,
            "wgpu_device": wg_device,
            "device_delta_pct": pct(wg_device, py_device),
            "python_readback": py_readback,
            "wgpu_readback": wg_readback,
            "readback_delta_pct": pct(wg_readback, py_readback),
            "wgpu_consumer": median([row.wgpu_consumer_ms for row in sessions]),
        },
        "first_request_ms": {
            "python_device": median([row.python_first_device_ms for row in sessions]),
            "python_readback": median([row.python_first_readback_ms for row in sessions]),
            "wgpu_device": median([row.wgpu_first_device_ms for row in sessions]),
            "wgpu_readback": median([row.wgpu_first_readback_ms for row in sessions]),
            "wgpu_consumer": median([row.wgpu_first_consumer_ms for row in sessions]),
        },
        "throughput": {
            "python_requests_per_second": 1000.0 / py_readback,
            "wgpu_requests_per_second": 1000.0 / wg_readback,
            "python_audio_seconds_per_wall_second": 1000.0 * output_seconds / py_readback,
            "wgpu_audio_seconds_per_wall_second": 1000.0 * output_seconds / wg_readback,
        },
        "load_wall_seconds": {
            "python": median([row.python_load_s for row in sessions]),
            "wgpu": median([row.wgpu_load_s for row in sessions]),
        },
        "persistent_vram_mib": {
            "python_allocated": median(
                [row.python_persistent_allocated_mib for row in sessions]
            ),
            "python_reserved": median(
                [row.python_persistent_reserved_mib for row in sessions]
            ),
            "wgpu_in_use": median([row.wgpu_persistent_in_use_mib for row in sessions]),
            "wgpu_reserved": median(
                [row.wgpu_persistent_reserved_mib for row in sessions]
            ),
        },
        "nvml_process_peak_mib": {
            "python": max(row.python_nvml_peak_mib for row in sessions),
            "wgpu": max(row.wgpu_nvml_peak_mib for row in sessions),
        },
        "accuracy": asdict(accuracy),
        "deterministic_across_fresh_sessions": {
            "python": True,
            "wgpu": True,
        },
        "speed_and_accuracy_pass": wg_readback < py_readback and accuracy.hard_pass,
        "sessions": [asdict(row) for row in sessions],
    }


def write_csv(path: Path, conditions: list[dict[str, Any]]) -> None:
    fields = [
        "frames",
        "output_seconds",
        "voice",
        "python_readback_ms",
        "wgpu_readback_ms",
        "readback_delta_pct",
        "python_first_readback_ms",
        "wgpu_first_readback_ms",
        "python_load_s",
        "wgpu_load_s",
        "python_nvml_peak_mib",
        "wgpu_nvml_peak_mib",
        "max_abs",
        "rmse",
        "snr_db",
        "cosine",
        "accuracy_status",
        "speed_and_accuracy_pass",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in conditions:
            writer.writerow(
                {
                    "frames": row["frames"],
                    "output_seconds": row["output_seconds"],
                    "voice": row["voice"],
                    "python_readback_ms": row["latency_ms"]["python_readback"],
                    "wgpu_readback_ms": row["latency_ms"]["wgpu_readback"],
                    "readback_delta_pct": row["latency_ms"]["readback_delta_pct"],
                    "python_first_readback_ms": row["first_request_ms"]["python_readback"],
                    "wgpu_first_readback_ms": row["first_request_ms"]["wgpu_readback"],
                    "python_load_s": row["load_wall_seconds"]["python"],
                    "wgpu_load_s": row["load_wall_seconds"]["wgpu"],
                    "python_nvml_peak_mib": row["nvml_process_peak_mib"]["python"],
                    "wgpu_nvml_peak_mib": row["nvml_process_peak_mib"]["wgpu"],
                    "max_abs": row["accuracy"]["max_abs"],
                    "rmse": row["accuracy"]["rmse"],
                    "snr_db": row["accuracy"]["snr_db"],
                    "cosine": row["accuracy"]["cosine"],
                    "accuracy_status": row["accuracy"]["status"],
                    "speed_and_accuracy_pass": row["speed_and_accuracy_pass"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = load_json(root / "protocol.json")
    expected_sessions = int(
        protocol["process_contract"]["fresh_sessions_per_runtime_condition"]
    )
    grouped: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for path in sorted((root / "sessions").iterdir()):
        if not path.is_dir():
            continue
        match = CONDITION_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected session directory: {path}")
        slug, voice, raw_session = match.groups()
        grouped.setdefault((slug, voice), []).append((int(raw_session), path))

    conditions: list[dict[str, Any]] = []
    for (slug, voice), entries in sorted(grouped.items()):
        if len(entries) != expected_sessions:
            raise RuntimeError(
                f"{slug}/{voice} has {len(entries)} sessions, expected {expected_sessions}"
            )
        session_ids = [session for session, _ in entries]
        if session_ids != list(range(1, expected_sessions + 1)):
            raise RuntimeError(f"non-contiguous sessions for {slug}/{voice}: {session_ids}")
        sessions = [session_metrics(path, voice, session) for session, path in entries]
        frames_values = {
            int(load_json(path / "wgpu/result.json")["items"][0]["frames"])
            for _, path in entries
        }
        if len(frames_values) != 1:
            raise RuntimeError(f"frame count changed across sessions: {slug}/{voice}")
        conditions.append(
            summarize_condition(slug, voice, sessions, next(iter(frames_values)))
        )

    if not conditions:
        raise RuntimeError("campaign has no conditions")
    hard_passes = sum(row["accuracy"]["hard_pass"] for row in conditions)
    target_passes = sum(row["accuracy"]["target_pass"] for row in conditions)
    wgpu_readback_wins = sum(
        row["latency_ms"]["wgpu_readback"]
        < row["latency_ms"]["python_readback"]
        for row in conditions
    )
    speed_and_accuracy = sum(row["speed_and_accuracy_pass"] for row in conditions)
    summary = {
        "format": "irodori-v4-40step-formal-summary-v1",
        "campaign_root": str(root),
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "status": "pass" if hard_passes == len(conditions) else "accuracy_failure",
        "grain": "median within 10 measured requests, then median across five fresh sessions per frames/voice/runtime",
        "condition_count": len(conditions),
        "hard_accuracy_passes": hard_passes,
        "target_accuracy_passes": target_passes,
        "wgpu_readback_wins": wgpu_readback_wins,
        "speed_and_accuracy_passes": speed_and_accuracy,
        "accuracy_policy": {
            "hard": {"waveform_snr_db_min": 80.0, "cosine_min": 0.99999999},
            "target": {"waveform_snr_db_min": 85.0, "max_abs_max": 2.0e-4},
            "finite_required": True,
        },
        "conditions": conditions,
    }
    if args.output.exists() or args.csv.exists():
        raise FileExistsError("summary outputs must be new")
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.csv, conditions)
    print(json.dumps({key: summary[key] for key in (
        "status", "condition_count", "hard_accuracy_passes", "target_accuracy_passes",
        "wgpu_readback_wins", "speed_and_accuracy_passes"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
