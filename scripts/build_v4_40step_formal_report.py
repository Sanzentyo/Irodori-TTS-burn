# /// script
# requires-python = ">=3.10,<3.13"
# ///
"""Build the canonical report artifact for a sealed v4 40-step campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

VOICE_LABELS = {
    "text": "Text-only",
    "design": "Voice design",
    "clone": "Prepared clone",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def sql_project_rows(
    rows: list[dict[str, Any]], dataset: str
) -> tuple[list[dict[str, Any]], str]:
    """Project validated JSON rows through the SQL used by report widgets."""
    if not dataset.replace("_", "").isalnum():
        raise ValueError(f"unsafe dataset name: {dataset}")
    sql = (
        f"WITH {dataset}_json(payload) AS (VALUES (?)) "
        f"SELECT value FROM {dataset}_json, json_each(payload) ORDER BY key"
    )
    with sqlite3.connect(":memory:") as connection:
        projected = [
            json.loads(value)
            for (value,) in connection.execute(
                sql, (json.dumps(rows, ensure_ascii=False, sort_keys=True),)
            )
        ]
    if projected != rows:
        raise RuntimeError(f"SQL projection changed {dataset} rows")
    return projected, sql


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    campaign = args.campaign.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"report artifact already exists: {output}")
    summary = load_json(campaign / "summary.json")
    protocol = load_json(campaign / "protocol.json")
    if not (campaign / "COMPLETE").is_file():
        raise RuntimeError("campaign is not sealed COMPLETE")
    if summary["condition_count"] != 18 or len(summary["conditions"]) != 18:
        raise RuntimeError("formal report requires exactly 18 conditions")
    if protocol["steps"] != 40 or protocol["precision"] != "fp32":
        raise RuntimeError("formal report requires the strict-FP32 40-step protocol")

    conditions: list[dict[str, Any]] = []
    session_pairs: list[dict[str, Any]] = []
    for condition in summary["conditions"]:
        accuracy = condition["accuracy"]
        latency = condition["latency_ms"]
        first = condition["first_request_ms"]
        persistent = condition["persistent_vram_mib"]
        peak = condition["nvml_process_peak_mib"]
        voice = condition["voice"]
        row = {
            "condition": f"{condition['frames']}f · {VOICE_LABELS[voice]}",
            "frames": condition["frames"],
            "output_seconds": condition["output_seconds"],
            "voice": VOICE_LABELS[voice],
            "voice_id": voice,
            "python_device_ms": round_float(latency["python_device"]),
            "wgpu_device_ms": round_float(latency["wgpu_device"]),
            "python_readback_ms": round_float(latency["python_readback"]),
            "wgpu_readback_ms": round_float(latency["wgpu_readback"]),
            "readback_delta_pct": round_float(latency["readback_delta_pct"]),
            "python_first_readback_ms": round_float(first["python_readback"]),
            "wgpu_first_readback_ms": round_float(first["wgpu_readback"]),
            "wgpu_consumer_ms": round_float(latency["wgpu_consumer"]),
            "python_load_seconds": round_float(
                condition["load_wall_seconds"]["python"]
            ),
            "wgpu_load_seconds": round_float(condition["load_wall_seconds"]["wgpu"]),
            "python_persistent_allocated_mib": round_float(
                persistent["python_allocated"]
            ),
            "python_persistent_reserved_mib": round_float(
                persistent["python_reserved"]
            ),
            "wgpu_persistent_in_use_mib": round_float(persistent["wgpu_in_use"]),
            "wgpu_persistent_reserved_mib": round_float(persistent["wgpu_reserved"]),
            "python_nvml_peak_mib": round_float(peak["python"]),
            "wgpu_nvml_peak_mib": round_float(peak["wgpu"]),
            "snr_db": round_float(accuracy["snr_db"]),
            "max_abs": accuracy["max_abs"],
            "rmse": accuracy["rmse"],
            "cosine": accuracy["cosine"],
            "accuracy_status": accuracy["status"],
            "hard_accuracy_pass": accuracy["hard_pass"],
            "target_accuracy_pass": accuracy["target_pass"],
            "measured_requests_per_runtime": condition["measured_requests_per_runtime"],
        }
        conditions.append(row)
        for session in condition["sessions"]:
            delta = 100.0 * (
                session["wgpu_readback_ms"] / session["python_readback_ms"] - 1.0
            )
            session_pairs.append(
                {
                    "condition": row["condition"],
                    "frames": row["frames"],
                    "voice": row["voice"],
                    "session": session["session"],
                    "python_readback_ms": round_float(session["python_readback_ms"]),
                    "wgpu_readback_ms": round_float(session["wgpu_readback_ms"]),
                    "readback_delta_pct": round_float(delta),
                }
            )

    deltas = [row["readback_delta_pct"] for row in conditions]
    hard_failures = [row for row in conditions if not row["hard_accuracy_pass"]]
    target_warnings = [
        row
        for row in conditions
        if row["hard_accuracy_pass"] and not row["target_accuracy_pass"]
    ]
    worst_accuracy = min(conditions, key=lambda row: row["snr_db"])
    smallest_gap = min(conditions, key=lambda row: row["readback_delta_pct"])
    largest_gap = max(conditions, key=lambda row: row["readback_delta_pct"])
    python_session_wins = sum(
        row["python_readback_ms"] < row["wgpu_readback_ms"] for row in session_pairs
    )
    max_python_peak = max(row["python_nvml_peak_mib"] for row in conditions)
    max_wgpu_peak = max(row["wgpu_nvml_peak_mib"] for row in conditions)
    total_vram = protocol["hardware"]["vram_mib"]
    load_python = statistics.median(row["python_load_seconds"] for row in conditions)
    load_wgpu = statistics.median(row["wgpu_load_seconds"] for row in conditions)
    persistent_python = max(
        row["python_persistent_allocated_mib"] for row in conditions
    )
    persistent_wgpu = max(row["wgpu_persistent_in_use_mib"] for row in conditions)

    vram_by_length: list[dict[str, Any]] = []
    for frames in sorted({row["frames"] for row in conditions}):
        matching = [row for row in conditions if row["frames"] == frames]
        seconds = matching[0]["output_seconds"]
        vram_by_length.extend(
            [
                {
                    "frames": frames,
                    "output_seconds": seconds,
                    "runtime": "PyTorch/CUDA",
                    "peak_mib": max(row["python_nvml_peak_mib"] for row in matching),
                    "voice_conditions": 3,
                    "fresh_sessions_per_voice": 5,
                },
                {
                    "frames": frames,
                    "output_seconds": seconds,
                    "runtime": "Rust/WGPU",
                    "peak_mib": max(row["wgpu_nvml_peak_mib"] for row in matching),
                    "voice_conditions": 3,
                    "fresh_sessions_per_voice": 5,
                },
            ]
        )

    campaign_name = campaign.name
    source_path = f"benchmark-artifacts/{campaign_name}/summary.json"
    checksums_path = f"benchmark-artifacts/{campaign_name}/SHA256SUMS"
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    hard_failure_labels = ", ".join(row["condition"] for row in hard_failures)
    warning_labels = ", ".join(row["condition"] for row in target_warnings)
    source_commit = protocol["source_commit"]
    checksum_digest = sha256_file(campaign / "SHA256SUMS")

    source = {
        "id": "formal-summary",
        "label": "Fresh strict-FP32 40-step campaign summary",
        "path": source_path,
        "query": {
            "engine": "local-json",
            "language": "python",
            "description": (
                "Validated condition and session aggregates generated from sealed raw "
                "Python, WGPU, wall-clock, audio, and NVML evidence."
            ),
            "executed_at": generated_at,
            "tables_used": [
                "summary.json",
                "condition-summary.csv",
                "90 Python session JSON files",
                "90 WGPU session JSON files",
                "180 NVML CSV files",
            ],
            "filters": [
                "strict FP32; TF32 and autocast disabled",
                "40 Euler evaluations; linear schedule; independent CFG",
                "2 warmups plus 10 measured requests per fresh process",
                "5 fresh processes per runtime/length/voice condition",
                "fixed shared FP32 initial noise; tail trim and watermark disabled",
                "no automatic retry and no sample pooling from older campaigns",
            ],
            "metric_definitions": [
                "Condition latency is the median of 10 measured requests within each process, then the median of five fresh-process medians.",
                "Readback-complete ends after an owned contiguous CPU F32 waveform exists for both runtimes.",
                "Hard accuracy requires finite output, waveform SNR at least 80 dB, and cosine at least 0.99999999.",
                "Target accuracy additionally requires SNR at least 85 dB and max absolute error at most 2e-4.",
                "NVML peak is 100 ms device memory.used sampling while the benchmark holds the GPU lock; it is not allocator-internal memory.",
            ],
        },
    }
    raw_source = {
        "id": "raw-evidence",
        "label": "Sealed raw campaign and checksums",
        "path": checksums_path,
        "query": {
            "engine": "local-files",
            "description": "1,723 sealed source, binary, fixture, model, session, audio, wall, and NVML files.",
            "executed_at": generated_at,
            "tables_used": [
                "SHA256SUMS",
                "protocol.json",
                "pins.sha256",
                "raw sessions",
            ],
        },
    }
    implementation_source = {
        "id": "implementation",
        "label": "Formal harness and sampler implementation",
        "path": "scripts/run_v4_40step_formal_compare.sh",
        "query": {
            "engine": "git",
            "description": f"Measurement source commit {source_commit}.",
            "executed_at": generated_at,
            "tables_used": [
                "src/rf/euler_sampler.rs",
                "src/bin/bench_v4_residency.rs",
                "scripts/bench_python_runtime_scenarios.py",
                "scripts/run_v4_40step_formal_compare.sh",
            ],
        },
    }

    headline = {
        "condition_count": len(conditions),
        "wgpu_wins": summary["wgpu_readback_wins"],
        "python_session_wins": python_session_wins,
        "session_pairs": len(session_pairs),
        "hard_accuracy_passes": summary["hard_accuracy_passes"],
        "target_accuracy_passes": summary["target_accuracy_passes"],
        "median_readback_delta_pct": round_float(statistics.median(deltas)),
        "max_wgpu_peak_mib": max_wgpu_peak,
        "max_python_peak_mib": max_python_peak,
        "vram_headroom_mib": total_vram - max_wgpu_peak,
    }
    headline_rows, headline_sql = sql_project_rows([headline], "headline")
    conditions, conditions_sql = sql_project_rows(conditions, "conditions")
    session_pairs, session_pairs_sql = sql_project_rows(session_pairs, "session_pairs")
    vram_by_length, vram_sql = sql_project_rows(vram_by_length, "vram_by_length")
    dataset_sources = [
        {
            "id": "headline-sql",
            "label": "Headline metrics SQLite projection",
            "query": {
                "engine": "sqlite3",
                "sql": headline_sql,
                "description": "Projects the validated headline JSON row without aggregation or coercion.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "conditions-sql",
            "label": "Condition results SQLite projection",
            "query": {
                "engine": "sqlite3",
                "sql": conditions_sql,
                "description": "Projects all 18 validated condition rows without aggregation or coercion.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "session-pairs-sql",
            "label": "Fresh-session pairs SQLite projection",
            "query": {
                "engine": "sqlite3",
                "sql": session_pairs_sql,
                "description": "Projects all 90 paired fresh-session rows without aggregation or coercion.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "vram-by-length-sql",
            "label": "VRAM envelope SQLite projection",
            "query": {
                "engine": "sqlite3",
                "sql": vram_sql,
                "description": "Projects the 12 validated runtime-by-length VRAM envelope rows.",
                "executed_at": generated_at,
            },
        },
    ]
    all_sources = [source, raw_source, implementation_source, *dataset_sources]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Irodori v4 strict-FP32 40-step formal comparison",
            "description": (
                "Fresh PyTorch/CUDA versus Rust/WGPU all-resident performance, "
                "accuracy, memory, load, and warmup evidence on a 12 GiB GPU."
            ),
            "generatedAt": generated_at,
            "sources": all_sources,
            "cards": [
                {
                    "id": "speed-card",
                    "dataset": "headline",
                    "sourceId": "headline-sql",
                    "description": "Condition-level readback wins; lower latency wins.",
                    "metrics": [
                        {
                            "label": "WGPU wins / 18",
                            "field": "wgpu_wins",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "gap-card",
                    "dataset": "headline",
                    "sourceId": "headline-sql",
                    "description": "Median across 18 heterogeneous condition-level deltas.",
                    "metrics": [
                        {
                            "label": "Median WGPU latency delta",
                            "field": "median_readback_delta_pct",
                            "format": "number",
                            "unit": "%",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "accuracy-card",
                    "dataset": "headline",
                    "sourceId": "headline-sql",
                    "description": "Hard gate: finite, SNR ≥80 dB, cosine ≥0.99999999.",
                    "metrics": [
                        {
                            "label": "Hard accuracy passes / 18",
                            "field": "hard_accuracy_passes",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "vram-card",
                    "dataset": "headline",
                    "sourceId": "headline-sql",
                    "description": "Maximum 100 ms NVML device-memory sample across all WGPU sessions.",
                    "metrics": [
                        {
                            "label": "Maximum WGPU peak",
                            "field": "max_wgpu_peak_mib",
                            "format": "number",
                            "unit": "MiB",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "latency-delta",
                    "title": "Readback-complete latency delta by output duration",
                    "subtitle": "WGPU minus PyTorch, percent; 18 condition medians, each from five fresh processes × 10 measured requests. Positive favors PyTorch.",
                    "intent": "relationship",
                    "question": "Does WGPU overtake PyTorch at any practical output length or voice topology?",
                    "rationale": "A scatter preserves all 18 condition estimates and exposes length scaling without treating the six lengths as a time series.",
                    "comparisonContext": {
                        "baseline": "PyTorch/CUDA",
                        "grain": "frames × voice",
                        "unit": "%",
                    },
                    "type": "scatter",
                    "dataset": "conditions",
                    "sourceId": "conditions-sql",
                    "encodings": {
                        "x": {
                            "field": "output_seconds",
                            "type": "quantitative",
                            "label": "Output duration",
                            "unit": "s",
                        },
                        "y": {
                            "field": "readback_delta_pct",
                            "type": "quantitative",
                            "label": "WGPU − PyTorch",
                            "unit": "%",
                        },
                        "color": {
                            "field": "voice",
                            "type": "nominal",
                            "label": "Voice condition",
                        },
                        "tooltip": [
                            {
                                "field": "condition",
                                "type": "nominal",
                                "label": "Condition",
                            },
                            {
                                "field": "python_readback_ms",
                                "type": "quantitative",
                                "label": "PyTorch",
                                "unit": "ms",
                            },
                            {
                                "field": "wgpu_readback_ms",
                                "type": "quantitative",
                                "label": "WGPU",
                                "unit": "ms",
                            },
                            {
                                "field": "readback_delta_pct",
                                "type": "quantitative",
                                "label": "Delta",
                                "unit": "%",
                            },
                        ],
                    },
                    "xAxisTitle": "Output duration (s)",
                    "yAxisTitle": "WGPU − PyTorch readback latency (%)",
                    "valueFormat": "number",
                    "unit": "%",
                    "layout": "full",
                    "palette": {"kind": "categorical", "name": "blue-gold-orange"},
                    "legend": {
                        "position": "bottom",
                        "sort": "spec",
                        "title": "Voice condition",
                    },
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 0,
                            "label": "Parity",
                            "color": "neutral",
                            "lineStyle": "dashed",
                        }
                    ],
                    "surface": {
                        "surface": "explorer",
                        "viewMode": "both",
                        "showControls": False,
                    },
                },
                {
                    "id": "accuracy-snr",
                    "title": "Waveform SNR by output duration",
                    "subtitle": "Rust/WGPU output versus PyTorch/CUDA with identical FP32 noise and sampler bits; 80 dB hard gate and 85 dB target.",
                    "intent": "benchmark",
                    "question": "Which practical shapes cross the numerical accuracy thresholds?",
                    "rationale": "All 18 condition-level SNR values remain visible against both declared gates.",
                    "comparisonContext": {
                        "baseline": "PyTorch/CUDA waveform",
                        "grain": "frames × voice",
                        "unit": "dB",
                    },
                    "type": "scatter",
                    "dataset": "conditions",
                    "sourceId": "conditions-sql",
                    "encodings": {
                        "x": {
                            "field": "output_seconds",
                            "type": "quantitative",
                            "label": "Output duration",
                            "unit": "s",
                        },
                        "y": {
                            "field": "snr_db",
                            "type": "quantitative",
                            "label": "Waveform SNR",
                            "unit": "dB",
                        },
                        "color": {
                            "field": "voice",
                            "type": "nominal",
                            "label": "Voice condition",
                        },
                        "tooltip": [
                            {
                                "field": "condition",
                                "type": "nominal",
                                "label": "Condition",
                            },
                            {
                                "field": "snr_db",
                                "type": "quantitative",
                                "label": "SNR",
                                "unit": "dB",
                            },
                            {
                                "field": "max_abs",
                                "type": "quantitative",
                                "label": "Max abs",
                            },
                            {
                                "field": "accuracy_status",
                                "type": "nominal",
                                "label": "Status",
                            },
                        ],
                    },
                    "xAxisTitle": "Output duration (s)",
                    "yAxisTitle": "Waveform SNR (dB)",
                    "valueFormat": "number",
                    "unit": "dB",
                    "layout": "full",
                    "palette": {"kind": "categorical", "name": "blue-gold-orange"},
                    "legend": {
                        "position": "bottom",
                        "sort": "spec",
                        "title": "Voice condition",
                    },
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 80,
                            "label": "Hard gate",
                            "color": "neutral",
                            "lineStyle": "solid",
                        },
                        {
                            "axis": "y",
                            "value": 85,
                            "label": "Target",
                            "color": "neutral",
                            "lineStyle": "dashed",
                        },
                    ],
                    "surface": {
                        "surface": "explorer",
                        "viewMode": "both",
                        "showControls": False,
                    },
                },
                {
                    "id": "vram-length",
                    "title": "NVML peak VRAM by output duration",
                    "subtitle": "Maximum across three voice conditions and five fresh sessions at each length; one benchmark process held the GPU lock.",
                    "intent": "trend",
                    "question": "Does the all-resident configuration fit the 12 GiB device across all measured lengths?",
                    "rationale": "Two lines show the conservative per-length peak envelope for each runtime.",
                    "comparisonContext": {
                        "baseline": "12,227 MiB physical VRAM",
                        "grain": "runtime × frames",
                        "unit": "MiB",
                    },
                    "type": "line",
                    "dataset": "vram_by_length",
                    "sourceId": "vram-by-length-sql",
                    "encodings": {
                        "x": {
                            "field": "output_seconds",
                            "type": "quantitative",
                            "label": "Output duration",
                            "unit": "s",
                        },
                        "y": {
                            "field": "peak_mib",
                            "type": "quantitative",
                            "label": "Peak VRAM",
                            "unit": "MiB",
                        },
                        "color": {
                            "field": "runtime",
                            "type": "nominal",
                            "label": "Runtime",
                        },
                        "tooltip": [
                            {"field": "runtime", "type": "nominal", "label": "Runtime"},
                            {
                                "field": "frames",
                                "type": "quantitative",
                                "label": "Frames",
                            },
                            {
                                "field": "peak_mib",
                                "type": "quantitative",
                                "label": "Peak",
                                "unit": "MiB",
                            },
                        ],
                    },
                    "xAxisTitle": "Output duration (s)",
                    "yAxisTitle": "Peak device memory used (MiB)",
                    "valueFormat": "number",
                    "unit": "MiB",
                    "layout": "full",
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "legend": {
                        "position": "bottom",
                        "sort": "spec",
                        "title": "Runtime",
                    },
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": total_vram,
                            "label": "Physical VRAM",
                            "color": "neutral",
                            "lineStyle": "dashed",
                        }
                    ],
                    "surface": {
                        "surface": "explorer",
                        "viewMode": "both",
                        "showControls": False,
                    },
                },
            ],
            "tables": [
                {
                    "id": "condition-results",
                    "title": "Exact condition-level results",
                    "subtitle": "Median of 10 measured requests within process, then median across five fresh processes; accuracy compares one deterministic waveform per runtime.",
                    "dataset": "conditions",
                    "sourceId": "conditions-sql",
                    "defaultSort": {"field": "frames", "direction": "asc"},
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {
                            "field": "frames",
                            "label": "Frames",
                            "type": "number",
                            "align": "right",
                        },
                        {"field": "voice", "label": "Voice", "type": "text"},
                        {
                            "field": "python_readback_ms",
                            "label": "PyTorch ms",
                            "type": "number",
                            "align": "right",
                        },
                        {
                            "field": "wgpu_readback_ms",
                            "label": "WGPU ms",
                            "type": "number",
                            "align": "right",
                        },
                        {
                            "field": "readback_delta_pct",
                            "label": "WGPU Δ %",
                            "type": "number",
                            "movement": True,
                            "align": "right",
                        },
                        {
                            "field": "snr_db",
                            "label": "SNR dB",
                            "type": "number",
                            "align": "right",
                        },
                        {
                            "field": "max_abs",
                            "label": "Max abs",
                            "type": "number",
                            "align": "right",
                        },
                        {
                            "field": "accuracy_status",
                            "label": "Accuracy",
                            "type": "text",
                        },
                        {
                            "field": "python_nvml_peak_mib",
                            "label": "Py peak MiB",
                            "type": "number",
                            "align": "right",
                        },
                        {
                            "field": "wgpu_nvml_peak_mib",
                            "label": "WGPU peak MiB",
                            "type": "number",
                            "align": "right",
                        },
                    ],
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "layout": "full",
                    "body": "# Irodori v4 strict-FP32 40-step formal comparison",
                },
                {
                    "id": "decision",
                    "type": "markdown",
                    "sourceId": "formal-summary",
                    "layout": "full",
                    "body": (
                        "## Decision\n\nThe current Rust/WGPU all-resident path is **not a 40-step performance-and-accuracy pass** against PyTorch/CUDA on this 12 GiB GPU. WGPU won 0 of 18 condition medians and 0 of 90 paired fresh sessions. Its median condition-level readback penalty was "
                        f"{statistics.median(deltas):.2f}%. Accuracy passed the hard gate in {summary['hard_accuracy_passes']}/18 conditions and the target in {summary['target_accuracy_passes']}/18. Therefore earlier 4-step wins remain diagnostic results, not product-path performance claims."
                    ),
                },
                {
                    "id": "headline",
                    "type": "metric-strip",
                    "cardIds": ["speed-card", "gap-card", "accuracy-card", "vram-card"],
                    "layout": "full",
                },
                {
                    "id": "performance-finding",
                    "type": "markdown",
                    "sourceId": "formal-summary",
                    "layout": "full",
                    "body": (
                        "## PyTorch remains faster across every measured length and voice topology\n\nThe smallest WGPU readback penalty was "
                        f"{smallest_gap['readback_delta_pct']:.2f}% at {smallest_gap['condition']}; the largest was {largest_gap['readback_delta_pct']:.2f}% at {largest_gap['condition']}. All 90 process-level pairs favored PyTorch, so the conclusion is not caused by one aggregated outlier. Device-complete and readback-complete are retained separately in the evidence; the chart uses readback-complete for both runtimes."
                    ),
                },
                {
                    "id": "latency-chart",
                    "type": "chart",
                    "chartId": "latency-delta",
                    "layout": "full",
                },
                {
                    "id": "accuracy-finding",
                    "type": "markdown",
                    "sourceId": "formal-summary",
                    "layout": "full",
                    "body": (
                        "## Four conditions fail the 80 dB numerical hard gate\n\nHard failures are "
                        f"{hard_failure_labels}. The worst is {worst_accuracy['condition']} at {worst_accuracy['snr_db']:.2f} dB, max_abs {worst_accuracy['max_abs']:.6g}, cosine {worst_accuracy['cosine']:.10f}. Five more conditions pass hard accuracy but miss the 85 dB / 2e-4 target: {warning_labels}. No auditory preference test was run, so these are numerical reproducibility results rather than claims of audible degradation."
                    ),
                },
                {
                    "id": "accuracy-chart",
                    "type": "chart",
                    "chartId": "accuracy-snr",
                    "layout": "full",
                },
                {
                    "id": "memory-finding",
                    "type": "markdown",
                    "sourceId": "formal-summary",
                    "layout": "full",
                    "body": (
                        "## All-resident fits 12 GiB, but WGPU retains and peaks higher\n\nThe maximum WGPU NVML peak was "
                        f"{max_wgpu_peak:.0f} MiB, leaving {total_vram - max_wgpu_peak:.0f} MiB below the reported {total_vram} MiB physical total. Python peaked at {max_python_peak:.0f} MiB. Allocator-internal persistent use was about {persistent_wgpu:.1f} MiB for WGPU versus {persistent_python:.1f} MiB allocated by PyTorch, a {persistent_wgpu - persistent_python:.1f} MiB difference. This proves feasibility on this device, not portability to other adapters."
                    ),
                },
                {
                    "id": "vram-chart",
                    "type": "chart",
                    "chartId": "vram-length",
                    "layout": "full",
                },
                {
                    "id": "results-table",
                    "type": "table",
                    "tableId": "condition-results",
                    "layout": "full",
                },
                {
                    "id": "warmup-load",
                    "type": "markdown",
                    "sourceId": "formal-summary",
                    "layout": "full",
                    "body": (
                        "## Load and fresh-process warmup are separate from steady latency\n\nMedian model/codec load wall was "
                        f"{load_python:.3f} s for Python and {load_wgpu:.3f} s for WGPU. One fresh campaign-local CubeCL environment and driver cache were primed, then imported into every WGPU session. Compute pipelines remained process-local and were rebuilt after each launch; the first request is stored separately from the 10-request steady median. This campaign does not report external launch-to-WAV-close cold E2E, because each process performs two warmup requests before its measured set."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "formal-summary",
                    "layout": "full",
                    "body": (
                        "## Methodology and semantic-work contract\n\nBoth runtimes used strict FP32, TF32/autocast off, 40 Euler evaluations, independent CFG, text/caption/speaker scales 3/4/5, CFG window 0.5–1.0, linear schedule, fixed shared FP32 noise, and no tail trim or watermark. The 41 schedule bits match exactly. Text-only uses 60 effective rows; design and prepared clone use 80. Each request performs 40 whole-model forwards, 12 layers, and 480 block calls. The runtimes do the same semantic work, not the same operator graph."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "formal-summary",
                    "layout": "full",
                    "body": (
                        "## Limitations\n\nThis is one NVIDIA/Vulkan/CUDA laptop GPU and one pinned driver. Durations are exact fixed frame counts, so duration prediction is excluded. Clone uses prepared references; raw reference encoding is measured only in preparation and excluded from online latency. Fixed noise is necessary for numerical comparison but does not reproduce an official sample's random seed, automatic duration, or every request input. NVML is device-level memory used under an exclusive GPU lock, not a per-allocation trace."
                    ),
                },
                {
                    "id": "next-work",
                    "type": "markdown",
                    "sourceId": "implementation",
                    "layout": "full",
                    "body": (
                        "## Recommended next work\n\n1. Stop performance adoption and localize the first numerical divergence for 333/489/685-frame clone and 489-frame design, preserving the 80 dB hard / 85 dB target split.\n2. Profile the 40-step RF path at the matched schedule; the product-path gap compounds across 40 forwards, so codec-only wins cannot offset it.\n3. Reduce prepared-weight residency only after verifying every required shape has a valid fallback-free route; the present all-resident path costs about 1.35 GiB more internal live memory than PyTorch.\n4. Run a separate cold external-process and duration-prediction campaign; do not mix it with this steady all-resident estimator."
                    ),
                },
                {
                    "id": "reproduction",
                    "type": "markdown",
                    "sourceId": "raw-evidence",
                    "layout": "full",
                    "body": (
                        "## Reproduction and pins\n\nUse source commit `"
                        f"{source_commit}`, model revision `{protocol['model_revision']}`, codec revision `{protocol['codec_revision']}`, GPU `{protocol['hardware']['gpu']}`, driver `{protocol['hardware']['driver']}`, PCI `{protocol['hardware']['pci_bus_id']}`, CUDA/NVML index 0, and WGPU adapter 0. The SHA256SUMS file digest is `{checksum_digest}` and covers 1,723 files. Run `scripts/run_v4_40step_formal_compare.sh --output-dir <fresh-path>`; existing paths are rejected and automatic retries are disabled."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "conditions": conditions,
                "session_pairs": session_pairs,
                "vram_by_length": vram_by_length,
            },
        },
        "sources": all_sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
