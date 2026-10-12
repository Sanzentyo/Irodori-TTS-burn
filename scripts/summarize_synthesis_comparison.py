# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Compare host-complete speech.cpp benchmark outputs with the official oracle.

uv run scripts/summarize_synthesis_comparison.py INPUT_DIR RUN_DIR [RUN_DIR ...]
Input arrays come from prepare_synthesis_comparison.py; no alignment or scaling
is applied to the waveforms. Prints JSON and writes metrics.json in each run.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np


def compare(expected, actual):
    expected, actual = expected.ravel(), actual.ravel()
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("comparison shape mismatch or non-finite output")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    noise = float(np.dot(difference, difference))
    signal = float(np.dot(expected.astype(np.float64), expected.astype(np.float64)))
    return {
        "elements": actual.size,
        "max_abs": float(np.abs(difference).max()),
        "rmse": float(np.sqrt(noise / actual.size)),
        "snr_db": float(10 * np.log10(signal / noise)) if noise else None,
        "bitwise_equal": expected.tobytes() == actual.tobytes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    results = {}
    for run in args.runs:
        timings = json.loads((run / "timings.json").read_text())
        samples = [s for s in timings["samples"] if not s["warmup"]]
        result = {
            "measured_requests": len(samples),
            "rf_seconds": median(s["rf_seconds"] for s in samples),
            "audio_seconds": median(s["consumer_seconds"] for s in samples),
            "condition_seconds": median(s["condition_seconds"] for s in samples),
        }
        for key, reference in [
            ("latent", "latent"),
            ("audio", "audio"),
            ("decode_reference", "audio"),
        ]:
            path = run / f"{key}.f32le"
            result[key] = compare(
                np.load(args.input / f"reference_{reference}.npy"),
                np.fromfile(path, dtype="<f4"),
            )
            result[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        results[run.name] = result
        (run / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
