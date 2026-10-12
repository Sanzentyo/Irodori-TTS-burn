# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors"]
# ///
"""Compare speech.cpp's native sampler, isolated forwards and decoder with Burn.

uv run scripts/compare_speech_cpp.py INPUT_DIR SPEECH_OUTPUT_DIR REPORT.json
This measures numerical agreement, not perceptual or transcription accuracy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def metrics(reference, candidate):
    if reference.dtype != np.dtype("<f4") or candidate.dtype != np.dtype("<f4"):
        raise ValueError("comparison requires float32 tensors")
    if reference.size != candidate.size or not reference.size:
        raise ValueError(
            f"element count mismatch: {reference.size} vs {candidate.size}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("non-finite value in comparison")
    ref, got = (
        reference.ravel().astype(np.float64),
        candidate.ravel().astype(np.float64),
    )
    difference = got - ref
    signal = float(np.dot(ref, ref))
    error = float(np.dot(difference, difference))
    denominator = math.sqrt(signal * float(np.dot(got, got)))
    return {
        "elements": int(ref.size),
        "bitwise_equal": bool(
            np.array_equal(
                reference.ravel().view(np.uint32), candidate.ravel().view(np.uint32)
            )
        ),
        "max_abs": float(np.abs(difference).max()),
        "rmse": math.sqrt(error / ref.size),
        "relative_l2": math.sqrt(error / signal) if signal else None,
        "snr_db": 10 * math.log10(signal / error) if signal and error else None,
        "cosine": float(np.dot(ref, got)) / denominator if denominator else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("speech_output_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--python-oracle",
        type=Path,
        help="optional strict-F32 diagnostic safetensors from the official runtime",
    )
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"input hash mismatch: {name}")

    def compare(reference, candidate):
        return metrics(
            np.load(args.input_dir / (reference + ".npy")),
            np.fromfile(args.speech_output_dir / (candidate + ".f32le"), dtype="<f4"),
        )

    comparisons = {
        name: compare(ref, got)
        for name, ref, got in [
            ("native_schedule", "times", "native_times"),
            ("native_final_latent", "reference_latent", "latent"),
            ("native_audio", "reference_audio", "audio"),
            ("decoder_same_input", "reference_audio", "decode_reference"),
        ]
    }
    comparisons["forwards_same_input"] = [
        {"step": i, **compare(f"output_{i:02}", f"forward_{i:02}")}
        for i in range(manifest["num_steps"])
    ]
    for condition in ("text", "caption"):
        if (args.speech_output_dir / f"{condition}_state.f32le").is_file():
            comparisons[f"{condition}_condition"] = compare(
                condition, f"{condition}_state"
            )
    payload = {
        "reference_manifest": manifest,
        "candidate_directory": str(args.speech_output_dir.resolve()),
        "candidate_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.speech_output_dir.glob("*.f32le"))
        },
        "timings": json.loads((args.speech_output_dir / "timings.json").read_text()),
        "comparisons": comparisons,
    }
    if args.python_oracle:
        oracle = load_file(args.python_oracle)
        payload["python_oracle"] = {
            "path": str(args.python_oracle.resolve()),
            "sha256": hashlib.sha256(args.python_oracle.read_bytes()).hexdigest(),
            "comparisons": {
                name: {
                    "burn": metrics(
                        oracle[key], np.load(args.input_dir / (reference + ".npy"))
                    ),
                    "speech_cpp": metrics(
                        oracle[key],
                        np.fromfile(
                            args.speech_output_dir / (candidate + ".f32le"), dtype="<f4"
                        ),
                    ),
                }
                for name, key, reference, candidate in [
                    ("final_latent", "rf_final_patched", "reference_latent", "latent"),
                    ("audio", "codec_output_untrimmed", "reference_audio", "audio"),
                ]
            },
        }
    with args.report.open("x") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {k: v for k, v in comparisons.items() if k != "forwards_same_input"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
