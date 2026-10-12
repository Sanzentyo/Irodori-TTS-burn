# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors"]
# ///
"""Export a bench_v4_residency diagnostic run for bench_speech_cpp.cpp.

Run with uv run scripts/prepare_speech_cpp_comparison.py REPORT.json OUTPUT_DIR.
Requires --designed --diagnostic-forward-ordinal 0 and independent Euler CFG
with text=3, caption=4, min_t=0.5, max_t=1, without truncation or rescaling.
The C++ runner consumes the same precomputed conditions, noise and F32 weights.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        help="original request fixture, to also benchmark condition encoding",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    trace = report["diagnostic_artifacts"]
    work = report["work_reports"][trace["request"] - 1]
    if not (
        report["strict_fp32"]
        and report["designed"]
        and not report["unconditioned"]
        and report["latency_results_valid"] is False
        and trace["excluded_from_latency_comparisons"]
        and report["cfg_caption"] == 4
        and work["method"] == "euler"
        and work["guidance_mode"] == "independent"
    ):
        raise ValueError(
            "runner requires a strict-FP32 designed-voice diagnostic with independent Euler CFG"
        )
    tensors = {t["name"]: t for t in trace["tensors"]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixture_hash = None
    if args.fixture:
        fixture_hash = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
        if report["fixture_sha256"] != [fixture_hash]:
            raise ValueError(
                "token export requires the exact single-fixture diagnostic"
            )
        fixture = load_file(args.fixture)
        for condition in ("text", "caption"):
            ids = fixture[f"inputs/{condition}_input_ids"].reshape(-1)
            mask = fixture[f"inputs/{condition}_mask"].reshape(-1).astype(bool)
            ids = ids[mask]
            if not ids.size or np.any(ids < 0) or np.any(ids >= 102400):
                raise ValueError("invalid compacted v4 token IDs")
            np.save(args.output_dir / f"{condition}_ids.npy", ids.astype("<i4"))

    def read(name):
        tensor = tensors[name]
        raw = Path(tensor["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != tensor["sha256"]:
            raise ValueError(f"hash mismatch: {name}")
        return np.frombuffer(raw, dtype="<f4").reshape(tensor["shape"])

    def save(name, values):
        np.save(args.output_dir / name, np.ascontiguousarray(values, dtype="<f4"))

    for condition in ("text", "caption"):
        save(condition, read(f"rf_selected_{condition}_state")[0])
    save("noise", read("rf_forward_input_00_step_00")[0])
    save("reference_latent", read("rf_final_patched")[0])
    times = np.asarray(work["schedule_f32_bits"], dtype="<u4").view("<f4")
    save("times", times)
    for step in range(work["num_steps"]):
        save(f"input_{step:02}", read(f"rf_forward_input_{step:02}_step_{step:02}")[0])
        save(f"output_{step:02}", read(f"rf_forward_{step:02}_step_{step:02}"))
    audio = next(
        a for a in report["audio_artifacts"] if a["request"] == trace["request"]
    )
    raw = Path(audio["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != audio["sha256"]:
        raise ValueError("audio hash mismatch")
    save("reference_audio", np.frombuffer(raw, dtype="<f4"))
    manifest = {
        "source_report": str(args.report.resolve()),
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "frames": int(read("rf_final_patched").shape[1]),
        "num_steps": work["num_steps"],
        "token_fixture_sha256": fixture_hash,
        "guidance": {"text": 3, "caption": 4, "speaker": 0, "min_t": 0.5, "max_t": 1},
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output_dir.glob("*.npy"))
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Exported {len(manifest['files'])} arrays to {args.output_dir}")


if __name__ == "__main__":
    main()
