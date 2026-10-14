# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare saved --dump-stages runs without loading a model or starting a GPU.

uv run scripts/compare_synthesis_stages.py REFERENCE_DIR CANDIDATE_DIR
Reports the first observed divergence; this does not establish its root cause.
"""

import argparse
import hashlib
import json
import math
import sys
from array import array
from pathlib import Path


def load_report(directory):
    report = json.loads((directory / "report.json").read_text())
    capture = report["stage_capture"]
    if capture["schema_version"] != 2 or not capture["tensors"]:
        raise ValueError("unsupported or empty stage capture")
    names = [t["name"] for t in capture["tensors"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate stage names")
    return report


def values(directory, tensor):
    filename = tensor["file"]
    if Path(filename).name != filename or not filename.endswith(".f32le"):
        raise ValueError("stage file must be a local .f32le filename")
    shape = tensor["shape"]
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid stage shape")
    data = (directory / filename).read_bytes()
    elements = math.prod(shape)
    if tensor["elements"] != elements or len(data) != elements * 4:
        raise ValueError("stage byte count does not match shape")
    if hashlib.sha256(data).hexdigest() != tensor["sha256"]:
        raise ValueError("stage checksum mismatch")
    output = array("f")
    output.frombytes(data)
    if sys.byteorder != "little":
        output.byteswap()
    if any(not math.isfinite(x) for x in output):
        raise ValueError("non-finite stage values")
    return output


def compare(reference, candidate, atol=0.0, rtol=0.0):
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    ref, actual = load_report(reference), load_report(candidate)
    if (
        ref["declared_source_checkpoint_sha256"]
        != actual["declared_source_checkpoint_sha256"]
    ):
        raise ValueError("different source checkpoints")
    left, right = ref["stage_capture"], actual["stage_capture"]
    if left["fixture_sha256"] != right["fixture_sha256"]:
        raise ValueError("different input fixtures")
    if left["codec_sha256"] != right["codec_sha256"]:
        raise ValueError("different codec checkpoints")
    for key in [
        "method",
        "num_steps",
        "guidance_mode",
        "schedule_f32_bits",
        "requested",
        "enabled_cfg",
    ]:
        if ref["sampler_work"][key] != actual["sampler_work"][key]:
            raise ValueError(f"different sampler contract: {key}")
    if [t["name"] for t in left["tensors"]] != [t["name"] for t in right["tensors"]]:
        raise ValueError("stage names/order differ")
    stages = []
    for a, b in zip(left["tensors"], right["tensors"], strict=True):
        if a["shape"] != b["shape"]:
            raise ValueError(f"stage shape differs: {a['name']}")
        x, y = values(reference, a), values(candidate, b)
        error = math.fsum((u - v) ** 2 for u, v in zip(x, y))
        energy = math.fsum(u * u for u in x)
        max_abs = max(abs(u - v) for u, v in zip(x, y))
        stages.append(
            {
                "name": a["name"],
                "elements": len(x),
                "bitwise_equal": a["sha256"] == b["sha256"],
                "within_tolerance": all(
                    abs(u - v) <= atol + rtol * abs(u) for u, v in zip(x, y)
                ),
                "max_abs": max_abs,
                "rmse": math.sqrt(error / len(x)),
                "reference_rms": math.sqrt(energy / len(x)),
                "snr_db": 10 * math.log10(energy / error)
                if energy > 0 and error > 0
                else None,
            }
        )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "new_gpu_measurements": False,
        "atol": atol,
        "rtol": rtol,
        "reference_precision": ref["precision"],
        "candidate_precision": actual["precision"],
        "reference_state": ref["sampler_work"]["state_precision"],
        "candidate_state": actual["sampler_work"]["state_precision"],
        "first_bitwise_difference": next(
            (s["name"] for s in stages if not s["bitwise_equal"]), None
        ),
        "first_tolerance_failure": next(
            (s["name"] for s in stages if not s["within_tolerance"]), None
        ),
        "stages": stages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(args.reference, args.candidate, args.atol, args.rtol),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
