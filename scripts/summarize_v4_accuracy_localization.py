# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = []
# ///
"""Summarize forward-by-forward RF divergence and codec amplification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def first_below(rows: list[dict[str, Any]], threshold: float) -> int | None:
    return next(
        (int(row["step"]) for row in rows if float(row["snr_db"]) < threshold),
        None,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    cases: list[dict[str, Any]] = []
    for path in sorted((args.root / "cases").glob("*/comparison.json")):
        with path.open(encoding="utf-8") as file:
            comparison = json.load(file)["comparisons"]
        forwards = []
        for name, value in comparison.items():
            if not name.startswith("rf_forward_"):
                continue
            ordinal_text, step_text = name.removeprefix("rf_forward_").split(
                "_step_", maxsplit=1
            )
            forwards.append(
                {
                    "ordinal": int(ordinal_text),
                    "step": int(step_text),
                    "batch_rows": int(value["shape"][0]),
                    "snr_db": float(value["snr_db"]),
                    "max_abs": float(value["max_abs"]),
                    "rmse": float(value["rmse"]),
                    "cosine": float(value["cosine"]),
                }
            )
        forwards.sort(key=lambda row: int(row["ordinal"]))
        if len(forwards) != 40 or [row["step"] for row in forwards] != list(range(40)):
            raise ValueError(f"{path}: expected exactly 40 ordered Euler forwards")
        codec_input = comparison["codec_input_unpatched"]
        waveform = comparison["codec_output_untrimmed"]
        cases.append(
            {
                "case": path.parent.name,
                "classification": (
                    "rf_iteration_and_codec_amplification"
                    if float(codec_input["snr_db"]) < 90.0
                    else "codec_amplification_dominant"
                ),
                "first_forward_snr_db": forwards[0]["snr_db"],
                "cfg_last_batched_step_19_snr_db": forwards[19]["snr_db"],
                "first_unbatched_step_20_snr_db": forwards[20]["snr_db"],
                "cfg_transition_drop_db": forwards[20]["snr_db"]
                - forwards[19]["snr_db"],
                "worst_forward": min(forwards, key=lambda row: row["snr_db"]),
                "first_step_below_db": {
                    str(threshold): first_below(forwards, threshold)
                    for threshold in (110, 100, 95, 90, 85, 80)
                },
                "rf_final": codec_input,
                "waveform": waveform,
                "codec_snr_delta_db": float(waveform["snr_db"])
                - float(codec_input["snr_db"]),
                "forwards": forwards,
            }
        )
    payload = {
        "format": "irodori-v4-accuracy-localization-analysis-v1",
        "latency_values_used": False,
        "interpretation": {
            "no_operator_catastrophe_at_step_0": all(
                case["first_forward_snr_db"] >= 110.0 for case in cases
            ),
            "cfg_transition": "steps 0-19 use batched CFG; steps 20-39 use one row",
            "limits": (
                "later forward comparisons include already-diverged latent inputs; "
                "they localize iterative accumulation, not one kernel in isolation"
            ),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


if __name__ == "__main__":
    main()
