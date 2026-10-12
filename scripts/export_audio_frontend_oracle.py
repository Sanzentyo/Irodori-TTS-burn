# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch==2.10.0", "torchaudio==2.10.0", "pyloudnorm==0.2.0"]
# ///
"""Reproduce tests/fixtures/audio-preprocessing.json on the CPU.

uv run scripts/export_audio_frontend_oracle.py /tmp/audio-preprocessing.json
Torch versions match the pinned Irodori comparison environment.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyloudnorm
import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = []
    for rate in (16000, 22050, 44100, 48000, 96000):
        count = rate // 2 + 123
        values = np.asarray(
            [
                (
                    0.15 * math.sin(2 * math.pi * 173 * i / rate)
                    + 0.07 * math.sin(2 * math.pi * 17731 * i / rate)
                )
                * (0.7 + 0.3 * math.sin(math.pi * i / count))
                for i in range(count)
            ],
            dtype=np.float32,
        )
        output = torchaudio.functional.resample(
            torch.from_numpy(values),
            rate,
            48000,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        ).numpy()
        indices = [
            0,
            1,
            2,
            3,
            11,
            44,
            123,
            321,
            777,
            len(output) // 2,
            *range(len(output) - 4, len(output)),
        ]
        cases.append(
            {
                "rate": rate,
                "count": count,
                "indices": indices,
                "expected": output[indices].tolist(),
                "output_count": len(output),
                "loudness": pyloudnorm.Meter(rate).integrated_loudness(values),
            }
        )
    with args.output.open("x") as stream:
        json.dump(
            {
                "source": "torchaudio.functional.resample and pyloudnorm.Meter",
                "torchaudio": torchaudio.__version__,
                "cases": cases,
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
