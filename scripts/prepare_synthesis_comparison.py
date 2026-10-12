# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors"]
# ///
"""Prepare bench_speech_cpp's oracle mode from export_meanflow_oracle.py output.

The checkpoint must also be converted with the pinned speech.cpp converter.
Masked positions are removed, as speech.cpp's public text encoder takes compact
token sequences. No model-derived condition values are substituted.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with safe_open(args.fixture, framework="np") as fixture:
        metadata = fixture.metadata()
        if metadata.get("reference"):
            raise ValueError(
                "oracle comparison harness currently supports text/caption without a reference voice"
            )
        for name in ("text", "caption"):
            ids = fixture.get_tensor(f"{name}_ids")
            mask = fixture.get_tensor(f"{name}_mask")
            assert ids.shape[0] == 1 and mask.shape == ids.shape
            kept = ids[mask].astype(np.int32)
            np.save(args.output / f"{name}_ids.npy", kept)
            # Only the first dimension is consumed in oracle mode. Conditions
            # themselves are encoded by speech.cpp inside the measured boundary.
            np.save(args.output / f"{name}.npy", np.empty((len(kept), 0), np.float32))
        noise = fixture.get_tensor("initial_noise")
        assert noise.shape[0] == 1
        np.save(args.output / "noise.npy", noise[0])
        np.save(
            args.output / "reference_latent.npy",
            fixture.get_tensor("expected_latent")[0],
        )
        np.save(
            args.output / "reference_audio.npy",
            fixture.get_tensor("expected_audio").ravel(),
        )
        # The C++ runtime uses its own native schedule. This array supplies only
        # the step count; no forward-isolation diagnostic runs in oracle mode.
        np.save(
            args.output / "times.npy", np.zeros(int(metadata["steps"]) + 1, np.float32)
        )
    metadata["fixture_sha256"] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
