# /// script
# requires-python = ">=3.12"
# dependencies = ["faster-whisper", "numpy", "scipy", "safetensors", "huggingface-hub"]
# ///
"""Compare an oracle and generated audio with the same Japanese ASR/CER metric.

This is an intelligibility proxy, not a listening test or perceptual quality
score. Inputs are a JSON array of {label, text, audio, sample_rate}; audio can
be .f32le or a safetensors fixture's expected_audio. CUDA libraries must be
available before starting uv. No dependency is added to the Rust runtime.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from safetensors import safe_open
from scipy.signal import resample_poly


def normalize(text):
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", text)
                   if not unicodedata.category(c).startswith(("P", "Z", "S", "C")))


def distance(expected, actual):
    row = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        next_row = [i]
        for j, right in enumerate(actual, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (left != right)))
        row = next_row
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face model commit")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-fA-F]{40}", args.revision) is None:
        raise ValueError("revision must be an immutable 40-digit model commit")
    if args.output.exists():
        raise FileExistsError(args.output)
    model_path = snapshot_download(args.model, revision=args.revision)
    model = WhisperModel(model_path, device=args.device,
                         compute_type="float16" if args.device == "cuda" else "int8", cpu_threads=4)
    results = []
    output = {"model": args.model, "revision": args.revision, "device": args.device,
              "faster_whisper_version": importlib.metadata.version("faster-whisper"),
              "ctranslate2_version": importlib.metadata.version("ctranslate2"),
              "language": "ja", "beam_size": 5, "temperature": 0,
              "normalization": "NFKC lowercase, omit punctuation/space/symbol/control", "results": results}
    for case in json.loads(args.manifest.read_text()):
        path = Path(case["audio"])
        if path.suffix == ".safetensors":
            with safe_open(path, framework="np") as tensors:
                audio = tensors.get_tensor("expected_audio").ravel()
        else:
            audio = np.fromfile(path, dtype="<f4")
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"invalid audio: {path}")
        rate = case["sample_rate"]
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        segments, _ = model.transcribe(samples, language="ja", beam_size=5, temperature=0,
                                      vad_filter=False, condition_on_previous_text=False)
        transcript = "".join(segment.text for segment in segments)
        expected, actual = normalize(case["text"]), normalize(transcript)
        if not expected:
            raise ValueError("empty normalized reference text")
        errors = distance(expected, actual)
        result = dict(case, transcript=transcript, normalized_expected=expected,
                      normalized_transcript=actual, edit_distance=errors,
                      reference_characters=len(expected), cer=errors / len(expected),
                      audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest())
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
