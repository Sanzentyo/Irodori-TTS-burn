"""Benchmark the Python DACVAE codec encode/decode speed.

Run with:
    cd ../Irodori-TTS  # relative to this project root
    uv run --extra dev python ../Irodori-TTS-burn/scripts/bench_codec_py.py \
        --model-path ~/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth

Prints median wall-clock time (ms) for each operation.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import torch

from irodori_tts.codec import DACVAECodec

SAMPLE_RATE = 48_000
HOP_LENGTH = 1_920


def make_sine(seconds: float, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Return a [1, 1, N] sine-wave tensor at 440 Hz, amplitude 0.01."""
    n = round(sample_rate * seconds)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    wave = 0.01 * torch.sin(2 * math.pi * 440.0 * t)
    return wave.reshape(1, 1, n)


def make_zero_latent(seconds: float) -> torch.Tensor:
    n_frames = round(SAMPLE_RATE * seconds) // HOP_LENGTH
    return torch.zeros(1, n_frames, 32)


def bench(name: str, fn, n_warmup: int = 3, n_runs: int = 10) -> float:
    """Run *fn* several times and return the median wall-clock time in ms."""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    median = statistics.median(times)
    mean = statistics.mean(times)
    print(f"[bench] {name}: median={median:.1f}ms  mean={mean:.1f}ms  runs={n_runs}")
    return median


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DACVAE Python codec.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to weights.pth for the DACVAE model.",
    )
    parser.add_argument("--device", default="cpu", help="torch device (cpu/cuda/mps)")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    # Load codec (normalize_db=None to match Rust which does no loudness normalization)
    codec = DACVAECodec.load(
        repo_id=str(args.model_path),
        device=args.device,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=None,
    )

    print(f"\n=== Python DACVAE codec benchmark (device={args.device}) ===\n")

    for duration in [1.0, 5.0]:
        audio = make_sine(duration).to(args.device)
        latent_pre = make_zero_latent(duration).to(args.device)

        bench(
            f"encode_{duration:.0f}s_sine",
            lambda a=audio: codec.encode_waveform(a, SAMPLE_RATE),
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
        )
        bench(
            f"decode_{duration:.0f}s_zero_latent",
            lambda lp=latent_pre: codec.decode_latent(lp),
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
        )
        bench(
            f"roundtrip_{duration:.0f}s",
            lambda a=audio: codec.decode_latent(codec.encode_waveform(a, SAMPLE_RATE)),
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
        )
        print()


if __name__ == "__main__":
    main()
