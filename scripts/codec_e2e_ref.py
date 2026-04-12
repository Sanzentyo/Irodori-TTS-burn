#!/usr/bin/env python3
"""
Codec E2E reference script.

Encodes a WAV file (or synthesises a sine tone) using the Python DACVAE codec
and saves the latent as a .npy file for comparison with the Rust implementation.

Usage:
    uv run scripts/codec_e2e_ref.py \
        --output /tmp/py_latent.npy \
        [--audio path/to/audio.wav] \
        [--save-audio target/test_audio.wav]

    Then:
    cargo run --bin codec_e2e --release -- \
        --weights target/dacvae_weights.safetensors \
        --ref-latent /tmp/py_latent.npy \
        [--audio target/test_audio.wav]
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "numpy",
#     "soundfile",
# ]
# ///

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Irodori-TTS"))

import argparse
import numpy as np
import soundfile as sf
import torch

from irodori_tts.codec import DACVAECodec


def make_sine(sample_rate: int = 48000, duration: float = 0.5, freq: float = 440.0) -> np.ndarray:
    """Generate a mono sine wave as float32 in [-1, 1]."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce DACVAE reference latent (Python)")
    parser.add_argument(
        "--output", default="/tmp/py_latent.npy", help="Path to write the .npy latent"
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to a WAV file to encode (default: synthesise a 0.5-s sine tone)",
    )
    parser.add_argument(
        "--model-dir",
        default="target/Aratako_Semantic-DACVAE-Japanese-32dim",
        help="Directory containing the DACVAE model checkpoint",
    )
    parser.add_argument(
        "--save-audio",
        default=None,
        help="If set, also save the input audio as a WAV file (for the Rust binary)",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[ref] Loading codec from {args.model_dir} …")
    codec = DACVAECodec.load(
        repo_id=args.model_dir,
        device="cpu",
        enable_watermark=False,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=None,  # disable loudness normalisation for parity test
    )
    print("[ref] Codec loaded.")

    # ── Load/synthesise audio ─────────────────────────────────────────────────
    sr = 48000
    if args.audio:
        wav_np, file_sr = sf.read(args.audio, dtype="float32", always_2d=False)
        if wav_np.ndim == 2:
            wav_np = wav_np[:, 0]  # take first channel
        if file_sr != sr:
            print(f"[ref] WARNING: audio is {file_sr} Hz, expected {sr} Hz — no resampling done")
    else:
        print("[ref] No audio supplied — synthesising 0.5 s sine tone at 440 Hz")
        wav_np = make_sine(sr)

    # ── Optionally save the audio for Rust ────────────────────────────────────
    if args.save_audio:
        sf.write(args.save_audio, wav_np, sr, subtype='FLOAT')
        print(f"[ref] Saved input audio (float32) to {args.save_audio}")

    print(f"[ref] Audio shape: {wav_np.shape}, dtype: {wav_np.dtype}")

    # ── Encode ────────────────────────────────────────────────────────────────
    # normalize_db=None: no loudness normalisation, ensures Rust gets the exact same samples
    wav_t = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    latent = codec.encode_waveform(wav_t, sample_rate=sr, normalize_db=None)  # [1, T', 32]

    print(f"[ref] Latent shape:   {tuple(latent.shape)}")
    print(f"[ref] Latent[0,0,:8]: {latent[0, 0, :8].tolist()}")

    latent_np = latent.cpu().numpy()
    np.save(args.output, latent_np)
    print(f"[ref] Saved reference latent to {args.output}")

    # ── Decode round-trip ─────────────────────────────────────────────────────
    recon = codec.decode_latent(latent)  # [1, 1, T']
    print(f"[ref] Reconstructed audio shape: {tuple(recon.shape)}")

    print("[ref] Done.")


if __name__ == "__main__":
    main()

