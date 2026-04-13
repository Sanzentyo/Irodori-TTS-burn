#!/usr/bin/env python3
"""
Encode a dataset of WAV files into latent safetensors files and produce a
JSONL manifest suitable for LoRA training.

Each line of the manifest has the form:
    {"text": "...", "latent_path": "relative/to/manifest.jsonl", "ref_latent_path": "..."}

``ref_latent_path`` is only written when ``--ref-audio-dir`` is supplied.

Usage
-----
Encode every WAV in an audio directory, using a tab-separated transcript file:

    uv run scripts/encode_dataset.py \\
        --audio-dir data/wavs \\
        --transcript data/transcripts.tsv \\
        --codec-weights target/dacvae_weights.safetensors \\
        --output-dir data/latents \\
        --manifest data/train.jsonl

The transcript TSV must have two columns: ``filename`` (without extension) and
``text``.  Example::

    utt001\\tHello world.
    utt002\\tこんにちは世界。

Latents are saved as ``<output-dir>/<filename>.safetensors`` under the key
``"latent"`` with shape ``[S, D]`` (float32).

If ``--ref-audio-dir`` is given, each utterance ID is looked up in that
directory as well; the encoded result is stored as ``<filename>_ref.safetensors``
and referenced in the manifest.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "safetensors>=0.4",
#     "soundfile",
#     "numpy",
#     "tqdm",
# ]
# ///

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Irodori-TTS"))


def _load_codec(weights_path: Path):
    """Load the Rust-side DACVAE codec via the Python reference implementation."""
    from irodori_tts.codec import DACVAECodec

    print(f"[encode] Loading DACVAE codec from {weights_path} …")
    codec = DACVAECodec.load(str(weights_path))
    codec.eval()
    return codec


def _encode_wav(codec, wav_path: Path):
    """Return a float32 numpy array of shape [S, D] for *wav_path*."""
    import numpy as np
    import soundfile as sf
    import torch

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    wav_t = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        latent = codec.encode_waveform(wav_t, sample_rate=sr, normalize_db=None)  # [1, S, D]
    return latent.squeeze(0).cpu().float().numpy()  # [S, D]


def _save_safetensors(array, out_path: Path) -> None:
    """Save a float32 numpy array as a single-tensor safetensors file (key "latent")."""
    import numpy as np
    from safetensors.numpy import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"latent": np.ascontiguousarray(array, dtype=np.float32)}, str(out_path))


def _load_transcripts(tsv_path: Path) -> dict[str, str]:
    """Read a two-column TSV (stem, text) and return {stem: text}."""
    result: dict[str, str] = {}
    with tsv_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed TSV line (expected tab-separated): {line!r}")
            result[parts[0].strip()] = parts[1].strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode WAV files to latent safetensors and write a training manifest."
    )
    parser.add_argument("--audio-dir", required=True, help="Directory containing input WAV files.")
    parser.add_argument(
        "--transcript",
        required=True,
        help="Tab-separated file: <stem>\\t<text> (one utterance per line).",
    )
    parser.add_argument(
        "--codec-weights",
        required=True,
        help="DACVAE safetensors checkpoint (dacvae_weights.safetensors).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-utterance latent .safetensors files are written.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--ref-audio-dir",
        default=None,
        help="Optional directory of reference/speaker-conditioning WAV files (same stems).",
    )
    parser.add_argument(
        "--ext",
        default=".wav",
        help="Audio file extension to look for (default: .wav).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for codec inference (default: cpu).",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    ref_audio_dir = Path(args.ref_audio_dir) if args.ref_audio_dir else None

    codec = _load_codec(Path(args.codec_weights))

    import torch

    codec = codec.to(args.device)

    transcripts = _load_transcripts(Path(args.transcript))
    if not transcripts:
        print("[encode] ERROR: transcript file is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    items = list(transcripts.items())
    iterator = tqdm(items, desc="encoding") if tqdm else items

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for stem, text in iterator:
            wav_path = audio_dir / f"{stem}{args.ext}"
            if not wav_path.exists():
                print(f"[encode] WARNING: {wav_path} not found — skipping.", file=sys.stderr)
                skipped += 1
                continue

            # Encode target latent
            latent = _encode_wav(codec, wav_path)
            lat_out = output_dir / f"{stem}.safetensors"
            _save_safetensors(latent, lat_out)

            record: dict = {
                "text": text,
                "latent_path": str(lat_out.relative_to(manifest_path.parent)),
            }

            # Encode reference (speaker conditioning) latent if available
            if ref_audio_dir is not None:
                ref_wav = ref_audio_dir / f"{stem}{args.ext}"
                if ref_wav.exists():
                    ref_latent = _encode_wav(codec, ref_wav)
                    ref_out = output_dir / f"{stem}_ref.safetensors"
                    _save_safetensors(ref_latent, ref_out)
                    record["ref_latent_path"] = str(ref_out.relative_to(manifest_path.parent))

            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(transcripts)
    encoded = total - skipped
    print(f"[encode] Done. Encoded {encoded}/{total} utterances → {manifest_path}")
    if skipped:
        print(f"[encode] Skipped {skipped} utterances (WAV not found).", file=sys.stderr)


if __name__ == "__main__":
    main()
