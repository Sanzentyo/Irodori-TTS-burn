# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "torchaudio>=2.0",
#     "safetensors>=0.4",
#     "soundfile",
#     "numpy",
#     "tqdm",
# ]
# ///
"""
Encode a dataset of WAV files into latent safetensors files and produce a
JSONL manifest suitable for LoRA training.

Each line of the manifest has the form::

    {"text": "...", "latent_path": "...", "ref_latent_path": "..."}

``ref_latent_path`` is only written when ``--ref-audio-dir`` is supplied.

Usage
-----
Encode every WAV in an audio directory using a tab-separated transcript file::

    uv run scripts/encode_dataset.py \\
        --audio-dir data/wavs \\
        --transcript data/transcripts.tsv \\
        --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \\
        --output-dir data/latents \\
        --manifest data/train.jsonl \\
        --apply

The transcript TSV must have two columns: ``stem`` (filename without extension) and
``text``.  Example::

    utt001\\tHello world.
    utt002\\tこんにちは世界。

Latents are saved as ``<output-dir>/<stem>.safetensors`` under the key
``"latent"`` with shape ``[S, D]`` (float32).

If ``--ref-audio-dir`` is given, each utterance is looked up in that directory
as well; the encoded result is stored as ``<stem>_ref.safetensors`` and
referenced in the manifest.

Dry-run mode
------------
By default the script only prints what it *would* do.  Pass ``--apply`` / ``-a``
to actually encode and write files.

Running inside the Irodori-TTS environment
------------------------------------------
The script imports ``irodori_tts.codec``.  Make sure the Irodori-TTS venv is
active, e.g. via the justfile recipe::

    just encode-dataset \\
        --audio-dir data/wavs \\
        --transcript data/transcripts.tsv \\
        --output-dir data/latents \\
        --manifest data/train.jsonl \\
        --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add Irodori-TTS to the Python path when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Irodori-TTS"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_codec(repo_or_path: str, device: str, dtype_str: str):
    """Load the DACVAECodec from a HuggingFace repo or local HF snapshot directory."""
    import torch
    from irodori_tts.codec import DACVAECodec  # type: ignore[import]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]
    print(f"[encode] Loading DACVAE codec from '{repo_or_path}' on {device} ({dtype_str}) …")
    codec = DACVAECodec.load(
        repo_or_path=repo_or_path,
        device=device,
        dtype=dtype,
        deterministic_encode=True,
    )
    print(f"[encode] latent_dim={codec.latent_dim}, sample_rate={codec.sample_rate}")
    return codec


def _encode_wav(codec, wav_path: Path):
    """Encode a WAV file and return a float32 numpy array of shape [S, D]."""
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
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed TSV at line {lineno} (expected tab-separated): {line!r}"
                )
            result[parts[0].strip()] = parts[1].strip()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode WAV files to DACVAE latent safetensors and write a training manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio-dir", required=True, help="Directory containing input WAV files.")
    p.add_argument(
        "--transcript",
        required=True,
        help="Tab-separated file: <stem>\\t<text> (one utterance per line).",
    )
    p.add_argument(
        "--codec-repo",
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
        help="HuggingFace repo or local HF snapshot directory for the DACVAE codec.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-utterance latent .safetensors files are written.",
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="Output JSONL manifest path.",
    )
    p.add_argument(
        "--ref-audio-dir",
        default=None,
        help="Optional directory of reference/speaker-conditioning WAV files (same stems).",
    )
    p.add_argument("--ext", default=".wav", help="Audio file extension to scan for.")
    p.add_argument("--device", default="cpu", help="Torch device for codec inference.")
    p.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Codec model precision.",
    )
    p.add_argument(
        "--apply", "-a",
        action="store_true",
        help="Actually encode and write files (default: dry-run only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    ref_audio_dir = Path(args.ref_audio_dir) if args.ref_audio_dir else None

    # -----------------------------------------------------------------------
    # Load transcript
    # -----------------------------------------------------------------------
    transcripts = _load_transcripts(Path(args.transcript))
    if not transcripts:
        print("[encode] ERROR: transcript file is empty.", file=sys.stderr)
        sys.exit(1)
    print(f"[encode] {len(transcripts)} utterances in transcript.")

    # -----------------------------------------------------------------------
    # Dry-run mode
    # -----------------------------------------------------------------------
    if not args.apply:
        print("=== DRY RUN — pass --apply / -a to actually encode ===")
        items = list(transcripts.items())[:5]
        for stem, text in items:
            lat = output_dir / f"{stem}.safetensors"
            ref = (ref_audio_dir / f"{stem}_ref.safetensors") if ref_audio_dir else None
            entry: dict = {"text": text[:60] + "…" if len(text) > 60 else text, "latent_path": str(lat)}
            if ref:
                entry["ref_latent_path"] = str(ref)
            print(f"  {entry}")
        if len(transcripts) > 5:
            print(f"  … and {len(transcripts) - 5} more")
        print(f"Output manifest: {manifest_path}")
        return

    # -----------------------------------------------------------------------
    # Apply: load codec and encode
    # -----------------------------------------------------------------------
    try:
        codec = _load_codec(args.codec_repo, args.device, args.precision)
    except ImportError as exc:
        sys.exit(
            f"Could not import irodori_tts: {exc}\n"
            "Run this script inside the Irodori-TTS venv or via:\n"
            "  just encode-dataset ..."
        )

    try:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(list(transcripts.items()), desc="encoding")
    except ImportError:
        iterator = list(transcripts.items())  # type: ignore[assignment]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for stem, text in iterator:
            wav_path = audio_dir / f"{stem}{args.ext}"
            if not wav_path.exists():
                print(f"[encode] SKIP: {wav_path} not found.", file=sys.stderr)
                skipped += 1
                continue

            # Encode target latent
            try:
                latent = _encode_wav(codec, wav_path)
            except Exception as exc:
                print(f"[encode] SKIP: encode failed for {wav_path}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            lat_out = output_dir / f"{stem}.safetensors"
            _save_safetensors(latent, lat_out)

            record: dict = {
                "text": text,
                "latent_path": str(lat_out),
            }

            # Encode reference (speaker conditioning) latent if available
            if ref_audio_dir is not None:
                ref_wav = ref_audio_dir / f"{stem}{args.ext}"
                if ref_wav.exists():
                    try:
                        ref_latent = _encode_wav(codec, ref_wav)
                        ref_out = output_dir / f"{stem}_ref.safetensors"
                        _save_safetensors(ref_latent, ref_out)
                        record["ref_latent_path"] = str(ref_out)
                    except Exception as exc:
                        print(
                            f"[encode] WARN: ref encode failed for {ref_wav}: {exc} — omitting ref.",
                            file=sys.stderr,
                        )

            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(transcripts)
    encoded = total - skipped
    print(f"[encode] Done. Encoded {encoded}/{total} utterances → {manifest_path}")
    if skipped:
        print(f"[encode] Skipped {skipped} utterances.", file=sys.stderr)


if __name__ == "__main__":
    main()

