# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub",
# ]
# ///
"""Download Aratako/Irodori-TTS-500M-v2 weights from HuggingFace."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Aratako/Irodori-TTS-500M-v2"
FILENAME = "model.safetensors"
OUT_DIR = Path("target/hf_model")

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / FILENAME

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"Already downloaded: {out_path} ({size_mb:.0f} MB)")
        return

    print(f"Downloading {REPO_ID}/{FILENAME} …")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(OUT_DIR),
    )
    size_mb = Path(local_path).stat().st_size / 1e6
    print(f"Saved to {local_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
