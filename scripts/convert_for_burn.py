#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "safetensors>=0.5",
# ]
# ///
"""
Convert a Python Irodori-TTS safetensors checkpoint for use with the Rust/burn port.

The only change needed is renaming the `cond_module` keys:
  cond_module.0.weight  →  cond_module.linear0.weight
  cond_module.2.weight  →  cond_module.linear1.weight
  cond_module.4.weight  →  cond_module.linear2.weight

All other keys match the Rust field names exactly.

Dry-run by default: pass --apply / -a to write the output file.

Usage:
    uv run scripts/convert_for_burn.py <input.safetensors> <output.safetensors>
    uv run scripts/convert_for_burn.py <input.safetensors> <output.safetensors> --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


RENAMES: dict[str, str] = {
    "cond_module.0.weight": "cond_module.linear0.weight",
    "cond_module.2.weight": "cond_module.linear1.weight",
    "cond_module.4.weight": "cond_module.linear2.weight",
}


def convert(input_path: Path, output_path: Path, *, apply: bool) -> None:
    tensors: dict = {}
    metadata: dict[str, str] | None = None

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            new_key = RENAMES.get(key, key)
            tensors[new_key] = f.get_tensor(key)
            if new_key != key:
                print(f"  RENAME: {key!r}  →  {new_key!r}")

    renamed_count = sum(1 for k in tensors if k in RENAMES.values())
    print(
        f"Tensors: {len(tensors)} total, "
        f"{renamed_count} renamed, "
        f"metadata keys: {list(metadata.keys()) if metadata else []}"
    )

    if apply:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(output_path), metadata=metadata)
        print(f"Written: {output_path}")
    else:
        print(
            f"\n[dry-run] Would write {len(tensors)} tensors to {output_path}.\n"
            "Pass --apply / -a to actually write the file."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Source safetensors checkpoint")
    parser.add_argument("output", type=Path, help="Destination safetensors checkpoint")
    parser.add_argument(
        "--apply",
        "-a",
        action="store_true",
        default=False,
        help="Actually write the output file (default: dry-run)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output, apply=args.apply)


if __name__ == "__main__":
    main()
