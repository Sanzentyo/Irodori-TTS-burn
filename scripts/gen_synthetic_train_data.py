# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "safetensors>=0.4",
#     "numpy",
# ]
# ///
"""
Generate synthetic training data for benchmarking LoRA training throughput.

Creates random latent safetensors + JSONL manifest, no real audio needed.

Usage:
    uv run scripts/gen_synthetic_train_data.py \
        --output-dir target/bench_data \
        --num-samples 100 \
        --latent-dim 32 \
        --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--min-steps", type=int, default=20, help="Min latent sequence length")
    parser.add_argument("--max-steps", type=int, default=100, help="Max latent sequence length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", "-a", action="store_true", help="Actually write files (dry-run otherwise)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    latent_dir = args.output_dir / "latents"
    manifest_path = args.output_dir / "train.jsonl"

    texts = [
        "こんにちは、今日はいい天気ですね。",
        "東京タワーは日本の象徴的なランドマークです。",
        "人工知能の研究が急速に進んでいます。",
        "春になると桜が美しく咲き誇ります。",
        "音声合成技術は近年大幅に進歩しました。",
        "富士山は日本で最も高い山です。",
        "この文章はテスト用のダミーデータです。",
        "機械学習モデルのトレーニングを行います。",
        "データセットの品質がモデル性能を左右します。",
        "ニューラルネットワークの学習率を調整します。",
    ]

    if not args.apply:
        print(f"[DRY-RUN] Would create {args.num_samples} samples in {args.output_dir}")
        print(f"  Latent dim: {args.latent_dim}, steps: {args.min_steps}-{args.max_steps}")
        print(f"  Manifest: {manifest_path}")
        print("Pass --apply to actually write files.")
        return

    latent_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i in range(args.num_samples):
        seq_len = rng.integers(args.min_steps, args.max_steps + 1)
        latent = rng.standard_normal((int(seq_len), args.latent_dim)).astype(np.float32)
        stem = f"utt{i:05d}"
        latent_path = latent_dir / f"{stem}.safetensors"
        save_file({"latent": latent}, str(latent_path))

        # Also create a reference latent (speaker conditioning)
        ref_len = rng.integers(args.min_steps, args.max_steps + 1)
        ref_latent = rng.standard_normal((int(ref_len), args.latent_dim)).astype(np.float32)
        ref_path = latent_dir / f"{stem}_ref.safetensors"
        save_file({"latent": ref_latent}, str(ref_path))

        records.append({
            "text": texts[i % len(texts)],
            "latent_path": str(latent_path),
            "ref_latent_path": str(ref_path),
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Created {args.num_samples} samples in {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
