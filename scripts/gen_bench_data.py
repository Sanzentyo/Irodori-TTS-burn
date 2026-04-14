# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "safetensors>=0.4",
#     "numpy",
# ]
# ///
"""
Generate synthetic training data in BOTH formats:
- .safetensors (for Rust training)
- .pt (for Python training)

And produce separate manifests for each.

Usage:
    uv run scripts/gen_bench_data.py \
        --output-dir target/bench_data \
        --num-samples 100 \
        --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as save_safetensors


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark training data (dual format)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--min-steps", type=int, default=20, help="Min latent sequence length")
    parser.add_argument("--max-steps", type=int, default=100, help="Max latent sequence length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", "-a", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    st_dir = out / "latents_st"
    pt_dir = out / "latents_pt"
    manifest_rust = out / "train_rust.jsonl"
    manifest_py = out / "train_py.jsonl"

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
        print(f"[DRY-RUN] Would create {args.num_samples} samples in {out}")
        print(f"  Safetensors: {st_dir}")
        print(f"  PyTorch .pt: {pt_dir}")
        print(f"  Manifests: {manifest_rust}, {manifest_py}")
        print("Pass --apply to write.")
        return

    st_dir.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)

    rust_records = []
    py_records = []

    for i in range(args.num_samples):
        seq_len = int(rng.integers(args.min_steps, args.max_steps + 1))
        latent_np = rng.standard_normal((seq_len, args.latent_dim)).astype(np.float32)
        stem = f"utt{i:05d}"

        # Save safetensors (Rust)
        st_path = st_dir / f"{stem}.safetensors"
        save_safetensors({"latent": latent_np}, str(st_path))

        # Save .pt (Python) — shape [S, D]
        pt_path = pt_dir / f"{stem}.pt"
        torch.save(torch.from_numpy(latent_np), str(pt_path))

        # Reference latent (speaker conditioning)
        ref_len = int(rng.integers(args.min_steps, args.max_steps + 1))
        ref_np = rng.standard_normal((ref_len, args.latent_dim)).astype(np.float32)
        ref_st = st_dir / f"{stem}_ref.safetensors"
        save_safetensors({"latent": ref_np}, str(ref_st))
        ref_pt = pt_dir / f"{stem}_ref.pt"
        torch.save(torch.from_numpy(ref_np), str(ref_pt))

        text = texts[i % len(texts)]
        rust_records.append({
            "text": text,
            "latent_path": str(st_path.resolve()),
            "ref_latent_path": str(ref_st.resolve()),
        })
        py_records.append({
            "text": text,
            "latent_path": str(pt_path.resolve()),
            "speaker_id": f"spk{i % 5:02d}",
        })

    for path, records in [(manifest_rust, rust_records), (manifest_py, py_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Created {args.num_samples} dual-format samples:")
    print(f"  Rust manifest: {manifest_rust}")
    print(f"  Python manifest: {manifest_py}")


if __name__ == "__main__":
    main()
