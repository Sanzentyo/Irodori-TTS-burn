# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch",
#   "safetensors",
#   "packaging",
#   "numpy",
#   "huggingface-hub",
# ]
# ///
"""Benchmark Python Irodori-TTS inference speed for comparison with Rust/burn.

Uses synthetic inputs (same shape as Rust bench) against the real model.
Mirrors the Rust bench_realmodel binary: batch=1, text_len=4, ref_frames=8,
seq_len=750, num_steps=40, Independent CFG.

Supports --dtype f32 | bf16 | autocast-bf16:
  f32             — default float32 weights + compute
  bf16            — cast model weights to bfloat16 (may crash on some ops)
  autocast-bf16   — mixed precision via torch.autocast (recommended for bf16)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
IRODORI_TTS_DIR = REPO_ROOT.parent / "Irodori-TTS"
sys.path.insert(0, str(IRODORI_TTS_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN = 750
NUM_STEPS = 40
WARMUP = 1
RUNS = 3
TEXT_LEN = 4
REF_FRAMES = 8
CFG_SCALE_TEXT = 3.0
CFG_SCALE_SPEAKER = 5.0
CFG_MIN_T = 0.5
CHECKPOINT = REPO_ROOT / "target" / "hf_model" / "model.safetensors"


def load_python_model(dtype_mode: str):
    from irodori_tts.model import TextToLatentRFDiT  # type: ignore
    from irodori_tts.config import ModelConfig  # type: ignore
    from dataclasses import fields as dc_fields

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    with safe_open(str(CHECKPOINT), framework="pt", device="cpu") as f:
        meta = f.metadata()
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    config_dict = json.loads(meta["config_json"])
    print(
        f"Config     : model_dim={config_dict['model_dim']}, "
        f"layers={config_dict['num_layers']}, "
        f"heads={config_dict['num_heads']}"
    )

    known = {f.name for f in dc_fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in config_dict.items() if k in known})

    model = TextToLatentRFDiT(cfg)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    if dtype_mode == "bf16":
        model = model.to(dtype=torch.bfloat16)
        print("dtype      : bfloat16 (native cast)")
    elif dtype_mode == "autocast-bf16":
        print("dtype      : autocast bfloat16 (mixed precision)")
    else:
        print("dtype      : float32")

    return model, cfg, device


def run_inference(model, cfg, device: torch.device, *, use_autocast: bool = False, num_steps: int = NUM_STEPS) -> None:
    """Run sample_euler_rf_cfg — mirrors Rust bench_realmodel inputs."""
    from irodori_tts.rf import sample_euler_rf_cfg  # type: ignore

    model_dtype = next(model.parameters()).dtype

    text_ids = torch.zeros(1, TEXT_LEN, dtype=torch.long, device=device)
    text_mask = torch.ones(1, TEXT_LEN, dtype=torch.bool, device=device)
    ref_latent = torch.zeros(1, REF_FRAMES, cfg.latent_dim, dtype=model_dtype, device=device)
    ref_mask = torch.ones(1, REF_FRAMES, dtype=torch.bool, device=device)

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else torch.no_grad()

    with torch.no_grad(), ctx:
        sample_euler_rf_cfg(
            model=model,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            sequence_length=SEQ_LEN,
            num_steps=num_steps,
            cfg_scale_text=CFG_SCALE_TEXT,
            cfg_scale_speaker=CFG_SCALE_SPEAKER,
            cfg_min_t=CFG_MIN_T,
            cfg_guidance_mode="independent",
            seed=0,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python Irodori-TTS benchmark")
    parser.add_argument(
        "--dtype",
        choices=["f32", "bf16", "autocast-bf16"],
        default="f32",
        help="Model precision: f32, bf16 (native cast), autocast-bf16 (mixed precision)",
    )
    parser.add_argument("--runs", type=int, default=RUNS, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Number of warmup runs")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help="Number of diffusion steps")
    args = parser.parse_args()

    dtype_mode = args.dtype
    runs = args.runs
    warmup = args.warmup
    num_steps = args.num_steps
    use_autocast = dtype_mode == "autocast-bf16"

    print(f"Python benchmark — seq_len={SEQ_LEN}, num_steps={num_steps}, dtype={dtype_mode}")
    print(f"Checkpoint : {CHECKPOINT}")

    t_load = time.perf_counter()
    model, cfg, device = load_python_model(dtype_mode)
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"Model loaded in {load_ms:.0f} ms")
    print(f"seq_len    : {SEQ_LEN}")
    print(f"num_steps  : {num_steps}")

    if warmup > 0:
        print(f"Warm-up ({warmup} run(s)) …")
        for _ in range(warmup):
            run_inference(model, cfg, device, use_autocast=use_autocast, num_steps=num_steps)

    print(f"Benchmarking ({runs} run(s)) …")
    times_ms: list[float] = []
    for i in range(runs):
        t = time.perf_counter()
        run_inference(model, cfg, device, use_autocast=use_autocast, num_steps=num_steps)
        elapsed_ms = (time.perf_counter() - t) * 1000
        times_ms.append(elapsed_ms)

    times_np = np.array(times_ms)
    dtype_label = {"f32": "f32", "bf16": "bf16 (native)", "autocast-bf16": "bf16 (autocast)"}[dtype_mode]
    print()
    print("=== Benchmark results ===")
    print(f"Backend    : PyTorch ({device.type.upper()}) {dtype_label}")
    print(f"seq_len    : {SEQ_LEN}")
    print(f"num_steps  : {num_steps}")
    print(f"runs       : {runs}")
    print(f"mean       : {times_np.mean():.1f} ms")
    print(f"min        : {times_np.min():.1f} ms")
    print(f"p50        : {np.median(times_np):.1f} ms")
    print(f"p95        : {np.percentile(times_np, 95):.1f} ms")
    print(f"max        : {times_np.max():.1f} ms")
    for i, t in enumerate(times_ms):
        print(f"  run[{i}]   : {t:.1f} ms")


if __name__ == "__main__":
    main()
