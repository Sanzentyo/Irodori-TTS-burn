# /// script
# requires-python = ">=3.12"
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
"""

from __future__ import annotations

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


def load_python_model():
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

    # Filter to only fields ModelConfig actually declares (checkpoint may contain
    # TrainConfig fields like max_text_len that ModelConfig doesn't know about).
    known = {f.name for f in dc_fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in config_dict.items() if k in known})

    model = TextToLatentRFDiT(cfg)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model, cfg, device


def run_inference(model, cfg, device: torch.device) -> None:
    """Run sample_euler_rf_cfg — mirrors Rust bench_realmodel inputs."""
    from irodori_tts.rf import sample_euler_rf_cfg  # type: ignore

    text_ids = torch.zeros(1, TEXT_LEN, dtype=torch.long, device=device)
    text_mask = torch.ones(1, TEXT_LEN, dtype=torch.bool, device=device)
    ref_latent = torch.zeros(1, REF_FRAMES, cfg.latent_dim, device=device)
    ref_mask = torch.ones(1, REF_FRAMES, dtype=torch.bool, device=device)

    with torch.no_grad():
        sample_euler_rf_cfg(
            model=model,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            sequence_length=SEQ_LEN,
            num_steps=NUM_STEPS,
            cfg_scale_text=CFG_SCALE_TEXT,
            cfg_scale_speaker=CFG_SCALE_SPEAKER,
            cfg_min_t=CFG_MIN_T,
            cfg_guidance_mode="independent",
            seed=0,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    print(f"Python benchmark — seq_len={SEQ_LEN}, num_steps={NUM_STEPS}")
    print(f"Checkpoint : {CHECKPOINT}")

    t_load = time.perf_counter()
    model, cfg, device = load_python_model()
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"Model loaded in {load_ms:.0f} ms")
    print(f"seq_len    : {SEQ_LEN}")
    print(f"num_steps  : {NUM_STEPS}")

    if WARMUP > 0:
        print(f"Warm-up ({WARMUP} run(s)) …")
        for _ in range(WARMUP):
            run_inference(model, cfg, device)

    print(f"Benchmarking ({RUNS} run(s)) …")
    times_ms: list[float] = []
    for i in range(RUNS):
        t = time.perf_counter()
        run_inference(model, cfg, device)
        elapsed_ms = (time.perf_counter() - t) * 1000
        times_ms.append(elapsed_ms)

    times_np = np.array(times_ms)
    print()
    print("=== Benchmark results ===")
    print(f"Backend    : PyTorch ({device.type.upper()})")
    print(f"seq_len    : {SEQ_LEN}")
    print(f"num_steps  : {NUM_STEPS}")
    print(f"runs       : {RUNS}")
    print(f"mean       : {times_np.mean():.1f} ms")
    print(f"min        : {times_np.min():.1f} ms")
    print(f"p50        : {np.median(times_np):.1f} ms")
    print(f"p95        : {np.percentile(times_np, 95):.1f} ms")
    print(f"max        : {times_np.max():.1f} ms")
    for i, t in enumerate(times_ms):
        print(f"  run[{i}]   : {t:.1f} ms")


if __name__ == "__main__":
    main()
