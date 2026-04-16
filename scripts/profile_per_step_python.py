# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "safetensors",
#   "numpy",
# ]
# ///
"""Per-step CUDA event timing for Python inference."""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Irodori-TTS"))
import torch
from safetensors import safe_open

SEQ_LEN = 750
TEXT_LEN = 4
REF_FRAMES = 8
NUM_STEPS = 40
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "target" / "hf_model" / "model.safetensors"

def main():
    device = torch.device("cuda")
    
    # Load model (same path as bench_python.py)
    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.config import ModelConfig
    from dataclasses import fields as dc_fields
    
    with safe_open(str(CHECKPOINT), framework="pt", device="cpu") as f:
        meta = f.metadata()
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    config_dict = json.loads(meta["config_json"])
    known = {f.name for f in dc_fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in config_dict.items() if k in known})
    
    model = TextToLatentRFDiT(cfg)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    
    dtype = next(model.parameters()).dtype
    text_ids = torch.zeros(1, TEXT_LEN, dtype=torch.long, device=device)
    text_mask = torch.ones(1, TEXT_LEN, dtype=torch.bool, device=device)
    ref_latent = torch.zeros(1, REF_FRAMES, cfg.latent_dim, dtype=dtype, device=device)
    ref_mask = torch.ones(1, REF_FRAMES, dtype=torch.bool, device=device)
    
    # Monkey-patch the rf module to add per-step timing
    import irodori_tts.rf as rf_mod
    original_fn = rf_mod.sample_euler_rf_cfg
    
    step_events = []
    
    # Warmup
    with torch.no_grad():
        original_fn(
            model=model, text_input_ids=text_ids, text_mask=text_mask,
            ref_latent=ref_latent, ref_mask=ref_mask,
            sequence_length=SEQ_LEN, num_steps=NUM_STEPS,
            cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_min_t=0.5,
            cfg_guidance_mode="independent", seed=0,
        )
        torch.cuda.synchronize()
    
    # Timed run using wall clock + CUDA events
    overall_start = torch.cuda.Event(enable_timing=True)
    overall_end = torch.cuda.Event(enable_timing=True)
    
    overall_start.record()
    wall_start = time.perf_counter_ns()
    with torch.no_grad():
        original_fn(
            model=model, text_input_ids=text_ids, text_mask=text_mask,
            ref_latent=ref_latent, ref_mask=ref_mask,
            sequence_length=SEQ_LEN, num_steps=NUM_STEPS,
            cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_min_t=0.5,
            cfg_guidance_mode="independent", seed=0,
        )
    overall_end.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter_ns()
    
    total_gpu_ms = overall_start.elapsed_time(overall_end)
    total_wall_ms = (wall_end - wall_start) / 1e6
    
    # Now do a step-count sweep to isolate setup vs per-step
    results = {}
    for n_steps in [1, 5, 10, 20, 40]:
        times = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                original_fn(
                    model=model, text_input_ids=text_ids, text_mask=text_mask,
                    ref_latent=ref_latent, ref_mask=ref_mask,
                    sequence_length=SEQ_LEN, num_steps=n_steps,
                    cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_min_t=0.5,
                    cfg_guidance_mode="independent", seed=0,
                )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        avg = sum(times) / len(times)
        results[n_steps] = avg
        print(f"  {n_steps:2d} steps: {avg:.1f} ms (median of 5)")
    
    # Linear regression: time = setup + per_step * n_steps
    import numpy as np
    xs = np.array(list(results.keys()), dtype=float)
    ys = np.array(list(results.values()), dtype=float)
    # Least squares: y = a + b*x
    A = np.vstack([np.ones_like(xs), xs]).T
    (setup, per_step), _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    
    print(f"\n=== Python Timing Breakdown ===")
    print(f"Total GPU ({NUM_STEPS} steps): {total_gpu_ms:.1f} ms")
    print(f"Total wall ({NUM_STEPS} steps): {total_wall_ms:.1f} ms")
    print(f"Linear fit: setup = {setup:.1f} ms, per_step = {per_step:.2f} ms")
    print(f"Predicted 40-step: {setup + 40 * per_step:.1f} ms")

if __name__ == "__main__":
    main()
