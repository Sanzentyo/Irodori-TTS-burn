#!/usr/bin/env python3
"""Per-step timing profiler for Python inference.

Monkey-patches sample_euler_rf_cfg to report per-step timings.
Usage: cd ../Irodori-TTS && uv run python ../Irodori-TTS-burn/scripts/profile_steps.py
"""
import sys, time, argparse
import torch
import numpy as np

SEQ_LEN = 750
TEXT_LEN = 4
REF_FRAMES = 8
NUM_STEPS = 40

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load model
    from irodori_tts.model import TextToLatentRFDiT
    from huggingface_hub import hf_hub_download
    import json, safetensors.torch

    ckpt_path = hf_hub_download("Aratako/Irodori-TTS-500M-v2", "model.safetensors")
    state = safetensors.torch.load_file(ckpt_path, device="cpu")
    cfg_json = safetensors.torch.load_file(ckpt_path).__metadata__
    # fallback — try loading config from metadata
    try:
        meta = safetensors.torch.load_file(ckpt_path).__metadata__
    except:
        pass

    # Load via the model's from_pretrained or manual instantiation
    from irodori_tts.rf import sample_euler_rf_cfg
    
    # Use the same loading path as bench_python.py
    cfg_path = hf_hub_download("Aratako/Irodori-TTS-500M-v2", "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    class Cfg:
        pass
    cfg = Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    
    model = TextToLatentRFDiT(cfg).to(device)
    state_dict = safetensors.torch.load_file(ckpt_path, device=str(device))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    model_dtype = next(model.parameters()).dtype
    text_ids = torch.zeros(1, TEXT_LEN, dtype=torch.long, device=device)
    text_mask = torch.ones(1, TEXT_LEN, dtype=torch.bool, device=device)
    ref_latent = torch.zeros(1, REF_FRAMES, cfg.latent_dim, dtype=model_dtype, device=device)
    ref_mask = torch.ones(1, REF_FRAMES, dtype=torch.bool, device=device)

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            sample_euler_rf_cfg(
                model=model, text_input_ids=text_ids, text_mask=text_mask,
                ref_latent=ref_latent, ref_mask=ref_mask,
                sequence_length=SEQ_LEN, num_steps=args.num_steps,
                cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_min_t=0.5,
                cfg_guidance_mode="independent", seed=0,
            )
            torch.cuda.synchronize()

    # Now instrument the rf.py to get per-step timing
    # We'll monkey-patch the model forward to add CUDA events
    step_times = []
    
    # Instead of monkey-patching, let's use CUDA events around the whole call
    # and also profile individual operations
    
    # Profile: total inference
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        sample_euler_rf_cfg(
            model=model, text_input_ids=text_ids, text_mask=text_mask,
            ref_latent=ref_latent, ref_mask=ref_mask,
            sequence_length=SEQ_LEN, num_steps=args.num_steps,
            cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_min_t=0.5,
            cfg_guidance_mode="independent", seed=0,
        )
    end_event.record()
    torch.cuda.synchronize()
    total_gpu_ms = start_event.elapsed_time(end_event)
    
    # Profile: setup (conditioning) vs loop
    # Run with 1 step to measure setup
    start_event.record()
    with torch.no_grad():
        sample_euler_rf_cfg(
            model=model, text_input_ids=text_ids, text_mask=text_mask,
            ref_latent=ref_latent, ref_mask=ref_mask,
            sequence_length=SEQ_LEN, num_steps=1,
            cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_min_t=0.5,
            cfg_guidance_mode="independent", seed=0,
        )
    end_event.record()
    torch.cuda.synchronize()
    one_step_ms = start_event.elapsed_time(end_event)
    
    # Profile: 10 steps
    start_event.record()
    with torch.no_grad():
        sample_euler_rf_cfg(
            model=model, text_input_ids=text_ids, text_mask=text_mask,
            ref_latent=ref_latent, ref_mask=ref_mask,
            sequence_length=SEQ_LEN, num_steps=10,
            cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_min_t=0.5,
            cfg_guidance_mode="independent", seed=0,
        )
    end_event.record()
    torch.cuda.synchronize()
    ten_step_ms = start_event.elapsed_time(end_event)
    
    # Profile: individual tensor ops
    x = torch.randn(1, SEQ_LEN, cfg.latent_dim, dtype=model_dtype, device=device)
    
    # cat [x; x; x] (cfg_batch_mult=3)
    n_cat = 1000
    start_event.record()
    for _ in range(n_cat):
        _ = torch.cat([x, x, x], dim=0)
    end_event.record()
    torch.cuda.synchronize()
    cat_per_op_us = start_event.elapsed_time(end_event) / n_cat * 1000
    
    # chunk(3, dim=0)
    x3 = torch.cat([x, x, x], dim=0)
    start_event.record()
    for _ in range(n_cat):
        _ = x3.chunk(3, dim=0)
    end_event.record()
    torch.cuda.synchronize()
    chunk_per_op_us = start_event.elapsed_time(end_event) / n_cat * 1000
    
    # clone
    start_event.record()
    for _ in range(n_cat):
        _ = x.clone()
    end_event.record()
    torch.cuda.synchronize()
    clone_per_op_us = start_event.elapsed_time(end_event) / n_cat * 1000
    
    # from_floats equivalent (tensor creation)
    start_event.record()
    for _ in range(n_cat):
        _ = torch.tensor([0.5], device=device)
    end_event.record()
    torch.cuda.synchronize()
    tensor_create_us = start_event.elapsed_time(end_event) / n_cat * 1000
    
    per_step = (total_gpu_ms - one_step_ms) / (args.num_steps - 1)
    setup = one_step_ms - per_step  # approximate setup cost
    
    print(f"\n=== Python Per-Step Profile (CUDA events) ===")
    print(f"Total ({args.num_steps} steps): {total_gpu_ms:.1f} ms")
    print(f"1-step run:  {one_step_ms:.1f} ms")
    print(f"10-step run: {ten_step_ms:.1f} ms")
    print(f"Approx setup:     {setup:.1f} ms")
    print(f"Approx per-step:  {per_step:.2f} ms")
    print(f"\n=== Micro-op costs (µs, avg over {n_cat} calls) ===")
    print(f"torch.cat([x]*3, 0):  {cat_per_op_us:.1f} µs")
    print(f"x3.chunk(3, 0):       {chunk_per_op_us:.1f} µs")
    print(f"x.clone():            {clone_per_op_us:.1f} µs")
    print(f"torch.tensor([0.5]):  {tensor_create_us:.1f} µs")

if __name__ == "__main__":
    main()
