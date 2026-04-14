"""
LoRA training throughput benchmark for Python PyTorch.

Runs N forward+backward steps on the Irodori-TTS model with LoRA adapters
using synthetic data.  Measures wall time and steps/sec.

Must be run from within the Irodori-TTS venv so that ``irodori_tts`` is importable.

Usage:
    cd ../Irodori-TTS && uv run --extra train python \
        ../Irodori-TTS-burn/scripts/bench_train_py.py \
        --manifest ../Irodori-TTS-burn/target/bench_data/train_py.jsonl \
        --model ~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2/.../model.safetensors \
        --steps 50 --batch-size 4 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def load_model_config_from_safetensors(path: Path):
    """Extract ModelConfig from embedded safetensors metadata."""
    from irodori_tts.config import ModelConfig

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    config_json = meta.get("config_json")
    if config_json is None:
        print("Warning: no config_json in safetensors metadata, using defaults")
        return ModelConfig()
    import dataclasses
    cfg_dict = json.loads(config_json)
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in field_names}
    return ModelConfig(**filtered)


def run_train_step(
    model,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_speaker: bool,
    use_bf16: bool,
):
    """One training step matching Irodori-TTS train.py logic."""
    from irodori_tts.rf import rf_interpolate, rf_velocity_target, sample_logit_normal_t

    text_ids = batch["text_ids"].to(device, non_blocking=True)
    text_mask = batch["text_mask"].to(device, non_blocking=True)
    x0 = batch["latent_patched"].to(device, non_blocking=True)
    x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
    x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)

    ref_latent = None
    ref_mask = None
    if use_speaker:
        ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
        ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
        has_speaker = batch["has_speaker"].to(device, non_blocking=True)
        ref_mask = ref_mask & has_speaker[:, None]
        ref_latent = ref_latent * has_speaker[:, None, None].to(ref_latent.dtype)

    bsz = x0.shape[0]
    t = sample_logit_normal_t(batch_size=bsz, device=device, mean=0.0, std=1.0)
    noise = torch.randn_like(x0)
    x_t = rf_interpolate(x0, noise, t)
    v_target = rf_velocity_target(x0, noise)

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    with ctx:
        v_pred = model(
            x_t=x_t,
            t=t,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            latent_mask=x_mask,
        )

    v_pred = v_pred.float()
    mask_f = x_mask.unsqueeze(-1).float()
    valid_f = x_mask_valid.unsqueeze(-1).float()
    diff2 = (v_pred - v_target.float()) ** 2
    loss = (diff2 * mask_f).sum() / valid_f.sum().clamp(min=1.0)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Python LoRA training throughput benchmark")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Base model safetensors")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps (excluded from timing)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    from irodori_tts.config import ModelConfig
    from irodori_tts.dataset import LatentTextDataset, TTSCollator
    from irodori_tts.lora import apply_lora
    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.tokenizer import PretrainedTextTokenizer

    device = torch.device(args.device)
    use_bf16 = args.dtype == "bfloat16"

    # 1. Load model config from safetensors metadata
    print(f"Loading model config from {args.model} ...")
    model_cfg = load_model_config_from_safetensors(args.model)
    print(f"  model_dim={model_cfg.model_dim}, layers={model_cfg.num_layers}, "
          f"latent_dim={model_cfg.latent_dim}, tokenizer={model_cfg.text_tokenizer_repo}")
    use_speaker = bool(model_cfg.speaker_dim) and bool(model_cfg.speaker_layers)

    # 2. Create model + load weights
    model = TextToLatentRFDiT(model_cfg).to(device)
    state = load_file(str(args.model), device=str(device))
    model.load_state_dict(state, strict=True)

    # 3. Apply LoRA (field names must match irodori_tts.config.TrainConfig)
    lora_config = {
        "lora_enabled": True,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "lora_bias": "none",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj"],
    }
    model = apply_lora(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")

    # 4. Tokenizer from HF repo (matches model config)
    print(f"Loading tokenizer from {model_cfg.text_tokenizer_repo} ...")
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        model_cfg.text_tokenizer_repo,
        add_bos=model_cfg.text_add_bos,
    )

    # 5. Dataset + collator
    dataset = LatentTextDataset(
        manifest_path=args.manifest,
        latent_dim=model_cfg.latent_dim,
        enable_speaker_condition=use_speaker,
        enable_caption_condition=False,
    )
    collator = TTSCollator(
        tokenizer=tokenizer,
        caption_tokenizer=None,
        latent_dim=model_cfg.latent_dim,
        latent_patch_size=model_cfg.latent_patch_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=0,
    )

    # 6. Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    model.train()
    print(f"\nBenchmarking {args.steps} steps (batch_size={args.batch_size}, "
          f"dtype={args.dtype}, device={args.device}) ...")

    # Warmup (excluded from timing)
    data_iter = iter(loader)
    for i in range(args.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        loss = run_train_step(model, batch, optimizer, device, use_speaker, use_bf16)
        print(f"  warmup {i+1}/{args.warmup_steps}  loss={loss:.4f}")

    if args.device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    start = time.perf_counter()
    step = 0
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        loss = run_train_step(model, batch, optimizer, device, use_speaker, use_bf16)
        step += 1

        if step % 10 == 0:
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f"  step {step}/{args.steps}  loss={loss:.4f}  "
                  f"elapsed={elapsed:.1f}s  steps/s={step/elapsed:.2f}")

    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"\n{'='*60}")
    print(f"Python benchmark: {args.steps} steps in {elapsed:.2f}s")
    print(f"Throughput: {args.steps/elapsed:.2f} steps/sec")
    print(f"Time per step: {elapsed/args.steps*1000:.1f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
