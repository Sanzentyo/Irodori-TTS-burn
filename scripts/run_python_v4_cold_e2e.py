# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "dacvae @ git+https://github.com/facebookresearch/dacvae@414c20785fc3a28373073ea8ef7a1316eeeaca6e",
#   "huggingface-hub==1.23.0",
#   "numpy==2.2.6",
#   "safetensors==0.7.0",
#   "sentencepiece==0.1.99",
#   "silentcipher @ git+https://github.com/SesameAILabs/silentcipher.git@d46d7d0893a583d8968ab3a6626e2289faec9152",
#   "soundfile==0.13.1",
#   "torch==2.10.0",
#   "torchaudio==2.10.0",
#   "torchcodec==0.10.0",
#   "transformers==5.12.1",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchaudio = { index = "pytorch-cu128" }
# ///
"""One strict-FP32 Python process from runtime construction through WAV close."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

TEXT = "これは音声合成の実運用に近い条件を確認するためのサンプルです。"
DESIGN = "落ち着いた自然な日本語の声で、明瞭かつ穏やかに話す。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--voice", choices=("text", "design", "clone"), required=True)
    parser.add_argument("--ref-wav", type=Path)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-pci", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--num-steps", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in (args.output_wav, args.output_json):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
    if args.voice == "clone" and args.ref_wav is None:
        raise ValueError("clone requires --ref-wav")
    if args.voice != "clone" and args.ref_wav is not None:
        raise ValueError("--ref-wav is valid only for clone")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.is_autocast_enabled():
        raise RuntimeError("autocast must be disabled")
    props = torch.cuda.get_device_properties(0)
    raw_pci = props.pci_bus_id
    if isinstance(raw_pci, int) and not isinstance(raw_pci, bool):
        pci = f"00000000:{raw_pci:02x}:00.0"
    else:
        pci = str(raw_pci).strip().lower()
    if pci.startswith("0000:"):
        pci = "00000000:" + pci.removeprefix("0000:")
    if props.name != args.expected_gpu_name or pci != args.expected_pci.lower():
        raise RuntimeError(f"GPU identity mismatch: {props.name} / {pci}")

    sys.path.insert(0, str(args.upstream))
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
    )

    process_started = time.perf_counter()
    load_started = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(args.checkpoint),
            model_device="cuda:0",
            codec_repo=str(args.codec),
            model_precision="fp32",
            codec_device="cuda:0",
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    runtime.watermarker.model = None
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    request = SamplingRequest(
        text=TEXT,
        caption=DESIGN if args.voice == "design" else None,
        ref_wav=str(args.ref_wav) if args.voice == "clone" else None,
        no_ref=args.voice != "clone",
        seconds=None,
        num_steps=args.num_steps,
        cfg_scale_text=3.0,
        cfg_scale_caption=4.0,
        cfg_scale_speaker=5.0,
        cfg_guidance_mode="independent",
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        context_kv_cache=True,
        seed=42,
        t_schedule_mode="linear",
        sway_coeff=-1.0,
        trim_tail=False,
    )
    synth_started = time.perf_counter()
    result = runtime.synthesize(request)
    audio = result.audio.detach().to(device="cpu", dtype=torch.float32).contiguous()
    torch.cuda.synchronize()
    cpu_audio_ready_seconds = time.perf_counter() - synth_started
    payload = {
        "format": "irodori-v4-python-cold-e2e-v1",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": props.name,
            "pci_bus_id": pci,
            "strict_fp32": True,
            "tf32": False,
            "autocast": False,
        },
        "request": {
            "voice": args.voice,
            "text": TEXT,
            "caption": DESIGN if args.voice == "design" else None,
            "raw_reference_encode": args.voice == "clone",
            "duration": "predict",
            "num_steps": args.num_steps,
            "schedule": "linear",
            "cfg": {"text": 3.0, "caption": 4.0, "speaker": 5.0},
            "seed": 42,
            "trim_tail": False,
        },
        "timing_seconds": {
            "model_codec_load": load_seconds,
            "request_to_cpu_audio_ready": cpu_audio_ready_seconds,
            "process_body_through_cpu_audio_ready": time.perf_counter()
            - process_started,
        },
        "result": {
            "sample_rate": result.sample_rate,
            "samples": int(audio.numel()),
            "output_seconds": int(audio.numel()) / result.sample_rate,
            "stage_timings": result.stage_timings,
            "wav": str(args.output_wav.resolve()),
        },
    }
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    # Keep WAV creation as the process's final material action. The parent
    # runner's external wall interval therefore ends immediately after close.
    sf.write(
        args.output_wav,
        audio.squeeze().numpy(),
        result.sample_rate,
        subtype="PCM_16",
    )
    with args.output_wav.open("rb") as file:
        os.fsync(file.fileno())


if __name__ == "__main__":
    main()
