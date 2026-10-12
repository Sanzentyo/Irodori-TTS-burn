# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.10.0",
#     "torchaudio==2.10.0",
#     "safetensors",
#     "transformers==5.12.1",
#     "irodori-tts",
#     "dacvae",
#     "silentcipher",
#     "soundfile>=0.14.0",
# ]
#
# [tool.uv.sources]
# irodori-tts = { git = "https://github.com/Aratako/Irodori-TTS", rev = "89f9d8fbd4d51ea019867ee1197725ede1df13c5" }
# dacvae = { git = "https://github.com/facebookresearch/dacvae", rev = "414c20785fc3a28373073ea8ef7a1316eeeaca6e" }
# silentcipher = { git = "https://github.com/SesameAILabs/silentcipher.git", rev = "d46d7d0893a583d8968ab3a6626e2289faec9152" }
# ///
"""Export an actual v4.1 RF/MF trajectory using the installed official package.

Use the pinned speech.cpp reference uv project with --no-sync for a comparison
against exactly that dependency revision. No compiler or watermark is enabled.
"""

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from irodori_tts.codec import DACVAECodec
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import (
    _load_checkpoint_for_inference,
    _move_inference_module,
)
from irodori_tts.meanflow import sample_euler_meanflow
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.rf import sample_euler_rf_cfg
from irodori_tts.text_normalization import normalize_text
from irodori_tts.tokenizer import PretrainedTextTokenizer
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default="こんにちは。")
    parser.add_argument("--caption", default="")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--additional-frames", type=int, nargs="*", default=[])
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", choices=["cuda", "xpu", "cpu"], default="cuda")
    parser.add_argument("--sdpa", choices=["auto", "math"], default="auto")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmups < 1 or args.repeat < 1:
        raise ValueError("warmups and repeat must be positive")
    device = torch.device(args.device)
    accelerator = getattr(torch, args.device) if args.device != "cpu" else None
    if accelerator is not None and not accelerator.is_available():
        raise RuntimeError(f"requested accelerator {args.device} is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    dtype = torch.float32 if args.precision == "fp32" else torch.float16
    state, config, _, encoder_config = _load_checkpoint_for_inference(args.checkpoint)
    config = merge_dataclass_overrides(
        ModelConfig(), config, section="checkpoint model_config"
    )
    model = TextToLatentRFDiT(
        config,
        pretrained_backbone_config=encoder_config,
        load_pretrained_backbone_weights=False,
    )
    model.load_state_dict(state)
    del state
    model = _move_inference_module(model, device=device, dtype=dtype).eval()
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        str(args.checkpoint.parent / "tokenizer"),
        add_bos=config.text_add_bos,
        local_files_only=True,
    )
    ids, mask = tokenizer.batch_encode([normalize_text(args.text)])
    caption_ids, caption_mask = tokenizer.batch_encode([args.caption])
    if not args.caption:
        caption_mask.zero_()
    ref = torch.zeros(1, 4, 32)
    ref_mask = torch.zeros(1, 4, dtype=torch.bool)
    codec = DACVAECodec.load(
        str(args.codec),
        device=str(device),
        dtype=torch.float32,
        deterministic_encode=True,
        deterministic_decode=True,
    )
    if args.reference:
        import soundfile as sf

        wav, rate = sf.read(args.reference, dtype="float32", always_2d=True)
        ref = codec.encode_waveform(
            torch.from_numpy(wav.T.copy()).unsqueeze(0), rate, normalize_db=-16.0
        ).cpu()
        ref_mask = torch.ones(ref.shape[:2], dtype=torch.bool)

    def export_case(frames: int, output: Path):
        # Identical FP32 source, cast only once for each runtime/precision.
        noise = torch.randn(1, frames, 32, generator=torch.Generator().manual_seed(42))
        native_noise = noise.to(device, dtype=dtype)
        steps = args.steps or (4 if config.flow_parameterization == "meanflow" else 40)
        kwargs = {
            "model": model,
            "text_input_ids": ids.to(device),
            "text_mask": mask.to(device),
            "ref_latent": ref.to(device, dtype=dtype),
            "ref_mask": ref_mask.to(device),
            "caption_input_ids": caption_ids.to(device),
            "caption_mask": caption_mask.to(device),
            "sequence_length": frames,
            "num_steps": steps,
            "seed": 42,
        }
        if config.flow_parameterization == "meanflow":
            sampler = sample_euler_meanflow
        else:
            sampler = sample_euler_rf_cfg
            kwargs.update(
                cfg_scale_text=3.0,
                cfg_scale_speaker=5.0 if args.reference else 0.0,
                cfg_scale_caption=3.0 if args.caption else 0.0,
                cfg_min_t=0.5,
                cfg_max_t=1.0,
            )
        forwards = []
        original_forward = model.forward_with_encoded_conditions

        def capture(*pos, **kw):
            result = original_forward(*pos, **kw)
            forwards.append(result.detach().float().cpu())
            return result

        original_randn = torch.randn

        def starting_noise(*pos, **kw):
            shape = (
                pos[0] if len(pos) == 1 and isinstance(pos[0], (list, tuple)) else pos
            )
            if tuple(shape) == tuple(noise.shape):
                return native_noise.clone()
            return original_randn(*pos, **kw)

        timings = []
        audio_timings = []
        attention_context = (
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
            if args.sdpa == "math"
            else nullcontext()
        )
        with (
            torch.inference_mode(),
            patch("torch.randn", starting_noise),
            attention_context,
        ):
            for index in range(args.warmups + args.repeat):
                if accelerator is not None:
                    accelerator.synchronize()
                start = time.perf_counter()
                latent = sampler(**kwargs)
                latent.float().cpu()
                rf_seconds = time.perf_counter() - start
                codec.decode_latent(latent.float()).float().cpu()
                audio_seconds = time.perf_counter() - start
                if index >= args.warmups:
                    timings.append(rf_seconds)
                    audio_timings.append(audio_seconds)
            model.forward_with_encoded_conditions = capture
            latent = sampler(**kwargs)
        model.forward_with_encoded_conditions = original_forward
        with torch.inference_mode():
            waveform = codec.decode_latent(latent.float()).float().cpu()
        tensors = {
            "text_ids": ids.to(torch.int32),
            "text_mask": mask,
            "caption_ids": caption_ids.to(torch.int32),
            "caption_mask": caption_mask,
            "ref_latent": ref,
            "ref_mask": ref_mask,
            "initial_noise": noise,
            "expected_latent": latent.float().cpu(),
            "expected_audio": waveform,
        }
        tensors.update({f"forward_{i:03}": value for i, value in enumerate(forwards)})
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata = {
            "precision": args.precision,
            "frames": str(frames),
            "steps": str(steps),
            "flow": config.flow_parameterization,
            "checkpoint_sha256": digest.hexdigest(),
            "rf_seconds": json.dumps(timings),
            "audio_seconds": json.dumps(audio_timings),
            "warmups": str(args.warmups),
            "device": str(device),
            "device_name": accelerator.get_device_name() if accelerator else "CPU",
            "torch_version": torch.__version__,
            "sdpa": args.sdpa,
            "text": args.text,
            "caption": args.caption,
            "reference": str(args.reference) if args.reference else "",
        }
        save_file(
            {key: value.contiguous() for key, value in tensors.items()},
            str(output),
            metadata=metadata,
        )
        print(json.dumps(metadata, ensure_ascii=False))

    cases = [(args.frames, args.output)] + [
        (
            frames,
            args.output.with_name(f"{args.output.stem}-{frames}{args.output.suffix}"),
        )
        for frames in args.additional_frames
    ]
    if any(frames < 1 or output.exists() for frames, output in cases):
        raise ValueError("frame counts must be positive and output paths must be new")
    if len({frames for frames, _ in cases}) != len(cases):
        raise ValueError("duplicate frame counts")
    for frames, output in cases:
        export_case(frames, output)


if __name__ == "__main__":
    main()
