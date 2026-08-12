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
"""Measure the official v4 duration predictor with explicit readback boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

UPSTREAM_COMMIT = "9f19d9a9048099a4b978a762d0509228fe624e3f"
MODEL_SHA256 = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593"
CODEC_SHA256 = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"
MODEL_REVISION = "e4aaac4df355ff560dcd35e0dae272c3a759317b"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
DEFAULT_TEXT = "こんにちは。"
WARMUPS = 5
MEASURED = 10
REPEATS = WARMUPS + MEASURED
DURATION_FIXTURE_FORMAT = "irodori-v4-duration-fixture-v1"
SAMPLE_RATE = 48_000
HOP_LENGTH = 1_920
LATENT_PATCH_SIZE = 1
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0


@dataclass(frozen=True)
class Timing:
    cuda_event_seconds: float
    device_complete_seconds: float
    readback_complete_seconds: float
    readback_elements: int
    readback_dtype: str
    readback_owned: bool
    readback_contiguous: bool


@dataclass(frozen=True)
class ScopeResult:
    repeat: int
    cold: bool
    scope: str
    timing: Timing
    log_frames: float
    predicted_frames: float
    output_sha256: str


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    static_self_test = "--static-self-test" in sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=root.parent / "Irodori-TTS")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=cache
        / "models--Aratako--Irodori-TTS-v4-Small"
        / "snapshots"
        / MODEL_REVISION
        / "model.safetensors",
    )
    parser.add_argument(
        "--codec",
        type=Path,
        default=cache
        / "models--Aratako--Semantic-DACVAE-Japanese-32dim"
        / "snapshots"
        / CODEC_REVISION
        / "weights.pth",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--fixture-out", type=Path, required=not static_self_test)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--json-out", type=Path, required=not static_self_test)
    parser.add_argument("--static-self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_f32(value: torch.Tensor) -> str:
    data = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(data.numpy().tobytes(order="C")).hexdigest()


def configure_math() -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    policy = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast": False,
    }
    if policy != {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "autocast": False,
    }:
        raise RuntimeError(f"strict fp32 math policy failed: {policy}")
    return policy


def timed_scalar(
    device: torch.device, operation: Callable[[], torch.Tensor]
) -> tuple[torch.Tensor, Timing]:
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    device_complete = time.perf_counter() - wall_start
    if output.shape != (1,) or output.dtype != torch.float32 or output.device != device:
        raise RuntimeError(
            f"duration output must be cuda fp32 [1], got {output.dtype} {tuple(output.shape)} {output.device}"
        )
    cpu = output.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    torch.cuda.synchronize(device)
    readback_complete = time.perf_counter() - wall_start
    return cpu, Timing(
        cuda_event_seconds=start.elapsed_time(end) / 1_000.0,
        device_complete_seconds=device_complete,
        readback_complete_seconds=readback_complete,
        readback_elements=cpu.numel(),
        readback_dtype="float32",
        readback_owned=cpu.data_ptr() != output.data_ptr(),
        readback_contiguous=cpu.is_contiguous(),
    )


def resolve_predicted_length(predicted_frames: float) -> dict[str, int | float]:
    if not math.isfinite(predicted_frames) or predicted_frames < 0.0:
        raise ValueError(f"invalid predicted frames: {predicted_frames}")
    min_frames = max(1, math.ceil(MIN_SECONDS * SAMPLE_RATE / HOP_LENGTH))
    max_frames = max(1, math.floor(MAX_SECONDS * SAMPLE_RATE / HOP_LENGTH))
    latent_frames = min(max(round(predicted_frames), min_frames), max_frames)
    target_samples = latent_frames * HOP_LENGTH
    return {
        "duration_scale": 1.0,
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "latent_patch_size": LATENT_PATCH_SIZE,
        "latent_frames": latent_frames,
        "patched_frames": math.ceil(latent_frames / LATENT_PATCH_SIZE),
        "target_samples": target_samples,
        "seconds": target_samples / SAMPLE_RATE,
    }


def create_inputs(
    runtime: Any, text: str, fixture_out: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], str, str]:
    from irodori_tts.text_normalization import normalize_text

    if fixture_out.exists():
        raise FileExistsError(fixture_out)
    normalized = normalize_text(text).strip()
    if not normalized:
        raise ValueError(
            "duration benchmark text must not be empty after normalization"
        )
    text_ids, text_mask = runtime.tokenizer.batch_encode(
        [normalized], max_length=runtime.default_text_max_len
    )
    if runtime.caption_tokenizer is None:
        raise RuntimeError("released v4 runtime unexpectedly lacks caption tokenizer")
    caption_ids, caption_mask = runtime.caption_tokenizer.batch_encode(
        [""], max_length=runtime.default_caption_max_len
    )
    caption_mask.zero_()
    tensors = {
        "inputs/text_input_ids": text_ids.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
        "inputs/text_mask": text_mask.detach()
        .to(device="cpu", dtype=torch.bool)
        .contiguous(),
        "inputs/caption_input_ids": caption_ids.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
        "inputs/caption_mask": caption_mask.detach()
        .to(device="cpu", dtype=torch.bool)
        .contiguous(),
    }
    if tensors["inputs/text_input_ids"].shape != (1, 256):
        raise RuntimeError("fixture text input shape mismatch")
    if tensors["inputs/caption_input_ids"].shape != (1, 512):
        raise RuntimeError("fixture caption input shape mismatch")
    metadata = {
        "format": DURATION_FIXTURE_FORMAT,
        "text": text,
        "normalized_text": normalized,
        "text_valid_tokens": int(tensors["inputs/text_mask"].sum().item()),
        "model_sha256": MODEL_SHA256,
    }
    fixture_out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(fixture_out),
        metadata={"duration_json": json.dumps(metadata, sort_keys=True)},
    )
    fixture_sha = sha256_file(fixture_out)
    return (
        {key: value.to(device=device) for key, value in tensors.items()},
        normalized,
        fixture_sha,
    )


def summarize(rows: list[ScopeResult], scope: str) -> dict[str, Any]:
    selected = [row for row in rows if row.scope == scope]
    measured = selected[WARMUPS:]

    def values(field: str) -> list[float]:
        return [getattr(row.timing, field) for row in measured]

    return {
        "warmups": WARMUPS,
        "measured": len(measured),
        "cuda_event_median_seconds": statistics.median(values("cuda_event_seconds")),
        "device_complete_min_seconds": min(values("device_complete_seconds")),
        "device_complete_median_seconds": statistics.median(
            values("device_complete_seconds")
        ),
        "device_complete_max_seconds": max(values("device_complete_seconds")),
        "readback_complete_min_seconds": min(values("readback_complete_seconds")),
        "readback_complete_median_seconds": statistics.median(
            values("readback_complete_seconds")
        ),
        "readback_complete_max_seconds": max(values("readback_complete_seconds")),
        "output_hashes_equal": len({row.output_sha256 for row in selected}) == 1,
        "log_frames": selected[0].log_frames,
        "predicted_frames": selected[0].predicted_frames,
    }


def static_self_test() -> None:
    sample = Timing(0.1, 0.2, 0.3, 1, "float32", True, True)
    rows = [
        ScopeResult(i + 1, i < 2, "head", sample, 1.0, math.e - 1.0, "a")
        for i in range(REPEATS)
    ]
    summary = summarize(rows, "head")
    assert summary["warmups"] == WARMUPS and summary["measured"] == MEASURED
    assert summary["output_hashes_equal"] is True
    expected = (
        (45.381_015_214_336_86, 45, 86_400, 1.8),
        (111.602_249_616_249_18, 112, 215_040, 4.48),
        (254.628_081_942_057_97, 255, 489_600, 10.2),
        (333.443_053_490_291_8, 333, 639_360, 13.32),
        (488.575_488_580_713_45, 489, 938_880, 19.56),
        (685.135_738_441_183_7, 685, 1_315_200, 27.4),
    )
    for predicted, frames, samples, seconds in expected:
        resolved = resolve_predicted_length(predicted)
        assert resolved["latent_frames"] == frames
        assert resolved["patched_frames"] == frames
        assert resolved["target_samples"] == samples
        assert resolved["seconds"] == seconds
    assert resolve_predicted_length(44.5)["latent_frames"] == 44
    assert resolve_predicted_length(45.5)["latent_frames"] == 46
    assert resolve_predicted_length(0.0)["latent_frames"] == 13
    assert resolve_predicted_length(1_000_000.0)["latent_frames"] == 750
    for invalid in (math.nan, -1.0):
        try:
            resolve_predicted_length(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid prediction accepted: {invalid}")
    print("duration_static_self_test=passed gpu_execution=false", flush=True)


def main() -> None:
    args = parse_args()
    if args.static_self_test:
        static_self_test()
        return
    if args.repeats != REPEATS:
        raise ValueError(f"duration benchmark requires exactly {REPEATS} repeats")
    if args.json_out.exists():
        raise FileExistsError(args.json_out)
    upstream = args.upstream.resolve()
    # Preserve the extension-bearing Hugging Face snapshot symlink paths.
    # The upstream loader selects safetensors vs. torch serialization from the
    # suffix, while the file hash below still binds the resolved blob contents.
    checkpoint = args.checkpoint.absolute()
    codec = args.codec.absolute()
    for path, expected in (
        (checkpoint, MODEL_SHA256),
        (codec, CODEC_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {head}")
    changes = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if changes:
        raise RuntimeError("upstream has tracked changes")

    device = torch.device(args.device)
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "duration benchmark requires exactly one visible CUDA device"
        )
    math_policy = configure_math()
    sys.path.insert(0, str(upstream))
    from irodori_tts.duration import build_duration_features
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device=str(device),
            codec_repo=str(codec),
            model_precision="fp32",
            codec_device=str(device),
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    model = runtime.model.eval()
    inputs, normalized_text, fixture_sha256 = create_inputs(
        runtime, args.text, args.fixture_out, device
    )
    text_ids = inputs["inputs/text_input_ids"].long()
    text_mask = inputs["inputs/text_mask"].bool()
    speaker_tokens = max(1, int(runtime.model_cfg.speaker_patch_size))
    ref_latent = torch.zeros(
        (
            1,
            speaker_tokens,
            runtime.model_cfg.latent_dim * runtime.model_cfg.latent_patch_size,
        ),
        dtype=torch.float32,
        device=device,
    )
    ref_mask = torch.zeros((1, speaker_tokens), dtype=torch.bool, device=device)
    caption_ids = inputs["inputs/caption_input_ids"].long()
    caption_mask = inputs["inputs/caption_mask"].bool()
    duration_features = build_duration_features(
        [normalized_text],
        token_counts=text_mask.sum(dim=1),
        max_text_len=text_ids.shape[1],
        has_speaker=torch.zeros((1,), dtype=torch.bool, device=device),
    ).to(device=device, dtype=torch.float32)
    has_speaker = torch.zeros((1,), dtype=torch.bool, device=device)
    has_caption = torch.zeros((1,), dtype=torch.bool, device=device)

    def encode() -> tuple[torch.Tensor, ...]:
        return model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
        )

    def predict(condition: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return model.predict_duration_log_frames(
            text_state=condition[0],
            text_mask=condition[1],
            speaker_state=condition[2],
            speaker_mask=condition[3],
            caption_state=condition[4],
            caption_mask=condition[5],
            duration_features=duration_features,
            has_speaker=has_speaker,
            has_caption=has_caption,
        ).float()

    with torch.inference_mode():
        cached_condition = encode()
        rows: list[ScopeResult] = []
        for repeat in range(1, args.repeats + 1):
            order = ("head", "full") if repeat % 2 else ("full", "head")
            for scope in order:
                operation = (
                    (lambda: predict(cached_condition))
                    if scope == "head"
                    else (lambda: predict(encode()))
                )
                cpu_output, timing = timed_scalar(device, operation)
                value = float(cpu_output.item())
                row = ScopeResult(
                    repeat=repeat,
                    cold=repeat <= WARMUPS,
                    scope=scope,
                    timing=timing,
                    log_frames=value,
                    predicted_frames=math.expm1(value),
                    output_sha256=sha256_f32(cpu_output),
                )
                rows.append(row)
                print(
                    "duration_repeat=" + json.dumps(asdict(row), sort_keys=True),
                    flush=True,
                )

    for scope in ("head", "full"):
        selected = [row for row in rows if row.scope == scope]
        if len({row.output_sha256 for row in selected}) != 1:
            raise RuntimeError(f"{scope} duration output is nondeterministic")
    if rows[0].output_sha256 != rows[1].output_sha256:
        raise RuntimeError("head-only and full duration predictions differ")

    resolved_length = resolve_predicted_length(
        next(row.predicted_frames for row in rows if row.scope == "full")
    )
    payload = {
        "format": "irodori-v4-python-duration-benchmark-v1",
        "pins": {
            "upstream_commit": UPSTREAM_COMMIT,
            "model_sha256": MODEL_SHA256,
            "codec_sha256": CODEC_SHA256,
            "fixture_sha256": fixture_sha256,
        },
        "math_policy": math_policy,
        "input": {
            "text": args.text,
            "normalized_text": normalized_text,
            "text_shape": list(text_ids.shape),
            "text_valid_tokens": int(text_mask.sum().item()),
            "speaker_request_shape": list(ref_latent.shape),
            "speaker_present": False,
            "caption_present": False,
            "duration_features_f32": duration_features.detach().cpu().tolist()[0],
        },
        "timer_contract": {
            "primary": "pre-sync to device complete; scalar readback excluded",
            "secondary": "owned contiguous float32 one-element CPU readback complete",
        },
        "resolved_length": resolved_length,
        "scopes": {scope: summarize(rows, scope) for scope in ("head", "full")},
        "repeats": [asdict(row) for row in rows],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
