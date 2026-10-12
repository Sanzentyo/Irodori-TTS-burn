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
"""Measure resident-runtime and speaker-switch scenarios for Irodori-TTS v4.

The runtime is loaded exactly once.  Every measured request records both a CUDA
event interval and a host wall interval that stops after the runtime-owned CPU
audio tensor exists.  Reference WAV and pre-encoded-latent cases are kept
separate so speaker preparation is not accidentally presented as model work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

FORMAT = "irodori-v4-python-runtime-scenarios-v1"
TEXT = "これは現在の実装を評価するための音声合成ベンチマークです。"
DESIGN_A = "落ち着いた低めの声で、明瞭かつ穏やかに話す。"
DESIGN_B = "明るく快活な高めの声で、テンポよく話す。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--ref1", type=Path, required=True)
    parser.add_argument("--ref2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.48)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-pci", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor_f32(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def median(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot summarize an empty sample set")
    return float(statistics.median(values))


def stage_map(stages: list[tuple[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, seconds in stages:
        if name in result:
            raise RuntimeError(f"duplicate runtime stage {name!r}")
        result[name] = float(seconds)
    return result


@dataclass(frozen=True)
class Row:
    scenario: str
    repetition: int
    warmup: bool
    selected_voice: str
    cuda_event_seconds: float
    cpu_audio_ready_wall_seconds: float
    audio_seconds: float
    audio_sha256_f32: str
    peak_allocated_mib: float
    peak_reserved_mib: float
    stages_seconds: dict[str, float]


def summarize_rows(rows: list[Row], seconds: float) -> dict[str, Any]:
    measured = [row for row in rows if not row.warmup]
    event = [row.cuda_event_seconds for row in measured]
    wall = [row.cpu_audio_ready_wall_seconds for row in measured]
    stage_names = sorted({name for row in measured for name in row.stages_seconds})
    stages: dict[str, dict[str, float]] = {}
    for name in stage_names:
        values = [
            row.stages_seconds[name] for row in measured if name in row.stages_seconds
        ]
        stages[name] = {
            "min_seconds": min(values),
            "median_seconds": median(values),
            "max_seconds": max(values),
        }
    hashes = sorted({row.audio_sha256_f32 for row in measured})
    hashes_by_voice = {
        voice: sorted(
            {row.audio_sha256_f32 for row in measured if row.selected_voice == voice}
        )
        for voice in sorted({row.selected_voice for row in measured})
    }
    return {
        "measured_requests": len(measured),
        "cuda_event": {
            "min_seconds": min(event),
            "median_seconds": median(event),
            "max_seconds": max(event),
        },
        "cpu_audio_ready_wall": {
            "min_seconds": min(wall),
            "median_seconds": median(wall),
            "max_seconds": max(wall),
        },
        "throughput": {
            "requests_per_second_from_wall_median": 1.0 / median(wall),
            "audio_seconds_per_wall_second": seconds / median(wall),
            "real_time_factor_from_wall_median": median(wall) / seconds,
        },
        "peak_allocated_mib": max(row.peak_allocated_mib for row in measured),
        "peak_reserved_mib": max(row.peak_reserved_mib for row in measured),
        "audio_f32_sha256_values": hashes,
        "audio_f32_sha256_by_voice": hashes_by_voice,
        "deterministic_per_voice": all(
            len(voice_hashes) == 1 for voice_hashes in hashes_by_voice.values()
        ),
        "distinct_voices": len(hashes_by_voice),
        "stages": stages,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.work_dir.exists():
        raise FileExistsError(f"work directory already exists: {args.work_dir}")
    if not (args.seconds > 0.0 and args.seconds <= 30.0):
        raise ValueError("--seconds must be in (0, 30]")
    if args.num_steps <= 0 or args.warmups < 1 or args.measured < 1:
        raise ValueError("steps and repetition counts must be positive")
    for path in (args.upstream, args.checkpoint, args.codec, args.ref1, args.ref2):
        if not path.exists():
            raise FileNotFoundError(path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly '0'")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.is_autocast_enabled():
        raise RuntimeError("autocast must be disabled")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible CUDA device, got {torch.cuda.device_count()}"
        )
    props = torch.cuda.get_device_properties(0)
    raw_pci = props.pci_bus_id
    if isinstance(raw_pci, int) and not isinstance(raw_pci, bool):
        pci = f"00000000:{raw_pci:02x}:00.0"
    else:
        pci = str(raw_pci).strip().lower()
        if pci.startswith("0000:"):
            pci = "00000000:" + pci.removeprefix("0000:")
    expected_pci = args.expected_pci.strip().lower()
    if pci != expected_pci:
        raise RuntimeError(f"expected visible PCI {expected_pci}, got {pci}")
    if props.name != args.expected_gpu_name:
        raise RuntimeError(
            f"expected GPU {args.expected_gpu_name!r}, got {props.name!r}"
        )

    args.work_dir.mkdir(parents=False)
    sys.path.insert(0, str(args.upstream))
    import irodori_tts.inference_runtime as runtime_module
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
    )

    key = RuntimeKey(
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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    runtime = InferenceRuntime.from_key(key)
    runtime.watermarker.model = None
    torch.cuda.synchronize()
    load_wall = time.perf_counter() - load_started
    load_peak_allocated = torch.cuda.max_memory_allocated() / 1048576.0
    load_peak_reserved = torch.cuda.max_memory_reserved() / 1048576.0
    load_idle_allocated = torch.cuda.memory_allocated() / 1048576.0
    load_idle_reserved = torch.cuda.memory_reserved() / 1048576.0

    # Persist raw, unpatched codec latents once. The public runtime accepts
    # these paths, but currently exposes no public prepare/cache API, so this
    # benchmark makes that ergonomic gap explicit rather than hiding it.
    prepared_paths: list[Path] = []
    prepared_times: list[float] = []
    for index, source in enumerate((args.ref1, args.ref2), start=1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        wav, sample_rate = runtime_module._load_audio(source)
        latent = runtime.codec.encode_waveform(
            wav.unsqueeze(0),
            sample_rate=int(sample_rate),
            normalize_db=-16.0,
            ensure_max=True,
        ).cpu()[0]
        torch.cuda.synchronize()
        prepared_times.append(time.perf_counter() - started)
        destination = args.work_dir / f"ref{index}-latent.pt"
        torch.save(latent, destination)
        prepared_paths.append(destination)

    def request(
        *,
        caption: str | None = None,
        ref_wav: Path | None = None,
        ref_latent: Path | None = None,
        no_ref: bool = False,
    ) -> SamplingRequest:
        return SamplingRequest(
            text=TEXT,
            caption=caption,
            ref_wav=None if ref_wav is None else str(ref_wav),
            ref_latent=None if ref_latent is None else str(ref_latent),
            no_ref=no_ref,
            seconds=args.seconds,
            num_steps=args.num_steps,
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0,
            cfg_guidance_mode="independent",
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            context_kv_cache=True,
            seed=args.seed,
            t_schedule_mode="linear",
            sway_coeff=-1.0,
            trim_tail=False,
        )

    scenarios: list[tuple[str, list[tuple[str, SamplingRequest]]]] = [
        ("text_only_fixed", [("none", request(no_ref=True))]),
        ("design_fixed", [("design-a", request(caption=DESIGN_A, no_ref=True))]),
        (
            "design_switch",
            [
                ("design-a", request(caption=DESIGN_A, no_ref=True)),
                ("design-b", request(caption=DESIGN_B, no_ref=True)),
            ],
        ),
        ("clone_raw_fixed", [("clone-ref1-wav", request(ref_wav=args.ref1))]),
        (
            "clone_raw_switch",
            [
                ("clone-ref1-wav", request(ref_wav=args.ref1)),
                ("clone-ref2-wav", request(ref_wav=args.ref2)),
            ],
        ),
        (
            "clone_prepared_fixed",
            [("clone-ref1-latent", request(ref_latent=prepared_paths[0]))],
        ),
        (
            "clone_prepared_switch",
            [
                ("clone-ref1-latent", request(ref_latent=prepared_paths[0])),
                ("clone-ref2-latent", request(ref_latent=prepared_paths[1])),
            ],
        ),
    ]
    all_rows: dict[str, list[Row]] = {}
    total_repetitions = args.warmups + args.measured
    for scenario, variants in scenarios:
        rows: list[Row] = []
        for repetition in range(total_repetitions):
            selected_voice, selected_request = variants[repetition % len(variants)]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_started = time.perf_counter()
            start_event.record()
            result = runtime.synthesize(selected_request)
            end_event.record()
            end_event.synchronize()
            wall = time.perf_counter() - wall_started
            event = start_event.elapsed_time(end_event) / 1000.0
            audio = result.audio
            if audio.device.type != "cpu" or audio.dtype != torch.float32:
                raise RuntimeError(
                    f"runtime result must be CPU float32, got {audio.device}/{audio.dtype}"
                )
            audio_seconds = float(audio.shape[-1]) / float(result.sample_rate)
            rows.append(
                Row(
                    scenario=scenario,
                    repetition=repetition + 1,
                    warmup=repetition < args.warmups,
                    selected_voice=selected_voice,
                    cuda_event_seconds=float(event),
                    cpu_audio_ready_wall_seconds=float(wall),
                    audio_seconds=audio_seconds,
                    audio_sha256_f32=sha256_tensor_f32(audio),
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 1048576.0,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 1048576.0,
                    stages_seconds=stage_map(result.stage_timings),
                )
            )
        all_rows[scenario] = rows

    payload = {
        "format": FORMAT,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": props.name,
            "pci_bus_id": pci,
            "strict_fp32": True,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "autocast": torch.is_autocast_enabled(),
        },
        "pins": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "codec": str(args.codec),
            "codec_sha256": sha256_file(args.codec),
            "ref1": str(args.ref1.resolve()),
            "ref1_sha256": sha256_file(args.ref1),
            "ref2": str(args.ref2.resolve()),
            "ref2_sha256": sha256_file(args.ref2),
        },
        "parameters": {
            "text": TEXT,
            "design_a": DESIGN_A,
            "design_b": DESIGN_B,
            "seconds": args.seconds,
            "num_steps": args.num_steps,
            "warmups": args.warmups,
            "measured": args.measured,
            "seed": args.seed,
            "trim_tail": False,
            "audio_readback_in_wall_interval": True,
            "runtime_reused_across_all_scenarios": True,
        },
        "load": {
            "wall_seconds": load_wall,
            "peak_allocated_mib": load_peak_allocated,
            "peak_reserved_mib": load_peak_reserved,
            "idle_allocated_mib": load_idle_allocated,
            "idle_reserved_mib": load_idle_reserved,
            "all_resident": True,
        },
        "prepared_reference": {
            "one_time_encode_wall_seconds": prepared_times,
            "paths": [str(path) for path in prepared_paths],
            "sha256": [sha256_file(path) for path in prepared_paths],
        },
        "scenarios": {
            name: {
                "summary": summarize_rows(rows, args.seconds),
                "rows": [asdict(row) for row in rows],
            }
            for name, rows in all_rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


if __name__ == "__main__":
    main()
