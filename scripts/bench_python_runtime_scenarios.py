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
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "irodori-v4-python-runtime-scenarios-v1"
TEXT = "これは現在の実装を評価するための音声合成ベンチマークです。"
DESIGN_A = "落ち着いた低めの声で、明瞭かつ穏やかに話す。"
DESIGN_B = "明るく快活な高めの声で、テンポよく話す。"
PRECISION_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
SCENARIO_NAMES = (
    "text_only_fixed",
    "design_fixed",
    "design_switch",
    "clone_raw_fixed",
    "clone_raw_switch",
    "clone_prepared_fixed",
    "clone_prepared_switch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--ref1", type=Path, required=True)
    parser.add_argument("--ref2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--audio-output-dir",
        type=Path,
        help=(
            "new directory receiving the first measured CPU waveform per scenario "
            "as raw little-endian f32; file I/O is outside the timing boundary"
        ),
    )
    parser.add_argument("--seconds", type=float, default=4.48)
    parser.add_argument(
        "--latent-frames",
        type=int,
        help=(
            "exact codec/RF frame count; when set, derives a duration whose "
            "48 kHz floor is exactly frames * 1920 samples"
        ),
    )
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-pci", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--precision", choices=sorted(PRECISION_DTYPES), default="fp32")
    parser.add_argument(
        "--source-fixture",
        type=Path,
        help="canonical safetensors containing FP32 key initial_noise",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="new directory for Rust-compatible text/design request fixtures",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIO_NAMES,
        help="scenario to execute; repeat the option, or omit it for all scenarios",
    )
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


def direct_cast_module(
    module: torch.nn.Module, *, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    """Apply the same explicit F16 conversion as the precision-oracle harness."""
    module.to(device=device)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(device=device, dtype=dtype)
            if parameter.grad is not None and parameter.grad.is_floating_point():
                parameter.grad.data = parameter.grad.data.to(device=device, dtype=dtype)
        for child in module.modules():
            for name, buffer in child._buffers.items():
                if buffer is None:
                    continue
                if buffer.is_floating_point() and buffer.dtype != dtype:
                    child._buffers[name] = buffer.to(device=device, dtype=dtype)
                elif buffer.device != device:
                    child._buffers[name] = buffer.to(device=device)
    return module


def verify_module_dtype(
    label: str, module: torch.nn.Module, dtype: torch.dtype
) -> None:
    mismatches = [
        (name, value.dtype)
        for name, value in [*module.named_parameters(), *module.named_buffers()]
        if value.is_floating_point() and value.dtype != dtype
    ]
    if mismatches:
        preview = ", ".join(f"{name}:{actual}" for name, actual in mismatches[:8])
        raise RuntimeError(f"{label} dtype conversion is incomplete: {preview}")


def load_source_noise(path: Path, latent_frames: int) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as fixture:
        canonical = fixture.get_tensor("initial_noise")
    if canonical.dtype != torch.float32 or tuple(canonical.shape) != (1, 50, 32):
        raise RuntimeError(
            "source fixture initial_noise must be contiguous FP32 [1, 50, 32]"
        )
    repeats = math.ceil(latent_frames / 50)
    tiled = canonical.repeat(1, repeats, 1)[:, :latent_frames, :].clone()
    block = torch.arange(latent_frames, dtype=torch.int64).div(
        50, rounding_mode="floor"
    )
    signs = torch.where(block.remainder(2) == 0, 1.0, -1.0).reshape(1, -1, 1)
    source = (tiled * signs).contiguous()
    if not bool(torch.isfinite(source).all().item()):
        raise RuntimeError("derived source noise contains non-finite values")
    return source


class FixedNoise:
    def __init__(self, source: torch.Tensor, dtype: torch.dtype) -> None:
        self.source = source
        self.dtype = dtype
        self.calls = 0

    def randn(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("sampler made more than one torch.randn call")
        if len(args) != 1:
            raise RuntimeError(f"unexpected torch.randn positional args: {args!r}")
        shape = tuple(int(value) for value in args[0])
        if shape != tuple(self.source.shape):
            raise RuntimeError(
                f"sampler requested noise shape {shape}, expected {tuple(self.source.shape)}"
            )
        requested_dtype = kwargs.get("dtype")
        requested_device = torch.device(kwargs.get("device"))
        if requested_dtype != self.dtype or requested_device != torch.device("cuda:0"):
            raise RuntimeError(
                "sampler noise dtype/device mismatch: "
                f"{requested_dtype}/{requested_device}"
            )
        return self.source.to(device=requested_device, dtype=self.dtype).clone()


def save_request_fixtures(
    directory: Path,
    runtime: Any,
    source_noise: torch.Tensor,
) -> dict[str, dict[str, str]]:
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"fixture directory already exists: {directory}")
    directory.mkdir(parents=True)
    text_ids, text_mask = runtime.tokenizer.batch_encode(
        [TEXT], max_length=runtime.default_text_max_len
    )
    if runtime.caption_tokenizer is None:
        raise RuntimeError("released v4 runtime lacks a caption tokenizer")

    outputs: dict[str, dict[str, str]] = {}
    for name, caption in (("text", ""), ("design", DESIGN_A)):
        caption_ids, caption_mask = runtime.caption_tokenizer.batch_encode(
            [caption], max_length=runtime.default_caption_max_len
        )
        if not caption:
            caption_mask.zero_()
        destination = directory / f"{name}.safetensors"
        save_file(
            {
                "inputs/text_input_ids": text_ids.detach().cpu().contiguous(),
                "inputs/text_mask": text_mask.detach().cpu().contiguous(),
                "inputs/caption_input_ids": caption_ids.detach().cpu().contiguous(),
                "inputs/caption_mask": caption_mask.detach().cpu().contiguous(),
                "noise/source_fp32": source_noise.detach().cpu().contiguous(),
            },
            str(destination),
            metadata={
                "format": "irodori-v4-runtime-request-fixture-v1",
                "voice": name,
            },
        )
        outputs[name] = {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
        }
    return outputs


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
    if args.fixture_dir is not None and (
        args.fixture_dir.exists() or args.fixture_dir.is_symlink()
    ):
        raise FileExistsError(f"fixture directory already exists: {args.fixture_dir}")
    if args.audio_output_dir is not None and (
        args.audio_output_dir.exists() or args.audio_output_dir.is_symlink()
    ):
        raise FileExistsError(
            f"audio output directory already exists: {args.audio_output_dir}"
        )
    if not (args.seconds > 0.0 and args.seconds <= 30.0):
        raise ValueError("--seconds must be in (0, 30]")
    if args.latent_frames is not None and not (1 <= args.latent_frames <= 750):
        raise ValueError("--latent-frames must be in [1, 750]")
    if args.num_steps <= 0 or args.warmups < 1 or args.measured < 1:
        raise ValueError("steps and repetition counts must be positive")
    for path in (args.upstream, args.checkpoint, args.codec, args.ref1, args.ref2):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.source_fixture is not None and not args.source_fixture.is_file():
        raise FileNotFoundError(args.source_fixture)
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
    if args.audio_output_dir is not None:
        args.audio_output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.upstream))
    import irodori_tts.inference_runtime as runtime_module
    import irodori_tts.rf as rf_module
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
    )

    key = RuntimeKey(
        checkpoint=str(args.checkpoint),
        model_device="cuda:0",
        codec_repo=str(args.codec),
        # The public runtime currently accepts fp32/bf16 only. Load the pinned
        # FP32 record and apply the same explicit F16 conversion as the oracle
        # exporter so FP16 remains a comparison-harness policy, not an upstream
        # source modification.
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
    dtype = PRECISION_DTYPES[args.precision]
    if args.precision == "fp16":
        runtime.model = direct_cast_module(
            runtime.model, device=torch.device("cuda:0"), dtype=dtype
        )
        runtime._model_dtype = dtype
        runtime.codec.model = direct_cast_module(
            runtime.codec.model, device=torch.device("cuda:0"), dtype=dtype
        )
        runtime.codec.dtype = dtype
    verify_module_dtype("model", runtime.model, dtype)
    verify_module_dtype("codec", runtime.codec.model, dtype)
    runtime.watermarker.model = None
    torch.cuda.synchronize()
    load_wall = time.perf_counter() - load_started
    load_peak_allocated = torch.cuda.max_memory_allocated() / 1048576.0
    load_peak_reserved = torch.cuda.max_memory_reserved() / 1048576.0
    load_idle_allocated = torch.cuda.memory_allocated() / 1048576.0
    load_idle_reserved = torch.cuda.memory_reserved() / 1048576.0
    if args.latent_frames is None:
        requested_seconds = args.seconds
        latent_frames = math.ceil(round(requested_seconds * 48_000) / 1_920)
    else:
        latent_frames = args.latent_frames
        target_samples = latent_frames * 1_920
        requested_seconds = math.nextafter(target_samples / 48_000, math.inf)
        if math.floor(requested_seconds * 48_000) != target_samples:
            raise RuntimeError("failed to derive an exact frame-aligned duration")
    output_seconds = latent_frames * 1_920 / 48_000
    source_noise = (
        load_source_noise(args.source_fixture, latent_frames)
        if args.source_fixture is not None
        else None
    )
    fixture_outputs = (
        save_request_fixtures(args.fixture_dir, runtime, source_noise)
        if args.fixture_dir is not None and source_noise is not None
        else {}
    )
    if args.fixture_dir is not None and source_noise is None:
        raise RuntimeError("--fixture-dir requires --source-fixture")

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
            seconds=requested_seconds,
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
    requested_scenarios = set(args.scenario or SCENARIO_NAMES)
    scenarios = [item for item in scenarios if item[0] in requested_scenarios]
    if len(scenarios) != len(requested_scenarios):
        raise RuntimeError("scenario selection did not resolve uniquely")
    all_rows: dict[str, list[Row]] = {}
    audio_artifacts: dict[str, dict[str, Any]] = {}
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
            if source_noise is None:
                result = runtime.synthesize(selected_request)
            else:
                fixed_noise = FixedNoise(source_noise, dtype)
                with patch.object(
                    rf_module.torch, "randn", side_effect=fixed_noise.randn
                ):
                    result = runtime.synthesize(selected_request)
                if fixed_noise.calls != 1:
                    raise RuntimeError(
                        f"sampler made {fixed_noise.calls} noise calls, expected one"
                    )
            end_event.record()
            audio = (
                result.audio.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )
            end_event.synchronize()
            wall = time.perf_counter() - wall_started
            event = start_event.elapsed_time(end_event) / 1000.0
            if audio.device.type != "cpu" or audio.dtype != torch.float32:
                raise RuntimeError(
                    f"runtime result must be CPU float32, got {audio.device}/{audio.dtype}"
                )
            audio_seconds = float(audio.shape[-1]) / float(result.sample_rate)
            if repetition == args.warmups and args.audio_output_dir is not None:
                audio_path = args.audio_output_dir / f"{scenario}.f32le"
                audio_bytes = audio.numpy().astype("<f4", copy=False).tobytes(order="C")
                with audio_path.open("xb") as file:
                    file.write(audio_bytes)
                    file.flush()
                    os.fsync(file.fileno())
                audio_artifacts[scenario] = {
                    "path": str(audio_path.resolve()),
                    "samples": int(audio.numel()),
                    "sha256": sha256_file(audio_path),
                    "excluded_from_cpu_audio_ready_wall": True,
                }
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
            "precision": args.precision,
            "strict_fp32": args.precision == "fp32",
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
            "seconds": requested_seconds,
            "output_seconds": output_seconds,
            "latent_frames": latent_frames,
            "num_steps": args.num_steps,
            "warmups": args.warmups,
            "measured": args.measured,
            "seed": args.seed,
            "trim_tail": False,
            "source_noise_injected": source_noise is not None,
            "source_fixture": (
                None
                if args.source_fixture is None
                else str(args.source_fixture.resolve())
            ),
            "source_fixture_sha256": (
                None
                if args.source_fixture is None
                else sha256_file(args.source_fixture)
            ),
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
        "rust_request_fixtures": fixture_outputs,
        "audio_output_dir": (
            None
            if args.audio_output_dir is None
            else str(args.audio_output_dir.resolve())
        ),
        "audio_artifacts": audio_artifacts,
        "scenarios": {
            name: {
                "summary": summarize_rows(rows, output_seconds),
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
