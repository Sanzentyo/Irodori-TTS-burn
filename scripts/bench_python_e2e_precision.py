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
"""Benchmark strict fp32/fp16/bf16 official Irodori-TTS v4 E2E inference.

One loaded upstream runtime is reused for four repeats. The canonical fp32 E2E
fixture is the sole noise source: it is cast once to the selected dtype and the
same effective tensor is injected into exactly one sampler ``torch.randn`` call
per repeat. CUDA matmul TF32 and cuDNN TF32 are disabled and autocast is never
used. Exactly one CUDA device must be visible, and its name and PCI bus ID are
verified before and after the benchmark. If any native/f32 output hash differs
between repeats, the JSON is preserved and the process exits nonzero.

Run only after confirming exclusive access to the visible CUDA device::

    CUDA_VISIBLE_DEVICES=1 uv run --directory ../Irodori-TTS --extra cu128 \
      --frozen --no-sync python \
      ../Irodori-TTS-wgpu/scripts/bench_python_e2e_precision.py \
      --precision fp16
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch

import torch
from safetensors import safe_open

UPSTREAM_COMMIT = "9f19d9a9048099a4b978a762d0509228fe624e3f"
MODEL_REPO = "Aratako/Irodori-TTS-v4-Small"
MODEL_REVISION = "e4aaac4df355ff560dcd35e0dae272c3a759317b"
MODEL_SHA256 = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
CODEC_SHA256 = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"
SOURCE_FIXTURE_SHA256 = (
    "8022b2baeed05e68dd2d335bebb10392b5817d1251e006413294ff597d363fc8"
)

TEXT = "こんにちは。"
DEFAULT_SECONDS = 2.0
SECONDS = DEFAULT_SECONDS
NUM_STEPS = 4
SEED = 0
CFG_SCALE_TEXT_REQUESTED = 3.0
CFG_SCALE_CAPTION_REQUESTED = 3.0
CFG_SCALE_SPEAKER_REQUESTED = 5.0
CFG_MIN_T = 0.5
CFG_MAX_T = 1.0
EXPECTED_SAMPLE_RATE = 48_000
EXPECTED_SAMPLES = 96_000
EXPECTED_DECODED_SAMPLES = 96_000
NOISE_SHAPE = (1, 50, 32)
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DETERMINISM_SUMMARY_KEYS = (
    "all_audio_native_hashes_equal",
    "all_audio_f32_hashes_equal",
    "all_latent_native_hashes_equal",
    "all_latent_f32_hashes_equal",
)
T = TypeVar("T")


@dataclass(frozen=True)
class CallTiming:
    cuda_event_seconds: float
    synchronized_wall_seconds: float
    synchronized_wall_with_readback_seconds: float
    clock: str
    pre_start_device_sync: bool
    stop_after_cuda_event_sync: bool
    final_latent_readback_included: bool
    cpu_readback_elements: int
    cpu_readback_dtype: str
    cpu_readback_owned: bool
    cpu_readback_contiguous: bool
    secondary_includes_cpu_readback: bool
    secondary_stops_after_cpu_readback: bool
    work_report_inside_timed_region: bool
    primary_metric: str
    secondary_metric: str


@dataclass(frozen=True)
class SamplerConditioningGeometry:
    batch_rows: int
    latent_sequence: int
    latent_dim: int
    text_tokens: int
    speaker_tokens: int | None
    caption_tokens: int | None
    joint_axis: int


@dataclass(frozen=True)
class SamplerContextKvBuildWork:
    ordinal: int
    batch_rows: int
    text_tokens: int
    speaker_tokens: int | None
    caption_tokens: int | None
    context_tokens: int
    layers: int


@dataclass(frozen=True)
class SamplerForwardWork:
    ordinal: int
    batch_rows: int
    latent_sequence: int
    latent_dim: int
    timestep_shape: list[int]
    timestep_dtype: str
    timestep_f32_bits: int
    cfg_active: bool
    text_tokens: int
    speaker_tokens: int | None
    caption_tokens: int | None
    joint_axis: int
    context_kv_layers: int
    context_kv_build_ordinal: int | None
    output_shape: list[int]


@dataclass(frozen=True)
class PythonSamplerWorkReport:
    schema_version: int
    num_steps: int
    schedule_f32_bits: list[int]
    guidance_mode: str
    enabled_cfg: list[str]
    requested: SamplerConditioningGeometry
    encoded: SamplerConditioningGeometry
    encode_calls: int
    context_kv_builds: list[SamplerContextKvBuildWork]
    context_kv_forward_hits: int
    cond_mlp_batches: list[int]
    forwards: list[SamplerForwardWork]
    whole_model_forwards: int
    forward_batches: list[int]
    effective_model_rows: int
    model_layers: int
    model_block_calls: int


@dataclass(frozen=True)
class RepeatResult:
    repeat: int
    cold: bool
    sample_rf_runtime_seconds: float
    sample_rf_probe: CallTiming
    sampler_work_report: PythonSamplerWorkReport
    decode_latent_runtime_seconds: float
    decode_latent_probe: CallTiming
    total_to_decode_seconds: float
    synthesize_cuda_span_seconds: float
    synthesize_wall_seconds: float
    stage_timings_seconds: list[tuple[str, float]]
    effective_noise_native_sha256: str
    final_latent_native_sha256: str
    final_latent_f32_sha256: str
    final_latent_shape: list[int]
    final_latent_dtype: str
    audio_native_sha256: str
    audio_f32_sha256: str
    audio_shape: list[int]
    audio_dtype: str
    global_cpu_rng_unchanged: bool
    global_cuda_rng_unchanged: bool
    sampler_randn_interceptions: int
    peak_cuda_allocated_mib: float
    peak_cuda_reserved_mib: float


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_snapshot = (
        cache_root
        / "models--Aratako--Irodori-TTS-v4-Small"
        / "snapshots"
        / MODEL_REVISION
    )
    codec_snapshot = (
        cache_root
        / "models--Aratako--Semantic-DACVAE-Japanese-32dim"
        / "snapshots"
        / CODEC_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=sorted(PRECISION_DTYPES), required=True)
    parser.add_argument(
        "--upstream", type=Path, default=repository_root.parent / "Irodori-TTS"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=model_snapshot / "model.safetensors"
    )
    parser.add_argument("--codec", type=Path, default=codec_snapshot / "weights.pth")
    parser.add_argument(
        "--source-fixture",
        type=Path,
        default=Path("/tmp/irodori-v4-e2e-oracle.safetensors"),
    )
    parser.add_argument(
        "--source-fixture-sha256",
        default=SOURCE_FIXTURE_SHA256,
        help="required SHA-256 for the supplied source fixture/oracle",
    )
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--codec-device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose-runtime", action="store_true")
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    return parser.parse_args()


def absolute_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous()
    return value.view(torch.uint8).numpy().tobytes(order="C")


def sha256_tensor_native(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def sha256_tensor_f32(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_tracked_changes(path: Path) -> list[str]:
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    return [line for line in output.splitlines() if line]


def package_versions() -> dict[str, str]:
    names = (
        "dacvae",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "sentencepiece",
        "soundfile",
        "torch",
        "torchaudio",
        "torchcodec",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in names}


def validate_pins(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, str]:
    upstream = args.upstream.expanduser().resolve()
    checkpoint = absolute_without_resolving_symlinks(args.checkpoint)
    codec = absolute_without_resolving_symlinks(args.codec)
    source_fixture = args.source_fixture.expanduser().resolve()
    source_fixture_sha256 = args.source_fixture_sha256.strip().lower()
    if len(source_fixture_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_fixture_sha256
    ):
        raise ValueError("--source-fixture-sha256 must contain 64 hexadecimal digits")
    if not upstream.is_dir():
        raise FileNotFoundError(upstream)
    upstream_head = git_head(upstream)
    upstream_changes = git_tracked_changes(upstream)
    problems = []
    if upstream_head != UPSTREAM_COMMIT:
        problems.append(
            f"upstream commit mismatch: expected {UPSTREAM_COMMIT}, got {upstream_head}"
        )
    if upstream_changes:
        problems.append("upstream has tracked changes: " + "; ".join(upstream_changes))
    if problems and not args.allow_upstream_mismatch:
        raise RuntimeError("\n".join(problems))
    if problems:
        print("WARNING: " + " | ".join(problems), file=sys.stderr, flush=True)

    for path, expected_sha in (
        (checkpoint, MODEL_SHA256),
        (codec, CODEC_SHA256),
        (source_fixture, source_fixture_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
    return upstream, checkpoint, codec, source_fixture, upstream_head


def load_source_noise(
    path: Path, expected_shape: tuple[int, int, int]
) -> tuple[torch.Tensor, dict[str, Any]]:
    with safe_open(str(path), framework="pt", device="cpu") as fixture:
        oracle_json = (fixture.metadata() or {}).get("oracle_json")
        if oracle_json is None:
            raise RuntimeError("source fixture metadata lacks oracle_json")
        metadata = json.loads(oracle_json)
        fixture_format = metadata.get("format")
        if fixture_format == "irodori-v4-e2e-oracle-v1":
            noise = fixture.get_tensor("initial_noise")
        elif fixture_format == "irodori-v4-precision-oracle-v1":
            noise = fixture.get_tensor("noise/source_fp32")
        else:
            raise RuntimeError("source fixture has an unsupported oracle format")
    if metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("source fixture upstream commit mismatch")
    if metadata.get("model_sha256") != MODEL_SHA256:
        raise RuntimeError("source fixture model SHA-256 mismatch")
    if metadata.get("codec_sha256") != CODEC_SHA256:
        raise RuntimeError("source fixture codec SHA-256 mismatch")
    if noise.dtype != torch.float32 or tuple(noise.shape) != expected_shape:
        raise RuntimeError(
            f"source noise must be fp32 {expected_shape}, got {noise.dtype} {tuple(noise.shape)}"
        )
    if not bool(torch.isfinite(noise).all().item()):
        raise RuntimeError("source noise contains non-finite values")
    source_hash = sha256_tensor_native(noise)
    manifest_key = (
        "initial_noise"
        if metadata["format"] == "irodori-v4-e2e-oracle-v1"
        else "noise/source_fp32"
    )
    manifest_hash = (
        metadata.get("tensor_manifest", {}).get(manifest_key, {}).get("sha256")
    )
    if source_hash != manifest_hash:
        raise RuntimeError(
            "source noise tensor SHA-256 differs from the canonical fixture manifest"
        )
    return noise.contiguous(), metadata


def configure_strict_math() -> tuple[dict[str, Any], dict[str, Any]]:
    imported = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    effective = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "autocast": False,
    }
    expected = {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if {key: effective[key] for key in expected} != expected:
        raise RuntimeError(f"failed to establish strict math settings: {effective}")
    return imported, effective


def autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def assert_no_autocast(*devices: torch.device) -> None:
    enabled = {device.type for device in devices if autocast_enabled(device.type)}
    if enabled or torch.is_autocast_enabled():
        raise RuntimeError(
            f"autocast must remain disabled, enabled for {sorted(enabled)}"
        )


def direct_cast_module(
    module: torch.nn.Module, *, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
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
        raise RuntimeError(f"{label} direct dtype cast is incomplete: {preview}")


def verify_module_device(
    label: str, module: torch.nn.Module, device: torch.device
) -> None:
    mismatches = [
        (name, value.device)
        for name, value in [*module.named_parameters(), *module.named_buffers()]
        if value.device.type != device.type
        or (device.index is not None and value.device.index != device.index)
    ]
    if mismatches:
        preview = ", ".join(f"{name}:{actual}" for name, actual in mismatches[:8])
        raise RuntimeError(f"{label} device placement is incomplete: {preview}")


def cuda_device_identity(
    model_device: torch.device, codec_device: torch.device
) -> dict[str, Any]:
    """Validate the single-visible-device contract and return stable identity data."""
    if model_device.type != "cuda" or codec_device.type != "cuda":
        raise ValueError("model and codec devices must both be CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible_device_count = int(torch.cuda.device_count())
    if visible_device_count != 1:
        raise RuntimeError(
            "exactly one CUDA device must be visible; set CUDA_VISIBLE_DEVICES to the target GPU"
        )
    device_index = int(torch.cuda.current_device())
    for label, device in (("model", model_device), ("codec", codec_device)):
        if device.index not in (None, device_index):
            raise RuntimeError(
                f"requested {label} device {device}, but the sole visible current "
                f"device is cuda:{device_index}"
            )
    model_index = device_index if model_device.index is None else model_device.index
    codec_index = device_index if codec_device.index is None else codec_device.index
    if model_index != codec_index:
        raise ValueError("model and codec must use the same visible CUDA device")

    properties = torch.cuda.get_device_properties(device_index)
    name = str(properties.name).strip()
    pci_bus_id = properties.pci_bus_id
    if not name:
        raise RuntimeError("CUDA device name is empty")
    if isinstance(pci_bus_id, bool) or not isinstance(pci_bus_id, (int, str)):
        raise TypeError(f"unsupported CUDA PCI bus ID {pci_bus_id!r}")
    if isinstance(pci_bus_id, str):
        pci_bus_id = pci_bus_id.strip().lower()
        if not pci_bus_id:
            raise RuntimeError("CUDA PCI bus ID is empty")
    return {
        "visible_device_count": visible_device_count,
        "device_index_after_visibility": device_index,
        "name": name,
        "pci_bus_id_after_visibility": pci_bus_id,
        "total_memory_mib": float(properties.total_memory) / (1024.0 * 1024.0),
    }


def verify_native_tensor(label: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise RuntimeError(f"{label} must have dtype {dtype}, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label} contains non-finite values")


def f32_bits(value: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", float(value)))[0])


def optional_sequence_tokens(value: torch.Tensor | None) -> int | None:
    return None if value is None else int(value.shape[1])


def sampler_geometry(
    *,
    batch_rows: int,
    latent_sequence: int,
    latent_dim: int,
    text_tokens: int,
    speaker_tokens: int | None,
    caption_tokens: int | None,
) -> SamplerConditioningGeometry:
    return SamplerConditioningGeometry(
        batch_rows=batch_rows,
        latent_sequence=latent_sequence,
        latent_dim=latent_dim,
        text_tokens=text_tokens,
        speaker_tokens=speaker_tokens,
        caption_tokens=caption_tokens,
        joint_axis=(
            latent_sequence
            + text_tokens
            + (0 if speaker_tokens is None else speaker_tokens)
            + (0 if caption_tokens is None else caption_tokens)
        ),
    )


class SamplerWorkProbe:
    """Record host-visible sampler work without cloning or reading GPU values in-timer."""

    def __init__(self, sampler_kwargs: dict[str, Any]) -> None:
        self.model = sampler_kwargs["model"]
        text_ids = sampler_kwargs["text_input_ids"]
        ref_latent = sampler_kwargs.get("ref_latent")
        caption_ids = sampler_kwargs.get("caption_input_ids")
        self.base_batch = int(text_ids.shape[0])
        self.num_steps = int(sampler_kwargs["num_steps"])
        self.guidance_mode = str(sampler_kwargs["cfg_guidance_mode"]).strip().lower()
        self.cfg_scale_text = float(sampler_kwargs["cfg_scale_text"])
        self.cfg_scale_speaker = float(sampler_kwargs["cfg_scale_speaker"])
        self.cfg_scale_caption = float(sampler_kwargs["cfg_scale_caption"])
        self.latent_sequence = int(sampler_kwargs["sequence_length"])
        self.latent_dim = int(self.model.cfg.patched_latent_dim)
        self.requested = sampler_geometry(
            batch_rows=self.base_batch,
            latent_sequence=self.latent_sequence,
            latent_dim=self.latent_dim,
            text_tokens=int(text_ids.shape[1]),
            speaker_tokens=optional_sequence_tokens(ref_latent),
            caption_tokens=optional_sequence_tokens(caption_ids),
        )
        self.encode_calls = 0
        self.encoded: SamplerConditioningGeometry | None = None
        self.encoded_speaker_mask: torch.Tensor | None = None
        self.encoded_caption_mask: torch.Tensor | None = None
        self.kv_builds: list[SamplerContextKvBuildWork] = []
        self.cache_ordinals: dict[int, int] = {}
        self.cond_mlp_batches: list[int] = []
        self.forward_records: list[dict[str, Any]] = []
        self.installed = False
        self.original_encode = self.model.encode_conditions
        self.original_build_kv = self.model.build_context_kv_cache
        self.original_forward = self.model.forward_with_encoded_conditions
        self.original_cond_forward = self.model.cond_module.forward

    def capture_encode(self, *args: Any, **kwargs: Any) -> Any:
        result = self.original_encode(*args, **kwargs)
        self.encode_calls += 1
        if self.encode_calls != 1:
            raise RuntimeError("official sampler encoded conditions more than once")
        if len(result) != 6:
            raise RuntimeError(
                f"unexpected encoded-condition result length {len(result)}"
            )
        (
            text_state,
            _text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        ) = result
        self.encoded_speaker_mask = speaker_mask
        self.encoded_caption_mask = caption_mask
        self.encoded = sampler_geometry(
            batch_rows=int(text_state.shape[0]),
            latent_sequence=self.latent_sequence,
            latent_dim=self.latent_dim,
            text_tokens=int(text_state.shape[1]),
            speaker_tokens=optional_sequence_tokens(speaker_state),
            caption_tokens=optional_sequence_tokens(caption_state),
        )
        return result

    def capture_build_kv(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError("context K/V cache build must use keyword arguments")
        result = self.original_build_kv(**kwargs)
        text_state = kwargs["text_state"]
        speaker_state = kwargs.get("speaker_state")
        caption_state = kwargs.get("caption_state")
        ordinal = len(self.kv_builds)
        speaker_tokens = optional_sequence_tokens(speaker_state)
        caption_tokens = optional_sequence_tokens(caption_state)
        text_tokens = int(text_state.shape[1])
        self.kv_builds.append(
            SamplerContextKvBuildWork(
                ordinal=ordinal,
                batch_rows=int(text_state.shape[0]),
                text_tokens=text_tokens,
                speaker_tokens=speaker_tokens,
                caption_tokens=caption_tokens,
                context_tokens=(
                    text_tokens
                    + (0 if speaker_tokens is None else speaker_tokens)
                    + (0 if caption_tokens is None else caption_tokens)
                ),
                layers=len(result),
            )
        )
        self.cache_ordinals[id(result)] = ordinal
        return result

    def capture_cond_module(self, *args: Any, **kwargs: Any) -> Any:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError(
                "condition MLP did not receive a positional timestep embedding"
            )
        output = self.original_cond_forward(*args, **kwargs)
        self.cond_mlp_batches.append(int(args[0].shape[0]))
        return output

    def capture_forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args:
            raise RuntimeError("encoded-condition forward must use keyword arguments")
        output = self.original_forward(**kwargs)
        x_t = kwargs["x_t"]
        timestep = kwargs["t"]
        text_state = kwargs["text_state"]
        speaker_state = kwargs.get("speaker_state")
        caption_state = kwargs.get("caption_state")
        context_kv = kwargs.get("context_kv_cache")
        speaker_tokens = optional_sequence_tokens(speaker_state)
        caption_tokens = optional_sequence_tokens(caption_state)
        text_tokens = int(text_state.shape[1])
        self.forward_records.append(
            {
                "batch_rows": int(x_t.shape[0]),
                "latent_sequence": int(x_t.shape[1]),
                "latent_dim": int(x_t.shape[2]),
                "timestep": timestep,
                "text_tokens": text_tokens,
                "speaker_tokens": speaker_tokens,
                "caption_tokens": caption_tokens,
                "joint_axis": (
                    int(x_t.shape[1])
                    + text_tokens
                    + (0 if speaker_tokens is None else speaker_tokens)
                    + (0 if caption_tokens is None else caption_tokens)
                ),
                "context_kv_layers": 0 if context_kv is None else len(context_kv),
                "context_kv_build_ordinal": (
                    None
                    if context_kv is None
                    else self.cache_ordinals.get(id(context_kv))
                ),
                "output_shape": [int(value) for value in output.shape],
            }
        )
        return output

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("sampler work probe is already installed")
        self.model.encode_conditions = self.capture_encode
        self.model.build_context_kv_cache = self.capture_build_kv
        self.model.forward_with_encoded_conditions = self.capture_forward
        self.model.cond_module.forward = self.capture_cond_module
        self.installed = True

    def restore(self) -> None:
        if not self.installed:
            return
        self.model.cond_module.forward = self.original_cond_forward
        self.model.forward_with_encoded_conditions = self.original_forward
        self.model.build_context_kv_cache = self.original_build_kv
        self.model.encode_conditions = self.original_encode
        self.installed = False

    def finalize(self) -> PythonSamplerWorkReport:
        if self.installed:
            raise RuntimeError(
                "sampler work probe must be restored before finalization"
            )
        if self.encoded is None or self.encode_calls != 1:
            raise RuntimeError(
                f"expected one encoded-condition call, got {self.encode_calls}"
            )
        forwards = []
        for ordinal, record in enumerate(self.forward_records):
            timestep = record["timestep"]
            timestep_value = float(
                timestep.reshape(-1)[0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .item()
            )
            forwards.append(
                SamplerForwardWork(
                    ordinal=ordinal,
                    batch_rows=record["batch_rows"],
                    latent_sequence=record["latent_sequence"],
                    latent_dim=record["latent_dim"],
                    timestep_shape=[int(value) for value in timestep.shape],
                    timestep_dtype=dtype_name(timestep.dtype),
                    timestep_f32_bits=f32_bits(timestep_value),
                    cfg_active=record["batch_rows"] > self.base_batch,
                    text_tokens=record["text_tokens"],
                    speaker_tokens=record["speaker_tokens"],
                    caption_tokens=record["caption_tokens"],
                    joint_axis=record["joint_axis"],
                    context_kv_layers=record["context_kv_layers"],
                    context_kv_build_ordinal=record["context_kv_build_ordinal"],
                    output_shape=record["output_shape"],
                )
            )

        batches = [forward.batch_rows for forward in forwards]
        model_layers = len(self.model.blocks)
        has_speaker_context = self.encoded_speaker_mask is not None and bool(
            self.encoded_speaker_mask.any().detach().to(device="cpu").item()
        )
        has_caption_context = self.encoded_caption_mask is not None and bool(
            self.encoded_caption_mask.any().detach().to(device="cpu").item()
        )
        enabled_cfg = []
        if self.cfg_scale_text > 0.0:
            enabled_cfg.append("text")
        if self.cfg_scale_speaker > 0.0 and has_speaker_context:
            enabled_cfg.append("speaker")
        if self.cfg_scale_caption > 0.0 and has_caption_context:
            enabled_cfg.append("caption")
        schedule_f32_bits = [forward.timestep_f32_bits for forward in forwards]
        schedule_f32_bits.append(f32_bits(0.0))
        report = PythonSamplerWorkReport(
            schema_version=1,
            num_steps=self.num_steps,
            schedule_f32_bits=schedule_f32_bits,
            guidance_mode=self.guidance_mode,
            enabled_cfg=enabled_cfg,
            requested=self.requested,
            encoded=self.encoded,
            encode_calls=self.encode_calls,
            context_kv_builds=self.kv_builds,
            context_kv_forward_hits=sum(
                forward.context_kv_layers > 0 for forward in forwards
            ),
            cond_mlp_batches=self.cond_mlp_batches,
            forwards=forwards,
            whole_model_forwards=len(forwards),
            forward_batches=batches,
            effective_model_rows=sum(batches),
            model_layers=model_layers,
            model_block_calls=model_layers * len(forwards),
        )
        expected_encoded = sampler_geometry(
            batch_rows=1,
            latent_sequence=self.latent_sequence,
            latent_dim=32,
            text_tokens=256,
            speaker_tokens=2,
            caption_tokens=512,
        )
        if report.encoded != expected_encoded:
            raise RuntimeError(
                f"Python sampler encoded geometry mismatch: {report.encoded}"
            )
        expected_schedule_f32_bits = [
            0x3F7FBE77,
            0x3F3FCED9,
            0x3EFFBE77,
            0x3E7FBE77,
            0,
        ]
        if (
            report.num_steps != 4
            or report.schedule_f32_bits != expected_schedule_f32_bits
        ):
            raise RuntimeError(
                "Python sampler exact schedule mismatch: "
                f"expected={expected_schedule_f32_bits}, "
                f"actual={report.schedule_f32_bits}"
            )
        if report.guidance_mode != "independent" or report.enabled_cfg != ["text"]:
            raise RuntimeError(
                "Python sampler effective guidance mismatch: "
                f"mode={report.guidance_mode}, enabled={report.enabled_cfg}"
            )
        if report.forward_batches != [2, 2, 1, 1]:
            raise RuntimeError(
                f"Python sampler forward batches mismatch: {report.forward_batches}"
            )
        if report.whole_model_forwards != 4 or report.effective_model_rows != 6:
            raise RuntimeError(
                "Python sampler must issue four whole-model forwards and six rows"
            )
        if report.model_layers != 12 or report.model_block_calls != 48:
            raise RuntimeError(
                "Python sampler must issue 48 block calls (4 forwards x 12 layers)"
            )
        if [forward.cfg_active for forward in forwards] != [True, True, False, False]:
            raise RuntimeError("Python sampler CFG activity mismatch")
        if any(
            forward.joint_axis != self.latent_sequence + 770
            or forward.context_kv_layers != 12
            or forward.output_shape
            != [forward.batch_rows, forward.latent_sequence, forward.latent_dim]
            for forward in forwards
        ):
            raise RuntimeError(f"Python sampler forward geometry mismatch: {forwards}")
        if len(report.context_kv_builds) != 2:
            raise RuntimeError(
                f"Python sampler K/V build count mismatch: {report.context_kv_builds}"
            )
        if [build.batch_rows for build in report.context_kv_builds] != [1, 2] or any(
            build.context_tokens != 770 or build.layers != 12
            for build in report.context_kv_builds
        ):
            raise RuntimeError(
                f"Python sampler K/V build work mismatch: {report.context_kv_builds}"
            )
        if [forward.context_kv_build_ordinal for forward in forwards] != [1, 1, 0, 0]:
            raise RuntimeError("Python sampler K/V cache reuse mismatch")
        if report.context_kv_forward_hits != 4:
            raise RuntimeError(
                "Python sampler must use a context K/V cache on every forward"
            )
        if report.cond_mlp_batches != [2, 2, 1, 1]:
            raise RuntimeError(
                f"Python sampler condition MLP work mismatch: {report.cond_mlp_batches}"
            )
        return report


class FixedNoiseInjection:
    def __init__(
        self,
        *,
        effective_noise: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.effective_noise = effective_noise
        self.dtype = dtype
        self.device = device
        self.calls = 0

    def randn(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError(
                "sampler made more than one intercepted torch.randn call"
            )
        if len(args) != 1 or not isinstance(args[0], (tuple, list, torch.Size)):
            raise RuntimeError(f"unexpected torch.randn positional arguments: {args!r}")
        expected_kwargs = {"device", "dtype", "generator"}
        if set(kwargs) != expected_kwargs:
            raise RuntimeError(
                "unexpected torch.randn keyword arguments: "
                f"got {sorted(kwargs)}, expected {sorted(expected_kwargs)}"
            )
        shape = tuple(int(value) for value in args[0])
        requested_dtype = kwargs["dtype"]
        requested_device = torch.device(kwargs["device"])
        if shape != NOISE_SHAPE:
            raise RuntimeError(
                f"intercepted randn shape {shape}, expected {NOISE_SHAPE}"
            )
        if requested_dtype != self.dtype:
            raise RuntimeError(
                f"intercepted randn dtype {requested_dtype}, expected {self.dtype}"
            )
        if requested_device != self.device:
            raise RuntimeError(
                f"intercepted randn device {requested_device}, expected {self.device}"
            )
        if not isinstance(kwargs["generator"], torch.Generator):
            raise TypeError("intercepted randn generator is not torch.Generator")
        return self.effective_noise.clone()


class RepeatProbe:
    def __init__(
        self,
        *,
        official_sampler: Any,
        rf_module: Any,
        original_decode: Any,
        effective_noise: torch.Tensor,
        dtype: torch.dtype,
        model_device: torch.device,
        codec_device: torch.device,
    ) -> None:
        self.official_sampler = official_sampler
        self.rf_module = rf_module
        self.original_decode = original_decode
        self.effective_noise = effective_noise
        self.dtype = dtype
        self.model_device = model_device
        self.codec_device = codec_device
        self.sampler_calls = 0
        self.decode_calls = 0
        self.injection: FixedNoiseInjection | None = None
        self.sample_timing: CallTiming | None = None
        self.sampler_work_probe: SamplerWorkProbe | None = None
        self.sampler_work_report: PythonSamplerWorkReport | None = None
        self.decode_timing: CallTiming | None = None
        self.final_latent: torch.Tensor | None = None
        self.decoded_output: torch.Tensor | None = None

    @staticmethod
    def timed_call(
        device: torch.device,
        operation: Any,
        *,
        work_report_inside_timed_region: bool = False,
    ) -> tuple[Any, CallTiming]:
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record()
        output = operation()
        end.record()
        end.synchronize()
        device_complete_wall_seconds = time.perf_counter() - wall_started
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"timed operation must return a torch.Tensor, got {type(output)}"
            )
        cpu_native = output.detach().to(device="cpu", copy=True).contiguous()
        cpu_readback = cpu_native.to(
            dtype=torch.float32,
            copy=cpu_native.dtype != torch.float32,
        ).contiguous()
        torch.cuda.synchronize(device)
        readback_inclusive_wall_seconds = time.perf_counter() - wall_started
        if cpu_readback.numel() != output.numel():
            raise RuntimeError("CPU readback element count changed")
        if cpu_readback.dtype != torch.float32 or not cpu_readback.is_contiguous():
            raise RuntimeError(
                "CPU readback must be an owned contiguous float32 buffer"
            )
        return output, CallTiming(
            cuda_event_seconds=start.elapsed_time(end) / 1_000.0,
            synchronized_wall_seconds=device_complete_wall_seconds,
            synchronized_wall_with_readback_seconds=readback_inclusive_wall_seconds,
            clock="time.perf_counter + torch.cuda.Event",
            pre_start_device_sync=True,
            stop_after_cuda_event_sync=True,
            final_latent_readback_included=False,
            cpu_readback_elements=cpu_readback.numel(),
            cpu_readback_dtype="float32",
            cpu_readback_owned=True,
            cpu_readback_contiguous=True,
            secondary_includes_cpu_readback=True,
            secondary_stops_after_cpu_readback=True,
            work_report_inside_timed_region=work_report_inside_timed_region,
            primary_metric="synchronized_wall_seconds",
            secondary_metric="synchronized_wall_with_readback_seconds",
        )

    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.sampler_calls += 1
        if self.sampler_calls != 1 or args:
            raise RuntimeError(
                "runtime must invoke the official sampler exactly once and by keyword"
            )
        assert_no_autocast(self.model_device)
        injection = FixedNoiseInjection(
            effective_noise=self.effective_noise,
            dtype=self.dtype,
            device=self.model_device,
        )
        work_probe = SamplerWorkProbe(kwargs)

        def operation() -> torch.Tensor:
            with patch.object(
                self.rf_module.torch, "randn", side_effect=injection.randn
            ):
                return self.official_sampler(**kwargs)

        work_probe.install()
        try:
            output, timing = self.timed_call(
                self.model_device,
                operation,
                work_report_inside_timed_region=True,
            )
        finally:
            work_probe.restore()
        if injection.calls != 1:
            raise RuntimeError(
                f"sampler made {injection.calls} intercepted randn calls, expected 1"
            )
        verify_native_tensor("final patched latent", output, self.dtype)
        if tuple(output.shape) != NOISE_SHAPE:
            raise RuntimeError(
                f"final patched latent shape {tuple(output.shape)}, expected {NOISE_SHAPE}"
            )
        self.injection = injection
        self.sample_timing = timing
        self.sampler_work_probe = work_probe
        self.final_latent = output.detach()
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        if self.decode_calls != 1:
            raise RuntimeError("runtime must invoke codec decode exactly once")
        assert_no_autocast(self.codec_device)
        verify_native_tensor("codec input latent", latent, self.dtype)
        output, timing = self.timed_call(
            self.codec_device, lambda: self.original_decode(latent)
        )
        verify_native_tensor("raw decoded waveform", output, self.dtype)
        expected_shape = (1, 1, EXPECTED_DECODED_SAMPLES)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"raw decoded waveform shape {tuple(output.shape)}, expected {expected_shape}"
            )
        self.decode_timing = timing
        self.decoded_output = output.detach()
        return output

    def validate(self) -> None:
        if self.sampler_calls != 1 or self.decode_calls != 1:
            raise RuntimeError(
                "unexpected runtime calls: "
                f"sampler={self.sampler_calls}, decode={self.decode_calls}"
            )
        if self.injection is None or self.injection.calls != 1:
            raise RuntimeError("fixed-noise injection did not occur exactly once")
        if self.sample_timing is None or self.decode_timing is None:
            raise RuntimeError("probe did not capture both RF and codec timings")
        if self.sampler_work_probe is None:
            raise RuntimeError("probe did not retain the RF work recorder")
        if self.sampler_work_report is None:
            self.sampler_work_report = self.sampler_work_probe.finalize()
        if self.final_latent is None or self.decoded_output is None:
            raise RuntimeError("probe did not retain RF and codec outputs")
        for label, timing in (
            ("sample_rf", self.sample_timing),
            ("decode_latent", self.decode_timing),
        ):
            for kind, seconds in (
                ("cuda_event", timing.cuda_event_seconds),
                ("synchronized_wall", timing.synchronized_wall_seconds),
                (
                    "synchronized_wall_with_readback",
                    timing.synchronized_wall_with_readback_seconds,
                ),
            ):
                if not math.isfinite(seconds) or seconds < 0.0:
                    raise RuntimeError(
                        f"{label} {kind} timing is invalid: {seconds} seconds"
                    )
            if (
                timing.synchronized_wall_with_readback_seconds
                < timing.synchronized_wall_seconds
            ):
                raise RuntimeError(
                    f"{label} readback timer stopped before device timer"
                )


def require_probe_value(value: T | None, label: str) -> T:
    if value is None:
        raise RuntimeError(f"probe value {label!r} is missing")
    return value


def find_stage(stage_timings: list[tuple[str, float]], name: str) -> float:
    matches = [float(seconds) for stage, seconds in stage_timings if stage == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name!r} stage, got {len(matches)}")
    return validated_seconds(name, matches[0])


def validated_seconds(label: str, seconds: float) -> float:
    value = float(seconds)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{label} timing is invalid: {value} seconds")
    return value


def median_fields(rows: list[RepeatResult]) -> dict[str, float] | None:
    if not rows:
        return None
    return {
        "sample_rf_runtime_seconds": statistics.median(
            row.sample_rf_runtime_seconds for row in rows
        ),
        "sample_rf_probe_cuda_seconds": statistics.median(
            row.sample_rf_probe.cuda_event_seconds for row in rows
        ),
        "sample_rf_probe_wall_seconds": statistics.median(
            row.sample_rf_probe.synchronized_wall_seconds for row in rows
        ),
        "sample_rf_probe_wall_with_readback_seconds": statistics.median(
            row.sample_rf_probe.synchronized_wall_with_readback_seconds for row in rows
        ),
        "decode_latent_runtime_seconds": statistics.median(
            row.decode_latent_runtime_seconds for row in rows
        ),
        "decode_latent_probe_cuda_seconds": statistics.median(
            row.decode_latent_probe.cuda_event_seconds for row in rows
        ),
        "decode_latent_probe_wall_seconds": statistics.median(
            row.decode_latent_probe.synchronized_wall_seconds for row in rows
        ),
        "decode_latent_probe_wall_with_readback_seconds": statistics.median(
            row.decode_latent_probe.synchronized_wall_with_readback_seconds
            for row in rows
        ),
        "total_to_decode_seconds": statistics.median(
            row.total_to_decode_seconds for row in rows
        ),
        "synthesize_cuda_span_seconds": statistics.median(
            row.synthesize_cuda_span_seconds for row in rows
        ),
        "synthesize_wall_seconds": statistics.median(
            row.synthesize_wall_seconds for row in rows
        ),
    }


def summarize(repeats: list[RepeatResult]) -> dict[str, Any]:
    audio_native_hashes = [row.audio_native_sha256 for row in repeats]
    audio_f32_hashes = [row.audio_f32_sha256 for row in repeats]
    latent_native_hashes = [row.final_latent_native_sha256 for row in repeats]
    latent_f32_hashes = [row.final_latent_f32_sha256 for row in repeats]
    steady_medians = median_fields(repeats[1:])
    return {
        "all_repeat_medians": median_fields(repeats),
        "steady_repeat_medians_excluding_first": steady_medians,
        "steady_realtime_factor": (
            None
            if steady_medians is None
            else steady_medians["synthesize_wall_seconds"] / SECONDS
        ),
        "steady_times_realtime": (
            None
            if steady_medians is None
            else SECONDS / steady_medians["synthesize_wall_seconds"]
        ),
        "all_audio_native_hashes_equal": len(set(audio_native_hashes)) == 1,
        "all_audio_f32_hashes_equal": len(set(audio_f32_hashes)) == 1,
        "all_latent_native_hashes_equal": len(set(latent_native_hashes)) == 1,
        "all_latent_f32_hashes_equal": len(set(latent_f32_hashes)) == 1,
        "audio_native_sha256": audio_native_hashes[0],
        "audio_f32_sha256": audio_f32_hashes[0],
        "final_latent_native_sha256": latent_native_hashes[0],
        "final_latent_f32_sha256": latent_f32_hashes[0],
        "maximum_peak_cuda_allocated_mib": max(
            row.peak_cuda_allocated_mib for row in repeats
        ),
        "maximum_peak_cuda_reserved_mib": max(
            row.peak_cuda_reserved_mib for row in repeats
        ),
    }


def failed_determinism_gates(summary: dict[str, Any]) -> list[str]:
    return [key for key in DETERMINISM_SUMMARY_KEYS if summary.get(key) is not True]


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    if not math.isfinite(seconds) or seconds <= 0.0 or sample_rate <= 0:
        raise ValueError("seconds and sample_rate must be positive and finite")
    return round(seconds * sample_rate)


def seconds_for_truncating_runtime(target_samples: int, sample_rate: int) -> float:
    if target_samples <= 0 or sample_rate <= 0:
        raise ValueError("target_samples and sample_rate must be positive")
    seconds = target_samples / sample_rate
    while int(seconds * sample_rate) < target_samples:
        seconds = math.nextafter(seconds, math.inf)
    if int(seconds * sample_rate) != target_samples:
        raise RuntimeError("cannot represent the requested integer sample target")
    return seconds


def main() -> None:
    global EXPECTED_DECODED_SAMPLES, EXPECTED_SAMPLES, NOISE_SHAPE, SECONDS
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not math.isfinite(args.seconds) or args.seconds <= 0.0:
        raise ValueError("--seconds must be finite and positive")
    SECONDS = float(args.seconds)
    EXPECTED_SAMPLES = seconds_to_samples(SECONDS, EXPECTED_SAMPLE_RATE)
    if EXPECTED_SAMPLES <= 0:
        raise ValueError("--seconds rounded to zero target samples")
    runtime_seconds = seconds_for_truncating_runtime(
        EXPECTED_SAMPLES, EXPECTED_SAMPLE_RATE
    )
    latent_steps = math.ceil(EXPECTED_SAMPLES / 1_920)
    NOISE_SHAPE = (1, latent_steps, 32)
    EXPECTED_DECODED_SAMPLES = latent_steps * 1_920
    precision = str(args.precision)
    target_dtype = PRECISION_DTYPES[precision]
    model_device = torch.device(args.model_device)
    codec_device = torch.device(args.codec_device)

    upstream, checkpoint, codec, source_fixture, upstream_head = validate_pins(args)
    source_noise, source_metadata = load_source_noise(source_fixture, NOISE_SHAPE)
    imported_math, effective_math = configure_strict_math()
    if torch.__version__ != "2.10.0+cu128":
        raise RuntimeError(f"torch must be 2.10.0+cu128, got {torch.__version__}")
    if importlib.metadata.version("transformers") != "5.12.1":
        raise RuntimeError(
            "transformers must be 5.12.1, got "
            + importlib.metadata.version("transformers")
        )
    initial_cuda_device = cuda_device_identity(model_device, codec_device)
    assert_no_autocast(model_device, codec_device)

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": initial_cuda_device["name"],
        "visible_device_count": initial_cuda_device["visible_device_count"],
        "device_index_after_visibility": initial_cuda_device[
            "device_index_after_visibility"
        ],
        "pci_bus_id_after_visibility": initial_cuda_device[
            "pci_bus_id_after_visibility"
        ],
        "total_memory_mib": initial_cuda_device["total_memory_mib"],
        "precision": precision,
        "native_dtype": dtype_name(target_dtype),
        "imported_math_defaults": imported_math,
        "effective_math": effective_math,
        "package_versions": package_versions(),
    }
    print("environment=" + json.dumps(environment, sort_keys=True), flush=True)

    sys.path.insert(0, str(upstream))
    import irodori_tts.inference_runtime as runtime_module
    import irodori_tts.rf as rf_module
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        resolve_cfg_scales,
    )
    from irodori_tts.rf import sample_euler_rf_cfg as official_sampler

    effective_text, effective_caption, effective_speaker, cfg_messages = (
        resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=CFG_SCALE_TEXT_REQUESTED,
            cfg_scale_caption=CFG_SCALE_CAPTION_REQUESTED,
            cfg_scale_speaker=CFG_SCALE_SPEAKER_REQUESTED,
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=False,
        )
    )
    effective_cfg = {
        "text": effective_text,
        "caption": effective_caption,
        "speaker": effective_speaker,
    }
    if effective_cfg != {"text": 3.0, "caption": 3.0, "speaker": 0.0}:
        raise RuntimeError(f"unexpected resolved CFG: {effective_cfg}")

    runtime_key = RuntimeKey(
        checkpoint=str(checkpoint),
        model_device=str(model_device),
        codec_repo=str(codec),
        model_precision="fp32",
        codec_device=str(codec_device),
        codec_precision="fp32",
        codec_deterministic_encode=True,
        codec_deterministic_decode=True,
        compile_model=False,
        compile_dynamic=False,
    )
    torch.cuda.synchronize(model_device)
    torch.cuda.reset_peak_memory_stats(model_device)
    load_started = time.perf_counter()
    runtime = InferenceRuntime.from_key(runtime_key)
    runtime.model = direct_cast_module(
        runtime.model, device=model_device, dtype=target_dtype
    )
    runtime._model_dtype = target_dtype
    runtime.codec.model = direct_cast_module(
        runtime.codec.model, device=codec_device, dtype=target_dtype
    )
    runtime.codec.dtype = target_dtype
    verify_module_dtype("model", runtime.model, target_dtype)
    verify_module_dtype("codec", runtime.codec.model, target_dtype)
    verify_module_device("model", runtime.model, model_device)
    verify_module_device("codec", runtime.codec.model, codec_device)
    runtime.watermarker.model = None
    effective_noise = source_noise.to(device=model_device, dtype=target_dtype)
    verify_native_tensor("effective initial noise", effective_noise, target_dtype)
    effective_noise_hash = sha256_tensor_native(effective_noise)
    torch.cuda.synchronize(model_device)
    load_wall_seconds = time.perf_counter() - load_started
    load_peak_allocated_mib = torch.cuda.max_memory_allocated(model_device) / (
        1024.0 * 1024.0
    )
    load_peak_reserved_mib = torch.cuda.max_memory_reserved(model_device) / (
        1024.0 * 1024.0
    )
    if runtime.watermarker.ready:
        raise RuntimeError("watermark must be disabled for the canonical comparison")
    assert_no_autocast(model_device, codec_device)

    request = SamplingRequest(
        text=TEXT,
        caption=None,
        no_ref=True,
        num_candidates=1,
        decode_mode="sequential",
        seconds=runtime_seconds,
        num_steps=NUM_STEPS,
        cfg_scale_text=effective_text,
        cfg_scale_caption=effective_caption,
        cfg_scale_speaker=effective_speaker,
        cfg_guidance_mode="independent",
        cfg_min_t=CFG_MIN_T,
        cfg_max_t=CFG_MAX_T,
        context_kv_cache=True,
        speaker_uncond_mode="mask",
        seed=SEED,
        t_schedule_mode="linear",
        sway_coeff=-1.0,
        trim_tail=False,
    )

    repeat_results = []
    for repeat_index in range(args.repeats):
        original_decode = runtime.codec.decode_latent
        probe = RepeatProbe(
            official_sampler=official_sampler,
            rf_module=rf_module,
            original_decode=original_decode,
            effective_noise=effective_noise,
            dtype=target_dtype,
            model_device=model_device,
            codec_device=codec_device,
        )
        previous_sampler = runtime_module.sample_euler_rf_cfg
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(model_device).clone()
        torch.cuda.synchronize(model_device)
        torch.cuda.reset_peak_memory_stats(model_device)
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        total_start.record()
        runtime_module.sample_euler_rf_cfg = probe.sample
        runtime.codec.decode_latent = probe.decode
        try:
            with torch.inference_mode():
                result = runtime.synthesize(
                    request,
                    log_fn=print if args.verbose_runtime else None,
                )
        finally:
            runtime.codec.decode_latent = original_decode
            runtime_module.sample_euler_rf_cfg = previous_sampler
        total_end.record()
        total_end.synchronize()
        synthesize_wall_seconds = time.perf_counter() - wall_started
        synthesize_cuda_span_seconds = total_start.elapsed_time(total_end) / 1_000.0
        peak_allocated_mib = torch.cuda.max_memory_allocated(model_device) / (
            1024.0 * 1024.0
        )
        peak_reserved_mib = torch.cuda.max_memory_reserved(model_device) / (
            1024.0 * 1024.0
        )
        probe.validate()
        assert_no_autocast(model_device, codec_device)
        cpu_rng_unchanged = torch.equal(cpu_rng_before, torch.random.get_rng_state())
        cuda_rng_unchanged = torch.equal(
            cuda_rng_before, torch.cuda.get_rng_state(model_device)
        )
        if not cpu_rng_unchanged or not cuda_rng_unchanged:
            raise RuntimeError(
                "global RNG state changed despite fixed-noise injection: "
                f"cpu={cpu_rng_unchanged}, cuda={cuda_rng_unchanged}"
            )
        if result.used_seed != SEED:
            raise RuntimeError(f"runtime used seed {result.used_seed}, expected {SEED}")
        if result.sample_rate != EXPECTED_SAMPLE_RATE:
            raise RuntimeError(
                f"runtime sample rate {result.sample_rate}, expected {EXPECTED_SAMPLE_RATE}"
            )
        if result.audio.numel() != EXPECTED_SAMPLES:
            raise RuntimeError(
                f"runtime produced {result.audio.numel()} samples, expected {EXPECTED_SAMPLES}"
            )
        expected_audio_shape = (1, EXPECTED_SAMPLES)
        if tuple(result.audio.shape) != expected_audio_shape:
            raise RuntimeError(
                f"runtime audio shape {tuple(result.audio.shape)}, expected {expected_audio_shape}"
            )
        verify_native_tensor("runtime result audio", result.audio, target_dtype)

        final_latent = require_probe_value(probe.final_latent, "final_latent")
        decoded_output = require_probe_value(probe.decoded_output, "decoded_output")
        sample_timing = require_probe_value(probe.sample_timing, "sample_timing")
        sampler_work_report = require_probe_value(
            probe.sampler_work_report, "sampler_work_report"
        )
        decode_timing = require_probe_value(probe.decode_timing, "decode_timing")
        injection = require_probe_value(probe.injection, "injection")
        if repeat_results and (
            sampler_work_report != repeat_results[0].sampler_work_report
        ):
            raise RuntimeError(
                "Python RF work manifest changed between repeats: "
                f"first={repeat_results[0].sampler_work_report}, "
                f"repeat={repeat_index + 1}:{sampler_work_report}"
            )
        captured_audio = decoded_output.detach().to(device="cpu")[
            0, :, :EXPECTED_SAMPLES
        ]
        result_audio = result.audio.detach().to(device="cpu")
        if not torch.equal(captured_audio, result_audio):
            difference = float(
                (captured_audio.float() - result_audio.float()).abs().max()
            )
            raise RuntimeError(
                f"captured raw decoder output differs from runtime audio: {difference}"
            )
        stages = [(name, float(seconds)) for name, seconds in result.stage_timings]
        total_to_decode_seconds = validated_seconds(
            "total_to_decode", float(result.total_to_decode)
        )
        validated_seconds("synthesize CUDA span", synthesize_cuda_span_seconds)
        validated_seconds("synthesize wall", synthesize_wall_seconds)
        row = RepeatResult(
            repeat=repeat_index + 1,
            cold=repeat_index == 0,
            sample_rf_runtime_seconds=find_stage(stages, "sample_rf"),
            sample_rf_probe=sample_timing,
            sampler_work_report=sampler_work_report,
            decode_latent_runtime_seconds=find_stage(stages, "decode_latent"),
            decode_latent_probe=decode_timing,
            total_to_decode_seconds=total_to_decode_seconds,
            synthesize_cuda_span_seconds=synthesize_cuda_span_seconds,
            synthesize_wall_seconds=synthesize_wall_seconds,
            stage_timings_seconds=stages,
            effective_noise_native_sha256=effective_noise_hash,
            final_latent_native_sha256=sha256_tensor_native(final_latent),
            final_latent_f32_sha256=sha256_tensor_f32(final_latent),
            final_latent_shape=list(final_latent.shape),
            final_latent_dtype=dtype_name(final_latent.dtype),
            audio_native_sha256=sha256_tensor_native(result.audio),
            audio_f32_sha256=sha256_tensor_f32(result.audio),
            audio_shape=list(result.audio.shape),
            audio_dtype=dtype_name(result.audio.dtype),
            global_cpu_rng_unchanged=cpu_rng_unchanged,
            global_cuda_rng_unchanged=cuda_rng_unchanged,
            sampler_randn_interceptions=injection.calls,
            peak_cuda_allocated_mib=peak_allocated_mib,
            peak_cuda_reserved_mib=peak_reserved_mib,
        )
        repeat_results.append(row)
        print("repeat=" + json.dumps(asdict(row), sort_keys=True), flush=True)

    final_cuda_device = cuda_device_identity(model_device, codec_device)
    if final_cuda_device != initial_cuda_device:
        raise RuntimeError(
            "CUDA device identity changed during E2E benchmark: "
            f"initial={initial_cuda_device}, final={final_cuda_device}"
        )
    summary = summarize(repeat_results)
    parameters = {
        "text": TEXT,
        "caption": None,
        "no_ref": True,
        "seconds": SECONDS,
        "num_steps": NUM_STEPS,
        "seed": SEED,
        "precision": precision,
        "model_precision": precision,
        "codec_precision": precision,
        "runtime_load_precision": "fp32_then_direct_cast",
        "autocast": False,
        "cfg_guidance_mode": "independent",
        "cfg_requested": {
            "text": CFG_SCALE_TEXT_REQUESTED,
            "caption": CFG_SCALE_CAPTION_REQUESTED,
            "speaker": CFG_SCALE_SPEAKER_REQUESTED,
        },
        "cfg_effective": effective_cfg,
        "cfg_min_t": CFG_MIN_T,
        "cfg_max_t": CFG_MAX_T,
        "context_kv_cache": True,
        "compile_model": False,
        "compile_dynamic": False,
        "codec_deterministic_encode": True,
        "codec_deterministic_decode": True,
        "t_schedule_mode": "linear",
        "sway_coeff": -1.0,
        "trim_tail": False,
        "watermark": False,
    }
    payload = {
        "format": "irodori-v4-python-e2e-precision-benchmark-v1",
        "precision": precision,
        "native_dtype": dtype_name(target_dtype),
        "strict_math": True,
        "no_autocast": True,
        "runtime_reused_across_repeats": True,
        "cuda_device_identity_verified_before_and_after": True,
        "repeats": args.repeats,
        "load_wall_seconds": load_wall_seconds,
        "load_peak_cuda_allocated_mib": load_peak_allocated_mib,
        "load_peak_cuda_reserved_mib": load_peak_reserved_mib,
        "environment": environment,
        "pins": {
            "upstream_commit": upstream_head,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_sha256": MODEL_SHA256,
            "codec_repo": CODEC_REPO,
            "codec_revision": CODEC_REVISION,
            "codec_sha256": CODEC_SHA256,
            "source_fixture_sha256": args.source_fixture_sha256.lower(),
        },
        "noise_contract": {
            "source_key": "initial_noise",
            "source_dtype": "float32",
            "source_shape": list(NOISE_SHAPE),
            "source_tensor_sha256": sha256_tensor_native(source_noise),
            "target_dtype": dtype_name(target_dtype),
            "effective_tensor_sha256": effective_noise_hash,
            "cast_count": 1,
            "same_effective_tensor_reused": True,
            "sampler_randn_interceptions_per_repeat": 1,
            "total_sampler_randn_interceptions": sum(
                row.sampler_randn_interceptions for row in repeat_results
            ),
        },
        "source_oracle": {
            "format": source_metadata["format"],
            "upstream_commit": source_metadata["upstream_commit"],
        },
        "length_contract": {
            "seconds": SECONDS,
            "target_samples": EXPECTED_SAMPLES,
            "latent_steps": NOISE_SHAPE[1],
            "decoded_samples": EXPECTED_DECODED_SAMPLES,
        },
        "parameters": parameters,
        "cfg_resolution_messages": cfg_messages,
        "repeat_results": [asdict(row) for row in repeat_results],
        "summary": summary,
    }
    output_path = args.json_out
    if output_path is None:
        output_path = Path(
            f"/tmp/irodori-python-e2e-{precision}-strict-fixed-noise.json"
        )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    print("summary=" + json.dumps(payload["summary"], sort_keys=True), flush=True)
    print(f"json_out={output_path}", flush=True)
    failed_gates = failed_determinism_gates(summary)
    if failed_gates:
        raise RuntimeError(
            "repeat output determinism gate failed after preserving the JSON result "
            f"at {output_path}: {', '.join(failed_gates)}"
        )
    print("repeat_output_determinism_gate=passed", flush=True)


if __name__ == "__main__":
    main()
