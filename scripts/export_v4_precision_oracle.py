# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "dacvae @ git+https://github.com/facebookresearch/dacvae@414c20785fc3a28373073ea8ef7a1316eeeaca6e",
#   "huggingface-hub==1.23.0",
#   "numpy==2.2.6",
#   "safetensors==0.7.0",
#   "sentencepiece==0.1.99",
#   "soundfile==0.13.1",
#   "torch==2.10.0",
#   "torchaudio==2.10.0",
#   "torchcodec==0.10.0",
#   "transformers==5.12.1",
# ]
# ///
"""Export strict same-noise fp32/fp16/bf16 Irodori-TTS v4 oracles.

The canonical fp32 E2E fixture is the only source of random noise. Its fp32
tensor is cast exactly once to the requested runtime dtype, then injected into
the unmodified upstream Euler sampler by intercepting its single ``torch.randn``
call. No autocast is used and both CUDA matmul and cuDNN TF32 are disabled.
Exactly one CUDA device must be visible; its name and PCI bus ID are recorded
and verified again before the fixture is written.

This script intentionally requires a CUDA model and codec device. Run it only
when exclusive access to the selected GPU has been confirmed::

    CUDA_VISIBLE_DEVICES=1 uv run --directory ../Irodori-TTS --extra cu128 \
      --frozen --no-sync python \
      ../Irodori-TTS-wgpu/scripts/export_v4_precision_oracle.py \
      --precision fp16
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

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
CFG_SCALE_TEXT = 3.0
CFG_SCALE_CAPTION = 3.0
CFG_SCALE_SPEAKER = 5.0
CFG_MIN_T = 0.5
CFG_MAX_T = 1.0
INIT_SCALE = 0.999
CANONICAL_NOISE_SHAPE = (1, 50, 32)
NOISE_SHAPE = CANONICAL_NOISE_SHAPE
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--precision", choices=sorted(PRECISION_DTYPES))
    mode.add_argument(
        "--static-self-test",
        action="store_true",
        help="validate dtype/noise/math contracts on CPU without loading the runtime",
    )
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
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="write a machine-readable manifest for the exported oracle",
    )
    parser.add_argument("--verification-wav", type=Path)
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="target audio duration; determines the latent/noise sequence length",
    )
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--codec-device", default="cuda:0")
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


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = cpu_contiguous(tensor)
    return {
        "shape": list(value.shape),
        "dtype": dtype_name(value.dtype),
        "elements": int(value.numel()),
        "bytes": len(tensor_bytes(value)),
        "sha256": sha256_tensor(value),
    }


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


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_tracked_changes(path: Path) -> list[str]:
    status = subprocess.check_output(
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
    return status.splitlines()


def configure_strict_fp32_math() -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "autocast": False,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


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
    """Match the official runtime's explicit parameter/buffer dtype conversion."""
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
        raise RuntimeError(
            "precision oracle export requires CUDA model and codec devices"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for precision oracle export")
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
        raise RuntimeError("model and codec must use the same visible CUDA device")

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
        raise RuntimeError(
            f"{label} must retain native dtype {dtype}, got {tensor.dtype}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label} contains non-finite values")


def cast_source_noise_once(
    source_noise: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Perform the sole source-to-effective target conversion.

    For fp32 the dtype component is intentionally a no-op; moving the canonical
    CPU tensor to the model device still happens through this same single call.
    """
    if (
        source_noise.device.type != "cpu"
        or source_noise.dtype != torch.float32
        or tuple(source_noise.shape) != NOISE_SHAPE
        or not source_noise.is_contiguous()
        or not bool(torch.isfinite(source_noise).all().item())
    ):
        raise RuntimeError(
            f"source noise must be finite contiguous CPU fp32 {NOISE_SHAPE}"
        )
    effective_noise = source_noise.to(device=device, dtype=dtype)
    if (
        effective_noise.device != device
        or effective_noise.dtype != dtype
        or tuple(effective_noise.shape) != NOISE_SHAPE
        or not effective_noise.is_contiguous()
        or not bool(torch.isfinite(effective_noise).all().item())
    ):
        raise RuntimeError(
            "one-time effective noise conversion produced an invalid tensor"
        )
    return effective_noise


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Resolve a decimal duration to the nearest integer sample.

    Multiplication by a decimal such as 10.2 can land one ULP below an exact
    integer. Truncation would then shorten the request by one sample and make
    the manifest disagree with the hop-aligned duration-predictor contract.
    """
    if not math.isfinite(seconds) or seconds <= 0.0 or sample_rate <= 0:
        raise ValueError("seconds and sample_rate must be positive and finite")
    return round(seconds * sample_rate)


def seconds_for_truncating_runtime(target_samples: int, sample_rate: int) -> float:
    """Encode an integer sample target for an upstream truncating runtime."""
    if target_samples <= 0 or sample_rate <= 0:
        raise ValueError("target_samples and sample_rate must be positive")
    seconds = target_samples / sample_rate
    while int(seconds * sample_rate) < target_samples:
        seconds = math.nextafter(seconds, math.inf)
    if int(seconds * sample_rate) != target_samples:
        raise RuntimeError("cannot represent the requested integer sample target")
    return seconds


def run_static_self_test() -> None:
    """Exercise all precision contracts without CUDA or upstream model imports."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the static self-test")
    if set(PRECISION_DTYPES) != {"fp32", "fp16", "bf16"}:
        raise RuntimeError(f"unexpected precision map: {sorted(PRECISION_DTYPES)}")
    expected_names = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    actual_names = {
        precision: dtype_name(dtype) for precision, dtype in PRECISION_DTYPES.items()
    }
    if actual_names != expected_names:
        raise RuntimeError(
            f"native dtype metadata map mismatch: {actual_names} != {expected_names}"
        )
    for seconds, samples in ((0.5, 24_000), (10.2, 489_600), (19.56, 938_880)):
        if seconds_to_samples(seconds, 48_000) != samples:
            raise RuntimeError(f"duration-to-sample conversion failed for {seconds}")
        runtime_seconds = seconds_for_truncating_runtime(samples, 48_000)
        if int(runtime_seconds * 48_000) != samples:
            raise RuntimeError(f"runtime duration encoding failed for {seconds}")

    source = torch.linspace(-1.0, 1.0, math.prod(NOISE_SHAPE), dtype=torch.float32)
    source = source.reshape(NOISE_SHAPE).contiguous()
    short = derive_source_noise(source, 13)
    long = derive_source_noise(source, 100)
    if tuple(short.shape) != (1, 13, 32) or not torch.equal(short, source[:, :13, :]):
        raise RuntimeError("short variable-length source derivation mismatch")
    if tuple(long.shape) != (1, 100, 32) or not torch.equal(long[:, 50:, :], -source):
        raise RuntimeError("long variable-length source derivation mismatch")
    source_hash = sha256_tensor(source)
    cast_results = {}
    for precision, dtype in PRECISION_DTYPES.items():
        effective = cast_source_noise_once(
            source, device=torch.device("cpu"), dtype=dtype
        )
        verify_native_tensor(f"static {precision} noise", effective, dtype)
        dtype_noop = dtype == source.dtype
        if dtype_noop != (precision == "fp32"):
            raise RuntimeError(f"unexpected dtype no-op policy for {precision}")
        if precision == "fp32" and (
            effective.data_ptr() != source.data_ptr()
            or sha256_tensor(effective) != source_hash
        ):
            raise RuntimeError("fp32 target conversion must be a bit-exact dtype no-op")
        cast_results[precision] = {
            "native_dtype": dtype_name(effective.dtype),
            "dtype_cast_is_noop": dtype_noop,
            "effective_sha256": sha256_tensor(effective),
        }

    for label in ("model", "codec"):
        module = torch.nn.Linear(4, 4, bias=True, dtype=torch.float32)
        parameter_pointers = [parameter.data_ptr() for parameter in module.parameters()]
        direct_cast_module(module, device=torch.device("cpu"), dtype=torch.float32)
        verify_module_dtype(label, module, torch.float32)
        if parameter_pointers != [
            parameter.data_ptr() for parameter in module.parameters()
        ]:
            raise RuntimeError(
                f"{label} fp32 direct cast unexpectedly replaced storage"
            )

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_float32_precision = torch.get_float32_matmul_precision()
    try:
        strict_math = configure_strict_fp32_math()
        expected_math = {
            "autocast": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        if strict_math != expected_math:
            raise RuntimeError(
                f"strict math metadata mismatch: {strict_math} != {expected_math}"
            )
        assert_no_autocast(torch.device("cpu"))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.set_float32_matmul_precision(previous_float32_precision)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized by the static self-test")
    print(
        "static_self_test="
        + json.dumps(
            {
                "cast_count": 1,
                "cuda_initialized": False,
                "precisions": cast_results,
                "strict_math": True,
            },
            sort_keys=True,
        )
    )


def load_canonical_source_noise(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    actual_sha = sha256_file(path)
    if actual_sha != SOURCE_FIXTURE_SHA256:
        raise RuntimeError(
            "canonical source fixture SHA-256 mismatch: "
            f"expected {SOURCE_FIXTURE_SHA256}, got {actual_sha}"
        )
    with safe_open(str(path), framework="pt", device="cpu") as fixture:
        metadata = fixture.metadata() or {}
        oracle_json = metadata.get("oracle_json")
        if oracle_json is None:
            raise RuntimeError("source fixture metadata lacks oracle_json")
        source_metadata = json.loads(oracle_json)
        noise = fixture.get_tensor("initial_noise")
    if source_metadata.get("format") != "irodori-v4-e2e-oracle-v1":
        raise RuntimeError("source fixture has an unsupported oracle format")
    if source_metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("source fixture upstream commit mismatch")
    if noise.dtype != torch.float32 or tuple(noise.shape) != CANONICAL_NOISE_SHAPE:
        raise RuntimeError(
            "canonical source noise must be fp32 "
            f"{CANONICAL_NOISE_SHAPE}, got {noise.dtype} {tuple(noise.shape)}"
        )
    if not bool(torch.isfinite(noise).all().item()):
        raise RuntimeError("source noise contains non-finite values")
    return noise.contiguous(), source_metadata


def derive_source_noise(
    canonical_noise: torch.Tensor, target_steps: int
) -> torch.Tensor:
    """Derive a deterministic variable-length source without invoking an RNG.

    The first 50 frames stay byte-identical to the canonical 2-second fixture.
    Longer requests tile those frames and alternate the sign of each 50-frame
    block. Performance is value-independent, while this definition keeps every
    benchmark input reproducible and exactly shareable across runtimes.
    """
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    if tuple(canonical_noise.shape) != CANONICAL_NOISE_SHAPE:
        raise RuntimeError("canonical noise shape changed before derivation")
    repeats = math.ceil(target_steps / CANONICAL_NOISE_SHAPE[1])
    tiled = canonical_noise.repeat(1, repeats, 1)[:, :target_steps, :].clone()
    block = torch.arange(target_steps, dtype=torch.int64).div(
        CANONICAL_NOISE_SHAPE[1], rounding_mode="floor"
    )
    signs = torch.where(block.remainder(2) == 0, 1.0, -1.0).reshape(1, -1, 1)
    derived = (tiled * signs).contiguous()
    if derived.dtype != torch.float32 or not bool(torch.isfinite(derived).all()):
        raise RuntimeError("variable-length source-noise derivation failed")
    return derived


class PrecisionSamplerTrace:
    """Observe official Euler arithmetic while replacing only its initial RNG call."""

    def __init__(
        self,
        *,
        effective_noise: torch.Tensor,
        dtype: torch.dtype,
        model_device: torch.device,
    ) -> None:
        self.effective_noise = effective_noise
        self.dtype = dtype
        self.model_device = model_device
        self.randn_calls = 0
        self.step_x: list[torch.Tensor] = []
        self.step_velocity: list[torch.Tensor] = []
        self.step_model_output: list[torch.Tensor] = []
        self.step_t: list[torch.Tensor] = []
        self.conditions: dict[str, torch.Tensor] = {}
        self.final_patched: torch.Tensor | None = None
        self.effective_sampler_kwargs: dict[str, Any] = {}
        self.recurrence_max_abs = 0.0

    def injected_randn(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.randn_calls += 1
        if self.randn_calls != 1:
            raise RuntimeError(
                "official sampler made more than one torch.randn call inside the scoped "
                "initial-noise interception"
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
        requested_dtype = kwargs.get("dtype")
        requested_device = torch.device(kwargs.get("device"))
        if shape != NOISE_SHAPE:
            raise RuntimeError(
                f"intercepted randn shape {shape}, expected {NOISE_SHAPE}"
            )
        if requested_dtype != self.dtype:
            raise RuntimeError(
                f"intercepted randn dtype {requested_dtype}, expected {self.dtype}"
            )
        if requested_device != self.model_device:
            raise RuntimeError(
                f"intercepted randn device {requested_device}, expected {self.model_device}"
            )
        if "generator" not in kwargs:
            raise RuntimeError(
                "official sampler randn call unexpectedly lacks a generator"
            )
        return self.effective_noise.clone()

    def run(
        self,
        *,
        official_sampler: Any,
        rf_module: Any,
        sampler_args: tuple[Any, ...],
        sampler_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if sampler_args:
            raise RuntimeError("official runtime must call the sampler only by keyword")
        model = sampler_kwargs["model"]
        batch = int(sampler_kwargs["text_input_ids"].shape[0])
        cfg_scale_text = float(sampler_kwargs["cfg_scale_text"])
        self.effective_sampler_kwargs = {
            key: value
            for key, value in sampler_kwargs.items()
            if key
            not in {
                "model",
                "text_input_ids",
                "text_mask",
                "caption_input_ids",
                "caption_mask",
                "ref_latent",
                "ref_mask",
                "speaker_state_override",
                "speaker_mask_override",
            }
        }

        original_encode = model.encode_conditions
        original_forward = model.forward_with_encoded_conditions

        def capture_encode(*args: Any, **kwargs: Any) -> Any:
            result = original_encode(*args, **kwargs)
            names = (
                "text_state_cond",
                "text_mask_cond",
                "speaker_state_cond",
                "speaker_mask_cond",
                "caption_state_cond",
                "caption_mask_cond",
            )
            for name, value in zip(names, result, strict=True):
                if value is not None:
                    self.conditions[name] = value.detach().clone()
            return result

        def capture_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            if args:
                raise RuntimeError(
                    "encoded-condition forward must use keyword arguments"
                )
            assert_no_autocast(self.model_device)
            output = original_forward(**kwargs)
            self.step_x.append(kwargs["x_t"][:batch].detach().clone())
            self.step_t.append(kwargs["t"][:batch].detach().clone())
            self.step_model_output.append(output.detach().clone())
            if int(output.shape[0]) == batch * 2:
                conditional, text_unconditional = output.chunk(2, dim=0)
                velocity = conditional + cfg_scale_text * (
                    conditional - text_unconditional
                )
            elif int(output.shape[0]) == batch:
                velocity = output
            else:
                raise RuntimeError(
                    f"unexpected model output shape {tuple(output.shape)}"
                )
            self.step_velocity.append(velocity.detach().clone())
            return output

        model.encode_conditions = capture_encode
        model.forward_with_encoded_conditions = capture_forward
        try:
            with patch.object(
                rf_module.torch, "randn", side_effect=self.injected_randn
            ):
                final = official_sampler(**sampler_kwargs)
        finally:
            model.encode_conditions = original_encode
            model.forward_with_encoded_conditions = original_forward

        if self.randn_calls != 1:
            raise RuntimeError(
                f"official sampler made {self.randn_calls} intercepted randn calls, expected 1"
            )
        self.final_patched = final.detach().clone()
        self.verify_trace()
        return final

    def verify_trace(self) -> None:
        if self.final_patched is None or len(self.step_x) != NUM_STEPS:
            raise RuntimeError(
                f"expected {NUM_STEPS} Euler calls, observed {len(self.step_x)}"
            )
        if not torch.equal(self.step_x[0], self.effective_noise):
            difference = float((self.step_x[0] - self.effective_noise).abs().max())
            raise RuntimeError(
                f"effective initial noise changed before step 0: {difference}"
            )
        schedule = (
            1.0 - torch.linspace(0.0, 1.0, NUM_STEPS + 1, device=self.model_device)
        ) * INIT_SCALE
        expected_next = [*self.step_x[1:], self.final_patched]
        for index, (x_t, velocity, next_x) in enumerate(
            zip(self.step_x, self.step_velocity, expected_next, strict=True)
        ):
            reconstructed = x_t + velocity * (schedule[index + 1] - schedule[index])
            difference = float((reconstructed - next_x).abs().max().item())
            self.recurrence_max_abs = max(self.recurrence_max_abs, difference)
        if self.recurrence_max_abs != 0.0:
            raise RuntimeError(
                "captured velocity does not exactly reproduce Euler recurrence: "
                f"max_abs={self.recurrence_max_abs:.9g}"
            )


def main() -> None:
    global NOISE_SHAPE, SECONDS
    args = parse_args()
    if args.static_self_test:
        run_static_self_test()
        return
    if args.precision is None:
        raise RuntimeError("precision is required outside the static self-test")
    precision = args.precision
    if not math.isfinite(args.seconds) or args.seconds <= 0.0:
        raise ValueError("--seconds must be finite and positive")
    SECONDS = float(args.seconds)
    target_samples = seconds_to_samples(SECONDS, 48_000)
    runtime_seconds = seconds_for_truncating_runtime(target_samples, 48_000)
    if target_samples <= 0:
        raise ValueError("--seconds rounded to zero target samples")
    latent_steps = math.ceil(target_samples / 1_920)
    NOISE_SHAPE = (1, latent_steps, 32)
    target_dtype = PRECISION_DTYPES[precision]
    upstream = args.upstream.expanduser().resolve()
    checkpoint = absolute_without_resolving_symlinks(args.checkpoint)
    codec_path = absolute_without_resolving_symlinks(args.codec)
    source_fixture = args.source_fixture.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path(f"/tmp/irodori-v4-{precision}-strict-oracle.safetensors")
    )

    upstream_head = git_head(upstream)
    upstream_changes = git_tracked_changes(upstream)
    upstream_problems = []
    if upstream_head != UPSTREAM_COMMIT:
        upstream_problems.append(f"HEAD must be {UPSTREAM_COMMIT}, got {upstream_head}")
    if upstream_changes:
        upstream_problems.append(
            "tracked worktree changes are present:\n"
            + "\n".join(f"  {change}" for change in upstream_changes)
        )
    if upstream_problems and not args.allow_upstream_mismatch:
        raise RuntimeError(
            "upstream checkout does not match the pinned clean source:\n"
            + "\n".join(upstream_problems)
        )
    if upstream_problems:
        print("WARNING: " + "\n".join(upstream_problems), file=sys.stderr)

    for path, expected_sha in (
        (checkpoint, MODEL_SHA256),
        (codec_path, CODEC_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
    canonical_noise, source_metadata = load_canonical_source_noise(source_fixture)
    source_noise = derive_source_noise(canonical_noise, latent_steps)
    strict_math = configure_strict_fp32_math()

    model_device = torch.device(args.model_device)
    codec_device = torch.device(args.codec_device)
    initial_cuda_device = cuda_device_identity(model_device, codec_device)
    assert_no_autocast(model_device, codec_device)

    sys.path.insert(0, str(upstream))
    import irodori_tts.inference_runtime as runtime_module
    import irodori_tts.rf as rf_module
    from irodori_tts.codec import unpatchify_latent
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        resolve_cfg_scales,
        save_wav,
    )
    from irodori_tts.rf import sample_euler_rf_cfg as official_sampler
    from irodori_tts.text_normalization import normalize_text

    if transformers.__version__ != "5.12.1":
        raise RuntimeError(
            f"transformers must be 5.12.1, got {transformers.__version__}"
        )
    if torch.__version__ != "2.10.0+cu128":
        raise RuntimeError(f"torch must be 2.10.0+cu128, got {torch.__version__}")

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device=str(model_device),
            codec_repo=str(codec_path),
            model_precision="fp32",
            codec_device=str(codec_device),
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    runtime.model = direct_cast_module(
        runtime.model, device=model_device, dtype=target_dtype
    )
    runtime._model_dtype = target_dtype
    runtime.codec.model = direct_cast_module(
        runtime.codec.model, device=codec_device, dtype=target_dtype
    )
    runtime.codec.dtype = target_dtype
    if runtime.model_cfg.latent_patch_size != 1:
        raise RuntimeError(
            "variable-length oracle currently requires released latent_patch_size=1"
        )
    verify_module_dtype("model", runtime.model, target_dtype)
    verify_module_dtype("codec", runtime.codec.model, target_dtype)
    verify_module_device("model", runtime.model, model_device)
    verify_module_device("codec", runtime.codec.model, codec_device)
    runtime.watermarker.model = None
    assert_no_autocast(model_device, codec_device)

    effective_noise = cast_source_noise_once(
        source_noise, device=model_device, dtype=target_dtype
    )
    effective_noise_cpu = cpu_contiguous(effective_noise)

    normalized_text = normalize_text(TEXT).strip()
    text_ids, text_mask = runtime.tokenizer.batch_encode(
        [normalized_text], max_length=runtime.default_text_max_len
    )
    if runtime.caption_tokenizer is None:
        raise RuntimeError("released v4 runtime unexpectedly lacks caption tokenizer")
    caption_ids, caption_mask = runtime.caption_tokenizer.batch_encode(
        [""], max_length=runtime.default_caption_max_len
    )
    caption_mask.zero_()

    effective_text, effective_caption, effective_speaker, scale_messages = (
        resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=CFG_SCALE_TEXT,
            cfg_scale_caption=CFG_SCALE_CAPTION,
            cfg_scale_speaker=CFG_SCALE_SPEAKER,
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=False,
        )
    )

    trace = PrecisionSamplerTrace(
        effective_noise=effective_noise,
        dtype=target_dtype,
        model_device=model_device,
    )

    def traced_sampler(*sampler_args: Any, **sampler_kwargs: Any) -> torch.Tensor:
        return trace.run(
            official_sampler=official_sampler,
            rf_module=rf_module,
            sampler_args=sampler_args,
            sampler_kwargs=sampler_kwargs,
        )

    decoded_inputs: list[torch.Tensor] = []
    decoded_outputs: list[torch.Tensor] = []
    original_decode = runtime.codec.decode_latent

    def traced_decode(latent: torch.Tensor) -> torch.Tensor:
        decoded_inputs.append(latent.detach().clone())
        decoded = original_decode(latent)
        decoded_outputs.append(decoded.detach().clone())
        return decoded

    previous_sampler = runtime_module.sample_euler_rf_cfg
    runtime_module.sample_euler_rf_cfg = traced_sampler
    runtime.codec.decode_latent = traced_decode
    try:
        with torch.inference_mode():
            result = runtime.synthesize(
                SamplingRequest(
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
                ),
                log_fn=print,
            )
    finally:
        runtime.codec.decode_latent = original_decode
        runtime_module.sample_euler_rf_cfg = previous_sampler

    assert_no_autocast(model_device, codec_device)
    if trace.final_patched is None:
        raise RuntimeError("official sampler did not produce a captured final latent")
    if len(decoded_inputs) != 1 or len(decoded_outputs) != 1:
        raise RuntimeError(
            "expected exactly one codec decode, got "
            f"inputs={len(decoded_inputs)} outputs={len(decoded_outputs)}"
        )

    hop_length = int(runtime.codec.model.hop_length)
    runtime_target_samples = seconds_to_samples(SECONDS, runtime.codec.sample_rate)
    if runtime_target_samples != target_samples:
        raise RuntimeError(
            "runtime sample rate changed the target sample count: "
            f"precomputed={target_samples}, runtime={runtime_target_samples}"
        )
    runtime_latent_steps = math.ceil(target_samples / hop_length)
    if runtime_latent_steps != latent_steps:
        raise RuntimeError(
            "runtime hop length changed the latent count: "
            f"precomputed={latent_steps}, runtime={runtime_latent_steps}"
        )
    final_unpatched = unpatchify_latent(
        trace.final_patched,
        patch_size=runtime.model_cfg.latent_patch_size,
        latent_dim=runtime.model_cfg.latent_dim,
    )[:, :latent_steps]
    if not torch.equal(final_unpatched, decoded_inputs[0]):
        difference = float((final_unpatched - decoded_inputs[0]).abs().max().item())
        raise RuntimeError(
            f"captured codec input differs from final latent: {difference}"
        )
    raw_waveform = decoded_outputs[0][0, :, :target_samples]
    raw_waveform_full = decoded_outputs[0][0]
    native_outputs = [
        ("effective initial noise", effective_noise),
        ("final patched latent", trace.final_patched),
        ("final unpatched latent", final_unpatched),
        ("raw decoded waveform", raw_waveform),
    ]
    native_outputs.extend(
        (f"Euler step {index} x_t", value) for index, value in enumerate(trace.step_x)
    )
    native_outputs.extend(
        (f"Euler step {index} velocity", value)
        for index, value in enumerate(trace.step_velocity)
    )
    native_outputs.extend(
        (f"Euler step {index} model output", value)
        for index, value in enumerate(trace.step_model_output)
    )
    native_outputs.extend(
        (f"Euler step {index} timestep", value)
        for index, value in enumerate(trace.step_t)
    )
    native_outputs.extend(
        (f"condition {name}", value)
        for name, value in trace.conditions.items()
        if value.is_floating_point()
    )
    for label, value in native_outputs:
        verify_native_tensor(label, value, target_dtype)
    result_waveform = result.audio.detach().to(device="cpu", dtype=torch.float32)
    if not torch.equal(cpu_contiguous(raw_waveform).float(), result_waveform):
        difference = float(
            (cpu_contiguous(raw_waveform).float() - result_waveform).abs().max().item()
        )
        raise RuntimeError(
            f"captured raw decoder output differs from result audio: {difference}"
        )

    schedule = (1.0 - torch.linspace(0.0, 1.0, NUM_STEPS + 1)) * INIT_SCALE
    x_states = [*trace.step_x, trace.final_patched]
    tensors: dict[str, torch.Tensor] = {
        "inputs/text_input_ids": cpu_contiguous(text_ids),
        "inputs/text_mask": cpu_contiguous(text_mask),
        "inputs/caption_input_ids": cpu_contiguous(caption_ids),
        "inputs/caption_mask": cpu_contiguous(caption_mask),
        "inputs/ref_latent_dummy": torch.zeros(
            [1, 1, runtime.model_cfg.latent_dim], dtype=target_dtype
        ),
        "inputs/ref_mask_dummy": torch.zeros([1, 1], dtype=torch.bool),
        "noise/source_fp32": source_noise,
        "noise/effective": effective_noise_cpu,
        "euler/t_schedule": cpu_contiguous(schedule),
        "euler/x_t": cpu_contiguous(torch.stack(x_states)),
        "euler/velocity": cpu_contiguous(torch.stack(trace.step_velocity)),
        "final_patched_latent": cpu_contiguous(trace.final_patched),
        "final_unpatched_latent": cpu_contiguous(final_unpatched),
        "raw_decoded_waveform": cpu_contiguous(raw_waveform),
        "raw_decoded_waveform_full": cpu_contiguous(raw_waveform_full),
    }
    for name, value in trace.conditions.items():
        tensors[f"conditions/{name}"] = cpu_contiguous(value)
    for index, (x_t, velocity, model_output, t_value) in enumerate(
        zip(
            trace.step_x,
            trace.step_velocity,
            trace.step_model_output,
            trace.step_t,
            strict=True,
        )
    ):
        tensors[f"euler/step_{index}/x_t"] = cpu_contiguous(x_t)
        tensors[f"euler/step_{index}/velocity"] = cpu_contiguous(velocity)
        tensors[f"euler/step_{index}/model_output"] = cpu_contiguous(model_output)
        tensors[f"euler/step_{index}/t"] = cpu_contiguous(t_value)

    if any(
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item())
        for tensor in tensors.values()
    ):
        raise RuntimeError("oracle contains a non-finite floating-point tensor")
    manifest = {key: tensor_summary(value) for key, value in sorted(tensors.items())}
    final_cuda_device = cuda_device_identity(model_device, codec_device)
    if final_cuda_device != initial_cuda_device:
        raise RuntimeError(
            "CUDA device identity changed during oracle export: "
            f"initial={initial_cuda_device}, final={final_cuda_device}"
        )
    requested_parameters = {
        "text": TEXT,
        "caption": None,
        "no_ref": True,
        "seconds": SECONDS,
        "num_steps": NUM_STEPS,
        "seed": SEED,
        "model_precision": precision,
        "codec_precision": precision,
        "cfg_guidance_mode": "independent",
        "cfg_scale_text": CFG_SCALE_TEXT,
        "cfg_scale_caption": CFG_SCALE_CAPTION,
        "cfg_scale_speaker": CFG_SCALE_SPEAKER,
        "cfg_min_t": CFG_MIN_T,
        "cfg_max_t": CFG_MAX_T,
        "t_schedule_mode": "linear",
        "context_kv_cache": True,
        "compile_model": False,
        "trim_tail": False,
        "watermark": False,
    }
    fixture_config = {
        "format_version": 1,
        "model_repo": MODEL_REPO,
        "codec_repo": CODEC_REPO,
        "normalized_text": normalized_text,
        "sample_rate": int(result.sample_rate),
        "hop_length": hop_length,
        "target_samples": target_samples,
        "decoded_samples": int(raw_waveform_full.shape[-1]),
        "latent_steps": latent_steps,
        "patched_steps": int(trace.final_patched.shape[1]),
        "euler_recurrence_max_abs": trace.recurrence_max_abs,
        "effective_cfg": {
            "text": effective_text,
            "caption": effective_caption,
            "speaker": effective_speaker,
        },
        "effective_sampler_kwargs": trace.effective_sampler_kwargs,
        "model_config": dataclasses.asdict(runtime.model_cfg),
        "train_config": runtime.train_cfg,
        "package_versions": package_versions(),
        "cuda_device": initial_cuda_device,
        "cuda_device_identity_verified_before_and_after": True,
    }
    metadata_payload = {
        "format": "irodori-v4-precision-oracle-v1",
        "upstream_commit": upstream_head,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "codec_revision": CODEC_REVISION,
        "codec_sha256": CODEC_SHA256,
        "precision": precision,
        "native_dtype": dtype_name(target_dtype),
        "math_policy": strict_math,
        "noise_contract": {
            "source_fixture_sha256": SOURCE_FIXTURE_SHA256,
            "source_key": "initial_noise",
            "source_dtype": "float32",
            "source_shape": list(NOISE_SHAPE),
            "source_tensor_sha256": sha256_tensor(source_noise),
            "source_derivation": "canonical_tile_alternating_sign_v1",
            "canonical_source_shape": list(CANONICAL_NOISE_SHAPE),
            "effective_key": "noise/effective",
            "effective_dtype": dtype_name(target_dtype),
            "effective_tensor_sha256": sha256_tensor(effective_noise_cpu),
            "cast_count": 1,
            "sampler_randn_interceptions": trace.randn_calls,
        },
        "parameters": requested_parameters,
        "config": fixture_config,
        "source_oracle": {
            "format": source_metadata["format"],
            "upstream_commit": source_metadata["upstream_commit"],
        },
        "tensor_manifest": manifest,
    }
    metadata = {
        "oracle_json": json.dumps(
            metadata_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata=metadata)
    output_sha256 = sha256_file(output)
    if args.manifest_out is not None:
        manifest_out = args.manifest_out.expanduser().resolve()
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        if manifest_out.exists() or manifest_out.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite oracle manifest: {manifest_out}"
            )
        export_manifest = {
            "format": "irodori-v4-precision-oracle-export-manifest-v1",
            "precision": precision,
            "artifact": {
                "path": str(output),
                "sha256": output_sha256,
            },
            "length": {
                "seconds": SECONDS,
                "sample_rate": int(result.sample_rate),
                "hop_length": hop_length,
                "target_samples": target_samples,
                "decoded_samples": int(raw_waveform_full.shape[-1]),
                "latent_steps": latent_steps,
                "patched_steps": int(trace.final_patched.shape[1]),
            },
            "noise": {
                "source_fp32_sha256": manifest["noise/source_fp32"]["sha256"],
                "effective_sha256": manifest["noise/effective"]["sha256"],
                "derivation": "canonical_tile_alternating_sign_v1",
            },
            "outputs": {
                "final_patched_latent_sha256": manifest["final_patched_latent"][
                    "sha256"
                ],
                "target_waveform_sha256": manifest["raw_decoded_waveform"]["sha256"],
                "full_waveform_sha256": manifest["raw_decoded_waveform_full"]["sha256"],
            },
            "tensor_manifest": manifest,
        }
        with manifest_out.open("x", encoding="utf-8") as manifest_file:
            json.dump(
                export_manifest,
                manifest_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            manifest_file.write("\n")
        print(f"manifest: {manifest_out}")
        print(f"manifest_sha256: {sha256_file(manifest_out)}")
    print(f"fixture: {output}")
    print(f"fixture_sha256: {output_sha256}")
    print(
        "noise_contract: "
        f"source_fp32={sha256_tensor(source_noise)} "
        f"effective_{precision}={sha256_tensor(effective_noise_cpu)} cast_count=1"
    )
    print(
        "cuda_device="
        + json.dumps(initial_cuda_device, ensure_ascii=False, sort_keys=True)
    )
    if args.verification_wav is not None:
        verification_wav = args.verification_wav.expanduser().resolve()
        verification_wav.parent.mkdir(parents=True, exist_ok=True)
        save_wav(verification_wav, result_waveform, result.sample_rate)
        print(f"verification_wav: {verification_wav}")
        print(f"verification_wav_sha256: {sha256_file(verification_wav)}")
    for message in scale_messages:
        print(f"cfg_resolution: {message}")
    for key, summary in manifest.items():
        print(f"  {key}: {summary['shape']} {summary['dtype']} {summary['sha256']}")


if __name__ == "__main__":
    main()
