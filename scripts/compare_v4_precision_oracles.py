# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy==2.2.6",
#   "safetensors==0.7.0",
#   "torch==2.10.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Fail-closed CPU comparison of two Irodori-TTS v4 precision oracles.

Both fixture SHA-256 values are required out of band. The script validates all
metadata pins, the canonical request, every tensor's manifest entry, native
precision dtypes, and the shared fp32-noise contract before calculating any
drift metric. It never imports the Irodori model runtime and refuses to proceed
if CUDA was initialized in the process. Device-identified fixtures must both
record the same single visible CUDA device; mixing legacy and device-identified
fixtures is rejected.

Example::

    uv run scripts/compare_v4_precision_oracles.py \
      --reference /tmp/irodori-v4-fp32-strict-oracle.safetensors \
      --reference-sha256 SHA256 \
      --candidate /tmp/irodori-v4-fp16-strict-oracle.safetensors \
      --candidate-sha256 SHA256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ORACLE_FORMAT = "irodori-v4-precision-oracle-v1"
UPSTREAM_COMMIT = "9f19d9a9048099a4b978a762d0509228fe624e3f"
MODEL_REVISION = "e4aaac4df355ff560dcd35e0dae272c3a759317b"
MODEL_SHA256 = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
CODEC_SHA256 = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"
MODEL_REPO = "Aratako/Irodori-TTS-v4-Small"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
SOURCE_FIXTURE_SHA256 = (
    "8022b2baeed05e68dd2d335bebb10392b5817d1251e006413294ff597d363fc8"
)
TEXT = "こんにちは。"
NUM_STEPS = 4
NOISE_SHAPE = [1, 50, 32]
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DTYPE_NAMES = {
    torch.bool: "bool",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


@dataclass(frozen=True)
class Metrics:
    count: int
    max_abs_error: float
    mean_abs_error: float
    root_mean_square_error: float
    relative_l2_error: float | None
    relative_l2_kind: str
    signal_to_noise_db: float | None
    signal_to_noise_kind: str
    cosine_similarity: float


@dataclass(frozen=True)
class TensorInfo:
    shape: list[int]
    dtype: str
    elements: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Fixture:
    path: Path
    file_sha256: str
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    manifest: dict[str, TensorInfo]

    @property
    def precision(self) -> str:
        return str(self.metadata["precision"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def normalized_sha256(value: str, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be exactly 64 hexadecimal characters")
    return normalized


def exact_json_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            exact_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


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


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return DTYPE_NAMES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported tensor dtype {dtype}") from error


def shape_elements(shape: list[int]) -> int:
    elements = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError(f"invalid tensor dimension {dimension!r}")
        elements *= dimension
    return elements


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def decode_oracle_json(value: str, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            value,
            parse_constant=reject_json_constant,
            object_pairs_hook=unique_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} oracle_json is not strict JSON") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} oracle_json must decode to an object")
    return decoded


def manifest_entry(name: str, raw: Any) -> TensorInfo:
    if not isinstance(raw, dict):
        raise TypeError(f"tensor manifest entry {name!r} must be an object")
    required = {"shape", "dtype", "elements", "bytes", "sha256"}
    if set(raw) != required:
        raise ValueError(
            f"tensor manifest entry {name!r} keys {sorted(raw)}, expected {sorted(required)}"
        )
    shape = raw["shape"]
    if not isinstance(shape, list):
        raise TypeError(f"tensor manifest entry {name!r} shape must be a list")
    dtype = raw["dtype"]
    if type(dtype) is not str or dtype not in set(DTYPE_NAMES.values()):
        raise ValueError(
            f"tensor manifest entry {name!r} has unsupported dtype {dtype!r}"
        )
    elements = raw["elements"]
    byte_count = raw["bytes"]
    if type(elements) is not int or elements < 0:
        raise ValueError(f"tensor manifest entry {name!r} has invalid element count")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"tensor manifest entry {name!r} has invalid byte count")
    if elements != shape_elements(shape):
        raise ValueError(
            f"tensor manifest entry {name!r} element count is inconsistent"
        )
    digest = normalized_sha256(raw["sha256"], f"manifest tensor {name}")
    return TensorInfo(
        shape=shape,
        dtype=str(dtype),
        elements=elements,
        bytes=byte_count,
        sha256=digest,
    )


def expected_parameters(precision: str) -> dict[str, Any]:
    return {
        "text": TEXT,
        "caption": None,
        "no_ref": True,
        "seconds": 2.0,
        "num_steps": 4,
        "seed": 0,
        "model_precision": precision,
        "codec_precision": precision,
        "cfg_guidance_mode": "independent",
        "cfg_scale_text": 3.0,
        "cfg_scale_caption": 3.0,
        "cfg_scale_speaker": 5.0,
        "cfg_min_t": 0.5,
        "cfg_max_t": 1.0,
        "t_schedule_mode": "linear",
        "context_kv_cache": True,
        "compile_model": False,
        "trim_tail": False,
        "watermark": False,
    }


def cuda_device_metadata(metadata: dict[str, Any], label: str) -> dict[str, Any] | None:
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"{label} config must be an object")
    raw = config.get("cuda_device")
    identity_verified = config.get("cuda_device_identity_verified_before_and_after")
    if raw is None:
        if identity_verified is not None:
            raise ValueError(
                f"{label} records CUDA identity verification without device metadata"
            )
        return None
    if identity_verified is not True:
        raise ValueError(
            f"{label} CUDA device identity was not verified before and after export"
        )
    if not isinstance(raw, dict):
        raise TypeError(f"{label} CUDA device metadata must be an object")
    expected_keys = {
        "visible_device_count",
        "device_index_after_visibility",
        "name",
        "pci_bus_id_after_visibility",
        "total_memory_mib",
    }
    if set(raw) != expected_keys:
        raise ValueError(
            f"{label} CUDA device metadata keys {sorted(raw)}, expected {sorted(expected_keys)}"
        )
    if raw["visible_device_count"] != 1:
        raise ValueError(
            f"{label} oracle was not exported with one visible CUDA device"
        )
    if raw["device_index_after_visibility"] != 0:
        raise ValueError(f"{label} sole visible CUDA device must be cuda:0")
    name = raw["name"]
    if type(name) is not str or not name.strip():
        raise ValueError(f"{label} CUDA device name is invalid")
    pci_bus_id = raw["pci_bus_id_after_visibility"]
    if isinstance(pci_bus_id, bool) or not isinstance(pci_bus_id, (int, str)):
        raise TypeError(f"{label} CUDA PCI bus ID is invalid")
    if isinstance(pci_bus_id, str) and not pci_bus_id.strip():
        raise ValueError(f"{label} CUDA PCI bus ID is empty")
    total_memory_mib = raw["total_memory_mib"]
    if (
        isinstance(total_memory_mib, bool)
        or not isinstance(total_memory_mib, (int, float))
        or not math.isfinite(float(total_memory_mib))
        or float(total_memory_mib) <= 0.0
    ):
        raise ValueError(f"{label} CUDA total memory is invalid")
    return dict(raw)


def validate_metadata(metadata: dict[str, Any], label: str) -> str:
    expected_keys = {
        "format",
        "upstream_commit",
        "model_revision",
        "model_sha256",
        "codec_revision",
        "codec_sha256",
        "precision",
        "native_dtype",
        "math_policy",
        "noise_contract",
        "parameters",
        "config",
        "source_oracle",
        "tensor_manifest",
    }
    if set(metadata) != expected_keys:
        raise ValueError(
            f"{label} oracle metadata keys {sorted(metadata)}, expected {sorted(expected_keys)}"
        )
    expected_pins = {
        "format": ORACLE_FORMAT,
        "upstream_commit": UPSTREAM_COMMIT,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "codec_revision": CODEC_REVISION,
        "codec_sha256": CODEC_SHA256,
    }
    for key, expected in expected_pins.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{label} metadata {key!r} mismatch: "
                f"got {metadata.get(key)!r}, expected {expected!r}"
            )
    precision = metadata.get("precision")
    if precision not in PRECISION_DTYPES:
        raise ValueError(f"{label} has unsupported precision {precision!r}")
    native_dtype = dtype_name(PRECISION_DTYPES[str(precision)])
    if metadata.get("native_dtype") != native_dtype:
        raise ValueError(f"{label} native dtype does not match precision {precision}")
    if not exact_json_value(
        metadata.get("parameters"), expected_parameters(str(precision))
    ):
        raise ValueError(f"{label} does not describe the canonical sampling request")

    math_policy = metadata.get("math_policy")
    if not isinstance(math_policy, dict):
        raise TypeError(f"{label} math_policy must be an object")
    strict_math = {
        "autocast": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if set(math_policy) != set(strict_math):
        raise ValueError(f"{label} math policy has unexpected keys")
    for key, expected in strict_math.items():
        if not exact_json_value(math_policy.get(key), expected):
            raise ValueError(f"{label} math policy {key!r} is not strict")

    config = metadata.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"{label} config must be an object")
    expected_config = {
        "format_version": 1,
        "model_repo": MODEL_REPO,
        "codec_repo": CODEC_REPO,
        "sample_rate": 48_000,
        "target_samples": 96_000,
        "latent_steps": 50,
        "patched_steps": 50,
        "euler_recurrence_max_abs": 0.0,
    }
    for key, expected in expected_config.items():
        if not exact_json_value(config.get(key), expected):
            raise ValueError(
                f"{label} config {key!r} mismatch: got {config.get(key)!r}, expected {expected!r}"
            )
    cuda_device_metadata(metadata, label)
    source_oracle = metadata.get("source_oracle")
    expected_source_oracle = {
        "format": "irodori-v4-e2e-oracle-v1",
        "upstream_commit": UPSTREAM_COMMIT,
    }
    if not exact_json_value(source_oracle, expected_source_oracle):
        raise ValueError(f"{label} source oracle pin mismatch")
    return str(precision)


def validate_noise_metadata(
    metadata: dict[str, Any], manifest: dict[str, TensorInfo], label: str
) -> None:
    noise = metadata.get("noise_contract")
    if not isinstance(noise, dict):
        raise TypeError(f"{label} noise_contract must be an object")
    precision = str(metadata["precision"])
    expected = {
        "source_fixture_sha256": SOURCE_FIXTURE_SHA256,
        "source_key": "initial_noise",
        "source_dtype": "float32",
        "source_shape": NOISE_SHAPE,
        "effective_key": "noise/effective",
        "effective_dtype": dtype_name(PRECISION_DTYPES[precision]),
        "cast_count": 1,
        "sampler_randn_interceptions": 1,
    }
    expected_keys = {
        *expected,
        "source_tensor_sha256",
        "effective_tensor_sha256",
    }
    if set(noise) != expected_keys:
        raise ValueError(f"{label} noise contract has unexpected keys")
    for key, value in expected.items():
        if not exact_json_value(noise.get(key), value):
            raise ValueError(
                f"{label} noise contract {key!r} mismatch: "
                f"got {noise.get(key)!r}, expected {value!r}"
            )
    source_hash = normalized_sha256(
        noise.get("source_tensor_sha256"), f"{label} source noise"
    )
    effective_hash = normalized_sha256(
        noise.get("effective_tensor_sha256"), f"{label} effective noise"
    )
    source_manifest = manifest.get("noise/source_fp32")
    effective_manifest = manifest.get("noise/effective")
    if source_manifest is None or effective_manifest is None:
        raise ValueError(f"{label} manifest lacks noise tensors")
    if source_manifest.sha256 != source_hash:
        raise ValueError(f"{label} source-noise contract hash differs from manifest")
    if effective_manifest.sha256 != effective_hash:
        raise ValueError(f"{label} effective-noise contract hash differs from manifest")


def validate_native_dtype(
    name: str, tensor: torch.Tensor, precision: str, label: str
) -> None:
    expected = PRECISION_DTYPES[precision]
    if name in {"noise/source_fp32", "euler/t_schedule"}:
        expected = torch.float32
    elif not tensor.is_floating_point():
        return
    if tensor.dtype != expected:
        raise ValueError(
            f"{label} tensor {name!r} dtype {tensor.dtype}, expected {expected}"
        )


def load_fixture(path: Path, expected_file_sha256: str, label: str) -> Fixture:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = normalized_sha256(expected_file_sha256, f"{label} fixture")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} fixture SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
        )

    with safe_open(str(path), framework="pt", device="cpu") as source:
        header_metadata = source.metadata() or {}
        if set(header_metadata) != {"oracle_json"}:
            raise ValueError(
                f"{label} fixture metadata keys {sorted(header_metadata)}, "
                "expected only 'oracle_json'"
            )
        oracle_json = header_metadata.get("oracle_json")
        if oracle_json is None:
            raise ValueError(f"{label} fixture metadata lacks oracle_json")
        metadata = decode_oracle_json(oracle_json, label)
        precision = validate_metadata(metadata, label)
        raw_manifest = metadata.get("tensor_manifest")
        if not isinstance(raw_manifest, dict):
            raise TypeError(f"{label} tensor_manifest must be an object")
        manifest = {
            name: manifest_entry(name, entry) for name, entry in raw_manifest.items()
        }
        keys = set(source.keys())
        if keys != set(manifest):
            missing = sorted(set(manifest) - keys)
            extra = sorted(keys - set(manifest))
            raise ValueError(
                f"{label} tensor/manifest key mismatch: missing={missing}, extra={extra}"
            )

        tensors: dict[str, torch.Tensor] = {}
        for name in sorted(keys):
            tensor = source.get_tensor(name).detach().clone().contiguous()
            entry = manifest[name]
            actual_info = TensorInfo(
                shape=list(tensor.shape),
                dtype=dtype_name(tensor.dtype),
                elements=int(tensor.numel()),
                bytes=len(tensor_bytes(tensor)),
                sha256=sha256_tensor(tensor),
            )
            if actual_info != entry:
                raise ValueError(
                    f"{label} tensor {name!r} differs from its manifest: "
                    f"actual={actual_info}, manifest={entry}"
                )
            validate_native_dtype(name, tensor, precision, label)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{label} tensor {name!r} contains non-finite values")
            tensors[name] = tensor
    validate_noise_metadata(metadata, manifest, label)
    fixture = Fixture(
        path=path,
        file_sha256=actual_hash,
        metadata=metadata,
        tensors=tensors,
        manifest=manifest,
    )
    validate_internal_consistency(fixture, label)
    return fixture


def require_tensor(
    fixture: Fixture, key: str, shape: list[int], label: str
) -> torch.Tensor:
    try:
        tensor = fixture.tensors[key]
    except KeyError as error:
        raise ValueError(f"{label} fixture lacks required tensor {key!r}") from error
    if list(tensor.shape) != shape:
        raise ValueError(
            f"{label} tensor {key!r} shape {list(tensor.shape)}, expected {shape}"
        )
    return tensor


def tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and torch.equal(left, right)


def expected_model_output_batch(
    fixture: Fixture, schedule: torch.Tensor, step: int, label: str
) -> int:
    """Derive the canonical independent-CFG batch from its recorded timestep."""
    parameters = fixture.metadata["parameters"]
    if parameters["caption"] is not None or parameters["no_ref"] is not True:
        raise ValueError(
            f"{label} canonical fixture must have text as its only active condition"
        )
    if parameters["cfg_guidance_mode"] != "independent":
        raise ValueError(f"{label} canonical fixture must use independent CFG")

    cfg_scale_text = parameters["cfg_scale_text"]
    cfg_min_t = parameters["cfg_min_t"]
    cfg_max_t = parameters["cfg_max_t"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (cfg_scale_text, cfg_min_t, cfg_max_t)
    ):
        raise TypeError(f"{label} canonical CFG metadata must be numeric")
    if not all(
        math.isfinite(float(value)) for value in (cfg_scale_text, cfg_min_t, cfg_max_t)
    ):
        raise ValueError(f"{label} canonical CFG metadata must be finite")
    if float(cfg_min_t) > float(cfg_max_t):
        raise ValueError(f"{label} canonical CFG interval is inverted")

    timestep = float(schedule[step].item())
    use_cfg = float(cfg_scale_text) != 0.0 and (
        float(cfg_min_t) <= timestep <= float(cfg_max_t)
    )
    # The canonical no-reference/no-caption request has one guided condition
    # (text), so independent CFG doubles only the active steps.
    return 2 if use_cfg else 1


def validate_internal_consistency(fixture: Fixture, label: str) -> None:
    text_ids = require_tensor(fixture, "inputs/text_input_ids", [1, 256], label)
    text_mask = require_tensor(fixture, "inputs/text_mask", [1, 256], label)
    caption_ids = require_tensor(fixture, "inputs/caption_input_ids", [1, 512], label)
    caption_mask = require_tensor(fixture, "inputs/caption_mask", [1, 512], label)
    ref_latent = require_tensor(fixture, "inputs/ref_latent_dummy", [1, 1, 32], label)
    ref_mask = require_tensor(fixture, "inputs/ref_mask_dummy", [1, 1], label)
    if text_ids.dtype != torch.int64 or caption_ids.dtype != torch.int64:
        raise ValueError(f"{label} token ID tensors must be int64")
    if text_mask.dtype != torch.bool or caption_mask.dtype != torch.bool:
        raise ValueError(f"{label} token masks must be bool")
    if ref_mask.dtype != torch.bool:
        raise ValueError(f"{label} reference mask must be bool")
    if not bool(text_mask.any()) or bool(caption_mask.any()) or bool(ref_mask.any()):
        raise ValueError(f"{label} canonical input masks are inconsistent")
    if bool(ref_latent.ne(0).any()):
        raise ValueError(f"{label} reference sentinel must contain only zeros")

    source_noise = require_tensor(fixture, "noise/source_fp32", NOISE_SHAPE, label)
    effective_noise = require_tensor(fixture, "noise/effective", NOISE_SHAPE, label)
    final_patched = require_tensor(fixture, "final_patched_latent", [1, 50, 32], label)
    require_tensor(fixture, "final_unpatched_latent", [1, 50, 32], label)
    require_tensor(fixture, "raw_decoded_waveform", [1, 96_000], label)
    schedule = require_tensor(fixture, "euler/t_schedule", [5], label)
    stacked_x = require_tensor(fixture, "euler/x_t", [5, 1, 50, 32], label)
    stacked_velocity = require_tensor(fixture, "euler/velocity", [4, 1, 50, 32], label)
    if source_noise.dtype != torch.float32:
        raise ValueError(f"{label} source noise is not fp32")
    expected_effective = source_noise.to(dtype=PRECISION_DTYPES[fixture.precision])
    if not tensor_equal(expected_effective, effective_noise):
        raise ValueError(
            f"{label} effective noise is not the one-time target cast of source fp32"
        )
    if not tensor_equal(stacked_x[0], effective_noise):
        raise ValueError(f"{label} Euler step 0 does not equal effective initial noise")
    if not tensor_equal(stacked_x[-1], final_patched):
        raise ValueError(f"{label} final Euler state differs from final patched latent")
    expected_schedule = (1.0 - torch.linspace(0.0, 1.0, NUM_STEPS + 1)) * 0.999
    if not tensor_equal(schedule, expected_schedule):
        raise ValueError(
            f"{label} Euler schedule is not the canonical four-step schedule"
        )

    for step in range(NUM_STEPS):
        step_x = require_tensor(fixture, f"euler/step_{step}/x_t", [1, 50, 32], label)
        step_velocity = require_tensor(
            fixture, f"euler/step_{step}/velocity", [1, 50, 32], label
        )
        model_output_batch = expected_model_output_batch(fixture, schedule, step, label)
        require_tensor(
            fixture,
            f"euler/step_{step}/model_output",
            [model_output_batch, 50, 32],
            label,
        )
        step_t = require_tensor(fixture, f"euler/step_{step}/t", [1], label)
        if not tensor_equal(step_x, stacked_x[step]):
            raise ValueError(f"{label} step {step} x_t differs from stacked Euler x_t")
        if not tensor_equal(step_velocity, stacked_velocity[step]):
            raise ValueError(
                f"{label} step {step} velocity differs from stacked Euler velocity"
            )
        expected_t = expected_schedule[step].to(dtype=step_t.dtype).reshape(1)
        if not tensor_equal(step_t, expected_t):
            raise ValueError(f"{label} step {step} timestep differs from schedule")


def validate_pair(reference: Fixture, candidate: Fixture) -> None:
    reference_parameters = dict(reference.metadata["parameters"])
    candidate_parameters = dict(candidate.metadata["parameters"])
    for parameters in (reference_parameters, candidate_parameters):
        parameters.pop("model_precision")
        parameters.pop("codec_precision")
    if reference_parameters != candidate_parameters:
        raise ValueError("reference and candidate sampling requests differ")

    reference_cuda_device = cuda_device_metadata(reference.metadata, "reference")
    candidate_cuda_device = cuda_device_metadata(candidate.metadata, "candidate")
    if (reference_cuda_device is None) != (candidate_cuda_device is None):
        raise ValueError(
            "reference and candidate must both include verified CUDA device metadata"
        )
    if (
        reference_cuda_device is not None
        and reference_cuda_device != candidate_cuda_device
    ):
        raise ValueError(
            "reference and candidate were exported on different CUDA devices: "
            f"reference={reference_cuda_device}, candidate={candidate_cuda_device}"
        )

    reference_noise = reference.metadata["noise_contract"]
    candidate_noise = candidate.metadata["noise_contract"]
    shared_noise_fields = (
        "source_fixture_sha256",
        "source_key",
        "source_dtype",
        "source_shape",
        "source_tensor_sha256",
        "cast_count",
        "sampler_randn_interceptions",
    )
    for field in shared_noise_fields:
        if reference_noise[field] != candidate_noise[field]:
            raise ValueError(f"source-noise contract differs at field {field!r}")
    if not tensor_equal(
        reference.tensors["noise/source_fp32"],
        candidate.tensors["noise/source_fp32"],
    ):
        raise ValueError("reference and candidate source fp32 noise differ")
    for key in (
        "inputs/text_input_ids",
        "inputs/text_mask",
        "inputs/caption_input_ids",
        "inputs/caption_mask",
        "inputs/ref_mask_dummy",
        "euler/t_schedule",
    ):
        if not tensor_equal(reference.tensors[key], candidate.tensors[key]):
            raise ValueError(f"reference and candidate tensor {key!r} differ")

    required_comparisons = comparison_keys()
    for key in required_comparisons:
        reference_tensor = reference.tensors[key]
        candidate_tensor = candidate.tensors[key]
        if list(reference_tensor.shape) != list(candidate_tensor.shape):
            raise ValueError(
                f"comparison tensor {key!r} shape mismatch: "
                f"reference={list(reference_tensor.shape)}, "
                f"candidate={list(candidate_tensor.shape)}"
            )


def comparison_keys() -> list[str]:
    keys = [
        "final_patched_latent",
        "final_unpatched_latent",
        "raw_decoded_waveform",
    ]
    for step in range(NUM_STEPS):
        keys.extend(
            (
                f"euler/step_{step}/x_t",
                f"euler/step_{step}/model_output",
                f"euler/step_{step}/velocity",
            )
        )
    return keys


def decode_float(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"cannot decode non-floating tensor dtype {tensor.dtype}")
    if target_dtype not in {torch.float32, torch.float64}:
        raise ValueError(f"unsupported decoded floating dtype {target_dtype}")
    value = tensor.detach().to(device="cpu", dtype=target_dtype).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("decoded tensor contains non-finite values")
    return value.reshape(-1)


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> Metrics:
    if list(reference.shape) != list(candidate.shape):
        raise ValueError("metric inputs have different shapes")
    # F16/BF16/F32 all have an exact representation in F32. Decode through
    # that common representation, then accumulate metrics in F64.
    expected = decode_float(reference, torch.float32).to(dtype=torch.float64)
    actual = decode_float(candidate, torch.float32).to(dtype=torch.float64)
    if expected.numel() == 0:
        raise ValueError("metric inputs must not be empty")
    difference = actual - expected
    count = int(expected.numel())
    absolute = difference.abs()
    squared_error = torch.dot(difference, difference).item()
    reference_energy = torch.dot(expected, expected).item()
    candidate_energy = torch.dot(actual, actual).item()
    dot_product = torch.dot(expected, actual).item()
    max_abs = absolute.max().item()
    mean_abs = absolute.mean().item()
    rmse = math.sqrt(squared_error / count)
    reference_norm = math.sqrt(reference_energy)
    error_norm = math.sqrt(squared_error)
    if reference_norm == 0.0 and error_norm == 0.0:
        relative_l2 = 0.0
        relative_l2_kind = "zero"
    elif reference_norm == 0.0:
        relative_l2 = None
        relative_l2_kind = "positive_infinity"
    else:
        relative_l2 = error_norm / reference_norm
        relative_l2_kind = "finite"

    if squared_error == 0.0 and reference_energy > 0.0:
        snr_db = None
        snr_kind = "positive_infinity"
    elif squared_error == 0.0 and reference_energy == 0.0:
        snr_db = None
        snr_kind = "undefined_zero_over_zero"
    elif reference_energy == 0.0:
        snr_db = None
        snr_kind = "negative_infinity"
    else:
        snr_db = 10.0 * math.log10(reference_energy / squared_error)
        snr_kind = "finite"

    denominator = math.sqrt(reference_energy * candidate_energy)
    if denominator == 0.0:
        cosine = 1.0 if reference_energy == candidate_energy else 0.0
    else:
        cosine = dot_product / denominator
    finite_values = (max_abs, mean_abs, rmse, cosine)
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError(f"non-finite comparison metric: {finite_values}")
    if relative_l2 is not None and not math.isfinite(relative_l2):
        raise ValueError(f"non-finite finite relative-L2 metric: {relative_l2}")
    if snr_db is not None and not math.isfinite(snr_db):
        raise ValueError(f"non-finite finite-SNR metric: {snr_db}")
    return Metrics(
        count=count,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        root_mean_square_error=rmse,
        relative_l2_error=relative_l2,
        relative_l2_kind=relative_l2_kind,
        signal_to_noise_db=snr_db,
        signal_to_noise_kind=snr_kind,
        cosine_similarity=cosine,
    )


def metric_line(label: str, metrics: Metrics) -> str:
    snr = (
        f"{metrics.signal_to_noise_db:.6f}"
        if metrics.signal_to_noise_db is not None
        else metrics.signal_to_noise_kind
    )
    relative_l2 = (
        f"{metrics.relative_l2_error:.9e}"
        if metrics.relative_l2_error is not None
        else metrics.relative_l2_kind
    )
    return (
        f"{label}: count={metrics.count} max_abs={metrics.max_abs_error:.9e} "
        f"mean_abs={metrics.mean_abs_error:.9e} "
        f"rmse={metrics.root_mean_square_error:.9e} "
        f"relative_l2={relative_l2} "
        f"snr_db={snr} cosine={metrics.cosine_similarity:.12f}"
    )


def worst_step_summary(
    step_metrics: list[dict[str, Metrics]], signal: str
) -> dict[str, Any]:
    selectors = {
        "max_abs_error": max,
        "mean_abs_error": max,
        "root_mean_square_error": max,
        "cosine_similarity": min,
    }
    summary: dict[str, Any] = {}
    for field, selector in selectors.items():
        indexed = [
            (step, getattr(metrics[signal], field))
            for step, metrics in enumerate(step_metrics)
        ]
        selected = selector(indexed, key=lambda item: item[1])
        summary[field] = {"step": selected[0], "value": selected[1]}
    relative_l2 = [
        (step, metrics[signal].relative_l2_error, metrics[signal].relative_l2_kind)
        for step, metrics in enumerate(step_metrics)
    ]
    infinite_relative_l2 = [item for item in relative_l2 if item[1] is None]
    if infinite_relative_l2:
        step, value, kind = infinite_relative_l2[0]
    else:
        step, value, kind = max(relative_l2, key=lambda item: item[1])
    summary["relative_l2_error"] = {"step": step, "value": value, "kind": kind}
    finite_snr = [
        (step, metrics[signal].signal_to_noise_db)
        for step, metrics in enumerate(step_metrics)
        if metrics[signal].signal_to_noise_db is not None
    ]
    summary["signal_to_noise_db"] = (
        None
        if not finite_snr
        else {
            "step": min(finite_snr, key=lambda item: item[1])[0],
            "value": min(value for _, value in finite_snr),
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only comparison")
    reference = load_fixture(args.reference, args.reference_sha256, "reference")
    candidate = load_fixture(args.candidate, args.candidate_sha256, "candidate")
    validate_pair(reference, candidate)

    top_level_metrics = {
        key: compare_tensors(reference.tensors[key], candidate.tensors[key])
        for key in (
            "final_patched_latent",
            "final_unpatched_latent",
            "raw_decoded_waveform",
        )
    }
    step_metrics: list[dict[str, Metrics]] = []
    for step in range(NUM_STEPS):
        metrics = {
            signal: compare_tensors(
                reference.tensors[f"euler/step_{step}/{signal}"],
                candidate.tensors[f"euler/step_{step}/{signal}"],
            )
            for signal in ("x_t", "model_output", "velocity")
        }
        step_metrics.append(metrics)

    source_noise = reference.tensors["noise/source_fp32"]
    noise_metrics = {
        "reference_effective_vs_source_fp32": compare_tensors(
            source_noise, reference.tensors["noise/effective"]
        ),
        "candidate_effective_vs_source_fp32": compare_tensors(
            source_noise, candidate.tensors["noise/effective"]
        ),
        "candidate_effective_vs_reference_effective": compare_tensors(
            reference.tensors["noise/effective"],
            candidate.tensors["noise/effective"],
        ),
    }
    worst_steps = {
        signal: worst_step_summary(step_metrics, signal)
        for signal in ("x_t", "model_output", "velocity")
    }

    for key, metrics in top_level_metrics.items():
        print(metric_line(key, metrics), flush=True)
    for step, metrics_by_signal in enumerate(step_metrics):
        for signal, metrics in metrics_by_signal.items():
            print(metric_line(f"euler/step_{step}/{signal}", metrics), flush=True)
    for key, metrics in noise_metrics.items():
        print(metric_line(f"noise/{key}", metrics), flush=True)

    payload = {
        "format": "irodori-v4-precision-oracle-comparison-v1",
        "cpu_only": True,
        "metric_input_decode_dtype": "float32",
        "metric_accumulation_dtype": "float64",
        "reference": {
            "path": str(reference.path),
            "file_sha256": reference.file_sha256,
            "precision": reference.precision,
            "native_dtype": reference.metadata["native_dtype"],
            "cuda_device": cuda_device_metadata(reference.metadata, "reference"),
        },
        "candidate": {
            "path": str(candidate.path),
            "file_sha256": candidate.file_sha256,
            "precision": candidate.precision,
            "native_dtype": candidate.metadata["native_dtype"],
            "cuda_device": cuda_device_metadata(candidate.metadata, "candidate"),
        },
        "source_noise_sha256": reference.metadata["noise_contract"][
            "source_tensor_sha256"
        ],
        "metrics": {key: asdict(value) for key, value in top_level_metrics.items()},
        "euler_steps": [
            {
                "step": step,
                **{key: asdict(value) for key, value in metrics.items()},
            }
            for step, metrics in enumerate(step_metrics)
        ],
        "euler_worst_steps": worst_steps,
        "effective_noise_quantization": {
            key: asdict(value) for key, value in noise_metrics.items()
        },
    }
    output_path = args.json_out
    if output_path is None:
        output_path = Path(
            f"/tmp/irodori-v4-precision-{reference.precision}-vs-{candidate.precision}.json"
        )
    output_path = output_path.expanduser().resolve()
    if output_path in {reference.path, candidate.path}:
        raise ValueError("JSON output path must not overwrite an input fixture")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was unexpectedly initialized during CPU-only comparison"
        )
    print(f"json_out={output_path}", flush=True)


if __name__ == "__main__":
    main()
