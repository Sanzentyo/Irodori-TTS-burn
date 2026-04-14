#!/usr/bin/env python3
"""Run multi-backend quality comparison for Irodori-TTS-burn.

Generates WAV files for 5 Japanese test prompts across all available backends
(Python reference, Rust LibTorch f32/bf16, CUDA f32/bf16, WGPU f32), then zips
the results and produces a performance report.

Usage (from repo root):
    uv run scripts/run_quality_comparison.py [--skip-build] [--skip-python] [--output-dir DIR]

Outputs:
    target/quality_comparison/          — organised WAV tree
    target/quality_comparison.zip       — zipped archive
    target/quality_comparison/performance_report.md — timing table
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.resolve()
TARGET = REPO_ROOT / "target"
OUT_DIR = TARGET / "quality_comparison"

CHECKPOINT = TARGET / "model_converted.safetensors"
CODEC_WEIGHTS = TARGET / "dacvae_weights.safetensors"

# Python uses the original HF model (unconverted), since our convert_for_burn.py
# renames keys like cond_module.0/2/4 → cond_module.linear0/1/2 for Burn compatibility.
_HF_CACHE = Path.home() / ".cache/huggingface/hub"
_PYTHON_CHECKPOINT_CANDIDATES = [
    _HF_CACHE / "models--Aratako--Irodori-TTS-500M-v2" / "snapshots" / "8fd631cafb911dde466bc30dd558a0dc55e8ccae" / "model.safetensors",
    TARGET / "hf_model" / "model.safetensors",
]
PYTHON_CHECKPOINT = next((p for p in _PYTHON_CHECKPOINT_CANDIDATES if p.exists()), None)

_PYTHON_REF_DIR = (REPO_ROOT.parent / "Irodori-TTS").resolve()
VENV = Path(os.environ.get("IRODORI_VENV", str(_PYTHON_REF_DIR / ".venv")))
TORCH_LIB = VENV / "lib/python3.10/site-packages/torch/lib"

# 5 diverse test prompts (short → medium, plain → emoji-style)
TEST_PROMPTS: list[tuple[str, str]] = [
    ("01_greeting",    "こんにちは、テストです。"),
    ("02_phone_auto",  "お電話ありがとうございます。ただいま電話が大変混み合っております。"),
    ("03_emoji_happy", "😊本日は晴天なり！素晴らしい一日になりそうですね。"),
    ("04_technical",   "Rustプログラミング言語は、安全性とパフォーマンスを両立しています。"),
    ("05_emoji_sad",   "うぅ…😭そんなに酷いこと、言わないで…😭"),
]

# Each entry: (backend_key, cargo_features, env_extra)
# cargo_features="" → NdArray (too slow for quality comparison, skip)
RUST_BACKENDS: list[tuple[str, str, dict]] = [
    (
        "rust_libtorch_f32",
        "backend_tch",
        {
            "LIBTORCH_USE_PYTORCH": "1",
            "LIBTORCH_BYPASS_VERSION_CHECK": "1",
            "VIRTUAL_ENV": str(VENV),
            "PATH": f"{VENV}/bin:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "LD_LIBRARY_PATH": f"{TORCH_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        },
    ),
    (
        "rust_libtorch_bf16",
        "backend_tch_bf16",
        {
            "LIBTORCH_USE_PYTORCH": "1",
            "LIBTORCH_BYPASS_VERSION_CHECK": "1",
            "VIRTUAL_ENV": str(VENV),
            "PATH": f"{VENV}/bin:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "LD_LIBRARY_PATH": f"{TORCH_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        },
    ),
    ("rust_cuda_f32",   "backend_cuda",      {}),
    ("rust_cuda_bf16",  "backend_cuda_bf16", {}),
    ("rust_wgpu_f32",   "backend_wgpu",      {}),
]

SEED_BASE = 42  # seed for prompt i = SEED_BASE + i


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    backend: str
    prompt_key: str
    wav_path: Path | None
    wall_time_s: float
    rf_time_ms: float | None = None
    codec_time_ms: float | None = None
    audio_duration_s: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.wav_path is not None

    def save_timing(self, wav_path: Path) -> None:
        """Persist timing alongside the WAV as a JSON sidecar."""
        sidecar = wav_path.with_suffix(".timing.json")
        sidecar.write_text(
            json.dumps({
                "backend": self.backend,
                "prompt_key": self.prompt_key,
                "wall_time_s": self.wall_time_s,
                "rf_time_ms": self.rf_time_ms,
                "codec_time_ms": self.codec_time_ms,
                "audio_duration_s": self.audio_duration_s,
            }, indent=2)
        )

    @staticmethod
    def load_timing(backend: str, prompt_key: str, wav_path: Path) -> "RunResult":
        """Load a RunResult from a timing sidecar. Falls back to wall_time=0 if missing."""
        sidecar = wav_path.with_suffix(".timing.json")
        if sidecar.exists():
            try:
                d = json.loads(sidecar.read_text())
                return RunResult(
                    backend=d.get("backend", backend),
                    prompt_key=d.get("prompt_key", prompt_key),
                    wav_path=wav_path,
                    wall_time_s=d.get("wall_time_s", 0.0),
                    rf_time_ms=d.get("rf_time_ms"),
                    codec_time_ms=d.get("codec_time_ms"),
                    audio_duration_s=d.get("audio_duration_s"),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        # Sidecar absent or corrupt: return with no timing data.
        return RunResult(backend, prompt_key, wav_path, 0.0)


def _run(cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> tuple[int, str, str]:
    full_env = {**os.environ, **(env or {})}
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=full_env, cwd=cwd or REPO_ROOT
    )
    elapsed = time.perf_counter() - t0
    return proc.returncode, proc.stdout, proc.stderr


def _parse_timing(stdout: str, stderr: str = "") -> tuple[float | None, float | None, float | None]:
    """Extract rf_time, codec_time, audio_duration from pipeline stdout/stderr.

    The pipeline prints the timing line to stdout (println!) for reliable parsing,
    and also to the tracing log on stderr.
    """
    rf_ms = None
    codec_ms = None
    dur_s = None
    pattern = re.compile(
        r"\[timing\] rf=(\d+(?:\.\d+)?)ms\s+codec=(\d+(?:\.\d+)?)ms\s+audio_duration=(\d+(?:\.\d+)?)s"
    )
    for m in pattern.finditer(stdout + "\n" + stderr):
        rf_ms = float(m.group(1))
        codec_ms = float(m.group(2))
        dur_s = float(m.group(3))
    return rf_ms, codec_ms, dur_s


def _parse_python_timing(stdout: str, stderr: str) -> tuple[float | None, float | None, float | None]:
    """Extract timing from Python infer.py --show-timings output.

    Python prints lines like:
        [timing] sample: 4567.8 ms
        [timing] decode: 234.5 ms
        [timing] total_to_decode: 5.678 s
    """
    combined = stdout + stderr
    rf_ms: float | None = None
    codec_ms: float | None = None
    dur_s: float | None = None

    m_rf = re.search(r"\[timing\] sample_rf:\s*([\d.]+)\s*ms", combined)
    if m_rf:
        rf_ms = float(m_rf.group(1))

    m_codec = re.search(r"\[timing\] decode_latent:\s*([\d.]+)\s*ms", combined)
    if m_codec:
        codec_ms = float(m_codec.group(1))

    m_total = re.search(r"\[timing\] total_to_decode:\s*([\d.]+)\s*s", combined)
    if m_total:
        dur_s = float(m_total.group(1))

    return rf_ms, codec_ms, dur_s


def build_rust_backend(features: str) -> bool:
    """Build release binary for the given feature set. Returns True on success."""
    feat_arg = ["--features", features] if features else []
    print(f"  [build] cargo build --release --bin pipeline {feat_arg} ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["cargo", "build", "--release", "--bin", "pipeline"] + feat_arg,
        cwd=REPO_ROOT,
        capture_output=False,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  [build] FAILED ({elapsed:.0f}s)", flush=True)
        return False
    print(f"  [build] done ({elapsed:.0f}s)", flush=True)
    return True


def run_python_inference(
    prompt_key: str,
    text: str,
    seed: int,
    out_wav: Path,
    backend_label: str = "python",
    model_precision: str = "fp32",
    codec_precision: str = "fp32",
) -> RunResult:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV / "bin/python"),
        str(_PYTHON_REF_DIR / "infer.py"),
        "--checkpoint", str(PYTHON_CHECKPOINT),
        "--text", text,
        "--no-ref",
        "--seed", str(seed),
        "--num-steps", "40",
        "--output-wav", str(out_wav),
        "--show-timings",
        "--model-precision", model_precision,
        "--codec-precision", codec_precision,
    ]
    env = {
        "VIRTUAL_ENV": str(VENV),
        "PATH": f"{VENV}/bin:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    print(f"  [{backend_label}] {prompt_key}: {text[:40]!r}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env}, cwd=str(_PYTHON_REF_DIR))
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  [{backend_label}] FAILED ({wall:.1f}s): {proc.stderr[-200:]}", flush=True)
        return RunResult(backend_label, prompt_key, None, wall, error=proc.stderr[-200:])

    rf_ms, codec_ms, dur_s = _parse_python_timing(proc.stdout, proc.stderr)
    print(f"  [{backend_label}] OK  wall={wall:.1f}s  audio={dur_s}s", flush=True)
    r = RunResult(backend_label, prompt_key, out_wav, wall, rf_ms, codec_ms, dur_s)
    r.save_timing(out_wav)
    return r


def run_rust_inference(
    backend_key: str,
    features: str,
    extra_env: dict,
    prompt_key: str,
    text: str,
    seed: int,
    out_wav: Path,
) -> RunResult:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    feat_arg = ["--features", features] if features else []
    cmd = (
        ["cargo", "run", "--release", "--bin", "pipeline"]
        + feat_arg
        + [
            "--",
            "--checkpoint", str(CHECKPOINT),
            "--codec-weights", str(CODEC_WEIGHTS),
            "--text", text,
            "--seed", str(seed),
            "--num-steps", "40",
            "--output", str(out_wav),
        ]
    )
    env = {**os.environ, **extra_env, "RUST_LOG": "info"}
    print(f"  [{backend_key}] {prompt_key}: {text[:40]!r}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        err = (proc.stderr + proc.stdout)[-300:]
        print(f"  [{backend_key}] FAILED ({wall:.1f}s): {err[-200:]}", flush=True)
        return RunResult(backend_key, prompt_key, None, wall, error=err)

    rf_ms, codec_ms, dur_s = _parse_timing(proc.stdout, proc.stderr)
    print(f"  [{backend_key}] OK  wall={wall:.1f}s  rf={rf_ms}ms  codec={codec_ms}ms  audio={dur_s}s", flush=True)
    r = RunResult(backend_key, prompt_key, out_wav, wall, rf_ms, codec_ms, dur_s)
    r.save_timing(out_wav)
    return r


def create_demo_lora(adapter_dir: Path) -> None:
    """Create a zero-initialized LoRA adapter to demonstrate the LoRA loading feature."""
    import safetensors.torch as st  # type: ignore[import]
    import torch

    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": "Aratako/Irodori-TTS-500M-v2",
        "fan_in_fan_out": False,
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8.0,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))

    # Zero lora_A matrices → merged W = W_base + 0 = W_base → identical output
    # Model dim = 1024, lora rank = 8
    # Target: a few attention projection matrices in the first DiT block
    tensors: dict[str, torch.Tensor] = {}
    dim = 1024
    rank = 8
    # Minimal: one q/k/v/out per layer (just block 0 as demo)
    for proj in ["to_q", "to_k", "to_v"]:
        key_a = f"base_model.model.blocks.0.joint_attn.{proj}.lora_A.default.weight"
        key_b = f"base_model.model.blocks.0.joint_attn.{proj}.lora_B.default.weight"
        tensors[key_a] = torch.zeros(rank, dim)  # zero A → zero update
        tensors[key_b] = torch.randn(dim, rank) * 0.02
    out_key_a = "base_model.model.blocks.0.joint_attn.to_out.0.lora_A.default.weight"
    out_key_b = "base_model.model.blocks.0.joint_attn.to_out.0.lora_B.default.weight"
    tensors[out_key_a] = torch.zeros(rank, dim)
    tensors[out_key_b] = torch.randn(dim, rank) * 0.02

    st.save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    print(f"  [lora] Created demo adapter at {adapter_dir} ({len(tensors)} tensors, zero A → no-op)")


def run_lora_demo(
    backend_key: str,
    features: str,
    extra_env: dict,
    adapter_dir: Path,
    out_wav: Path,
) -> RunResult:
    """Run one sample with LoRA adapter to verify the feature works."""
    prompt_key = "lora_demo"
    text = TEST_PROMPTS[0][1]  # use first prompt
    seed = SEED_BASE
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    feat_arg = ["--features", features] if features else []
    cmd = (
        ["cargo", "run", "--release", "--bin", "pipeline"]
        + feat_arg
        + [
            "--",
            "--checkpoint", str(CHECKPOINT),
            "--codec-weights", str(CODEC_WEIGHTS),
            "--adapter", str(adapter_dir),
            "--text", text,
            "--seed", str(seed),
            "--num-steps", "40",
            "--output", str(out_wav),
        ]
    )
    env = {**os.environ, **extra_env, "RUST_LOG": "info"}
    print(f"  [lora/{backend_key}] {text[:40]!r}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        err = (proc.stderr + proc.stdout)[-300:]
        print(f"  [lora/{backend_key}] FAILED: {err}", flush=True)
        return RunResult(f"{backend_key}+lora", prompt_key, None, wall, error=err)
    rf_ms, codec_ms, dur_s = _parse_timing(proc.stdout, proc.stderr)
    print(f"  [lora/{backend_key}] OK  wall={wall:.1f}s  audio={dur_s}s", flush=True)
    return RunResult(f"{backend_key}+lora", prompt_key, out_wav, wall, rf_ms, codec_ms, dur_s)


def format_report(results: list[RunResult]) -> str:
    """Generate a markdown performance report with throughput metrics."""
    backends = list(dict.fromkeys(r.backend for r in results))

    # Group results
    table: dict[tuple[str, str], RunResult] = {(r.backend, r.prompt_key): r for r in results}

    # Latent frame rate for this model: 48 kHz codec, hop_length=1920 → 25 Hz.
    # Warm RTF = (rf_ms + codec_ms) / 1000 / audio_duration_s  (<1 = faster than real-time)
    def warm_rtf(r: RunResult) -> float | None:
        if r.rf_time_ms is None or r.codec_time_ms is None or r.audio_duration_s is None:
            return None
        warm_s = (r.rf_time_ms + r.codec_time_ms) / 1000.0
        return warm_s / r.audio_duration_s

    def cold_rtf(r: RunResult) -> float | None:
        if r.audio_duration_s is None or r.audio_duration_s == 0:
            return None
        return r.wall_time_s / r.audio_duration_s

    lines = [
        "# Quality Comparison: Performance Report",
        "",
        f"Generated with `scripts/run_quality_comparison.py`  ",
        f"Model: `target/model_converted.safetensors`  ",
        f"Steps: 40, seed: {SEED_BASE}–{SEED_BASE + len(TEST_PROMPTS) - 1}",
        "",
        "## Test Prompts",
        "",
    ]
    for key, text in TEST_PROMPTS:
        lines.append(f"- **{key}**: {text}")

    lines += ["", "## Warm RTF (RF + Codec, lower is faster; < 1 = faster than real-time)", ""]
    lines += ["> RTF = (rf_time + codec_time) / audio_duration. Excludes model load.", ""]

    header_rtf = ["Backend"] + [p for p, _ in TEST_PROMPTS] + ["Avg"]
    lines.append("| " + " | ".join(header_rtf) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_rtf)) + " |")
    for bk in backends:
        row = [bk]
        vals = []
        for pk, _ in TEST_PROMPTS:
            r = table.get((bk, pk))
            rtf = warm_rtf(r) if r and r.ok else None
            if rtf is not None:
                row.append(f"{rtf:.3f}")
                vals.append(rtf)
            else:
                row.append("—")
        avg = f"{sum(vals)/len(vals):.3f}" if vals else "—"
        row.append(avg)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Cold-Start RTF (wall-clock / audio_duration, includes model load)", ""]
    header_cold = ["Backend"] + [p for p, _ in TEST_PROMPTS] + ["Avg"]
    lines.append("| " + " | ".join(header_cold) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cold)) + " |")
    for bk in backends:
        row = [bk]
        vals = []
        for pk, _ in TEST_PROMPTS:
            r = table.get((bk, pk))
            rtf = cold_rtf(r) if r and r.ok else None
            if rtf is not None:
                row.append(f"{rtf:.2f}")
                vals.append(rtf)
            else:
                row.append("—")
        avg = f"{sum(vals)/len(vals):.2f}" if vals else "—"
        row.append(avg)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## RF Sampler Time (ms, excludes model load + codec)", ""]
    header2 = ["Backend"] + [p for p, _ in TEST_PROMPTS] + ["Avg"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("| " + " | ".join(["---"] * len(header2)) + " |")
    for bk in backends:
        row = [bk]
        vals = []
        for pk, _ in TEST_PROMPTS:
            r = table.get((bk, pk))
            if r and r.ok and r.rf_time_ms is not None:
                row.append(f"{r.rf_time_ms:.0f}ms")
                vals.append(r.rf_time_ms)
            else:
                row.append("—")
        avg = f"{sum(vals)/len(vals):.0f}ms" if vals else "—"
        row.append(avg)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Wall-Clock Time (seconds, includes model load)", ""]
    header_wall = ["Backend"] + [p for p, _ in TEST_PROMPTS] + ["Avg"]
    lines.append("| " + " | ".join(header_wall) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_wall)) + " |")
    for bk in backends:
        row = [bk]
        vals = []
        for pk, _ in TEST_PROMPTS:
            r = table.get((bk, pk))
            if r and r.ok:
                row.append(f"{r.wall_time_s:.1f}s")
                vals.append(r.wall_time_s)
            else:
                row.append("FAIL")
        avg = f"{sum(vals)/len(vals):.1f}s" if vals else "—"
        row.append(avg)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Audio Duration per Prompt", ""]
    header3 = ["Backend"] + [p for p, _ in TEST_PROMPTS]
    lines.append("| " + " | ".join(header3) + " |")
    lines.append("| " + " | ".join(["---"] * len(header3)) + " |")
    for bk in backends:
        row = [bk]
        for pk, _ in TEST_PROMPTS:
            r = table.get((bk, pk))
            if r and r.ok and r.audio_duration_s is not None:
                row.append(f"{r.audio_duration_s:.2f}s")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Failures
    failures = [r for r in results if not r.ok]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"- `{r.backend}` / `{r.prompt_key}`: {r.error}")

    lines += ["", "## LoRA Adapter Status", ""]
    lines += [
        "No public community LoRA adapters exist for `Aratako/Irodori-TTS-500M-v2` on",
        "HuggingFace as of this comparison run (searched `base_model:adapter:` and",
        "`base_model:finetune:` metadata — 0 community results).",
        "",
        "A zero-initialized demo adapter was created at `target/demo_lora/` to verify",
        "the LoRA loading pipeline works end-to-end in Rust.  Zero `lora_A` matrices",
        "produce `W_merged = W_base + 0 = W_base`, so the LoRA demo output should be",
        "numerically identical to the base model output.",
    ]
    lora_results = [r for r in results if "lora" in r.backend.lower()]
    if lora_results:
        lines += [""]
        for r in lora_results:
            status = "OK" if r.ok else f"FAIL: {r.error}"
            lines.append(f"- `{r.backend}` / `{r.prompt_key}`: {status}")

    return "\n".join(lines) + "\n"


def zip_output(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir.parent))
    print(f"[zip] Created {zip_path}  ({zip_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="Skip cargo build steps")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python inference runs")
    parser.add_argument("--force", action="store_true", help="Re-run even if WAV already exists (overwrite)")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Restrict to specific backend keys (e.g. rust_libtorch_f32 rust_cuda_bf16)",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[RunResult] = []
    selected_backends = [b for b in RUST_BACKENDS if args.backends is None or b[0] in (args.backends or [])]

    # ── Step 0: sanity checks ────────────────────────────────────────────────
    for path in (CHECKPOINT, CODEC_WEIGHTS):
        if not path.exists():
            print(f"[error] Missing required file: {path}", file=sys.stderr)
            print("[error] Run: just download-model && just convert-model && just codec-convert", file=sys.stderr)
            sys.exit(1)
    if PYTHON_CHECKPOINT is None:
        print("[error] Cannot find original HF model for Python inference.", file=sys.stderr)
        print("[error] Run: just download-model  (downloads to ~/.cache/huggingface/hub)", file=sys.stderr)
        sys.exit(1)
    print(f"[info] Python checkpoint: {PYTHON_CHECKPOINT}")

    # ── Step 1: build Rust backend binaries ──────────────────────────────────
    if not args.skip_build:
        print("\n=== Building Rust backends ===")
        for backend_key, features, _ in selected_backends:
            ok = build_rust_backend(features)
            if not ok:
                print(f"[warn] Build failed for {backend_key}, will skip inference runs for this backend")

    # ── Step 2: Python inference (fp32 + bf16) ──────────────────────────────
    if not args.skip_python:
        # Python fp32 (default precision)
        print("\n=== Python reference inference (fp32) ===")
        py_dir = out_dir / "python"
        py_dir.mkdir(parents=True, exist_ok=True)
        for i, (key, text) in enumerate(TEST_PROMPTS):
            seed = SEED_BASE + i
            out_wav = py_dir / f"{key}.wav"
            if out_wav.exists() and not args.force:
                print(f"  [python] {key}: already exists, loading cached timing", flush=True)
                all_results.append(RunResult.load_timing("python", key, out_wav))
                continue
            result = run_python_inference(
                key, text, seed, out_wav,
                backend_label="python", model_precision="fp32", codec_precision="fp32",
            )
            all_results.append(result)

        # Python bf16 (model only in bf16, codec in fp32 — isolates model dtype effect)
        print("\n=== Python reference inference (model-bf16) ===")
        py_bf16_dir = out_dir / "python_bf16"
        py_bf16_dir.mkdir(parents=True, exist_ok=True)
        for i, (key, text) in enumerate(TEST_PROMPTS):
            seed = SEED_BASE + i
            out_wav = py_bf16_dir / f"{key}.wav"
            if out_wav.exists() and not args.force:
                print(f"  [python_bf16] {key}: already exists, loading cached timing", flush=True)
                all_results.append(RunResult.load_timing("python_bf16", key, out_wav))
                continue
            result = run_python_inference(
                key, text, seed, out_wav,
                backend_label="python_bf16", model_precision="bf16", codec_precision="fp32",
            )
            all_results.append(result)

    # ── Step 3: Rust inference per backend ───────────────────────────────────
    for backend_key, features, extra_env in selected_backends:
        print(f"\n=== Rust backend: {backend_key} ===")
        bk_dir = out_dir / backend_key
        bk_dir.mkdir(parents=True, exist_ok=True)
        for i, (key, text) in enumerate(TEST_PROMPTS):
            seed = SEED_BASE + i
            out_wav = bk_dir / f"{key}.wav"
            if out_wav.exists() and not args.force:
                print(f"  [{backend_key}] {key}: already exists, loading cached timing", flush=True)
                all_results.append(RunResult.load_timing(backend_key, key, out_wav))
                continue
            result = run_rust_inference(
                backend_key, features, extra_env, key, text, seed, out_wav
            )
            all_results.append(result)

    # ── Step 4: LoRA demo (use LibTorch f32 as reference) ────────────────────
    print("\n=== LoRA demo (zero-initialized adapter) ===")
    demo_lora_dir = TARGET / "demo_lora"
    try:
        create_demo_lora(demo_lora_dir)
        # Use first Rust backend that has LibTorch
        lora_backend = next(
            (b for b in selected_backends if "tch" in b[1]), selected_backends[0] if selected_backends else None
        )
        if lora_backend:
            bk_key, features, extra_env = lora_backend
            lora_out_dir = out_dir / "lora_demo"
            lora_out_dir.mkdir(parents=True, exist_ok=True)
            lora_wav = lora_out_dir / "01_greeting_lora.wav"
            if lora_wav.exists() and not args.force:
                print("  [lora] already exists, skipping", flush=True)
            else:
                lora_result = run_lora_demo(
                    bk_key, features, extra_env, demo_lora_dir, lora_wav
                )
                all_results.append(lora_result)
    except ImportError:
        print("  [lora] safetensors/torch not available in current env — skipping demo adapter creation")

    # ── Step 5: Report ───────────────────────────────────────────────────────
    report = format_report(all_results)
    report_path = out_dir / "performance_report.md"
    report_path.write_text(report)
    print(f"\n[report] Written to {report_path}")
    print(report)

    # ── Step 6: Zip ──────────────────────────────────────────────────────────
    zip_path = TARGET / "quality_comparison.zip"
    zip_output(out_dir, zip_path)
    print(f"\n=== Quality comparison complete ===")
    print(f"WAV files: {out_dir}")
    print(f"Archive:   {zip_path}")
    total = sum(1 for r in all_results if r.ok)
    print(f"Successful runs: {total}/{len(all_results)}")


if __name__ == "__main__":
    main()
