# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Measure explicit synthesis variants serially in fresh processes.

Run: uv run scripts/benchmark_synthesis_variants.py SPEC.json NEW_OUTPUT_DIR
The spec contains binary, cases (name, command, seed_cache, reference_output),
and variants (name, arguments, environment, optional binary override).
Arguments map flags to values; true adds a switch and null removes it.
Each variant is seeded before timing,
then run in rotating order. Closed, local tuning snapshots are required.
Set prepare_binary to prepare_projection_retune and each case's environment_file
to export/import CubeCL bundles into a fresh cache for every process. This avoids
copying Turso's process coordination files and freezes logical tuning decisions.
Outputs, logs, commands, private caches and failures are retained. There are no
retries. Every process must reproduce the frozen latent and PCM byte for byte.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def set_arg(command, flag, value):
    if flag in command:
        index = command.index(flag)
        del command[index]
        if index < len(command) and not command[index].startswith("--"):
            del command[index]
    if value is not None and value is not False:
        command.append(flag)
        if value is not True:
            command.append(str(value))


def cache_tool(prepare, *arguments):
    result = subprocess.run(
        [str(prepare), *map(str, arguments)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"cache tool failed: {result.stderr}")
    return json.loads(result.stdout)


def snapshot_cache(prepare, source, output):
    decisions = cache_tool(prepare, "inspect", "--source", source)
    receipt = cache_tool(prepare, "snapshot", "--source", source, "--output", output)
    receipt["decisions"] = decisions
    save(output / "benchmark-snapshot.json", receipt)
    return receipt


def import_cache(prepare, snapshot, cache, relative, precision):
    receipt = cache_tool(
        prepare,
        "import",
        "--bundle",
        snapshot["baseline"],
        "--cache-dir",
        (cache / relative).parent,
        "--precision",
        precision,
    )
    if (
        Path(receipt["environment"]["environment_path"]).resolve()
        != (cache / relative).resolve()
        or receipt["imported"]["imported"]
        != snapshot["projection_entries"] + snapshot["retained_entries"]
        or receipt["imported"]["skipped"] != 0
    ):
        raise ValueError("cache import was incomplete or used a different environment")
    decisions = cache_tool(prepare, "inspect", "--source", cache / relative)
    if decisions != snapshot["decisions"]:
        raise ValueError("cache import changed logical tuning decisions")
    save(cache / "benchmark-import.json", receipt)


def summarize(rows):
    groups = {}
    for row in rows:
        if row["phase"] == "measure":
            groups.setdefault((row["case"], row["variant"]), []).append(row)
    summary = []
    for (case, variant), group in groups.items():
        metrics = {}
        for name in group[0]["metrics"]:
            values = sorted(row["metrics"][name] for row in group)
            position = (len(values) - 1) * 0.95
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            metrics[name] = {
                "median": statistics.median(values),
                "p95": values[lower]
                + (values[upper] - values[lower]) * (position - lower),
                "min": values[0],
                "max": values[-1],
            }
        summary.append(
            {
                "case": case,
                "variant": variant,
                "processes": len(group),
                "metrics": metrics,
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    spec = read(args.spec)
    prepare = (
        Path(spec["prepare_binary"]).resolve(strict=True)
        if "prepare_binary" in spec
        else None
    )
    binaries = {
        Path(variant.get("binary", spec["binary"])).resolve(strict=True)
        for variant in spec["variants"]
    }
    if prepare:
        binaries.add(prepare)
    binary_hashes = {binary: digest(binary) for binary in binaries}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "spec.json", spec)
    save(args.output / "binaries.json", {str(p): h for p, h in binary_hashes.items()})
    rows = []

    def trial(case, variant, cache, phase, index):
        name = f"{case['name']}-{phase}-{index:02}-{variant['name']}"
        output = args.output / name
        binary = Path(variant.get("binary", spec["binary"])).resolve(strict=True)
        command = list(case["command"])
        command[0] = str(binary)
        for flag, value in {
            "--warmups": 1,
            "--repeat": 3,
            **variant.get("arguments", {}),
        }.items():
            set_arg(command, flag, value)
        set_arg(command, "--output", output)
        tuning_record = args.output / f"{name}.autotune.json.log"
        set_arg(command, "--autotune-record", tuning_record)
        environment = {
            "IRODORI_TTS_BURN_CACHE_DIR": str(cache),
            "CUBECL_WGPU_NATIVE_PIPELINE_CACHE": "0",
            "CUBECL_WGPU_SPIRV_CACHE": "0",
            "RUST_LOG": "warn,cubecl_wgpu::compute::native_pipeline_cache=debug",
            "RUST_BACKTRACE": "1",
            "NO_COLOR": "1",
            **case.get("environment", {}),
            **variant.get("environment", {}),
        }
        inherited = dict(os.environ)
        for key in ["__GL_SHADER_DISK_CACHE", "MESA_SHADER_CACHE_DISABLE"]:
            inherited.pop(key, None)
        rss_path = args.output / f"{name}.rss-kib"
        launch = ["/usr/bin/time", "-f", "%M", "-o", str(rss_path), *command]
        save(
            args.output / f"{name}-command.json",
            {
                "command": launch,
                "environment": environment,
                "binary_sha256": binary_hashes[binary],
            },
        )
        print(name, "start", flush=True)
        started = time.monotonic()
        with (args.output / f"{name}.log").open("x") as log:
            result = subprocess.run(
                launch,
                env=dict(inherited, **environment),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        wall = time.monotonic() - started
        if result.returncode:
            save(
                args.output / "failure.json",
                {"name": name, "exit_code": result.returncode},
            )
            raise RuntimeError(
                f"{name}: child failed ({result.returncode}); see retained log"
            )
        report = read(output / "report.json")
        hashes = {
            kind: digest(output / f"{kind}.f32le") for kind in ["latent", "audio"]
        }
        for kind, actual in hashes.items():
            expected = digest(Path(case["reference_output"]) / f"{kind}.f32le")
            if actual != expected:
                raise RuntimeError(f"{name}: {kind} differs from the frozen reference")
        for field in ["streamed_vs_full", "trimmed_vs_full"]:
            if not report[field]["bitwise_equal"]:
                raise RuntimeError(f"{name}: {field} differs")
        fresh = sum(bool(line.strip()) for line in tuning_record.read_text().splitlines())
        if phase == "measure" and fresh:
            raise RuntimeError(
                f"{name}: {fresh} fresh tuning decisions confound timing"
            )
        decisions = None
        if prepare and phase == "measure":
            decisions = cache_tool(
                prepare, "inspect", "--source", cache / case["environment_file"]
            )
            if decisions != snapshot["decisions"]:
                raise ValueError(f"{name}: measured tuning decisions changed")
        startup = report["startup"]
        metrics = {
            "process_seconds": wall,
            "max_rss_kib": int(rss_path.read_text()),
            **{
                key: startup[key]
                for key in [
                    "device_seconds",
                    "model_seconds",
                    "codec_seconds",
                    "ready_seconds",
                ]
            },
            "first_request_seconds": startup["warmup_timings"][0]["audio_seconds"],
            "ready_plus_first_request_seconds": startup["ready_seconds"]
            + startup["warmup_timings"][0]["audio_seconds"],
            "steady_audio_seconds": statistics.median(
                t["audio_seconds"] for t in report["timings"]
            ),
            "steady_sampling_seconds": statistics.median(
                t["rf_seconds"] for t in report["timings"]
            ),
        }
        if "streaming_requests" in report:
            for field in ["first_audio_seconds", "audio_seconds"]:
                metrics[f"streaming_warm_{field}"] = statistics.median(
                    t[field] for t in report["streaming_requests"]
                )
                metrics[f"streaming_cold_{field}"] = report["streaming_warmups"][0][
                    field
                ]
            metrics.update(
                {
                    key: report["streaming_memory"][key]
                    for key in ["device_bytes_in_use", "device_bytes_reserved"]
                }
            )
        checkpoints = [
            dict(re.findall(r"(\w+)=(\w+)", line))
            for line in (args.output / f"{name}.log").read_text().splitlines()
            if "native_pipeline_cache checkpoint" in line
        ]
        row = {
            "name": name,
            "case": case["name"],
            "variant": variant["name"],
            "phase": phase,
            "round": index,
            "metrics": metrics,
            "fresh_decisions": fresh,
            "cache_decisions": decisions,
            "output_sha256": hashes,
            "reference_bitwise_equal": True,
            "native_checkpoints": checkpoints,
            "audio": report["audio"],
            "latent": report["latent"],
            "cache_file_bytes": sum(
                p.stat().st_size for p in cache.rglob("*") if p.is_file()
            ),
        }
        rows.append(row)
        save(args.output / "results.json", rows)
        save(args.output / "summary.json", summarize(rows))
        print(
            name,
            "PASS",
            f"first request {metrics['first_request_seconds']:.6f}s",
            flush=True,
        )

    for case in spec["cases"]:
        cache = args.output / f"{case['name']}-cache"
        if prepare:
            relative = Path(case["environment_file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("environment_file must stay inside seed_cache")
            precision = case["command"][case["command"].index("--precision") + 1]
            snapshot = snapshot_cache(
                prepare,
                Path(case["seed_cache"]) / relative,
                args.output / f"{case['name']}-initial-snapshot",
            )
        else:
            shutil.copytree(case["seed_cache"], cache)
        variants = [
            v
            for v in spec["variants"]
            if v["name"] in case.get("variants", [v["name"] for v in spec["variants"]])
        ]
        if not variants:
            raise ValueError(f"{case['name']}: no variants")
        for index, variant in enumerate(variants):
            if prepare:
                cache = args.output / f"{case['name']}-seed-{index:02}-cache"
                import_cache(prepare, snapshot, cache, relative, precision)
            trial(case, variant, cache, "seed", index)
            if prepare:
                snapshot = snapshot_cache(
                    prepare,
                    cache / relative,
                    args.output / f"{case['name']}-seed-{index:02}-snapshot",
                )
        for index in range(args.rounds):
            offset = index % len(variants)
            for variant in variants[offset:] + variants[:offset]:
                if prepare:
                    cache = args.output / (
                        f"{case['name']}-measure-{index:02}-{variant['name']}-cache"
                    )
                    import_cache(prepare, snapshot, cache, relative, precision)
                trial(case, variant, cache, "measure", index)
    for binary, expected in binary_hashes.items():
        if digest(binary) != expected:
            raise RuntimeError(f"benchmark binary changed: {binary}")


if __name__ == "__main__":
    main()
