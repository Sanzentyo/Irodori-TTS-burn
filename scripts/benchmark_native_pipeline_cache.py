# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Serial, alternating fresh-process cache measurements with exact output checks.

Run: uv run scripts/benchmark_native_pipeline_cache.py SPEC.json NEW_OUTPUT_DIR
The spec records a frozen binary and cases containing name, command (a complete
validate_synthesis argv), seed_cache, reference_output, and cold_driver_env.
The output directory must not exist. Seed caches must be closed, local snapshots.
This command intentionally creates private caches and benchmark outputs there.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def set_arg(command, key, value):
    if key in command:
        command[command.index(key) + 1] = str(value)
    else:
        command.extend([key, str(value)])


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(results):
    groups = {}
    for row in results:
        key = (row["case"], row["driver_cache"], row["mode"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (case, driver, mode), rows in groups.items():
        metrics = {}
        for field in ["process_seconds", "ready_seconds", "first_audio_seconds",
                      "first_request_seconds", "steady_audio_seconds", "max_rss_kib",
                      "pipeline_seconds", "restore_seconds", "save_seconds"]:
            values = [row[field] for row in rows]
            metrics[field] = {"median": statistics.median(values),
                              "p95": percentile(values, 0.95),
                              "min": min(values), "max": max(values)}
        summary.append({"case": case, "driver_cache": driver, "mode": mode,
                        "processes": len(rows), "metrics": metrics})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    spec = read(args.spec)
    binary = Path(spec["binary"]).resolve(strict=True)
    binary_hash = digest(binary)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "spec.json", spec)
    results = []

    def trial(case, cache, driver, mode, pair):
        name = f'{case["name"]}-{driver}-{pair:02}-{mode}'
        output = args.output / name
        command = list(case["command"])
        command[0] = str(binary)
        set_arg(command, "--output", output)
        set_arg(command, "--warmups", 1)
        set_arg(command, "--repeat", 3)
        record = args.output / f"{name}.json.log"
        set_arg(command, "--autotune-record", record)
        environment = {
            "IRODORI_TTS_BURN_CACHE_DIR": str(cache),
            "CUBECL_WGPU_NATIVE_PIPELINE_CACHE": "0" if mode == "off" else "1",
            "RUST_LOG": "warn,cubecl_wgpu::compute::native_pipeline_cache=debug",
            "RUST_BACKTRACE": "1",
            "NO_COLOR": "1",
        }
        if driver == "disabled":
            environment.update(case["cold_driver_env"])
        # The caller must not silently override the controlled cache policy.
        inherited = dict(os.environ)
        for key in case["cold_driver_env"]:
            inherited.pop(key, None)
        rss_path = args.output / f"{name}.rss-kib"
        launch = ["/usr/bin/time", "-f", "%M", "-o", str(rss_path), *command]
        save(args.output / f"{name}-command.json",
             {"command": launch, "environment": environment, "binary_sha256": binary_hash})
        print(name, "start", flush=True)
        started = time.monotonic()
        with (args.output / f"{name}.log").open("x") as log:
            subprocess.run(launch, env=dict(inherited, **environment), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        wall = time.monotonic() - started
        report = read(output / "report.json")
        checkpoints = [dict(re.findall(r"(\w+)=(\w+)", line))
                       for line in (args.output / f"{name}.log").read_text().splitlines()
                       if "native_pipeline_cache checkpoint" in line]
        if not checkpoints:
            raise RuntimeError("native cache metrics missing; check build features")
        metrics = checkpoints[-1]
        if mode == "on" and int(metrics["restored"]) == 0:
            raise RuntimeError("restored trial did not supply any native cache data")
        hashes = {}
        for kind in ["latent", "audio"]:
            hashes[kind] = digest(output / f"{kind}.f32le")
            expected = digest(Path(case["reference_output"]) / f"{kind}.f32le")
            if hashes[kind] != expected:
                raise RuntimeError(f"{name}: {kind} differs from the frozen reference")
        for field in ["streamed_vs_full", "trimmed_vs_full"]:
            if not report[field]["bitwise_equal"]:
                raise RuntimeError(f"{name}: {field} changed")
        fresh = sum(bool(line.strip()) for line in record.read_text().splitlines()) if record.exists() else 0
        if fresh:
            raise RuntimeError(f"{name}: {fresh} fresh autotune decisions confound comparison")
        startup = report["startup"]
        first_request = startup["warmup_timings"][0]["audio_seconds"]
        result = {
            "name": name, "case": case["name"], "driver_cache": driver, "mode": mode,
            "pair": pair, "binary_sha256": binary_hash,
            "process_seconds": wall, "max_rss_kib": int(rss_path.read_text()),
            "ready_seconds": startup["ready_seconds"],
            "first_audio_seconds": startup["ready_seconds"] + first_request,
            "first_request_seconds": first_request,
            "steady_audio_seconds": statistics.median(t["audio_seconds"] for t in report["timings"]),
            "pipeline_seconds": int(metrics["pipeline_ns"]) / 1e9,
            "restore_seconds": int(metrics["restore_ns"]) / 1e9,
            "save_seconds": int(metrics["save_ns"]) / 1e9,
            "startup": startup, "timings": report["timings"], "native_metrics": metrics,
            "all_native_checkpoints": checkpoints, "fresh_decisions": fresh,
            "output_sha256": hashes, "reference_bitwise_equal": True,
            "audio": report["audio"], "latent": report["latent"],
            "database_bytes": sum(path.stat().st_size for path in cache.rglob("*.db")),
            "cache_file_bytes": sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()),
        }
        results.append(result)
        save(args.output / "results.json", results)
        save(args.output / "summary.json", summarize(results))
        print(name, "PASS", f"first request {first_request:.6f}s", flush=True)

    for case in spec["cases"]:
        cache = args.output / f'{case["name"]}-cache'
        shutil.copytree(case["seed_cache"], cache)
        trial(case, cache, "normal", "seed", 0)
        for driver in ["normal", "disabled"]:
            for pair in range(args.pairs):
                for mode in (["off", "on"] if pair % 2 == 0 else ["on", "off"]):
                    trial(case, cache, driver, mode, pair)
    if digest(binary) != binary_hash:
        raise RuntimeError("the supposedly frozen benchmark binary changed")


if __name__ == "__main__":
    main()
