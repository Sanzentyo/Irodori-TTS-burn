# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Re-evaluate projection choices in an isolated, persistent CubeCL cache.

Run under exclusive GPU ownership:
  uv run scripts/retune_projection_cache.py SPEC.json NEW_OUTPUT_DIR

SPEC supplies binary, prepare_binary, precision, seed_cache, environment_file
(relative to seed_cache), and cases [{name, command}]. Commands invoke
validate_synthesis with the same precision.
Only the existing projection namespace is remeasured; compilation and other
tuning decisions are retained through CubeCL's export API. The output receipt
names the selected persistent cache; pass it to production's --cache-dir.
No user cache is replaced. Failed/rejected trials retain their evidence.
"""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path

from benchmark_synthesis_variants import digest, read, save, set_arg


def decision_digest(path, prepare):
    """Inspect the runtime's database with CubeCL, including its Turso WAL."""
    result = subprocess.run(
        [str(prepare), "inspect", "--source", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["sha256"]


def recorded_decisions(path, *, measured):
    """A missing or nonempty measured record cannot prove a warm cache."""
    count = sum(bool(line.strip()) for line in Path(path).read_text().splitlines())
    if measured and count:
        raise ValueError(f"{path}: {count} fresh tuning decisions after seeding")
    return count


def accuracy_failure(report, minimum_snr=85.0, maximum_latent_error=0.001):
    latent = report["latent"]["max_abs"]
    audio = report["audio"]
    if (
        type(latent) not in (int, float)
        or not math.isfinite(latent)
        or not 0 <= latent <= maximum_latent_error
    ):
        return "latent error exceeds the existing strict FP32 gate"
    if (
        type(audio["rmse"]) not in (int, float)
        or not math.isfinite(audio["rmse"])
        or audio["rmse"] < 0
    ):
        return "invalid audio RMSE"
    snr = audio["snr_db"]
    if snr is None:
        if audio["rmse"] != 0:
            return "missing finite audio SNR"
    elif type(snr) not in (int, float) or not math.isfinite(snr) or snr < minimum_snr:
        return "audio SNR misses the existing strict FP32 gate"
    return None


def decide(rows, cases, sessions, minimum_improvement):
    reasons = []
    comparisons = []
    for case in cases:
        medians = {}
        for variant in ["baseline", "candidate"]:
            group = [r for r in rows if r["case"] == case and r["variant"] == variant]
            if len(group) != sessions or {r["session"] for r in group} != set(
                range(sessions)
            ):
                raise ValueError(
                    f"incomplete or duplicate sessions for {case}/{variant}"
                )
            for row in group:
                if (
                    not math.isfinite(row["median_seconds"])
                    or row["median_seconds"] <= 0
                ):
                    raise ValueError(f"invalid duration for {case}/{variant}")
                if row["accuracy_failure"]:
                    reasons.append(f"{case}/{variant}: {row['accuracy_failure']}")
            if len({(r["audio_sha256"], r["latent_sha256"]) for r in group}) != 1:
                reasons.append(f"{case}/{variant}: output changed between sessions")
            # Each fresh process has equal weight; warm samples are correlated.
            medians[variant] = statistics.median(r["median_seconds"] for r in group)
        ratio = medians["candidate"] / medians["baseline"]
        comparisons.append(
            {"case": case, "seconds": medians, "candidate_over_baseline": ratio}
        )
        if ratio > 1.02:
            reasons.append(f"{case}: more than 2% regression")
    ratio = math.exp(
        statistics.mean(math.log(c["candidate_over_baseline"]) for c in comparisons)
    )
    if ratio > 1.0 - minimum_improvement:
        reasons.append("insufficient geometric-mean improvement across workloads")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "comparisons": comparisons,
        "geometric_mean_ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--minimum-improvement", type=float, default=0.02)
    args = parser.parse_args()
    if args.sessions < 3 or args.repeat < 3 or not 0 < args.minimum_improvement < 1:
        parser.error(
            "require at least three sessions and requests, and improvement in (0, 1)"
        )
    spec = read(args.spec)
    binary = Path(spec["binary"]).resolve(strict=True)
    prepare = Path(spec["prepare_binary"]).resolve(strict=True)
    seed = Path(spec["seed_cache"]).resolve(strict=True)
    relative = Path(spec["environment_file"])
    precision = spec["precision"]
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be fp32 or fp16")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("environment_file must stay inside seed_cache")
    source_hash = digest(seed / relative)
    cases = spec["cases"]
    for case in cases:
        command = case["command"]
        if (
            "--precision" not in command
            or command[command.index("--precision") + 1] != precision
        ):
            raise ValueError("every case must specify the campaign precision")
    names = [case["name"] for case in cases]
    if (
        not names
        or len(names) != len(set(names))
        or any(not name or Path(name).name != name for name in names)
    ):
        raise ValueError("case names must be nonempty, unique path components")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "spec.json", spec)
    result = subprocess.run(
        [
            str(prepare),
            "snapshot",
            "--source",
            str(seed / relative),
            "--output",
            str(args.output / "seed"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    caches = {}
    for variant in ["baseline", "candidate"]:
        cache = args.output / f"{variant}-cache"
        shutil.copytree(
            seed, cache, ignore=shutil.ignore_patterns("*.db", "*.db-wal", "*.db-shm")
        )
        (cache / relative).parent.mkdir(parents=True, exist_ok=True)
        # An exported bundle is a transport artifact, not a writable runtime
        # database. Import through CubeCL so the runtime owns its WAL and rows.
        installed = subprocess.run(
            [
                str(prepare),
                "import",
                "--bundle",
                receipt[variant],
                "--cache-dir",
                str((cache / relative).parent),
                "--precision",
                precision,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        installation = json.loads(installed.stdout)
        expected_entries = receipt["retained_entries"] + (
            receipt["projection_entries"] if variant == "baseline" else 0
        )
        if (
            Path(installation["environment"]["environment_path"]).resolve()
            != (cache / relative).resolve()
            or installation["imported"]["imported"] != expected_entries
            or installation["imported"]["skipped"] != 0
        ):
            raise ValueError(f"incomplete or misdirected cache import: {variant}")
        save(args.output / f"{variant}-import.json", installation)
        caches[variant] = cache.resolve()
    rows = []
    binary_hash = digest(binary)
    reports = args.output / "requests"
    reports.mkdir()
    environment = dict(os.environ, **spec.get("environment", {}))
    environment.update(
        CUBECL_WGPU_NATIVE_PIPELINE_CACHE="1",
        CUBECL_WGPU_SPIRV_CACHE="1",
        CUBECL_WGPU_MAX_TASKS="32",
    )

    def run(case, variant, phase, session):
        label = f"{phase}-{session}-{case['name']}-{variant}"
        output = reports / label
        command = list(case["command"])
        command[0] = str(binary)
        for flag in [
            "--device-profile",
            "--dump-stages",
            "--dump-text-layers",
            "--condition-graph",
            "--graph-replay",
            "--stream-graph-entries",
            "--route-profile",
            "--portable",
        ]:
            set_arg(command, flag, None)
        for flag, value in {
            "--output": output,
            "--warmups": 2,
            "--repeat": args.repeat,
            "--stream-repeat": 0,
            "--autotune": True,
            "--autotune-record": reports / f"{label}.tuning.json.log",
        }.items():
            set_arg(command, flag, value)
        env = dict(environment, IRODORI_TTS_BURN_CACHE_DIR=str(caches[variant]))
        save(
            reports / f"{label}.command.json",
            {
                "command": command,
                "cache": str(caches[variant]),
                "binary_sha256": binary_hash,
            },
        )
        print(label, flush=True)
        with (reports / f"{label}.log").open("x") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        report = read(output / "report.json")
        durations = [r["audio_seconds"] for r in report["timings"]]
        if len(durations) != args.repeat or any(
            not math.isfinite(v) or v <= 0 for v in durations
        ):
            raise ValueError(f"invalid measured request durations: {label}")
        record = reports / f"{label}.tuning.json.log"
        fresh_decisions = recorded_decisions(record, measured=phase == "measure")
        failure = accuracy_failure(report)
        # All autotune decisions must stay frozen throughout measured sessions.
        if (
            phase == "measure"
            and decision_digest(caches[variant] / relative, prepare) != frozen[variant]
        ):
            raise ValueError(f"cache changed during measured session: {label}")
        if phase == "measure":
            rows.append(
                {
                    "case": case["name"],
                    "variant": variant,
                    "session": session,
                    "median_seconds": statistics.median(durations),
                    "seconds": durations,
                    "accuracy_failure": failure,
                    "audio": report["audio"],
                    "latent": report["latent"],
                    "audio_sha256": digest(output / "audio.f32le"),
                    "latent_sha256": digest(output / "latent.f32le"),
                    "fresh_decisions": fresh_decisions,
                }
            )
            save(args.output / "measurements.json", rows)

    # Populate every exact workload before freezing either set of decisions.
    for case in cases:
        for variant in ["baseline", "candidate"]:
            run(case, variant, "prepare", 0)
    frozen = {
        variant: decision_digest(cache / relative, prepare)
        for variant, cache in caches.items()
    }
    for session in range(args.sessions):
        for index, case in enumerate(cases):
            order = (
                ["baseline", "candidate"]
                if (session + index) % 2 == 0
                else ["candidate", "baseline"]
            )
            for variant in order:
                run(case, variant, "measure", session)
    if digest(binary) != binary_hash:
        raise ValueError("benchmark binary changed")
    if digest(seed / relative) != source_hash:
        raise ValueError("source environment changed during the isolated campaign")
    decision = decide(rows, names, args.sessions, args.minimum_improvement)
    decision.update(
        binary_sha256=binary_hash,
        tuning_values_sha256=frozen,
        seed=receipt,
        source_environment_sha256=source_hash,
        selected_cache=str(caches["candidate" if decision["accepted"] else "baseline"]),
        gate={"audio_snr_db_min": 85.0, "latent_max_abs": 0.001},
    )
    save(args.output / "result.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
