# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Serial, warm-filesystem, fresh-process comparisons; uv run this file --help."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/bench_checkpoint_load"))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--devices", nargs="+", default=["discrete:0", "integrated:0"])
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"], default=["fp32", "fp16"])
    parser.add_argument("--modes", nargs="+", choices=["safetensors", "file", "borrowed", "shared"],
                        default=["safetensors", "file", "borrowed", "shared"])
    parser.add_argument("--packed-prefix", default="mf")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two samples are required")
    args.checkpoint = args.checkpoint.resolve()
    args.packed_dir = args.packed_dir.resolve()
    args.binary = args.binary.resolve()
    inputs = ([args.checkpoint] if "safetensors" in args.modes else [])
    if any(mode != "safetensors" for mode in args.modes):
        inputs.extend(args.packed_dir / f"{args.packed_prefix}-{precision}.wgslpack"
                      for precision in args.precisions)
    # Hashing also primes the page cache. Pin evidence to these exact files,
    # and reject results if inputs or the executable change during the run.
    fingerprints = {path: digest(path) for path in [args.binary, *inputs]}
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for device in args.devices:
        for precision in args.precisions:
            for repeat in range(args.repeats):
                # Reverse each pair, including the two in-memory alternatives.
                modes = args.modes.copy()
                if repeat % 2:
                    modes.reverse()
                for mode in modes:
                    path = args.checkpoint if mode == "safetensors" else args.packed_dir / f"{args.packed_prefix}-{precision}.wgslpack"
                    label = f"{device.replace(':', '-')}-{precision}-{mode}-{repeat}"
                    command = [str(args.binary), "--checkpoint", str(path), "--device", device,
                               "--precision", precision, "--mode", "file" if mode == "safetensors" else mode]
                    if repeat == 0:
                        command.append("--verify")
                    result = subprocess.run(command, text=True, capture_output=True, check=False)
                    (args.output / f"{label}.log").write_text(result.stderr)
                    if result.returncode:
                        (args.output / f"{label}.json").write_text(result.stdout)
                        raise RuntimeError(f"{label}: exit {result.returncode}; inspect log")
                    case = json.loads(result.stdout)
                    if Path(case["checkpoint"]).resolve() != path:
                        raise RuntimeError(f"{label}: benchmark returned a different checkpoint")
                    case.update(requested_device=device, kind=mode, repeat=repeat,
                                checkpoint_sha256=fingerprints[path])
                    (args.output / f"{label}.json").write_text(json.dumps(case, indent=2) + "\n")
                    cases.append(case)
                    print(f"{label}: ready={case['ready_ms']:.1f}ms total={case['total_ms']:.1f}ms rss={case['peak_rss_kib']}", flush=True)
    summaries = []
    for device in args.devices:
        for precision in args.precisions:
            selected = [case for case in cases if case["requested_device"] == device and case["precision"] == precision]
            digests = {case["parameter_sha256"] for case in selected if case["parameter_sha256"] is not None}
            if len(digests) != 1:
                raise RuntimeError(f"parameter mismatch on {device} {precision}: {digests}")
            for mode in args.modes:
                samples = [case for case in selected if case["kind"] == mode]
                rss = [case["peak_rss_kib"] for case in samples]
                summaries.append(dict(device=device, precision=precision, mode=mode, n=len(samples),
                                      **{key: statistics.median(case[key] for case in samples)
                                         for key in ["ready_ms", "total_ms", "host_read_ms", "setup_ms"]},
                                      peak_rss_kib=statistics.median(rss) if all(value is not None for value in rss) else None,
                                      min_total_ms=min(case["total_ms"] for case in samples),
                                      max_total_ms=max(case["total_ms"] for case in samples)))
    for path, expected in fingerprints.items():
        if digest(path) != expected:
            raise RuntimeError(f"input or binary changed during measurement: {path}")
    result = dict(schema=2, filesystem="warm OS page cache; no privileged cache drop", process="fresh for every sample",
                  binary_sha256=fingerprints[args.binary],
                  cases=cases, summaries=summaries)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
