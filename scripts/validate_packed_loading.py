# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay an oracle with both loaders after checking complete parameter digests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/validate_synthesis"))
    parser.add_argument("--devices", nargs="+", default=["discrete:0", "integrated:0"])
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"], default=["fp32", "fp16"])
    args = parser.parse_args()
    cases = json.loads(args.campaign.read_text())["cases"]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for device in args.devices:
        for precision in args.precisions:
            selected = [case for case in cases if case["requested_device"] == device
                        and case["precision"] == precision and case["repeat"] == 0]
            old = next(case for case in selected if case["kind"] == "safetensors")
            new = next(case for case in selected if case["kind"] == "file")
            if not old["parameter_sha256"] or old["parameter_sha256"] != new["parameter_sha256"]:
                raise RuntimeError("all-parameter digest verification is required before rebinding a fixture")
            label = f"{device.replace(':', '-')}-{precision}"
            with args.fixture.open("rb") as source:
                header_len = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(header_len))
                metadata = header["__metadata__"]
                original_sha = digest(Path(old["checkpoint"]))
                if metadata["checkpoint_sha256"] != original_sha:
                    raise RuntimeError("oracle does not refer to the verified original checkpoint")
                # Preserve every oracle tensor byte and record provenance. The
                # validator still checks the exact SHA of the file it actually loads.
                metadata["original_checkpoint_sha256"] = original_sha
                metadata["parameter_sha256"] = old["parameter_sha256"]
                metadata["checkpoint_sha256"] = digest(Path(new["checkpoint"]))
                encoded = json.dumps(header, separators=(",", ":")).encode()
                encoded += b" " * (-len(encoded) % 8)
                fixture = args.output / f"{label}.safetensors"
                with fixture.open("xb") as destination:
                    destination.write(struct.pack("<Q", len(encoded)))
                    destination.write(encoded)
                    shutil.copyfileobj(source, destination)
            outputs = []
            for mode, case, oracle in [("original", old, args.fixture), ("packed", new, fixture)]:
                output = args.output / f"{label}-{mode}"
                command = [str(args.binary), "--checkpoint", case["checkpoint"], "--codec", str(args.codec),
                           "--fixture", str(oracle), "--output", str(output), "--device", device,
                           "--precision", precision, "--autotune", "--warmups", "1", "--repeat", "1"]
                with (args.output / f"{label}-{mode}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                outputs.append(output)
            hashes = {}
            for name in ["latent.f32le", "audio.f32le"]:
                first, second = (digest(output / name) for output in outputs)
                if first != second:
                    raise RuntimeError(f"{label}: {name} differs between loading paths")
                hashes[name] = first
            report = json.loads((outputs[1] / "report.json").read_text())
            result = dict(device=device, precision=precision, bit_identical=True,
                          parameter_sha256=old["parameter_sha256"], outputs=hashes,
                          oracle_latent=report["latent"], oracle_audio=report["audio"])
            results.append(result)
            print(json.dumps(result), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
