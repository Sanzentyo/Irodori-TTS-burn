# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Exercise real predicted duration, speed, stdin text and stdout WAV.

uv run scripts/smoke_synthesis_cli.py --binary target/release/irodori \
  --checkpoint MODEL --codec CODEC --voice VOICE --output NEW_DIRECTORY
"""

import argparse
import json
import subprocess
import wave
from io import BytesIO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.binary.resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--codec",
        str(args.codec),
        "--wgpu-adapter-index",
        "0",
        "--precision",
        args.precision,
        "--add-voice",
        f"reference={args.voice.resolve()}",
        "tts",
        "--voice",
        "reference",
        "--instructions",
        "落ち着いた声で。",
        "--seed",
        "42",
        "--keep-tail",
        "--output",
        "-",
        "--warmup",
        "1",
        "--repeat",
        "2",
    ]
    records = []
    for speed in [1, 2]:
        run = subprocess.run(
            command + ["--speed", str(speed)],
            input="今日は音声を比較します。".encode(),
            capture_output=True,
            check=True,
            timeout=240,
        )
        (args.output / f"speed-{speed}.stderr").write_bytes(run.stderr)
        (args.output / f"speed-{speed}.wav").write_bytes(run.stdout)
        with wave.open(BytesIO(run.stdout)) as audio:
            assert audio.getframerate() == 48000 and audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            samples = audio.getnframes()
            assert 24960 <= samples <= 1440000 and samples % 1920 == 0
        reports = [
            json.loads(line)
            for line in run.stderr.splitlines()
            if line.startswith(b'{"iteration":')
        ]
        assert len(reports) == 2, run.stderr.decode()
        assert all(
            r["report"]["samples"] == samples and r["report"]["segments"] == 1
            for r in reports
        )
        records.append({"speed": speed, "samples": samples, "reports": reports})
    normal, fast = (record["samples"] for record in records)
    if normal > 2 * 24960:
        assert abs(normal - 2 * fast) <= 2 * 1920, (
            "speed must divide the predicted duration"
        )
    else:
        assert fast == 24960, "short predictions must respect the 13-frame floor"
    (args.output / "report.json").write_text(json.dumps(records, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
