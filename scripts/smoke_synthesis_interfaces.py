# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Exercise real resident worker/HTTP inference, cancellation and protocol PCM.

uv run scripts/smoke_synthesis_interfaces.py --binary target/release/irodori \
  --checkpoint MODEL --codec CODEC --output ARTIFACT_DIRECTORY
"""

import argparse
import base64
import json
import queue
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from io import BytesIO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--voice", type=Path, help="Also load and synthesize with a saved voice"
    )
    parser.add_argument(
        "--long-text",
        action="store_true",
        help="Exercise token-limit splitting through the real worker",
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
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
    ]
    with (args.output / "worker.stderr").open("w") as log:
        worker_started = time.monotonic()
        worker = subprocess.Popen(
            command + ["worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        messages = queue.Queue()

        def read():
            for line in worker.stdout:
                try:
                    messages.put(json.loads(line))
                except json.JSONDecodeError as error:
                    messages.put(error)
            messages.put(EOFError("worker stdout closed"))

        threading.Thread(target=read, daemon=True).start()
        records = []

        def receive(timeout=180):
            record = messages.get(timeout=timeout)
            if isinstance(record, Exception):
                raise record
            records.append(
                {key: value for key, value in record.items() if key != "pcm"}
            )
            return record

        def send(value):
            worker.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
            worker.stdin.flush()

        try:
            ready = receive()
            assert ready["type"] == "ready"
            assert ready["model"]["streaming"] and not ready["model"]["incremental"]
            assert ready["model"]["architecture"] == "irodori-tts"
            ready_seconds = time.monotonic() - worker_started
            flow = ready["model"]["flow_parameterization"]
            send(
                {
                    "type": "synthesize",
                    "id": "cancel-me",
                    "text": "音声合成のキャンセルを確認します。",
                    "seconds": 10,
                    "steps": 1000,
                    "seed": 42,
                }
            )
            time.sleep(0.3)
            start = time.monotonic()
            send({"type": "count_tokens", "id": "query", "text": "こんにちは。"})
            while True:
                record = receive(timeout=10)
                if record.get("id") == "query":
                    assert record["type"] == "end" and record["tokens"] > 0
                    break
            query_seconds = time.monotonic() - start
            send({"type": "cancel", "id": "cancel-me"})
            while True:
                record = receive()
                if record.get("id") == "cancel-me" and record["type"] in (
                    "end",
                    "error",
                    "cancelled",
                ):
                    assert record["type"] == "cancelled", record
                    break
            # A rejected unknown option must not poison the next request.
            send(
                {
                    "type": "synthesize",
                    "id": "bad",
                    "text": "test",
                    "unknown_option": True,
                }
            )
            send(
                {
                    "type": "synthesize",
                    "id": "ok",
                    "text": "こんにちは。",
                    "seconds": 2,
                    "seed": 42,
                    "keep_tail": True,
                }
            )
            samples = 0
            seq = 0
            terminals = {}
            while len(terminals) < 2:
                record = receive()
                if record["type"] == "chunk":
                    assert record["id"] == "ok" and record["seq"] == seq
                    pcm = base64.b64decode(record["pcm"], validate=True)
                    assert len(pcm) % 2 == 0
                    samples += len(pcm) // 2
                    seq += 1
                elif record["type"] in ("end", "error", "cancelled"):
                    assert record["id"] not in terminals, record
                    terminals[record["id"]] = record
            assert terminals["bad"]["type"] == "error"
            assert terminals["ok"]["samples"] == samples == 96000
            assert seq >= 2

            def synthesize(identifier, **options):
                send(
                    {
                        "type": "synthesize",
                        "id": identifier,
                        "text": "今日は音声を比較します。",
                        "seconds": 2,
                        "seed": 42,
                        "keep_tail": True,
                        **options,
                    }
                )
                pcm = bytearray()
                sequence = 0
                while True:
                    message = receive()
                    assert message.get("id") == identifier, message
                    if message["type"] == "chunk":
                        assert message["seq"] == sequence
                        sequence += 1
                        pcm.extend(base64.b64decode(message["pcm"], validate=True))
                    else:
                        assert message["type"] == "end", message
                        assert message["samples"] * 2 == len(pcm)
                        break
                (args.output / f"worker-{identifier}.pcm").write_bytes(pcm)
                return pcm

            if args.voice:
                send(
                    {
                        "type": "add_voice",
                        "id": "load-voice",
                        "name": "reference",
                        "path": str(args.voice.resolve()),
                    }
                )
                assert receive() == {"type": "end", "id": "load-voice"}
                send({"type": "info", "id": "voice-info"})
                assert "reference" in [
                    voice["name"] for voice in receive()["model"]["voices"]
                ]
                mask_audio = synthesize(
                    "voice", voice="reference", instructions="落ち着いた声で。"
                )
                assert len(mask_audio) == 192000
                if flow == "rf_velocity":
                    noise_audio = synthesize(
                        "voice-noise",
                        voice="reference",
                        instructions="落ち着いた声で。",
                        speaker_uncond_mode="noise",
                    )
                    assert (
                        len(noise_audio) == len(mask_audio)
                        and noise_audio != mask_audio
                    )
                    controlled = synthesize(
                        "rf-controls",
                        voice="reference",
                        instructions="落ち着いた声で。",
                        steps=4,
                        sway_coeff=-1,
                        cfg_guidance_mode="alternating",
                        rescale_k=0.5,
                        rescale_sigma=0.6,
                        truncation_factor=0.9,
                        speaker_kv_scale=1.2,
                        speaker_kv_max_layers=4,
                    )
                    assert len(controlled) == 192000
            if args.long_text:
                long_text = "明日の東京は晴れです。今日は音声を比較します。" * 32
                send({"type": "count_tokens", "id": "long-count", "text": long_text})
                assert receive()["tokens"] > ready["model"]["max_text_tokens"]
                long_audio = synthesize("long", text=long_text, seconds=0.52, steps=4)
                assert len(long_audio) > 2 * 24960, (
                    "long text must emit more than one segment"
                )
                # Every piece ends a sentence, so the accumulated length must
                # contain exactly 0.52 s per segment plus 0.9 s between pieces.
                segment_samples, pause_samples = 24960, 43200
                segments, remainder = divmod(
                    len(long_audio) // 2 + pause_samples,
                    segment_samples + pause_samples,
                )
                assert remainder == 0 and segments > 1
            worker.stdin.close()
            assert worker.wait(timeout=30) == 0
            (args.output / "worker.json").write_text(
                json.dumps(
                    {
                        "ready_seconds": ready_seconds,
                        "query_seconds_while_busy": query_seconds,
                        "records": records,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=30)
    if args.voice:
        # EOF arrives before initialization finishes. The accepted voice job
        # still has to finish and its GPU work must drain before process exit.
        early_eof = subprocess.run(
            command + ["worker"],
            input=json.dumps(
                {
                    "type": "add_voice",
                    "id": "voice-before-eof",
                    "name": "reference",
                    "path": str(args.voice.resolve()),
                }
            )
            + "\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        (args.output / "worker-voice-eof.json").write_text(
            json.dumps(
                {
                    "exit_code": early_eof.returncode,
                    "stdout": early_eof.stdout,
                    "stderr": early_eof.stderr,
                },
                indent=2,
            )
            + "\n"
        )
        assert early_eof.returncode == 0, early_eof.stderr
        eof_records = [json.loads(line) for line in early_eof.stdout.splitlines()]
        assert [record["type"] for record in eof_records] == ["ready", "end"]
        assert eof_records[-1]["id"] == "voice-before-eof"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with (args.output / "http.stderr").open("w") as log:
        server_started = time.monotonic()
        server_command = command + (
            ["--add-voice", f"reference={args.voice.resolve()}"] if args.voice else []
        )
        server = subprocess.Popen(
            server_command + ["serve", "--listen", f"127.0.0.1:{port}"],
            stdout=log,
            stderr=log,
        )

        def request(path, data=None):
            body = None if data is None else json.dumps(data).encode()
            req = urllib.request.Request(
                base + path, data=body, headers={"Content-Type": "application/json"}
            )
            return urllib.request.urlopen(req, timeout=180)

        try:
            deadline = time.monotonic() + 180
            while True:
                try:
                    with request("/health") as result:
                        assert json.load(result)["status"] == "ok"
                    break
                except urllib.error.URLError:
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            health_seconds = time.monotonic() - server_started
            with request("/v1/models") as result:
                assert json.load(result)["data"][0]["id"] == "irodori"
            data = {
                "model": "irodori",
                "input": "こんにちは。",
                "voice": "reference" if args.voice else "none",
                "instructions": "落ち着いた声で。" if args.voice else "",
                "seed": 42,
                "seconds": 2,
                "keep_tail": True,
            }
            with request("/v1/audio/speech", data) as result:
                wav = result.read()
            with wave.open(BytesIO(wav)) as decoded:
                assert decoded.getframerate() == 48000 and decoded.getnframes() == 96000
                wav_pcm = decoded.readframes(decoded.getnframes())
            with request(
                "/v1/audio/speech", {**data, "response_format": "pcm"}
            ) as result:
                assert result.headers["X-Sample-Rate"] == "48000"
                pcm = result.read()
            assert len(pcm) == 192000
            assert pcm == wav_pcm, (
                "WAV and streamed PCM must reproduce with the same seed"
            )
            for update in [
                {"response_format": "mp3"},
                {"speed": 0},
                {"voice": "missing"},
            ]:
                try:
                    request("/v1/audio/speech", {**data, **update})
                    raise AssertionError("invalid request accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                    assert "error" in json.load(error)
            (args.output / "http.wav").write_bytes(wav)
            (args.output / "http.json").write_text(
                json.dumps(
                    {
                        "health_seconds": health_seconds,
                        "sample_rate": 48000,
                        "samples": 96000,
                        "pcm_matches_wav": True,
                        "rejections": 3,
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            server.terminate()
            server.wait(timeout=30)
    print(args.output)


if __name__ == "__main__":
    main()
