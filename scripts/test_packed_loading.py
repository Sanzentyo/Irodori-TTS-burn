# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Regression checks for checkpoint evidence; no model or GPU is required."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import bench_packed_loading as bench
import validate_packed_loading as validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / "model.safetensors"
        self.packed = self.root / "mf-fp32.wgslpack"
        self.original.write_bytes(b"original weights")
        self.packed.write_bytes(b"native weights A")
        self.binary = self.root / "validator"
        self.binary.write_bytes(b"test executable identity")
        self.fixture = self.root / "oracle.safetensors"
        header = json.dumps({"__metadata__": {"checkpoint_sha256": sha(self.original)}}).encode()
        self.oracle_payload = b"unchanged oracle tensor bytes"
        self.fixture.write_bytes(struct.pack("<Q", len(header)) + header + self.oracle_payload)
        self.campaign = self.root / "campaign.json"
        self.cases = [
            dict(requested_device="discrete:0", precision="fp32", repeat=0, kind=kind,
                 checkpoint=str(path), checkpoint_sha256=sha(path), parameter_sha256="a" * 64)
            for kind, path in [("safetensors", self.original), ("file", self.packed)]
        ]
        self.output = self.root / "output"

    def replay(self):
        self.campaign.write_text(json.dumps(dict(schema=2, cases=self.cases)))
        arguments = ["validate_packed_loading.py", "--campaign", str(self.campaign),
                     "--fixture", str(self.fixture), "--codec", str(self.root / "codec"),
                     "--output", str(self.output), "--binary", str(self.binary),
                     "--devices", "discrete:0", "--precisions", "fp32"]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            validate.main()

    def fake_validator(self, command, **_):
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        for name in ["audio.f32le", "latent.f32le"]:
            (output / name).write_bytes(b"identical tensor")
        (output / "report.json").write_text(json.dumps(dict(latent={}, audio={})))
        return subprocess.CompletedProcess(command, 0)

    def test_replaced_checkpoint_is_rejected_before_oracle_rebinding(self):
        # Same path and length, but the cached all-parameter digest is now stale.
        self.packed.write_bytes(b"native weights B")
        with patch.object(validate.subprocess, "run", side_effect=self.fake_validator) as run:
            with self.assertRaisesRegex(RuntimeError, "checkpoint.*changed"):
                self.replay()
            run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_file_identity_cannot_reuse_a_parameter_digest(self):
        del self.cases[1]["checkpoint_sha256"]
        with patch.object(validate.subprocess, "run", side_effect=self.fake_validator) as run:
            with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256"):
                self.replay()
            run.assert_not_called()

    def test_verified_replay_preserves_oracle_payload_and_provenance(self):
        with patch.object(validate.subprocess, "run", side_effect=self.fake_validator) as run:
            self.replay()
            self.assertEqual(run.call_count, 2)
        rewritten = (self.output / "discrete-0-fp32.safetensors").read_bytes()
        length = struct.unpack("<Q", rewritten[:8])[0]
        metadata = json.loads(rewritten[8:8 + length])["__metadata__"]
        self.assertEqual(metadata["checkpoint_sha256"], sha(self.packed))
        self.assertEqual(metadata["original_checkpoint_sha256"], sha(self.original))
        self.assertEqual(rewritten[8 + length:], self.oracle_payload)

    def benchmark(self, mutate=None, rss=1):
        def run(command, **_):
            path = Path(command[command.index("--checkpoint") + 1])
            case = dict(checkpoint=str(path), precision="fp32", ready_ms=1.0, total_ms=1.0,
                        host_read_ms=0.0, setup_ms=0.0, peak_rss_kib=rss,
                        parameter_sha256="a" * 64 if "--verify" in command else None)
            if mutate is not None:
                mutate.write_bytes(b"changed during measurement")
            return subprocess.CompletedProcess(command, 0, json.dumps(case), "")
        arguments = ["bench_packed_loading.py", "--checkpoint", str(self.original),
                     "--packed-dir", str(self.root), "--output", str(self.output),
                     "--binary", str(self.binary), "--devices", "discrete:0",
                     "--precisions", "fp32", "--modes", "safetensors", "file", "--repeats", "2"]
        with patch.object(sys, "argv", arguments), patch.object(bench.subprocess, "run", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()):
            bench.main()

    def test_benchmark_binds_every_case_to_the_measured_file(self):
        self.benchmark()
        result = json.loads((self.output / "results.json").read_text())
        for case in result["cases"]:
            self.assertEqual(case["checkpoint_sha256"], sha(Path(case["checkpoint"])))
        self.assertEqual(result["binary_sha256"], sha(self.binary))

    def test_input_change_during_measurement_does_not_publish_a_campaign(self):
        with self.assertRaisesRegex(RuntimeError, "changed during"):
            self.benchmark(mutate=self.packed)
        self.assertFalse((self.output / "results.json").exists())

    def test_binary_change_during_measurement_does_not_publish_a_campaign(self):
        with self.assertRaisesRegex(RuntimeError, "changed during"):
            self.benchmark(mutate=self.binary)
        self.assertFalse((self.output / "results.json").exists())

    def test_missing_rss_preserves_timing_results(self):
        self.benchmark(rss=None)
        result = json.loads((self.output / "results.json").read_text())
        for summary in result["summaries"]:
            self.assertEqual(summary["ready_ms"], 1.0)
            self.assertIsNone(summary["peak_rss_kib"])


if __name__ == "__main__":
    unittest.main()
