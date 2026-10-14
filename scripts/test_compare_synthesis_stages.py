# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""CPU-only fixture/manifest regression tests; no model or GPU is used."""

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from compare_synthesis_stages import compare


class StageComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reference = Path(self.directory.name) / "reference"
        self.candidate = Path(self.directory.name) / "candidate"
        self.report = {
            "declared_source_checkpoint_sha256": "a" * 64,
            "precision": "fp32",
            "sampler_work": {
                "method": "rf",
                "num_steps": 1,
                "guidance_mode": "joint",
                "schedule_f32_bits": [1065353216, 0],
                "requested": {},
                "enabled_cfg": {},
                "state_precision": "fp32",
            },
            "stage_capture": {
                "schema_version": 2,
                "fixture_sha256": "b" * 64,
                "codec_sha256": "c" * 64,
                "tensors": [],
            },
        }

    def write(self, directory, stages):
        directory.mkdir(exist_ok=True)
        report = copy.deepcopy(self.report)
        for name, values in stages:
            data = struct.pack(f"<{len(values)}f", *values)
            filename = f"{name}.f32le"
            (directory / filename).write_bytes(data)
            report["stage_capture"]["tensors"].append(
                {
                    "name": name,
                    "file": filename,
                    "shape": [1, len(values)],
                    "elements": len(values),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        self.save_report(directory, report)
        return report

    def save_report(self, directory, report):
        (directory / "report.json").write_text(json.dumps(report))

    def pair(self):
        stages = [("input-noise", [1.0, -0.0]), ("latent", [2.0, 3.0])]
        self.write(self.reference, stages)
        return self.write(self.candidate, stages)

    def test_first_observed_difference_and_tolerance_are_separate(self):
        self.write(
            self.reference, [("input-noise", [-0.0, 1.0]), ("latent", [2.0, 3.0])]
        )
        self.write(
            self.candidate, [("input-noise", [0.0, 1.0]), ("latent", [2.0, 3.25])]
        )
        result = compare(self.reference, self.candidate, atol=0.2)
        self.assertEqual(result["first_bitwise_difference"], "input-noise")
        self.assertEqual(result["first_tolerance_failure"], "latent")
        self.assertEqual(result["stages"][1]["max_abs"], 0.25)
        self.assertAlmostEqual(result["stages"][1]["rmse"], (0.0625 / 2) ** 0.5)
        self.assertIsNone(
            compare(self.reference, self.candidate, rtol=0.1)["first_tolerance_failure"]
        )
        self.assertTrue(result["diagnostic_only"])

    def test_identical_inputs_and_different_precisions_are_comparable(self):
        report = self.pair()
        report["precision"] = "fp16"
        report["sampler_work"]["state_precision"] = "fp16"
        self.save_report(self.candidate, report)
        result = compare(self.reference, self.candidate)
        self.assertIsNone(result["first_bitwise_difference"])
        self.assertTrue(all(s["within_tolerance"] for s in result["stages"]))
        self.assertFalse(result["new_gpu_measurements"])

    def test_different_workload_and_stage_contracts_are_rejected(self):
        original = self.pair()
        for change in (
            "source",
            "fixture",
            "codec",
            "steps",
            "order",
            "duplicate",
            "shape",
            "schema",
        ):
            with self.subTest(change=change):
                report = copy.deepcopy(original)
                capture = report["stage_capture"]
                if change == "source":
                    report["declared_source_checkpoint_sha256"] = "c" * 64
                elif change == "fixture":
                    capture["fixture_sha256"] = "d" * 64
                elif change == "codec":
                    capture["codec_sha256"] = "e" * 64
                elif change == "steps":
                    report["sampler_work"]["num_steps"] = 2
                elif change == "order":
                    capture["tensors"].reverse()
                elif change == "duplicate":
                    capture["tensors"][1]["name"] = "input-noise"
                elif change == "shape":
                    capture["tensors"][1]["shape"] = [2, 1]
                else:
                    capture["schema_version"] = 1
                self.save_report(self.candidate, report)
                with self.assertRaises(ValueError):
                    compare(self.reference, self.candidate)

    def test_corruption_truncation_and_path_escape_are_rejected(self):
        for change in ("checksum", "truncation", "path", "elements"):
            with self.subTest(change=change):
                report = self.pair()
                tensor = report["stage_capture"]["tensors"][0]
                if change == "checksum":
                    tensor["sha256"] = "0" * 64
                elif change == "truncation":
                    (self.candidate / tensor["file"]).write_bytes(b"\x00" * 4)
                elif change == "path":
                    tensor["file"] = "../outside.f32le"
                else:
                    tensor["elements"] = 0
                self.save_report(self.candidate, report)
                with self.assertRaises(ValueError):
                    compare(self.reference, self.candidate)

    def test_nonfinite_values_and_invalid_tolerances_are_rejected(self):
        self.write(self.reference, [("latent", [1.0])])
        self.write(self.candidate, [("latent", [float("nan")])])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare(self.reference, self.candidate)
        for tolerance in [-1, float("inf"), float("nan")]:
            with self.assertRaisesRegex(ValueError, "tolerances"):
                compare(self.reference, self.candidate, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
