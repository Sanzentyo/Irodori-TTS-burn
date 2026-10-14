# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import tempfile
import unittest
from pathlib import Path

from retune_projection_cache import accuracy_failure, decide, recorded_decisions


class AcceptanceTests(unittest.TestCase):
    def test_rejects_missing_or_fresh_measured_tuning_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json.log"
            with self.assertRaises(FileNotFoundError):
                recorded_decisions(path, measured=True)
            path.write_text("")
            self.assertEqual(recorded_decisions(path, measured=True), 0)
            path.write_text('{"fastest_index": 3}\n')
            self.assertEqual(recorded_decisions(path, measured=False), 1)
            with self.assertRaises(ValueError):
                recorded_decisions(path, measured=True)

    def rows(self, ratio):
        return [
            {
                "case": "rf",
                "variant": variant,
                "session": session,
                "median_seconds": 1.0 if variant == "baseline" else ratio,
                "accuracy_failure": None,
                "audio_sha256": "audio",
                "latent_sha256": "latent",
            }
            for session in range(3)
            for variant in ["baseline", "candidate"]
        ]

    def test_accepts_measured_improvement_but_rejects_bad_precision(self):
        rows = self.rows(0.87)
        self.assertTrue(decide(rows, ["rf"], 3, 0.02)["accepted"])
        rows[-1]["accuracy_failure"] = "audio precision regression"
        self.assertFalse(decide(rows, ["rf"], 3, 0.02)["accepted"])

    def test_rejects_noise_sized_gain_missing_or_duplicate_sessions(self):
        self.assertFalse(decide(self.rows(0.995), ["rf"], 3, 0.02)["accepted"])
        with self.assertRaises(ValueError):
            decide(self.rows(0.87)[:-1], ["rf"], 3, 0.02)
        rows = self.rows(0.87)
        rows[-1]["session"] = 0
        with self.assertRaises(ValueError):
            decide(rows, ["rf"], 3, 0.02)

    def test_nonfinite_metrics_cannot_pass_accuracy(self):
        report = {
            "latent": {"max_abs": 0.0002},
            "audio": {"snr_db": 94.3, "rmse": 0.00001},
        }
        self.assertIsNone(accuracy_failure(report))
        report["audio"]["snr_db"] = None
        self.assertIsNotNone(accuracy_failure(report))
        report["audio"]["rmse"] = 0
        self.assertIsNone(accuracy_failure(report))
        report["latent"]["max_abs"] = float("nan")
        self.assertIsNotNone(accuracy_failure(report))

    def test_rejects_unstable_outputs_and_invalid_durations(self):
        rows = self.rows(0.8)
        rows[-1]["audio_sha256"] = "different"
        self.assertFalse(decide(rows, ["rf"], 3, 0.02)["accepted"])
        rows[-1]["median_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            decide(rows, ["rf"], 3, 0.02)


if __name__ == "__main__":
    unittest.main()
