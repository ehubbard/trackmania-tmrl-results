"""Parser and report smoke tests. Run from learning-story/: python -m tests.test_learning_story"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from generate_report import build_html, lap_embed_html  # noqa: E402
from parse_trainer_log import missing_epochs, parse_trainer_log_file  # noqa: E402


class ParserTests(unittest.TestCase):
    def test_sample_log_keeps_epoch_10(self) -> None:
        rows = parse_trainer_log_file(_HERE / "sample_trainer.log")
        epochs = [r["epoch"] for r in rows]
        self.assertEqual(epochs, [0, 9, 10, 100])
        self.assertEqual(missing_epochs(rows), list(range(1, 9)) + list(range(11, 100)))
        ten = next(r for r in rows if r["epoch"] == 10)
        self.assertEqual(ten["return_train"], 6.0)


class ReportTests(unittest.TestCase):
    def test_html_has_hood_and_gap(self) -> None:
        payload = json.loads(
            (_ROOT / "example" / "SAC_4_LIDAR_train_overnight.json").read_text(
                encoding="utf-8"
            )
        )
        html = build_html(payload)
        self.assertIn("What this experiment is", html)
        self.assertIn("19 LIDAR rays", html)
        self.assertIn("Missing epochs", html)
        self.assertIn("Waiting on a capture pass", html)
        self.assertIn("Then vs now", html)
        self.assertIn("v1 vs this shakedown", html)
        self.assertIn('id="curves"', html)

    def test_lap_embed_includes_progress(self) -> None:
        lap = {
            "x": [0.0, 1.0, 2.0],
            "z": [0.0, 0.5, 1.0],
            "speed": [10, 20, 30],
            "lap_progress": [0.0, 0.5, 1.0],
            "dt": 0.05,
            "duration_s": 0.15,
            "finished": True,
            "track": [[0.0, 0.0], [2.0, 1.0]],
            "note": "unit test lap",
        }
        html = lap_embed_html(lap, video_href="fastest-lap.mp4")
        self.assertIn("Fastest recorded lap", html)
        self.assertIn("lap-progress", html)
        self.assertIn("fastest-lap.mp4", html)
        self.assertIn("finished", html)


if __name__ == "__main__":
    unittest.main()
