"""Parser and report smoke tests. Run from learning-story/: python -m tests.test_learning_story"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from generate_report import build_html, lap_embed_html, progress_title  # noqa: E402
from parse_trainer_log import missing_epochs, parse_trainer_log_file  # noqa: E402
from progress_report import (  # noqa: E402
    dummy_rows,
    parse_args,
    redact_config,
    run,
    track_slug,
    window_metrics,
)


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

    def test_progress_title_zero_pads_report(self) -> None:
        title = progress_title(3, 24, 412.4)
        self.assertEqual(
            title,
            "TrackMania training progress: Report 03 · Attempts 24 · Episode 412",
        )
        self.assertEqual(
            progress_title(3, 24, 412.4, track="Summer 2026-01"),
            "TrackMania training progress: Summer 2026-01 · Report 03 · Attempts 24 · Episode 412",
        )
        payload = {
            "track": "Summer 2026-01",
            "run_name": "SAC_4_LIDAR_summer2026_01",
            "omit_lap": True,
            "progress": {"report": 1, "attempts": 8, "episode": 90.0},
            "rounds": dummy_rows(0, 4, 0.0),
            "missing_epochs": [],
        }
        html = build_html(payload)
        self.assertIn(
            "TrackMania training progress: Summer 2026-01 · Report 01 · Attempts 8 · Episode",
            html,
        )
        self.assertIn('id="legend"', html)
        self.assertIn("not laps", html)
        self.assertIn("map Summer 2026-01", html)
        self.assertNotIn("map tmrl-test", html)
        self.assertIn("trainer rounds since the last report", html)
        self.assertNotIn("Four phases", html)

    def test_window_metrics(self) -> None:
        rows = dummy_rows(0, 4, 0.0)
        m = window_metrics(rows)
        self.assertGreater(m["episode"], 0)
        self.assertGreater(m["return_mean"], 0)

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


class TrackReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_track_slug(self) -> None:
        self.assertEqual(track_slug("Summer 2026-01"), "summer-2026-01")
        self.assertEqual(track_slug("tmrl-test"), "tmrl-test")
        with self.assertRaises(SystemExit):
            track_slug("  ")

    def test_track_required(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--dry-run", "--run-name", "demo"])

    def test_empty_track_refused(self) -> None:
        args = parse_args(
            ["--dry-run", "--track", "  ", "--run-name", "demo", "--output", str(self.tmp)]
        )
        with self.assertRaises(SystemExit):
            run(args)

    def test_redact_config(self) -> None:
        redacted = redact_config({"WANDB_KEY": "secret", "RUN_NAME": "x", "PASSWORD": "p"})
        self.assertEqual(redacted["WANDB_KEY"], "<redacted>")
        self.assertEqual(redacted["PASSWORD"], "<redacted>")
        self.assertEqual(redacted["RUN_NAME"], "x")

    def test_layout_and_legend(self) -> None:
        args = parse_args(
            [
                "--dry-run",
                "--track",
                "Summer 2026-01",
                "--run-name",
                "demo",
                "--output",
                str(self.tmp),
                "--new-run",
            ]
        )
        out = run(args)
        self.assertEqual(out, self.tmp / "summer-2026-01" / "demo")
        self.assertTrue((out / "report-01.html").exists())
        self.assertTrue((out / "state.json").exists())
        self.assertTrue((out / "manifest.json").exists())
        self.assertTrue((out / "slices" / "report-01.json").exists())
        html = (out / "report-01.html").read_text(encoding="utf-8")
        self.assertIn("Summer 2026-01", html)
        self.assertIn('id="legend"', html)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["track"], "Summer 2026-01")
        self.assertEqual(manifest["run_name"], "demo")
        self.assertTrue(manifest["from_scratch"])

    def test_two_tracks_isolated(self) -> None:
        run(
            parse_args(
                [
                    "--dry-run",
                    "--track",
                    "tmrl-test",
                    "--run-name",
                    "same-name",
                    "--output",
                    str(self.tmp),
                    "--new-run",
                ]
            )
        )
        run(
            parse_args(
                [
                    "--dry-run",
                    "--track",
                    "Summer 2026-01",
                    "--run-name",
                    "same-name",
                    "--output",
                    str(self.tmp),
                    "--new-run",
                ]
            )
        )
        a = self.tmp / "tmrl-test" / "same-name" / "state.json"
        b = self.tmp / "summer-2026-01" / "same-name" / "state.json"
        self.assertTrue(a.exists())
        self.assertTrue(b.exists())
        self.assertNotEqual(a, b)

    def test_data_only_does_not_advance_watermark(self) -> None:
        common = [
            "--dry-run",
            "--track",
            "Summer 2026-01",
            "--run-name",
            "demo",
            "--output",
            str(self.tmp),
        ]
        run(parse_args([*common, "--new-run"]))
        state = json.loads(
            (self.tmp / "summer-2026-01" / "demo" / "state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state["last_report"], 1)
        run(parse_args([*common, "--data-only"]))
        state = json.loads(
            (self.tmp / "summer-2026-01" / "demo" / "state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state["last_report"], 1)
        self.assertFalse((self.tmp / "summer-2026-01" / "demo" / "report-02.html").exists())
        run(parse_args(common))
        self.assertTrue((self.tmp / "summer-2026-01" / "demo" / "report-02.html").exists())


if __name__ == "__main__":
    unittest.main()
