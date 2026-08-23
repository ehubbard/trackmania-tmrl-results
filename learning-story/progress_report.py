"""On-demand progress reports from a continuing tmrl train.

You ask; we slice trainer rounds collected since the last report.
Restarting from scratch is --new-run (report 01, empty watermark).
Does not start or stop tmrl. Does not emit a page on a timer.

A report is a view over saved run data. HTML can be rebuilt; the trainer
log, reward pickle, and config snapshot cannot. Layout:

    reports/<track-slug>/<run-name>/
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from generate_report import build_html, progress_title
from parse_trainer_log import parse_trainer_log_file

STATE_NAME = "state.json"
MANIFEST_NAME = "manifest.json"
TMRL_DATA = Path.home() / "TmrlData"
SECRET_KEYS = {"WANDB_KEY", "PASSWORD", "WANDB_API_KEY"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def track_slug(name: str) -> str:
    text = (name or "").strip()
    if not text:
        raise SystemExit(
            "--track is required (map name, e.g. 'Summer 2026-01' or 'tmrl-test')"
        )
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        raise SystemExit("--track must contain letters or numbers")
    return slug


def run_dir(output: Path, track: str, run_name: str) -> Path:
    return output / track_slug(track) / run_name


def env_label(interface: str) -> str:
    upper = (interface or "").upper()
    if "FULL" in upper:
        return "FULL"
    if "LIDAR" in upper:
        return "LIDAR"
    return (interface or "unknown").strip() or "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if str(key).upper() in SECRET_KEYS:
                out[key] = "<redacted>"
            else:
                out[key] = redact_config(value)
        return out
    if isinstance(obj, list):
        return [redact_config(item) for item in obj]
    return obj


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return list(data.get("rounds") or [])
    return parse_trainer_log_file(path)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def window_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "episode": mean(rows, "episode_length_train"),
        "return_mean": mean(rows, "return_train"),
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_report": 0,
            "consumed_rounds": 0,
            "episode": None,
            "return_mean": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def dummy_rows(start: int, n: int, bump: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "epoch": start + i,
                "round": 0,
                "return_train": 3.0 + bump + i * 0.8,
                "episode_length_train": 90.0 + bump * 20 + i * 12,
                "memory_len": 1000 + (start + i) * 200,
            }
        )
    return rows


def snapshot_config(src: Path, dest: Path) -> dict[str, Any]:
    data = json.loads(src.read_text(encoding="utf-8"))
    redacted = redact_config(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(redacted, indent=2) + "\n", encoding="utf-8")
    return redacted


def archive_run(
    dest: Path,
    *,
    track: str,
    run_name: str,
    input_path: Path | None,
    config_src: Path | None,
    reward_src: Path | None,
    dry_run: bool,
    new_run: bool,
    from_scratch: bool,
    env: str,
    interface: str,
    algorithm: str = "SAC",
) -> dict[str, Any]:
    """Copy log / config / reward into the run folder. HTML is optional."""
    dest.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    warnings: list[str] = []
    slug = track_slug(track)
    reward_hash: str | None = None

    if input_path and input_path.exists():
        name = "metrics.json" if input_path.suffix.lower() == ".json" else "trainer.log"
        copy_file(input_path, dest / name)
        files["log"] = name

    snapshot: dict[str, Any] | None = None
    if config_src and config_src.exists():
        snapshot = snapshot_config(config_src, dest / "config.snapshot.json")
        files["config"] = "config.snapshot.json"
        nested = snapshot.get("ENV") if isinstance(snapshot.get("ENV"), dict) else {}
        if not interface:
            interface = str(
                nested.get("RTGYM_INTERFACE")
                or snapshot.get("RTGYM_INTERFACE")
                or ""
            )
        if not env:
            env = env_label(interface)

    dest_reward = dest / "reward.pkl"
    if reward_src and reward_src.exists():
        if new_run or not dest_reward.exists():
            copy_file(reward_src, dest_reward)
        elif file_sha256(reward_src) != file_sha256(dest_reward):
            warnings.append(
                "active reward.pkl does not match the pickle archived for this "
                "run; you may be mixing maps"
            )
        files["reward"] = "reward.pkl"
        reward_hash = file_sha256(dest_reward)
        if not dry_run:
            named = TMRL_DATA / "reward" / "tracks" / f"{slug}.pkl"
            if new_run or not named.exists():
                copy_file(dest_reward, named)
            elif file_sha256(named) != reward_hash:
                warnings.append(
                    f"{named} differs from this run's archived reward.pkl"
                )
    elif not dry_run and not dest_reward.exists():
        raise SystemExit(
            "no reward.pkl to archive for this track. Record a new reward on "
            "this map (tmrl overwrites TmrlData/reward/reward.pkl), then re-run. "
            "Pass --reward if the pickle lives somewhere else."
        )

    if not env:
        env = env_label(interface or "TM20LIDAR")
    if not interface:
        interface = "TM20LIDAR"

    manifest_path = dest / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"created_at": utc_now()}
    manifest.update(
        {
            "track": track.strip(),
            "track_slug": slug,
            "run_name": run_name,
            "env": env,
            "interface": interface,
            "algorithm": algorithm,
            "from_scratch": from_scratch,
            "updated_at": utc_now(),
            "files": {**dict(manifest.get("files") or {}), **files},
            "reward_sha256": reward_hash or manifest.get("reward_sha256"),
            "tmrl_paths": {
                "data": str(TMRL_DATA),
                "reward_active": str(TMRL_DATA / "reward" / "reward.pkl"),
                "reward_named": str(TMRL_DATA / "reward" / "tracks" / f"{slug}.pkl"),
                "config": str(TMRL_DATA / "config" / "config.json"),
                "weights": str(TMRL_DATA / "weights" / f"{run_name}.tmod"),
                "checkpoint": str(TMRL_DATA / "checkpoints" / f"{run_name}_t.tcpt"),
            },
            "note": (
                "HTML is a view. Rebuild from trainer.log (or metrics.json) "
                f"and {STATE_NAME} consumed_rounds. Do not mix maps in one "
                "RUN_NAME, reward.pkl, or replay buffer."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return manifest


def write_slice(dest: Path, report_n: int, payload: dict[str, Any]) -> Path:
    folder = dest / "slices"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"report-{report_n:02d}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_index(
    out: Path,
    track: str,
    run_name: str,
    reports: list[dict[str, Any]],
) -> None:
    items = []
    for rec in reports:
        items.append(
            "<tr>"
            f'<td><a href="{html_lib.escape(rec["html"])}">{html_lib.escape(rec["title"])}</a></td>'
            f"<td>{rec['attempts']}</td>"
            f"<td>{int(round(rec['episode']))}</td>"
            "</tr>"
        )
    track_html = html_lib.escape(track)
    run_html = html_lib.escape(run_name)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>TrackMania training progress — {track_html} — {run_html}</title>
  <style>
    body {{ margin: 0; font: 16px/1.5 Georgia, serif; color: #1c1917; background: #fafaf9; }}
    main {{ max-width: 920px; margin: 0 auto; padding: 48px 24px; }}
    table {{ width: 100%; border-collapse: collapse; font-family: ui-sans-serif, system-ui, sans-serif; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e7e5e4; }}
    a {{ color: #c2410c; }}
    .caption {{ color: #78716c; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <main>
    <h1>TrackMania training progress</h1>
    <p>{track_html} · {run_html}. Each page is a slice since the previous report. Same train unless you pass --new-run.</p>
    <p class="caption">Attempts = trainer rounds in that slice. Episode = mean episode_length_train (steps); 1000 is the cap, usually a finish. Not official lap times.</p>
    <table>
      <thead><tr><th>Report</th><th>Attempts</th><th>Episode</th></tr></thead>
      <tbody>
        {''.join(items) if items else '<tr><td colspan="3">No reports yet.</td></tr>'}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")


def emit_report(
    *,
    out: Path,
    track: str,
    run_name: str,
    label: str,
    interface: str,
    env: str,
    from_scratch: bool,
    slice_rows: list[dict[str, Any]],
    report_n: int,
    consumed: int,
    prev_episode: float | None,
    prev_return: float | None,
) -> dict[str, Any]:
    metrics = window_metrics(slice_rows)
    progress = {
        "report": report_n,
        "attempts": len(slice_rows),
        "episode": metrics["episode"],
        "episode_prev": prev_episode,
        "return_mean": metrics["return_mean"],
        "return_mean_prev": prev_return,
        "consumed_rounds": consumed,
    }
    payload: dict[str, Any] = {
        "track": track,
        "run_name": run_name,
        "label": label,
        "interface": interface or "TM20LIDAR",
        "env": env,
        "algorithm": "SAC",
        "from_scratch": from_scratch,
        "omit_lap": True,
        "progress": progress,
        "rounds": slice_rows,
        "missing_epochs": [],
    }
    write_slice(out, report_n, payload)
    html = build_html(payload)
    name = f"report-{report_n:02d}.html"
    (out / name).write_text(html, encoding="utf-8")
    rec = {
        "report": report_n,
        "attempts": len(slice_rows),
        "episode": metrics["episode"],
        "return_mean": metrics["return_mean"],
        "html": name,
        "title": progress_title(report_n, len(slice_rows), metrics["episode"], track=track)
        if not label
        else f"TrackMania training progress: {label}",
        "written_at": utc_now(),
        "slice": f"slices/report-{report_n:02d}.json",
    }
    return rec


def resolve_sources(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    config_src = args.config
    if config_src is None and not args.dry_run:
        default_cfg = TMRL_DATA / "config" / "config.json"
        config_src = default_cfg if default_cfg.exists() else None
    reward_src = args.reward
    if reward_src is None and not args.dry_run:
        default_reward = TMRL_DATA / "reward" / "reward.pkl"
        reward_src = default_reward if default_reward.exists() else None
    return config_src, reward_src


def run(args: argparse.Namespace) -> Path:
    track = str(args.track or "").strip()
    slug = track_slug(track)
    run_name = str(args.run_name or "").strip()
    if not run_name:
        raise SystemExit("--run-name is required and must match config.json RUN_NAME")

    out = run_dir(args.output, track, run_name)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / STATE_NAME
    state = load_state(state_path)
    reports: list[dict[str, Any]] = list(state.get("reports") or [])

    if args.new_run:
        state = {
            "last_report": 0,
            "consumed_rounds": 0,
            "episode": None,
            "return_mean": None,
        }
        reports = []

    from_scratch = bool(args.new_run or state.get("from_scratch"))
    config_src, reward_src = resolve_sources(args)
    interface = str(args.interface or "").strip()
    env = str(args.env or "").strip()

    manifest = archive_run(
        out,
        track=track,
        run_name=run_name,
        input_path=args.input,
        config_src=config_src,
        reward_src=reward_src,
        dry_run=args.dry_run,
        new_run=args.new_run,
        from_scratch=from_scratch,
        env=env,
        interface=interface,
    )
    interface = str(manifest.get("interface") or "TM20LIDAR")
    env = str(manifest.get("env") or env_label(interface))

    if args.data_only:
        state = {
            **state,
            "track": track,
            "track_slug": slug,
            "run_name": run_name,
            "from_scratch": from_scratch,
            "env": env,
            "interface": interface,
            "reports": reports,
            "updated_at": utc_now(),
        }
        save_state(state_path, state)
        if reports:
            write_index(out, track, run_name, reports)
        print(f"archived {out} (watermark unchanged, last_report={int(state.get('last_report') or 0):02d})")
        print(f"manifest {out / MANIFEST_NAME}")
        return out

    if args.dry_run:
        if int(state.get("last_report") or 0) == 0:
            slice_rows = dummy_rows(0, 8, 0.0)
            consumed_after = 8
        else:
            slice_rows = dummy_rows(8, 8, 6.0)
            consumed_after = 16
    else:
        if not args.input:
            raise SystemExit("pass a trainer log or metrics JSON (or --dry-run)")
        all_rows = load_rows(args.input)
        consumed = int(state.get("consumed_rounds") or 0)
        slice_rows = all_rows[consumed:]
        consumed_after = consumed + len(slice_rows)

    if not slice_rows:
        raise SystemExit("no new trainer rounds since the last report")

    report_n = int(state.get("last_report") or 0) + 1
    rec = emit_report(
        out=out,
        track=track,
        run_name=run_name,
        label=args.label,
        interface=interface,
        env=env,
        from_scratch=from_scratch,
        slice_rows=slice_rows,
        report_n=report_n,
        consumed=consumed_after,
        prev_episode=state.get("episode"),
        prev_return=state.get("return_mean"),
    )
    reports.append(rec)
    metrics = window_metrics(slice_rows)
    state = {
        "track": track,
        "track_slug": slug,
        "run_name": run_name,
        "from_scratch": from_scratch,
        "env": env,
        "interface": interface,
        "last_report": report_n,
        "consumed_rounds": consumed_after,
        "episode": metrics["episode"],
        "return_mean": metrics["return_mean"],
        "reports": reports,
        "updated_at": utc_now(),
    }
    save_state(state_path, state)
    write_index(out, track, run_name, reports)
    print(rec["title"])
    print(f"wrote {out / rec['html']}")
    print(f"slice {out / rec['slice']}")
    print(f"index {out / 'index.html'}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="trainer log or metrics JSON (not required with --dry-run)",
    )
    p.add_argument(
        "--track",
        required=True,
        help="map name (required). Example: 'Summer 2026-01' or 'tmrl-test'",
    )
    p.add_argument(
        "--run-name",
        required=True,
        help="must match TmrlData/config/config.json RUN_NAME; unique per track",
    )
    p.add_argument("--output", type=Path, default=_REPO / "reports")
    p.add_argument(
        "--new-run",
        action="store_true",
        help="training restarted from scratch; next page is report 01",
    )
    p.add_argument("--label", default="", help="optional H1 override (e.g. 'night 1')")
    p.add_argument("--dry-run", action="store_true", help="write dummy report 01 (or 02 if state exists)")
    p.add_argument(
        "--data-only",
        action="store_true",
        help="copy log/config/reward into the run folder; do not write HTML or move the watermark",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.json to snapshot (default: ~/TmrlData/config/config.json)",
    )
    p.add_argument(
        "--reward",
        type=Path,
        default=None,
        help="reward.pkl for THIS track (default: ~/TmrlData/reward/reward.pkl)",
    )
    p.add_argument(
        "--env",
        default="",
        help="LIDAR or FULL (default: inferred from config RTGYM_INTERFACE)",
    )
    p.add_argument(
        "--interface",
        default="",
        help="override RTGYM_INTERFACE in the report payload",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
