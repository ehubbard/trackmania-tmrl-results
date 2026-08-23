"""Record the trained policy's fastest finished lap for the learning-story page.

Overnight training did not keep TrackMania replays (save_replays was off).
This captures the car as it is now — the end-of-run weights — by watching
OpenPlanet position/speed while tmrl --test drives.

Requires TrackMania 2020 open on tmrl-test, window titled Trackmania,
cockpit camera with the car hidden. Click the game window after this starts.

Writes:
  fastest-lap.json   telemetry + downsampled demo track (always)
  fastest-lap.mp4    window capture if OpenCV can encode (optional)

Then rebuild the HTML:

  python generate_report.py example/SAC_4_LIDAR_train_overnight.json \\
      --lap example/fastest-lap.json -o ../docs/index.html
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def downsample_xy(points: np.ndarray, n: int = 400) -> list[list[float]]:
    if len(points) <= n:
        return [[float(p[0]), float(p[1])] for p in points]
    idx = np.linspace(0, len(points) - 1, n).astype(int)
    return [[float(points[i, 0]), float(points[i, 1])] for i in idx]


def try_write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> bool:
    """Write frames, then remux to H.264 if ffmpeg is available (Chrome-safe)."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_name(path.stem + ".raw.mp4")
    opened = False
    for fourcc in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            opened = True
            break
        writer.release()
    if not opened:
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    if not raw.exists() or raw.stat().st_size == 0:
        return False
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and path.exists() and path.stat().st_size > 0:
            raw.unlink(missing_ok=True)
            return True
        print(f"ffmpeg remux failed: {result.stderr[-400:]}", flush=True)
    raw.replace(path)
    return path.exists() and path.stat().st_size > 0


def find_interface(env):
    u = env.unwrapped
    if hasattr(u, "interface"):
        return u.interface
    raise RuntimeError("Could not find rtgym interface on the environment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "example" / "fastest-lap.json",
    )
    args = parser.parse_args()

    import tmrl.config.config_constants as cfg
    import tmrl.config.config_objects as cfg_obj
    from tmrl.envs import GenericGymEnv
    from tmrl.networking import RolloutWorker
    from tmrl.util import partial

    reward_path = Path(cfg.REWARD_PATH)
    if not reward_path.exists():
        raise SystemExit(f"Missing reward trajectory: {reward_path}")
    demo = pickle.loads(reward_path.read_bytes())
    demo = np.asarray(demo)
    track = downsample_xy(np.stack([demo[:, 0], demo[:, 2]], axis=1))

    config = dict(cfg_obj.CONFIG_DICT)
    config["interface_kwargs"] = dict(config.get("interface_kwargs") or {})
    config["interface_kwargs"]["save_replays"] = True

    print("Open TrackMania, tmrl-test, camera with car hidden, then click the window.", flush=True)
    rw = RolloutWorker(
        env_cls=partial(
            GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}
        ),
        actor_module_cls=cfg_obj.POLICY,
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
        device="cuda" if cfg.CUDA_INFERENCE else "cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
        crc_debug=cfg.CRC_DEBUG,
        standalone=True,
    )
    iface = find_interface(rw.env)
    orig_grab = iface.grab_lidar_speed_and_data
    dt = float(cfg.ENV_CONFIG["RTGYM_CONFIG"]["time_step_duration"])

    current: dict = {"points": [], "frames": []}

    def grab():
        img = iface.window_interface.screenshot()[:, :, :3]
        data = iface.client.retrieve_data()
        speed = np.array([data[0]], dtype="float32")
        lidar = iface.lidar.lidar_20(img=img, show=False)
        progress = 0.0
        if getattr(iface, "reward_function", None) is not None and iface.reward_function.datalen:
            progress = iface.reward_function.cur_idx / iface.reward_function.datalen
        current["points"].append(
            {
                "x": float(data[2]),
                "z": float(data[4]),
                "speed": float(data[0]),
                "finished": bool(data[8]),
                "progress": float(progress),
            }
        )
        small = cv2.resize(img, (480, 244))
        current["frames"].append(small)
        return lidar, speed, data

    iface.grab_lidar_speed_and_data = grab

    best = None
    attempts: list[dict] = []
    for ep in range(args.episodes):
        current = {"points": [], "frames": []}
        t0 = time.time()
        rw.run_episode(cfg.RW_MAX_SAMPLES_PER_EPISODE, train=False)
        elapsed = time.time() - t0
        pts = current["points"]
        finished = any(p["finished"] for p in pts)
        duration = len(pts) * dt
        progress = pts[-1]["progress"] if pts else 0.0
        print(
            f"episode {ep + 1}/{args.episodes}  steps={len(pts)}  "
            f"{duration:.2f}s  finished={finished}  progress={progress:.3f}  wall={elapsed:.1f}s",
            flush=True,
        )
        if not pts:
            continue
        attempts.append(
            {
                "episode": ep + 1,
                "steps": len(pts),
                "duration_s": round(duration, 2),
                "finished": finished,
                "progress": round(progress, 4),
            }
        )
        candidate = {
            "finished": finished,
            "duration_s": duration,
            "progress": progress,
            "points": pts,
            "frames": current["frames"],
        }
        if best is None:
            best = candidate
            continue
        # Prefer a finished lap; among those, shortest time; else most progress.
        def key(c):
            return (not c["finished"], c["duration_s"] if c["finished"] else -c["progress"])

        if key(candidate) < key(best):
            best = candidate

    if best is None:
        raise SystemExit("No episode recorded")

    video_name = None
    video_path = args.output.with_suffix(".mp4")
    if try_write_mp4(video_path, best["frames"], fps=1.0 / dt):
        video_name = video_path.name
        print(f"wrote {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        print("no browser-safe mp4 encoder; telemetry-only", flush=True)

    payload = {
        "run_name": cfg.RUN_NAME,
        "weights": Path(cfg.MODEL_PATH_WORKER).name,
        "map": "tmrl-test",
        "dt": dt,
        "duration_s": round(best["duration_s"], 3),
        "finished": best["finished"],
        "steps": len(best["points"]),
        "progress": round(best["progress"], 4),
        "max_speed": round(max(p["speed"] for p in best["points"]), 2),
        "attempts": attempts,
        "note": (
            "Recorded from the trained policy after the overnight run. "
            "Training itself did not save TrackMania replays."
        ),
        "video": video_name,
        "track": track,
        "x": [p["x"] for p in best["points"]],
        "z": [p["z"] for p in best["points"]],
        "speed": [round(p["speed"], 2) for p in best["points"]],
        "lap_progress": [round(p["progress"], 4) for p in best["points"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    print(
        "Rebuild the report with:\n"
        f"  python generate_report.py example/SAC_4_LIDAR_train_overnight.json "
        f"--lap {args.output} -o ../docs/index.html",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        print(
            "Start TrackMania 2020, load tmrl-test, then run this again.",
            file=sys.stderr,
        )
        sys.exit(1)
