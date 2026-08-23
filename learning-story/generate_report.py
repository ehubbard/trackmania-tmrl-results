"""Build a standalone HTML learning story from tmrl trainer metrics.

The page is one file: no CDN, no build step. Open it locally or drop it on
GitHub Pages. Charts plot against real epoch numbers and break the line
across missing data instead of faking a jump.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from parse_trainer_log import missing_epochs, parse_trainer_log_file


def load_lap(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_filmstrip(mp4: Path, count: int = 20) -> list[str]:
    """JPEG data URIs sampled across a window-capture mp4 (browser-safe stills)."""
    try:
        import base64

        import cv2
    except ImportError:
        return []
    if not mp4.exists():
        return []
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n < 2:
        cap.release()
        return []
    idxs = [int(round(i * (n - 1) / (count - 1))) for i in range(count)]
    out: list[str] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > 360:
            frame = cv2.resize(frame, (360, max(1, int(h * 360 / w))))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 52])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        out.append("data:image/jpeg;base64," + b64)
    cap.release()
    return out


def attempts_html(lap: dict[str, Any]) -> str:
    attempts = lap.get("attempts") or []
    if not attempts:
        return ""
    rows = []
    for a in attempts:
        fin = "yes" if a.get("finished") else "no"
        rows.append(
            "<tr>"
            f"<td>{int(a.get('episode') or 0)}</td>"
            f"<td>{float(a.get('duration_s') or 0):.2f}s</td>"
            f"<td>{int(a.get('steps') or 0)}</td>"
            f"<td>{fin}</td>"
            f"<td>{float(a.get('progress') or 0):.0%}</td>"
            "</tr>"
        )
    return f"""
    <p class="caption">Five test episodes after the overnight weights. Best finished lap is the ghost below.</p>
    <table>
      <thead><tr><th>Ep</th><th>Time</th><th>Steps</th><th>Finished</th><th>Demo progress</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def lap_embed_html(
    lap: dict[str, Any],
    video_href: str | None,
    filmstrip: list[str] | None = None,
) -> str:
    xs = lap.get("x") or []
    zs = lap.get("z") or []
    if len(xs) < 2:
        return ""
    track = lap.get("track") or []
    all_x = xs + [p[0] for p in track]
    all_z = zs + [p[1] for p in track]
    min_x, max_x = min(all_x), max(all_x)
    min_z, max_z = min(all_z), max(all_z)
    pad = 0.06 * max(max_x - min_x, max_z - min_z, 1.0)
    min_x -= pad
    max_x += pad
    min_z -= pad
    max_z += pad
    duration = float(lap.get("duration_s") or 0)
    finished = bool(lap.get("finished"))
    title = "Fastest recorded lap" if finished else "Best recorded run"
    max_speed = max((float(s) for s in (lap.get("speed") or [0])), default=0)
    caption = (
        "End-of-shakedown policy on tmrl-test, recorded after training "
        f"(not a TrackMania replay file). Peak {max_speed:.0f} km/h. "
        "Orange is the car; gray is the reward demo line."
    )
    frames = [f for f in (filmstrip or []) if f]
    payload = {
        "x": [round(v, 2) for v in xs],
        "z": [round(v, 2) for v in zs],
        "speed": lap.get("speed") or [],
        "progress": lap.get("lap_progress") or [],
        "dt": float(lap.get("dt") or 0.05),
        "duration": duration,
        "minX": min_x,
        "maxX": max_x,
        "minZ": min_z,
        "maxZ": max_z,
        "track": [[round(p[0], 2), round(p[1], 2)] for p in track],
        "nFrames": len(frames),
    }
    blob = json.dumps(payload).replace("<", "\\u003c")
    video = ""
    if video_href:
        video = f"""
        <video class="lap-video" controls playsinline>
          <source src="{video_href}" type="video/mp4" />
        </video>
        <p class="caption">Window capture may not play in Chrome (OpenCV mp4v). The stills and ghost below always work.</p>"""
    stills = ""
    if frames:
        stills = (
            '<img id="lap-frame" class="lap-film" alt="Cockpit still from the recorded lap" />'
        )
        stills_js = json.dumps(frames).replace("<", "\\u003c")
    else:
        stills_js = "[]"
    status = (
        f"{duration:.2f}s · finished"
        if finished
        else f"{duration:.2f}s · progress {float(lap.get('progress') or 0):.0%}"
    )
    return f"""
    <h2>{title}</h2>
    <p class="caption">{caption} {status}.</p>
    {attempts_html(lap)}
    <div class="lap">
      {video}
      {stills}
      <div class="lap-stage">
        <svg id="lap-map" viewBox="0 0 640 360" role="img" aria-label="Overhead lap replay">
          <rect width="640" height="360" fill="#fafaf9" />
          <polyline id="lap-track" fill="none" stroke="#d6d3d1" stroke-width="7" stroke-linecap="round" stroke-linejoin="round" />
          <polyline id="lap-path" fill="none" stroke="#c2410c" stroke-width="2.5" />
          <circle id="lap-car" r="6" fill="#c2410c" />
        </svg>
        <div class="lap-hud">
          <span id="lap-time">0.00s</span>
          <span id="lap-speed">0 km/h</span>
          <span id="lap-progress">0%</span>
          <button type="button" id="lap-toggle">Pause</button>
        </div>
      </div>
    </div>
    <script>
    (function () {{
      const L = {blob};
      const FRAMES = {stills_js};
      const svg = document.getElementById("lap-map");
      const toX = (x) => (x - L.minX) / (L.maxX - L.minX) * 640;
      const toY = (z) => (1 - (z - L.minZ) / (L.maxZ - L.minZ)) * 360;
      const pts = (arr) => arr.map((p) => toX(p[0]).toFixed(1) + "," + toY(p[1]).toFixed(1)).join(" ");
      document.getElementById("lap-track").setAttribute("points", pts(L.track));
      const car = document.getElementById("lap-car");
      const path = document.getElementById("lap-path");
      const timeEl = document.getElementById("lap-time");
      const speedEl = document.getElementById("lap-speed");
      const progEl = document.getElementById("lap-progress");
      const btn = document.getElementById("lap-toggle");
      let i = 0, playing = true, acc = 0, last = null;
      function draw(n) {{
        const sliced = L.x.slice(0, n + 1).map((x, k) => [x, L.z[k]]);
        path.setAttribute("points", pts(sliced));
        car.setAttribute("cx", toX(L.x[n]).toFixed(1));
        car.setAttribute("cy", toY(L.z[n]).toFixed(1));
        timeEl.textContent = (n * L.dt).toFixed(2) + "s";
        const sp = L.speed[n] || 0;
        speedEl.textContent = Math.round(sp) + " km/h";
        const pr = (L.progress && L.progress[n]) || 0;
        progEl.textContent = Math.round(pr * 100) + "% of demo";
        if (FRAMES.length) {{
          const img = document.getElementById("lap-frame");
          if (img) {{
            const fi = Math.min(
              FRAMES.length - 1,
              Math.round(n / Math.max(L.x.length - 1, 1) * (FRAMES.length - 1))
            );
            if (img.dataset.i !== String(fi)) {{
              img.src = FRAMES[fi];
              img.dataset.i = String(fi);
            }}
          }}
        }}
      }}
      function tick(ts) {{
        if (!playing) {{ last = ts; requestAnimationFrame(tick); return; }}
        if (last == null) last = ts;
        acc += (ts - last) / 1000;
        last = ts;
        while (acc >= L.dt && i < L.x.length - 1) {{ acc -= L.dt; i += 1; }}
        if (i >= L.x.length - 1) {{ i = 0; acc = 0; }}
        draw(i);
        requestAnimationFrame(tick);
      }}
      btn.addEventListener("click", () => {{
        playing = !playing;
        btn.textContent = playing ? "Pause" : "Play";
      }});
      svg.addEventListener("click", (ev) => {{
        const r = svg.getBoundingClientRect();
        i = Math.min(L.x.length - 1, Math.max(0, Math.floor((ev.clientX - r.left) / r.width * L.x.length)));
        draw(i);
      }});
      draw(0);
      requestAnimationFrame(tick);
    }})();
    </script>
    """


def load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    rows = parse_trainer_log_file(path)
    return {
        "run_name": path.stem,
        "rounds": rows,
        "missing_epochs": missing_epochs(rows),
    }


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def epoch_series(rows: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is None:
            continue
        buckets[int(row["epoch"])].append(float(val))
    return sorted((ep, mean(vals)) for ep, vals in buckets.items())


def first_round(rows: list[dict[str, Any]], pred) -> dict[str, Any] | None:
    for row in rows:
        if pred(row):
            return row
    return None


def svg_chart(
    series: list[tuple[int, float]],
    *,
    width: int = 920,
    height: int = 280,
    y_max: float | None = None,
    y_ref: float | None = None,
    y_ref_label: str = "",
    color: str = "#c2410c",
    y_label: str,
    x_label: str = "Epoch",
) -> str:
    if not series:
        return ""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 40
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1
    ymax = y_max if y_max is not None else max(ys) * 1.08
    ymin = 0.0
    if ymax <= ymin:
        ymax = 1.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def px(x: float, y: float) -> tuple[float, float]:
        return (
            pad_l + (x - xmin) / (xmax - xmin) * plot_w,
            pad_t + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h,
        )

    # Break the polyline wherever epochs skip more than 1.
    segments: list[list[tuple[int, float]]] = [[series[0]]]
    for prev, cur in zip(series, series[1:]):
        if cur[0] - prev[0] > 1:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    paths = []
    for seg in segments:
        if len(seg) == 1:
            x, y = px(*seg[0])
            paths.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" />'
            )
            continue
        d = "M " + " L ".join(f"{px(x, y)[0]:.1f},{px(x, y)[1]:.1f}" for x, y in seg)
        fill_pts = [px(seg[0][0], 0)] + [px(x, y) for x, y in seg] + [px(seg[-1][0], 0)]
        fill_d = "M " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in fill_pts) + " Z"
        paths.append(
            f'<path d="{fill_d}" fill="{color}" fill-opacity="0.12" stroke="none" />'
        )
        paths.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" />'
        )

    ref = ""
    if y_ref is not None:
        _, y = px(xmin, y_ref)
        ref = (
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="#78716c" stroke-dasharray="4 4" />'
            f'<text x="{width - pad_r - 4}" y="{y - 6:.1f}" text-anchor="end" '
            f'class="muted tiny">{y_ref_label}</text>'
        )

    y_ticks = [0, ymax / 2, ymax]
    y_grid = []
    for tick in y_ticks:
        x, y = px(xmin, tick)
        y_grid.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="#e7e5e4" />'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" class="muted tiny">'
            f"{tick:.0f}</text>"
        )

    x_ticks = [xmin]
    for t in (100, 150, 192):
        if xmin < t < xmax:
            x_ticks.append(t)
    x_ticks.append(xmax)
    x_labels = []
    for tick in x_ticks:
        x, _ = px(tick, ymin)
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" class="muted tiny">'
            f"{int(tick)}</text>"
        )

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{y_label} versus {x_label}">
  <text x="14" y="18" class="muted tiny" transform="rotate(-90 14 140)">{y_label}</text>
  {''.join(y_grid)}
  {ref}
  {''.join(paths)}
  {''.join(x_labels)}
  <text x="{width / 2:.0f}" y="{height - 2}" text-anchor="middle" class="muted tiny">{x_label}</text>
</svg>
"""


def fmt(n: float, digits: int = 1) -> str:
    return f"{n:.{digits}f}"


def build_html(
    payload: dict[str, Any],
    lap: dict[str, Any] | None = None,
    video_href: str | None = None,
    filmstrip: list[str] | None = None,
) -> str:
    rows = payload["rounds"]
    rows = sorted(rows, key=lambda r: (int(r["epoch"]), int(r.get("round") or 0)))
    returns = epoch_series(rows, "return_train")
    lengths = epoch_series(rows, "episode_length_train")
    memory = epoch_series(rows, "memory_len")
    missing = payload.get("missing_epochs") or missing_epochs(rows)

    peak = max((float(r["return_train"]) for r in rows), default=0.0)
    last = rows[-1]
    wall = payload.get("wall_hours")
    finish_bonus = float(payload.get("finish_bonus") or 100)
    ep_cap = float(payload.get("ep_max_length") or 1000)
    run_name = payload.get("run_name") or "tmrl run"

    m_ret10 = first_round(rows, lambda r: float(r.get("return_train") or 0) >= 10)
    m_len500 = first_round(rows, lambda r: float(r.get("episode_length_train") or 0) >= 500)
    m_len1000 = first_round(rows, lambda r: float(r.get("episode_length_train") or 0) >= ep_cap)
    m_ret200 = first_round(rows, lambda r: float(r.get("return_train") or 0) >= 200)

    def mile(row: dict[str, Any] | None) -> str:
        if not row:
            return "—"
        return (
            f"epoch {int(row['epoch'])} · return {fmt(row['return_train'])} · "
            f"length {fmt(row['episode_length_train'], 0)}"
        )

    lap_html = (
        lap_embed_html(lap, video_href, filmstrip=filmstrip)
        if lap
        else """
    <h2>Fastest lap</h2>
    <aside class="slot">
      <strong>Waiting on a capture pass.</strong>
      Overnight training did not save TrackMania replays. The final weights
      still exist. With the game on tmrl-test (car hidden), run
      <code>record_fastest_lap.py --episodes 5</code>. This block becomes an
      overhead ghost (gray demo line, orange car) and a window video if an
      mp4 encodes. That is the end-of-shakedown policy, not a lap from hour 3.
    </aside>
    """
    )

    gap_note = ""
    if missing:
        gap_note = f"""
        <aside class="callout">
          <strong>Missing epochs {missing[0]}–{missing[-1]}.</strong>
          The source log skipped this band. Charts leave that stretch empty
          instead of connecting epoch {missing[0] - 1} to {missing[-1] + 1}.
        </aside>
        """

    hours = f"{wall:g} hours" if wall else f"{int(last['epoch']) + 1} epochs"
    mem_k = float(last.get("memory_len") or 0) / 1000

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>trackmania-tmrl-results — {run_name}</title>
  <style>
    :root {{
      --bg: #fafaf9;
      --ink: #1c1917;
      --muted: #78716c;
      --line: #e7e5e4;
      --accent: #c2410c;
      --card: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 16px/1.5 "Source Serif 4", "Iowan Old Style", Georgia, serif;
    }}
    header, main {{ max-width: 920px; margin: 0 auto; padding: 0 24px; }}
    header {{ padding-top: 48px; padding-bottom: 8px; }}
    h1 {{ font-size: 2.1rem; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 8px; }}
    h2 {{ font-size: 1.25rem; margin: 40px 0 12px; }}
    p.lead {{ color: var(--muted); margin: 0 0 24px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
    .stat {{ background: var(--card); border: 1px solid var(--line); padding: 16px 18px; }}
    .stat b {{ display: block; font-size: 1.6rem; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .stat span {{ color: var(--muted); font-size: 0.85rem; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .callout {{
      border-left: 3px solid var(--accent);
      background: #fff7ed;
      padding: 12px 16px;
      margin: 16px 0 28px;
    }}
    .slot {{
      border-left: 3px solid #a8a29e;
      background: #f5f5f4;
      padding: 12px 16px;
      margin: 8px 0 28px;
      color: var(--muted);
    }}
    .hood {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 8px 0 28px; }}
    .hood div {{ background: var(--card); border: 1px solid var(--line); padding: 16px 18px; }}
    .hood h3 {{ margin: 0 0 6px; font-size: 1rem; }}
    nav {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.85rem; color: var(--muted); margin-bottom: 16px; }}
    nav a {{ color: var(--muted); margin-right: 14px; }}
    .chart {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 12px 8px 0;
      margin: 8px 0 28px;
    }}
    .caption {{ color: var(--muted); font-size: 0.85rem; margin: 0 0 8px; }}
    .phases {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .phase {{ background: var(--card); border: 1px solid var(--line); padding: 16px 18px; }}
    .phase h3 {{ margin: 0 0 4px; font-size: 1rem; }}
    .phase .when {{ color: var(--muted); font-size: 0.8rem; font-family: ui-sans-serif, system-ui, sans-serif; }}
    table {{ width: 100%; border-collapse: collapse; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.95rem; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-weight: 500; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .muted {{ fill: #78716c; color: #78716c; }}
    .tiny {{ font: 11px ui-sans-serif, system-ui, sans-serif; }}
    .lap {{ display: grid; gap: 12px; margin: 8px 0 28px; }}
    .lap-video {{ width: 100%; background: #000; }}
    .lap-film {{ width: 100%; background: #111; display: block; }}
    .lap-stage {{ background: var(--card); border: 1px solid var(--line); }}
    .lap-hud {{
      display: flex; gap: 16px; align-items: center;
      padding: 8px 12px; font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 0.9rem; border-top: 1px solid var(--line);
    }}
    .lap-hud button {{
      margin-left: auto; background: var(--ink); color: #fff;
      border: 0; padding: 4px 12px; cursor: pointer;
    }}
    footer {{ color: var(--muted); font-size: 0.9rem; margin: 48px 0 64px; }}
    a {{ color: var(--accent); }}
    @media (max-width: 720px) {{
      .stats, .phases, .hood {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 480px) {{
      .stats, .phases, .hood {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>How the car learned overnight</h1>
    <p class="lead">
      Shakedown run on tmrl-test. {run_name} · {payload.get("interface") or "TM20LIDAR"} · {payload.get("algorithm") or "SAC"} ·
      {hours} · epochs 0–{int(last["epoch"])}.
      We keep this as baseline. v1 is a later train with snapshots.
    </p>
    <nav>
      <a href="#hood">Under the hood</a>
      <a href="#lap">The car</a>
      <a href="#curves">Curves</a>
      <a href="#later">After v1</a>
    </nav>
  </header>
  <main>
    <h2 id="hood">What this experiment is</h2>
    <div class="hood">
      <div>
        <h3>Sees</h3>
        <p>19 LIDAR rays from a cockpit screenshot (car hidden) plus speed from OpenPlanet. Not a full camera.</p>
      </div>
      <div>
        <h3>Does</h3>
        <p>Throttle, brake, steer on a virtual Xbox pad. One TrackMania window, 20&nbsp;Hz, map tmrl-test.</p>
      </div>
      <div>
        <h3>Learns</h3>
        <p>SAC. Paid for moving along a recorded road line; +100 at the finish. Trainer on GPU, policy on CPU.</p>
      </div>
    </div>
    <p class="caption">LIDAR only works on black-border asphalt. Other Club maps need a new reward recording and usually a new train. Companion to <a href="https://github.com/trackmania-rl/tmrl">tmrl</a>. Public notes: <a href="https://github.com/ehubbard/trackmania-tmrl-results">trackmania-tmrl-results</a>.</p>
    <div class="stats">
      <div class="stat"><b>{hours.split()[0] if wall else int(last["epoch"])+1}</b><span>{"Hours" if wall else "Epochs"}</span></div>
      <div class="stat"><b>{int(last["epoch"])}</b><span>Last epoch</span></div>
      <div class="stat"><b>{fmt(peak, 0)}</b><span>Peak train return</span></div>
      <div class="stat"><b>{fmt(mem_k, 0)}k</b><span>Replay samples</span></div>
    </div>
    {gap_note}
    <div id="lap">{lap_html}</div>
    <h2 id="curves">Train return</h2>
    <p class="caption">
      Mean return per epoch. Dashed line is the +{fmt(finish_bonus, 0)} finish-line bonus.
      Values near 200 mean completed laps, not just staying on the road.
      Source: trainer log · x = epoch.
    </p>
    <div class="chart">
      {svg_chart(returns, y_ref=finish_bonus, y_ref_label=f"Finish bonus ({fmt(finish_bonus, 0)})", y_label="Train return")}
    </div>
    <h2>Episode length</h2>
    <p class="caption">
      Environment steps before crash, stray, or timeout. ~80 is a spin-out.
      {fmt(ep_cap, 0)} is the configured cap — pinning there means the episode finished.
      Source: trainer log · x = epoch.
    </p>
    <div class="chart">
      {svg_chart(lengths, y_max=ep_cap, y_ref=ep_cap, y_ref_label="Episode cap", color="#0f766e", y_label="Episode length (steps)")}
    </div>
    <h2>Four phases</h2>
    <div class="phases">
      <div class="phase">
        <div class="when">Epochs 0–9</div>
        <h3>Spin-outs</h3>
        <p>Return near 0–5. Episodes die around 80–230 steps. The policy has not learned to stay on the black-border road yet.</p>
      </div>
      <div class="phase">
        <div class="when">Epochs {missing[0] if missing else "—"}–{missing[-1] if missing else "—"}</div>
        <h3>Not in the graph</h3>
        <p>Missing from the uploaded log. Do not read the jump after the gap as one-step genius. Training was still running.</p>
      </div>
      <div class="phase">
        <div class="when">Epochs 100–108</div>
        <h3>Breakthrough</h3>
        <p>First long episodes, then full-length runs. Return crosses 200 — finish bonus plus progress down the track.</p>
      </div>
      <div class="phase">
        <div class="when">Epochs 109–{int(last["epoch"])}</div>
        <h3>Consistent driving</h3>
        <p>Episode length mostly glued to the cap. Return hangs around 190–230 (peak {fmt(peak, 0)}). Noise is SAC, not a broken chart.</p>
      </div>
    </div>
    <h2>Milestones</h2>
    <table>
      <thead><tr><th>What changed</th><th>When</th></tr></thead>
      <tbody>
        <tr><td>First return ≥ 10</td><td>{mile(m_ret10)}</td></tr>
        <tr><td>First episode ≥ 500 steps</td><td>{mile(m_len500)}</td></tr>
        <tr><td>First full-length episode</td><td>{mile(m_len1000)}</td></tr>
        <tr><td>Return ≥ 200 (laps)</td><td>{mile(m_ret200)}</td></tr>
        <tr><td>End of run</td><td>epoch {int(last["epoch"])} · return {fmt(last["return_train"])} · length {fmt(last["episode_length_train"], 0)}</td></tr>
      </tbody>
    </table>
    <h2>Replay buffer</h2>
    <p class="caption">Samples stored for training. Only goes up. The GPU was rarely the limit — one game at 20 FPS was. Units: samples.</p>
    <div class="chart">
      {svg_chart(memory, color="#44403c", y_label="Replay memory (samples)")}
    </div>
    <h2 id="later">What we still cannot show</h2>
    <aside class="slot">
      <strong>Then vs now.</strong>
      Overnight <code>SAVE_MODEL_EVERY</code> was 0, so there is no hour-2 vs hour-12
      policy. After v1 (snapshots every 10 epochs) this slot gets two ghosts or
      two clips: early wobble vs later drive. Same map, same camera.
    </aside>
    <aside class="slot">
      <strong>v1 vs this shakedown.</strong>
      Overlay two trains on the same epoch axis once v1 exists. Until then this
      page is one night of data, not a comparison.
    </aside>
    <p>
      Wandb stays the live logger
      (<a href="https://wandb.ai/models-acme/tmrl/runs/SAC_4_LIDAR_train_overnight">shakedown run</a>).
      This page is the story you can share. TrackMania <code>.Replay.Gbx</code>
      files do not embed on GitHub Pages; the 2D ghost plus a window video is
      the public artifact.
    </p>
    <footer>
      Generated from {len(rows)} trainer rounds
      {" · " + payload["wandb_run"] if payload.get("wandb_run") else ""}.
      Companion to <a href="https://github.com/trackmania-rl/tmrl">tmrl</a>, not a replacement.
    </footer>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trainer log or metrics JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/index.html"),
        help="HTML path",
    )
    parser.add_argument(
        "--lap",
        type=Path,
        default=None,
        help="fastest-lap.json from record_fastest_lap.py",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Embed the local mp4 (often not Chrome-safe; omitted from GitHub Pages)",
    )
    args = parser.parse_args()
    payload = load_payload(args.input)
    lap_path = args.lap
    if lap_path is None:
        candidate = args.output.parent / "fastest-lap.json"
        if candidate.exists():
            lap_path = candidate
    lap = load_lap(lap_path)
    video_href = None
    mp4 = None
    if lap_path:
        mp4 = lap_path.with_suffix(".mp4")
        if not mp4.exists():
            mp4 = None
    if args.video and mp4 is not None:
        video_href = mp4.name
    strip: list[str] = []
    if mp4 is not None:
        strip = extract_filmstrip(mp4)
        print(f"filmstrip {len(strip)} frames from {mp4}")
    html = build_html(payload, lap=lap, video_href=video_href, filmstrip=strip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
