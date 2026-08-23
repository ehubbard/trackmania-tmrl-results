# Learning story for tmrl

tmrl already trains a TrackMania car. This folder turns the **trainer log** into a page you can open or publish: how long the car stayed on the road, when it started finishing laps, and where the log has holes.

It is a companion to [tmrl](https://github.com/trackmania-rl/tmrl), not a fork. The public repo for this experiment is [ehubbard/trackmania-tmrl-results](https://github.com/ehubbard/trackmania-tmrl-results).

## Why this exists

Wandb is the right live logger. It is a weak place to *feel* the learning. A story page plots reward against **epoch** (not upload time), leaves missing epochs empty instead of drawing a fake leap, and can later point at saved policies from hour 2 vs hour 12.

The overnight LIDAR run in `example/` is the first cut of that page. The copy people actually open is [`docs/index.html`](../docs/index.html), served on GitHub Pages so nobody has to download a file. After a capture pass it includes an overhead ghost, cockpit stills, and the test episodes — not a TrackMania `.Replay.Gbx`.

## On-demand progress reports

Do not emit a page on a timer. When you want an update, run this against the current trainer log. It remembers how far it already reported and only charts **new** rounds. **`--track` is required.** Layout: `reports/<track-slug>/<run-name>/`.

HTML is a view. That folder also copies (or documents) the data you cannot rebuild: `trainer.log`, redacted `config.snapshot.json`, `reward.pkl` for this map, `manifest.json`. `--data-only` refreshes those files without writing HTML or moving the watermark.

```powershell
# first page of a from-scratch train
python progress_report.py path\to\trainer.log --track "Summer 2026-01" --run-name SAC_4_LIDAR_summer2026_01 --new-run

# later, same train, only the new slice
python progress_report.py path\to\trainer.log --track "Summer 2026-01" --run-name SAC_4_LIDAR_summer2026_01

# copy log/config/reward; do not emit HTML or advance the watermark
python progress_report.py path\to\trainer.log --track "Summer 2026-01" --run-name SAC_4_LIDAR_summer2026_01 --data-only

# naming check without tmrl
python progress_report.py --dry-run --track "Summer 2026-01" --run-name demo --new-run
python progress_report.py --dry-run --track "Summer 2026-01" --run-name demo
```

Swap `--track` (and `RUN_NAME` / `reward.pkl`) for tmrl-test, campaign 03, an editor map, etc. Exact Club steps: [docs/next-run.md](../docs/next-run.md).

Titles look like `TrackMania training progress: Summer 2026-01 · Report 03 · Attempts 24 · Episode 412`.

- **Report** — 01, 02, … for this run. `--new-run` resets to 01.
- **Attempts** — trainer rounds in this slice (the log’s unit; not laps).
- **Episode** — mean episode length (steps) in this slice. Toward 1000 means it is finishing. Not an official lap time.

`--label "night 1"` replaces the whole H1 if you want a custom name. Each map has its own `state.json` watermark; do not share `reward.pkl` or logs across tracks.

## Fastest lap on the page

Training did **not** keep TrackMania replays (`save_replays` was off), so the overnight session has no `.Replay.Gbx` to embed. We still have the **final** weights. After training, a capture pass recorded the policy on tmrl-test:

- 5 test episodes; 3 finished
- Fastest finish: **47.45 s** (949 steps, peak **73 km/h**)
- Episode 1 spun out; episode 2 hit the 1000-step cap at 60% of the demo line

The HTML plays an overhead ghost (demo line + car) and cockpit stills sampled from the window capture. The raw `fastest-lap.mp4` is local-only (gitignored; OpenCV `mp4v` often will not play in Chrome). GitHub Pages cannot play a native `.Replay.Gbx`.

With TrackMania open on tmrl-test (camera with the car hidden), click the game window, then:

```bash
python record_fastest_lap.py --episodes 5 -o example/fastest-lap.json
python generate_report.py example/SAC_4_LIDAR_train_overnight.json --lap example/fastest-lap.json -o ../docs/index.html
```

## Generate a report

From this folder:

```bash
python parse_trainer_log.py path/to/trainer.log -o metrics.json --run-name SAC_4_LIDAR_train
python generate_report.py metrics.json -o ../docs/index.html
```

`generate_report.py` also accepts a raw trainer log. The parser handles tmrl 0.7.x headers, including the width change that skips epochs 10–99 if you use a naive regex (`=== epoch 9/10000 = round` vs `=== epoch 10/10000  round`).

No extra packages. Python 3.10+.

## Make the next run worth watching

This overnight run only kept the **final** weights (`SAVE_MODEL_EVERY` was 0). You cannot replay hour 2 vs hour 12 from it.

In `TmrlData/config/config.json`:

- Set `"SAVE_MODEL_EVERY": 10`. The worker then writes a timestamped `.tmod` every 10 new policies. LIDAR weights are about 0.36 MB, so a 12-hour run is tens of megabytes, not hundreds.
- Train with `--wandb` if you want live numbers.
- After training, copy older `.tmod` files over the current worker weights and run `python -m tmrl --test` to watch that snapshot drive.
- Optional: `"save_replays": true` in `ENV.RTGYM_CONFIG.interface_kwargs`, or `python -m tmrl.tools.save_replays`, for TrackMania replay files. That can fill disk overnight; prefer snapshots plus a few recorded laps.

## What is not here yet

- Hour-2 vs hour-12 video (needs v1 snapshots, then another capture pass).
- A plugin that talks to wandb directly. This reads the same metrics tmrl already prints.

If this gets good enough to publish, the repo is this folder plus the example report — not the local `.venv`, wandb cache, or `config.json` API keys.
