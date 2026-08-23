# TrackMania tmrl results

A public notebook for one experiment: a **LIDAR SAC** agent learning to drive **TrackMania 2020** (free/Starter) on the official **tmrl-test** map, using [tmrl](https://github.com/trackmania-rl/tmrl).

This is not a fork of tmrl. tmrl is the trainer. This repo is what happened, why, and how to read it — including a shakedown overnight run we are keeping as baseline.

Live numbers (when training): [wandb `models-acme/tmrl`](https://wandb.ai/models-acme/tmrl).  
Shakedown story page: [learning-story/example/index.html](learning-story/example/index.html).

## Under the hood

The car does not get a camera full of pixels. LIDAR mode screenshots the cockpit (car hidden), then casts **19 rays** along the road in that image. OpenPlanet streams **speed** (and finish/position) over a local socket. The policy outputs throttle, brake, and steering to a **virtual Xbox pad** (ViGEm).

Reward is not “win the race.” A recorded demo trajectory is the spine of the track. The agent is paid for moving forward along that line, penalized for sitting still or straying, and given **+100** for crossing the finish. Episode cap is 1000 steps at 20 Hz (about 50 seconds).

Training is **SAC**. On this machine the trainer uses the GPU; the live policy runs on CPU. One TrackMania window is one worker. The 4080 Super spends most of the night waiting on new samples — the game at 20 FPS is the bottleneck, not the net.

We train locally: server + trainer + worker. Next real run will use `python -m tmrl --trainer --wandb` so graphs land in the existing wandb project. We are **not** wiping last night’s wandb run; a new `RUN_NAME` is a new run beside it.

## Shakedown (keep this)

First overnight pass, 21–22 Aug 2026, run name `SAC_4_LIDAR_train`. About **12.4 hours**, epochs **0–192**, **~482k** replay samples, peak train return **~320**. Late episodes often hit the 1000-step cap (it was finishing the episode, not just spinning out). Early return sat near 0.

Honest hole: the wandb backfill is missing **epochs 10–99** because the first log parser missed a header-width change. Charts in this repo leave that stretch empty instead of drawing a fake leap. After epoch 100 the record is dense and the car is clearly better.

What that run did **not** save: hourly policies (`SAVE_MODEL_EVERY` was 0) and TrackMania replay files. We still have the **final** weights. A later capture pass (`record_fastest_lap.py`, 5 episodes) got a **47.45 s** finished lap (3 of 5 finished, peak ~73 km/h). That is the end-of-shakedown car, not hour 3. We still cannot replay hour 2 vs hour 12 from shakedown.

## Next run (not started)

Same LIDAR path, new name (planned: `SAC_4_LIDAR_v1`). `SAVE_MODEL_EVERY` is already **10** so the worker keeps timestamped `.tmod` files (~0.36 MB each). After training: a short capture pass (`record_fastest_lap.py`) and `--test` on a few snapshots for a then-vs-now strip. Do not turn on overnight `save_replays`; it fills disk. Details: [docs/next-run.md](docs/next-run.md).

## Tools

Python 3.10+, no extra packages for the HTML report.

```text
learning-story/parse_trainer_log.py   trainer stdout → JSON
learning-story/generate_report.py     JSON/log → standalone HTML
learning-story/record_fastest_lap.py  fastest finished lap (needs the game open)
```

Do not commit `.venv/`, the wandb cache, `TmrlData/config/config.json` (API keys), or weight/checkpoint binaries.
