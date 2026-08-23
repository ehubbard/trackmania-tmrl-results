# Next run checklist

Shakedown weights stay on disk as `SAC_4_LIDAR_train`. Do not point `RUN_NAME` at v1 until you are ready to train, or `--test` will look for the wrong `.tmod`.

## Config (`TmrlData/config/config.json`)

When starting v1:

- `RUN_NAME`: `SAC_4_LIDAR_v1`
- `RESET_TRAINING`: `true` only for a true fresh start (then set it back to `false` so crashes resume)
- `SAVE_MODEL_EVERY`: `10` (already set)
- `WANDB_ENTITY`: `models-acme`
- `WANDB_PROJECT`: `tmrl` (leave; do not wipe the shakedown run)
- `save_replays`: keep `false` overnight

## Commands

Same three processes as shakedown, trainer with wandb:

```powershell
C:\Users\cursor\tmrl\.venv\Scripts\python.exe -m tmrl --server
C:\Users\cursor\tmrl\.venv\Scripts\python.exe -m tmrl --trainer --wandb
C:\Users\cursor\tmrl\.venv\Scripts\python.exe -m tmrl --worker
```

TrackMania: windowed, **tmrl-test**, cockpit camera with the car hidden, click the game after the worker starts.

## After training

1. Keep the final v1 checkpoint.
2. `python learning-story/record_fastest_lap.py --episodes 5`
3. `--test` a few timestamped `.tmod` files (copy over the worker weights, then restore the latest).
4. Rebuild the HTML with `generate_report.py --lap ...`
5. Compare to shakedown in the README, do not delete shakedown artifacts.
