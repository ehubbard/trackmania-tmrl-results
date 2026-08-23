# Train a new map (Club)

Shakedown weights stay on disk as `SAC_4_LIDAR_train` on **tmrl-test**. Do not reuse that `RUN_NAME`. tmrl keys weights and the replay checkpoint by `RUN_NAME` only — two maps with the same name overwrite each other even if reports live in different folders.

**Summer 2026-01** is the first extra Club map. The same steps work for any later map: change `--track`, `RUN_NAME`, and the reward pickle. Do not hardcode this campaign into the tools.

tmrl always reads one file: `%USERPROFILE%\TmrlData\reward\reward.pkl`. Mixing maps in that file, or in one replay buffer, wastes the train.

## Before you touch Summer 2026-01

Keep the tmrl-test pickle under a name you can restore:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\TmrlData\reward\tracks" | Out-Null
Copy-Item "$env:USERPROFILE\TmrlData\reward\reward.pkl" "$env:USERPROFILE\TmrlData\reward\tracks\tmrl-test.pkl"
```

`--record-reward` overwrites `reward.pkl`. Record a new line on the map you will train. Do not reuse the tmrl-test pickle.

## LIDAR checklist (any map)

- Solo / local load, windowed, title `Trackmania`
- Cockpit camera, car hidden, LIDAR view (**L** on this machine)
- Plain asphalt with **black borders** the whole way
- Ice, dirt, water, or borderless platform → skip, or use FULL camera (different experiment)
- Human lap well under ~50 s (episode cap is 1000 steps at 20 Hz) unless you raise `ep_max_length`

## Config (`TmrlData/config/config.json`)

| Key | Summer 2026-01 example | Later maps |
| --- | --- | --- |
| `RUN_NAME` | `SAC_4_LIDAR_summer2026_01` | new unique name |
| `RESET_TRAINING` | `true` on first launch, then `false` so a crash resumes | same |
| `SAVE_MODEL_EVERY` | `10` (already set) | keep |
| `WANDB_ENTITY` / `WANDB_PROJECT` | `models-acme` / `tmrl` | leave; new `RUN_NAME` is a new wandb run |
| `ENV.RTGYM_INTERFACE` | `TM20LIDAR` | `TM20FULL` only if you switch cameras |
| `save_replays` | `false` overnight | keep false |

## Record reward, then train

1. Load **Summer 2026-01** (or the next map) solo, windowed, camera as above. Click the game.
2. Record a **new** reward line (this overwrites `reward.pkl`):

```powershell
C:\Users\cursor\tmrl\.venv\Scripts\python.exe -m tmrl --record-reward --use-keyboard
```

E to start, drive a full lap (slow is fine), finish or Q.

3. Keep a named copy **before** you train anything else:

```powershell
Copy-Item "$env:USERPROFILE\TmrlData\reward\reward.pkl" "$env:USERPROFILE\TmrlData\reward\tracks\summer-2026-01.pkl"
```

For another map, use that map’s slug (`campaign-03.pkl`, `editor-foo.pkl`, …).

4. `python -m tmrl --check-environment` with the game still on that map.
5. Set `RUN_NAME` and `RESET_TRAINING: true`. Start the three processes. **Redirect trainer stdout** — tmrl does not write `trainer.log` itself:

```powershell
$py = "C:\Users\cursor\tmrl\.venv\Scripts\python.exe"
$run = "C:\Users\cursor\tmrl\reports\summer-2026-01\SAC_4_LIDAR_summer2026_01"
New-Item -ItemType Directory -Force $run | Out-Null

# terminal 1
& $py -m tmrl --server

# terminal 2 — keep this file; HTML can be rebuilt from it
& $py -m tmrl --trainer --wandb *>&1 | Tee-Object -FilePath "$run\trainer.log"

# terminal 3
& $py -m tmrl --worker
```

Click TrackMania after the worker starts. After the first epoch, set `RESET_TRAINING` back to `false`.

To train a different map later: copy `tracks\<slug>.pkl` back onto `reward.pkl`, change `RUN_NAME`, record if you do not already have a pickle, `--new-run` on the first report.

## Reports (on demand, not on a timer)

`--track` is required. Output is `reports/<track-slug>/<run-name>/`. That folder is the archive: `trainer.log`, `config.snapshot.json` (secrets stripped), `reward.pkl`, `manifest.json`, `state.json` (watermark), `slices/report-NN.json`. HTML is a view.

First page of a **from-scratch** train:

```powershell
cd C:\Users\cursor\tmrl\learning-story
python progress_report.py `
  ..\reports\summer-2026-01\SAC_4_LIDAR_summer2026_01\trainer.log `
  --track "Summer 2026-01" `
  --run-name SAC_4_LIDAR_summer2026_01 `
  --new-run
```

Later, same train, only new trainer rounds:

```powershell
python progress_report.py `
  ..\reports\summer-2026-01\SAC_4_LIDAR_summer2026_01\trainer.log `
  --track "Summer 2026-01" `
  --run-name SAC_4_LIDAR_summer2026_01
```

Snapshot log/config/reward **without** writing HTML or moving the watermark (skip pages for days, then generate report 04 from the new slice):

```powershell
python progress_report.py `
  ..\reports\summer-2026-01\SAC_4_LIDAR_summer2026_01\trainer.log `
  --track "Summer 2026-01" `
  --run-name SAC_4_LIDAR_summer2026_01 `
  --data-only
```

Same commands with `--track "tmrl-test"` or `--track "Campaign 03"` after you swap `reward.pkl` and `RUN_NAME`.

Titles: `TrackMania training progress: Summer 2026-01 · Report 03 · Attempts 24 · Episode 412`.

- **Attempts** = trainer rounds in that slice (what the log counts), not laps.
- **Episode** = mean `episode_length_train` in steps. Toward 1000 ≈ finishing. Not an official lap time.

`--label` still replaces the whole H1 if you want a custom name.

## After training

1. Keep the final `.tmod` and any `SAVE_MODEL_EVERY` snapshots (paths are in `manifest.json`).
2. Capture with the **same** map loaded: `python record_fastest_lap.py --episodes 5`.
3. Do not point `--test` at shakedown weights while `RUN_NAME` is the new run.
4. Compare to shakedown in the README. Do not delete shakedown artifacts.
