# Under the hood

Short version of what this experiment actually is. For the library itself, see [tmrl](https://github.com/trackmania-rl/tmrl).

## Game

TrackMania 2020 with Club Access. User data lives under `Documents\Trackmania2020`, not `Documents\Trackmania`. Window about 958×488. OpenPlanet plugin `TMRL_GrabData` must be loaded. LIDAR still needs black-border asphalt; Club does not waive that. Each map gets its own `reward.pkl` and `RUN_NAME`.

## Observation (LIDAR)

Each step:

1. Screenshot the TrackMania window.
2. Cast 19 rays from a cockpit road point until they hit dark track border.
3. Read speed (and position / finished flag) from OpenPlanet on localhost.

The policy never sees the full frame. If the camera is chase-cam or the car is visible, the rays are junk. Camera 3 is the LIDAR view; on this machine that binding is **L**, not the number-row 3.

## Action

Virtual Xbox 360 pad via ViGEm: forward, back, steer. Keyboard is a fallback. Click the game window so tmrl’s window finder (`title = Trackmania`) actually drives that instance.

## Reward

A pickle of a human-ish trajectory (`TmrlData/reward/reward.pkl`) is the track spine. tmrl always reads that **one** path, so keep named copies under `TmrlData/reward/tracks/<slug>.pkl` and restore before you train that map. Reward is progress along that polyline. Too far off, or no progress for a countdown, ends the episode. Finish line adds 100. Constant penalty is 0 in our config. `ep_max_length` 1000 × 0.05 s ≈ 50 s.

## Learning

SAC (Soft Actor-Critic), small MLP. Trainer on CUDA, inference on CPU. Replay buffer grows toward hundreds of thousands of samples. `UPDATE_MODEL_INTERVAL` 1000 training steps ≈ one new policy per round, 10 rounds per epoch.

One worker unless you add another **game**. `NB_WORKERS: -1` only means the server will accept more; it does not spawn them.

## What we log

tmrl prints `epoch`, `round`, `return_train`, `episode_length_train`, `memory_len`, losses. `--wandb` ships those live. This repo’s parser reads the same stdout (including the padded `=== epoch 10/10000  round` header that naive regexes skip) and builds an HTML story with **epoch** on the x-axis.
