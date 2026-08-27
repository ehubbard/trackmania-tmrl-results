# Learning results

Notebooks from two experiments: a [LIDAR SAC](https://github.com/trackmania-rl/tmrl) agent on **TrackMania 2020**, and a PPO + Nature CNN agent on **Gymnasium Space Invaders**.

**[Open the reports](https://ehubbard.github.io/trackmania-tmrl-results/)** · TrackMania live numbers on [wandb](https://wandb.ai/models-acme/tmrl)

## Space Invaders

| Game | Run | What happened |
| --- | --- | --- |
| Space Invaders | [v1_ppo_invaders](https://ehubbard.github.io/trackmania-tmrl-results/space-invaders/v1_ppo_invaders/) | PPO + CNN on ALE Space Invaders. Snapshot ~1/3 of a 100M-step train. Capture-window peak 2825. Training mean is the eight exploring envs; the window is greedy play. |

On that page: learning loop, training-mean curve vs env steps, last finished window games, entropy, and 2600 scoring. Overlay last-score waits for the on-screen game to end.

## TrackMania 2020

| Map | Run | What happened |
| --- | --- | --- |
| tmrl-test | [SAC_4_LIDAR_train](https://ehubbard.github.io/trackmania-tmrl-results/tmrl-test/SAC_4_LIDAR_train/) | Shakedown overnight (~12.4h). Peak return ~320. Later capture: 47.45s finished lap. |
| Summer 2026-01 | [SAC_4_LIDAR_summer2026_01](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_LIDAR_summer2026_01/) | Learned, then collapsed into a crash loop. |
| Summer 2026-01 | [SAC_4_LIDAR_summer2026_01_v2](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_LIDAR_summer2026_01_v2/) | Restart from the peak snapshot with an empty replay buffer. First ~30 epochs. |
| Summer 2026-01 | [SAC_4_FULL_summer2026_01](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_FULL_summer2026_01/) | Switched to FULL camera. |

On a TrackMania report page, **Attempts** are trainer rounds (not laps). **Episode** is mean length in steps; 1000 is the cap and usually means a finish.
