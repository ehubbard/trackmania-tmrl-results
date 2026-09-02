# Learning results

Notebooks from three experiments: a [LIDAR SAC](https://github.com/trackmania-rl/tmrl) agent on **TrackMania 2020**, a PPO + Nature CNN agent on **Gymnasium Space Invaders**, and a PPO + Nature CNN agent on **NES Super Mario Bros**.

**[Open the reports](https://ehubbard.github.io/trackmania-tmrl-results/)** · **[Live stream](https://www.twitch.tv/northeast22)** · TrackMania numbers on [wandb](https://wandb.ai/models-acme/tmrl)

## Super Mario Bros

| Game | Run | What happened |
| --- | --- | --- |
| Super Mario Bros | [v3_ppo_mario_full_fresh](https://ehubbard.github.io/trackmania-tmrl-results/mario/v3_ppo_mario_full_fresh/) | First 100 million PPO steps on the full NES cart. World 1 cleared in order; World 2 open. Explainer: [How the AI learns](https://ehubbard.github.io/trackmania-tmrl-results/mario/). |

On that page: greedy vs the 16 learners, why train return is not skill, the 1-3 pit farm, sequential cart %, and the usual questions (why the video is fast, why it looks like it forgot).

## Space Invaders

| Game | Run | What happened |
| --- | --- | --- |
| Space Invaders | [v1_ppo_invaders](https://ehubbard.github.io/trackmania-tmrl-results/space-invaders/v1_ppo_invaders/) | PPO + CNN on ALE Space Invaders. Snapshot ~1/3 of a 100M-step train. Watch live: [twitch.tv/northeast22](https://www.twitch.tv/northeast22). |

On that page: learning loop, training-mean curve vs env steps, last finished window games, entropy, and 2600 scoring. Overlay last-score waits for the on-screen game to end.

## TrackMania 2020

| Map | Run | What happened |
| --- | --- | --- |
| tmrl-test | [SAC_4_LIDAR_train](https://ehubbard.github.io/trackmania-tmrl-results/tmrl-test/SAC_4_LIDAR_train/) | Shakedown overnight (~12.4h). Peak return ~320. Later capture: 47.45s finished lap. |
| Summer 2026-01 | [SAC_4_LIDAR_summer2026_01](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_LIDAR_summer2026_01/) | Learned, then collapsed into a crash loop. |
| Summer 2026-01 | [SAC_4_LIDAR_summer2026_01_v2](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_LIDAR_summer2026_01_v2/) | Restart from the peak snapshot with an empty replay buffer. First ~30 epochs. |
| Summer 2026-01 | [SAC_4_FULL_summer2026_01](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_FULL_summer2026_01/) | Switched to FULL camera. |

On a TrackMania report page, **Attempts** are trainer rounds (not laps). **Episode** is mean length in steps; 1000 is the cap and usually means a finish.
