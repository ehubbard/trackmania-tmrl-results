# TrackMania tmrl results

A [LIDAR SAC](https://github.com/trackmania-rl/tmrl) agent learning to drive **TrackMania 2020**. These pages are the notebooks for that experiment.

**[Open the reports](https://ehubbard.github.io/trackmania-tmrl-results/)** · live numbers on [wandb](https://wandb.ai/models-acme/tmrl)

| Map | Run | What happened |
| --- | --- | --- |
| tmrl-test | [SAC_4_LIDAR_train](https://ehubbard.github.io/trackmania-tmrl-results/tmrl-test/SAC_4_LIDAR_train/) | Shakedown overnight (~12.4h). Peak return ~320. Later capture: 47.45s finished lap. |
| Summer 2026-01 | [SAC_4_LIDAR_summer2026_01](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_LIDAR_summer2026_01/) | Learned, then collapsed into a crash loop. |
| Summer 2026-01 | [SAC_4_LIDAR_summer2026_01_v2](https://ehubbard.github.io/trackmania-tmrl-results/summer-2026-01/SAC_4_LIDAR_summer2026_01_v2/) | Restart from the peak snapshot with an empty replay buffer. First ~30 epochs. |

On a report page, **Attempts** are trainer rounds (not laps). **Episode** is mean length in steps; 1000 is the cap and usually means a finish.
