# Continual learning results: five tasks

Results of six methods trained on the task sequence circle → rectangle → figure8 → star → vertical_circle with
setpoint observations (single seed, 2026-10-23/24). This extends
[continual_learning_results.md](continual_learning_results.md), which describes the setup, the methods and the
three-task results; only the differences are described here. All numbers come from the `session.json` file of each
run unless stated otherwise.

## What is new

- **Two tasks**, appended to the sequence. Both run once every 6 s around the point 1 m above the environment origin:
  - star: five-pointed star outline flown at constant speed (0.60 m/s); tips 0.5 m and inner corners 0.2 m from the
    center, so each tip is an acute 36° corner;
  - vertical_circle: circle of radius 0.5 m in the vertical x-z plane (height 0.5 to 1.5 m). It is the only task that
    needs height changes.
- **Checkpoint selection.** Two variants were run:
  - **best** (main result, sessions `*_setpoints_best`): every method keeps the *best checkpoint* of each task. It is
    the stage's evaluated policy and what the next task builds on: the base of `residual`, the unit of `componet`, the
    policy the KAE of `kae_moe` is fitted to, the fast learner that `fame` distills into its meta learner. The best
    checkpoint is skrl's checkpoint with the highest mean training episode return. Candidates come every 20 iterations,
    each scored by the episodes that ended in the 10 iterations before it. For `fame`, the states used for the
    distillation are re-collected with the best fast learner when the best checkpoint is not the final one.
  - **last** (sessions `*_star-vertical_circle_setpoints_all`): as in the three-task results, only `finetune` keeps
    the best checkpoint; the other methods keep the agent at the end of training. These runs continued the three-task
    sessions with `train.py --resume` (see *Caveats*).
- `scratch` was not rerun with the best checkpoint; its numbers are from the **last** variant. It carries nothing
  between tasks, so only its own per-task numbers could change.

## Main result: best checkpoint

### Error on each task right after training on it

Policy saved at that stage, evaluated on the task it was just trained on. Cells give error in cm, then crash rate
where above 0. In brackets: the **last** variant, where it differs by more than 0.5 cm.

| method | circle | rectangle | figure8 | star | vertical_circle |
|---|---|---|---|---|---|
| scratch (last) | 1.7 | 5.2 | 15.9 | 7.6 | 2.8 |
| finetune | 1.7 | 6.7 | 3.7 | 7.8 | 8.0 |
| residual | 1.7 | 10.7 (13.2) | 15.5 (18.7) | 11.3 (19.4) | 4.1 (4.9) |
| componet | 1.7 | 9.3 (23.5) | 14.5 (17.9) | 12.8 (32.9, 100%) | 3.4 |
| kae_moe | 1.7 | 3.0 (3.9) | 3.0 | 3.9 (4.8) | 0.75 (1.7) |
| fame (fast learner) | **0.51** | **0.94** | **1.1** | **1.3** | **0.47** |

### Final policy (after training on vertical_circle) on all tasks

Cells give error in cm, then crash rate where above 0.

| method | circle | rectangle | figure8 | star | vertical_circle | mean error | mean error, last |
|---|---|---|---|---|---|---|---|
| scratch (last) | 44.2, 3% | 45.4, 7% | 56.4, 41% | 41.3 | 2.8 | 38.0 | 38.0 |
| finetune | 43.2 | 41.9 | 57.8 | 38.3 | 8.0 | 37.8 | 37.8 |
| residual | 44.5 | 44.3 | 61.4, 4% | 39.9 | 4.1 | 38.8 | 36.5 |
| componet | 44.3 | 44.9 | 64.8 | 42.1 | 3.4 | 39.9 | 37.9 |
| kae_moe | 43.2, 15% | 44.8 | 51.1, 40% | 40.9, 4% | 0.75 | 36.1 | 37.0 |
| fame, fast learner | 25.8, 59% | 24.4, 69% | 23.7, 81% | 31.7, 38% | 0.47 | 21.2 | 27.5 |
| **fame, meta learner**\* | **0.62** | **3.3** | **1.3** | **3.4** | **0.49** | **1.82** | 1.92 |

\* From `scripts/skrl/evaluate.py` on `4_vertical_circle/meta_policy.pt`, with 1024 environments instead of 4096
(`meta_policy_eval.json` in the session directory).

### Full transfer matrices

Rows: policy after training through that task. Columns: task evaluated on. Cells give error in cm, then crash rate
where above 0.

| method | after | circle | rectangle | figure8 | star | vertical_circle |
|---|---|---|---|---|---|---|
| finetune | circle | 1.7 | 29.1 | 44.0, 100% | 16.9 | 42.6, 27% |
|  | rectangle | 29.8 | 6.7 | 44.7, 10% | 27.5 | 44.4, 2% |
|  | figure8 | 49.2, 100% | 42.3 | 3.7 | 49.4 | 62.4, 1% |
|  | star | 15.7 | 24.2 | 48.4, 97% | 7.8 | 42.6 |
|  | vertical_circle | 43.2 | 41.9 | 57.8 | 38.3 | 8.0 |
| residual | circle | 1.7 | 29.1 | 44.0, 100% | 16.9 | 42.6, 27% |
|  | rectangle | 32.6 | 10.7 | 41.0 | 30.0 | 43.5 |
|  | figure8 | 66.7 | 45.8 | 15.5 | 57.1 | 61.5, 23% |
|  | star | 14.0 | 29.7 | 50.9, 55% | 11.3 | 42.3 |
|  | vertical_circle | 44.5 | 44.3 | 61.4, 4% | 39.9 | 4.1 |
| componet | circle | 1.7 | 29.1 | 44.0, 100% | 16.9 | 42.6, 27% |
|  | rectangle | 23.9 | 9.3 | 48.8 | 23.0 | 42.1 |
|  | figure8 | 60.6, 13% | 44.3 | 14.5 | 50.8 | 60.6 |
|  | star | 12.9 | 28.9 | 44.0, 100% | 12.8 | 42.5 |
|  | vertical_circle | 44.3 | 44.9 | 64.8 | 42.1 | 3.4 |
| kae_moe | circle | 1.7 | 29.1 | 44.0, 100% | 16.9 | 42.7, 26% |
|  | rectangle | 29.0 | 3.0 | 46.5 | 27.7 | 43.0 |
|  | figure8 | 51.4, 100% | 44.7 | 3.0 | 51.3, 11% | 62.5, 3% |
|  | star | 15.7 | 26.1 | 48.6, 41% | 3.9 | 42.4 |
|  | vertical_circle | 43.2, 15% | 44.8 | 51.1, 40% | 40.9, 4% | 0.75 |
| fame (fast) | circle | 0.51 | 22.1, 52% | 17.9, 100% | 16.3 | 40.4, 31% |
|  | rectangle | 28.0 | 0.94 | 33.5, 48% | 27.0 | 39.6, 58% |
|  | figure8 | 39.0, 100% | 46.2, 4% | 1.1 | 45.1, 100% | 46.2, 100% |
|  | star | 16.1 | 26.0, 10% | 31.6, 59% | 1.3 | 40.3, 43% |
|  | vertical_circle | 25.8, 59% | 24.4, 69% | 23.7, 81% | 31.7, 38% | 0.47 |
| fame (meta)\* | circle | 0.53 | 23.4, 43% | 20.5, 100% | 16.2 | 40.1, 33% |
|  | rectangle | 0.69 | 3.5 | 34.0, 100% | 16.4 | 45.7, 33% |
|  | figure8 | 0.66 | 2.8 | 1.3 | 16.9 | 45.2, 23% |
|  | star | 0.62 | 3.8, 2% | 1.3 | 2.2 | 40.1, 42% |
|  | vertical_circle | 0.62 | 3.3 | 1.3 | 3.4 | 0.49 |

### Method details

- **Where the best checkpoints are** (timestep of 24000 per task, tasks in order):

  | method | circle | rectangle | figure8 | star | vertical_circle |
  |---|---|---|---|---|---|
  | finetune | 24000 | 14880 | 21120 | 19680 | 16320 |
  | residual | 24000 | 4320 | 8160 | 2400 | 24000 |
  | componet | 24000 | 5280 | 5760 | 3360 | 4800 |
  | kae_moe | 24000 | 21120 | 19200 | 24000 | 24000 |
  | fame | 23040 | 14880 | 23520 | 21600 | 23040 |

  On most later tasks, the training return of `residual` and `componet` peaks in the first 10–35% of the task and
  then declines.
- **kae_moe.** The KAEs explain 99.9%, 99.6%, 98.7%, 99.2% and 99.9% of the variation in their policy's actions on
  visited states, and took 36–42 min each to fit (the GPU was shared).
- **fame.** At the four task switches the fast learner started from the meta learner, a fresh learner, the meta
  learner and the meta learner (detection return, meta vs fresh: 1.9 vs 0.64 on the rectangle, −0.73 vs 0.38 on the
  figure 8, 22.3 vs 1.0 on the star, 1.0 vs 0.74 on the vertical circle).

## Findings

1. **The new tasks are new to every controller.** Before training on them, the figure8 policies miss the star by
   45–57 cm and the vertical circle by 46–62 cm. The vertical circle is the most distinct task: every policy trained only
   on horizontal curves misses it by 40 cm or more. The star is closest to the circle: the circle policies fly it at
   16–17 cm.
2. **The best checkpoint mainly helps `residual` and `componet` on the current task.** `componet`'s star stage, which
   collapsed at the end of training in the last variant (32.9 cm, 100% crashes), reaches 12.8 cm; its rectangle error
   drops from 23.5 to 9.3 cm. Even so, both remain worse than `scratch` on the rectangle and the star. Their training
   return peaks early in most tasks, so their training is unstable, not just their final checkpoint.
3. **Forgetting is unchanged.** Every final policy except FAME's meta learner misses the earlier tasks by 38–65 cm
   (fast learner: 24–32 cm, with 38–81% crashes); the mean final error moves by at most 2.3 cm between variants for
   the PPO methods.
4. **FAME's meta learner again retains every task in one policy:** within 3.4 cm on all five tasks, with no crashes.
   After each task, the meta learner flies that task and the earlier ones within 3.8 cm (with 2% crashes on the
   rectangle after the star). Compared with the last
   variant it is better on the vertical circle (0.49 vs 2.2 cm) and worse on the rectangle (3.3 vs 1.9 cm); with one
   seed these differences are not established.
5. **Forward transfer on the new tasks.** Compared with `scratch` (star 7.6 cm, vertical circle 2.8 cm):
   - `kae_moe` (3.9 / 0.75 cm) and FAME's fast learner (1.3 / 0.47 cm) are better on both;
   - `finetune` matches scratch on the star (7.8 cm) and is worse on the vertical circle (8.0 cm);
   - `residual` (11.3 / 4.1 cm) and `componet` (12.8 / 3.4 cm) are worse on both.

## Caveats

- **One seed.** Differences of a few cm between methods are not established.
- **`scratch` uses the last checkpoint.** Its per-task numbers would likely improve with the best checkpoint (its
  figure-8 training error swung between 2.4 and 16 cm in the three-task runs).
- **The last variant was resumed, not run in one go.** Its star and vertical_circle stages continued the three-task
  sessions in a new process (`--resume`), so the random-number state differs from an uninterrupted run. `finetune`
  gives the same numbers in the resumed and the uninterrupted run (within 0.1 cm), which supports that the resume
  restores the method state. For `fame`, the buffer of earlier-task states is not saved; it was refilled by flying
  each earlier stage's fast learner on its task.
- **FAME's earlier best-checkpoint record was wrong.** Its agent persists across tasks, and skrl's best-checkpoint
  record was not reset between tasks: a task's `best_agent.pt` was written only if its training return beat every
  earlier task. This did not affect earlier results, which did not use `best_agent.pt`; it is fixed for the best
  variant.
- **Selection uses the training return.** The best checkpoint is chosen from the mean return of the stochastic
  policy during training, not from an evaluation of the deterministic policy.
- **Run times are not comparable.** Three runs shared the GPU with each other and with another user's jobs. Total
  times for the five tasks (best variant): finetune 225 min, residual 148, componet 169, kae_moe 361 (KAE fitting is
  190 of those minutes), fame 189.
- The caveats of the three-task results (untuned reward weights, `kae_moe` starting at about 1/16 of the earlier
  policy's action, the KAE target being nearly constant) still apply.

## Reproduction

```bash
# best variant: all five tasks in one run
for m in finetune residual componet kae_moe fame; do
  python scripts/skrl/train.py --tasks circle rectangle figure8 star vertical_circle --method $m --headless \
    --run_name setpoints_best
done
# FAME meta learner on all tasks
python scripts/skrl/evaluate.py --headless --checkpoint logs/skrl/crazyflie_tracking/<fame session>/*/meta_policy.pt \
  --output logs/skrl/crazyflie_tracking/<fame session>/meta_policy_eval.json
# last variant: continue a three-task session (made before every method kept the best checkpoint)
python scripts/skrl/train.py --tasks star vertical_circle --method <m> --headless --run_name setpoints_all \
  --resume logs/skrl/crazyflie_tracking/<three-task session>
```

The last variant was made with the code before the best-checkpoint change; with the current code, `--resume` keeps
the best checkpoint for every method.

Runs used here, under `logs/skrl/crazyflie_tracking/` (logs are not in git):

| method | best | last |
|---|---|---|
| finetune | `2026-10-24_11-36-42_finetune_circle-rectangle-figure8-star-vertical_circle_setpoints_best` | `2026-10-23_20-53-17_finetune_circle-rectangle-figure8-star-vertical_circle_setpoints_all` |
| scratch | - | `2026-10-24_00-28-21_scratch_circle-rectangle-figure8-star-vertical_circle_setpoints_all` |
| residual | `2026-10-24_15-39-18_residual_circle-rectangle-figure8-star-vertical_circle_setpoints_best` | `2026-10-23_23-40-25_residual_circle-rectangle-figure8-star-vertical_circle_setpoints_all` |
| componet | `2026-10-24_14-59-40_componet_circle-rectangle-figure8-star-vertical_circle_setpoints_best` | `2026-10-23_23-19-19_componet_circle-rectangle-figure8-star-vertical_circle_setpoints_all` |
| kae_moe | `2026-10-24_11-36-42_kae_moe_circle-rectangle-figure8-star-vertical_circle_setpoints_best` | `2026-10-23_20-53-17_kae_moe_circle-rectangle-figure8-star-vertical_circle_setpoints_all` |
| fame | `2026-10-24_11-36-42_fame_circle-rectangle-figure8-star-vertical_circle_setpoints_best` | `2026-10-23_20-53-17_fame_circle-rectangle-figure8-star-vertical_circle_setpoints_all` |

The last-variant sessions list the circle, rectangle and figure8 stages of the three-task sessions they continue
(`resumed_from` in `session.json`).
