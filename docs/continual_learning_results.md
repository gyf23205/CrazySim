# Continual learning results: circle → rectangle → figure8

Results of six methods trained on the task sequence circle → rectangle → figure8 in this repository
(single seed, 2026-10-21/22). All numbers come from the `session.json` file of each run unless stated otherwise.

## Setup

- **Robot and control.** Crazyflie 2.x in Isaac Lab; the policy sets the four motor thrusts directly at 100 Hz.
- **Tasks.** Follow a reference point that moves around a closed curve in a horizontal plane at 1 m height, one loop
  every 6 s:
  - circle: radius 0.5 m;
  - rectangle: 1.0 × 0.6 m, constant speed;
  - figure 8: lemniscate of Gerono, half-widths 0.6 × 0.3 m.
- **Reward.** Mainly `10 · (1 − tanh(e / 0.25 m))` per second, where `e` is the distance to the reference point. Small
  penalties on velocity error, angular velocity, yaw and action change. Episodes last 10 s.
- **Training budget.** 1000 PPO iterations per task (24 steps × 4096 environments each).
- **Evaluation after each task.** The stage's deterministic policy is flown for one full episode in 4096 environments
  on every task. Reported:
  - *error*: mean distance to the reference until the episode ends, in cm;
  - *crash rate*: share of episodes ending by a crash, a flip, or drifting more than 1 m from the reference.

### Observation modes

The tasks were run with two ways of telling the policy where the reference is:

| mode | reference information in the observation | tasks different for the controller? |
|---|---|---|
| `lookahead` | the reference point now and 0.2, 0.4, …, 1.0 s ahead | **no**: following the next 1 s of path works on any shape |
| `setpoints` (current default) | 16 fixed points spread over the whole loop, plus the loop phase (sin, cos) | **yes**: the policy has to find its target from the shape and the phase |

The **setpoints** results are the main continual-learning result. The lookahead results are kept only to show why
that mode was replaced.

### Methods

| method | first task | each later task | kept from earlier tasks |
|---|---|---|---|
| `finetune` | MLP policy (PPO) | continues training the agent with the best checkpoint of the previous task\* | everything (one policy) |
| `scratch` | MLP policy | new MLP policy | nothing |
| `residual` | MLP policy | frozen previous policy + new residual MLP (output starts at 0) | all earlier policies, frozen inside the new one |
| `componet` | MLP policy | new CompoNet unit attending over the frozen earlier units | all earlier units, frozen |
| `kae_moe` | MLP policy | gate between a mixture of the eigen-mode experts of the earlier tasks' Koopman autoencoders (KAEs) and a new MLP | one frozen KAE fitted to each earlier policy |
| `fame` | fast learner | new fast learner, initialized from the meta learner when that scores higher; the meta learner is distilled from the fast learner after every task | meta learner (distilled policy) + states from earlier tasks |

\* "Best" means skrl's checkpoint with the highest mean training episode return, checked every 20 iterations. In the
lookahead runs, finetune still carried the last checkpoint.

**Shared settings and resets:**
- `finetune`, `scratch`, `residual`, `componet` and `kae_moe` use the same skrl PPO settings and differ only in the
  policy network.
- `residual`, `componet` and `kae_moe` start every task with a new value network and the initial action noise
  (std 0.61).
- `fame` uses its own PPO variant, as in the reference implementation: tanh-squashed actions, state-dependent action
  noise, and an entropy weight tuned automatically.

## Main result: setpoint observations

### Error on each task right after training on it

Policy saved at that stage, evaluated on the task it was just trained on. Cells give error in cm; the crash rate
was 0 in every case.

| method | circle | rectangle | figure8 |
|---|---|---|---|
| finetune | 1.7 | 6.7 | 3.7 |
| scratch | 1.7 | 5.2 | 15.9 |
| residual | 1.7 | 13.2 | 18.7 |
| componet | 1.7 | 23.5 | 17.9 |
| kae_moe | 1.7 | 3.9 | 2.9 |
| fame (fast learner) | **0.72** | **0.82** | **0.82** |

### Final policy (after training on figure8) on all tasks

Cells give error in cm, then crash rate where above 0.

| method | circle | rectangle | figure8 | mean error |
|---|---|---|---|---|
| finetune | 49.2, 100% | 42.4 | 3.7 | 31.8 |
| scratch (figure8 policy) | 54.5, 100% | 50.8, 2% | 15.9 | 40.4 |
| residual | 51.8, 100% | 46.0 | 18.7 | 38.8 |
| componet | 46.1, 100% | 47.4 | 17.9 | 37.1 |
| kae_moe | 52.8, 100% | 47.4 | 2.9 | 34.3 |
| fame, fast learner | 41.6, 100% | 43.7, 9% | 0.82 | 28.7 |
| **fame, meta learner**\*\* | **0.95** | **2.2** | **0.89** | **1.35** |

\*\* From `scripts/skrl/evaluate.py` on `2_figure8/meta_policy.pt`, with 1024 environments instead of 4096.

### Full transfer matrices

Rows: policy after training through that task. Columns: task evaluated on. Cells give error in cm, then crash rate
where above 0.

| method | after | circle | rectangle | figure8 |
|---|---|---|---|---|
| finetune | circle | 1.7 | 29.1 | 44.0, 100% |
| | rectangle | 29.8 | 6.7 | 44.7, 10% |
| | figure8 | 49.2, 100% | 42.4 | 3.7 |
| scratch | circle | 1.7 | 29.1 | 44.0, 100% |
| | rectangle | 27.0 | 5.2 | 45.6, 1% |
| | figure8 | 54.5, 100% | 50.8, 2% | 15.9 |
| residual | circle | 1.7 | 29.1 | 44.0, 100% |
| | rectangle | 34.8 | 13.2 | 49.1 |
| | figure8 | 51.8, 100% | 46.0 | 18.7 |
| componet | circle | 1.7 | 29.1 | 44.0, 100% |
| | rectangle | 40.8, 23% | 23.5 | 40.6, 100% |
| | figure8 | 46.1, 100% | 47.4 | 17.9 |
| kae_moe | circle | 1.7 | 29.1 | 44.0, 100% |
| | rectangle | 27.3 | 3.9 | 47.3 |
| | figure8 | 52.8, 100% | 47.4 | 2.9 |
| fame (fast) | circle | 0.72 | 28.4, 5% | 30.4, 100% |
| | rectangle | 29.1 | 0.82 | 35.0, 43% |
| | figure8 | 41.6, 100% | 43.7, 9% | 0.82 |
| fame (meta)\*\* | circle | 2.0 | 26.5 | 58.8 |
| | rectangle | 1.1 | 2.8 | 40.6, 10% |
| | figure8 | 0.95 | 2.2 | 0.89 |

### Method details

- **finetune.** The best checkpoints came from timesteps 24000 (end), 14880 and 21120. On figure8, training ended at
  20.3 cm, but the carried best checkpoint evaluated at 3.7 cm.
- **kae_moe.** The KAEs explain 99.9%, 99.6% and 97.0% of the variation in their policy's actions on visited states,
  and took 24, 22 and 18 min to fit. During training the gate shifts toward the KAE pathway.
- **fame.** At both task switches the meta learner was chosen over a fresh learner (detection return 14.9 vs 0.64 on
  the rectangle, 3.8 vs 0.38 on the figure 8).

## Findings

1. **The tasks interfere.** A policy trained on one task misses the others by 27–55 cm and often crashes. Every method
   whose deployed policy is a single network trained on the latest task forgets the earlier tasks: the circle error
   rises from 0.7–1.7 cm to 42–55 cm with 100% crashes.
2. **Only FAME's meta learner retains all tasks in one policy:** 0.95 / 2.2 / 0.89 cm after the whole sequence. Its
   distillation, which matches the previous meta learner on stored states from earlier tasks, is what keeps old tasks.
3. **Freezing earlier networks does not protect the final policy.** `residual`, `componet` and `kae_moe` keep the
   earlier stages' networks exactly. Nothing makes the new trainable part hand control back to an old network on an
   old task, although the setpoints identify the task. If the evaluation protocol lets each task use the policy saved
   at its own stage (task identity known at test time), these methods retain the earlier tasks exactly by
   construction.
4. **Forward transfer.** Compared with `scratch` on the later tasks:
   - `kae_moe` (3.9 / 2.9 cm) and `finetune` (6.7 / 3.7 cm) learn the figure 8 much better than scratch (15.9 cm);
   - `residual` (13.2 / 18.7 cm) and `componet` (23.5 / 17.9 cm) are worse than scratch.
5. **FAME's lead on the current task is partly its PPO variant.** Its fast learner reaches 0.7–0.8 cm on every task,
   against 1.7–23.5 cm for the methods on the shared PPO settings. The shared settings train unstably on setpoint
   observations (scratch's figure-8 training error swung between 2.4 and 16 cm). The per-task numbers therefore do not
   isolate the transfer mechanisms.

## Contrast: lookahead observations

Same sequence with the lookahead observation. These runs predate two changes: finetune carried the last checkpoint,
and FAME did not yet have the numerical fix described under *Implementation notes*. Final policy after figure8:

| method | circle | rectangle | figure8 |
|---|---|---|---|
| finetune | 0.64 | 1.2 | 0.36 |
| scratch (figure8 policy) | 1.8 | 2.4 | 2.2 |
| residual | 2.5 | 2.8 | 2.7 |
| componet | 3.0 | 3.3 | 1.9 |
| kae_moe | 2.0 | 2.4 | 1.8 |
| fame, fast learner | 0.61 | 1.2 | 0.38 |

No crashes, and no forgetting: every final policy flies every task within 3.3 cm. Even the circle-only policy flew
the other tasks within 8.1 cm without crashing. With lookahead observations the tasks are
effectively one task for the controller, so this mode is not suitable for studying continual learning.

## Caveats

- **One seed.** Differences of a few cm between methods are not established.
- **Run times are not comparable.** Most runs shared the GPU with two others. Total times: finetune 65 min, scratch 53,
  residual 55, componet 56, kae_moe 94 (KAE fitting is 64 of those minutes), fame 66 (partly run alone).
- **The reward weights were set by judgment and are untuned.** The rectangle's instantaneous 90° turns put a floor on
  its error for every method.
- **`kae_moe` does not start from the earlier policy.** It starts at about 1/16 of the earlier policy's action, because
  the mode weights are L1-normalized. This is inherited from the reference implementation.
- **The KAE is not a true Koopman model here.** The KAE target (4 action values padded with 65 ones) is nearly
  constant, so the latent consistency `K·z_x ≈ z_y` carries little information.

## Implementation notes

- The methods are ported from `less_leg_walking_1`; each file in `source/crazysim/crazysim/methods/` lists its
  differences from the reference.
- One change fixes a crash the reference FAME shares. Its PPO update recomputed the probability of a saturated
  (tanh = ±1) action through atanh. For a pre-squash mean of 10 and noise 0.0067, that gives a log-probability of about
  −84,000 instead of 22; the probability ratio overflows and the parameters become NaN. The update now uses the stored
  pre-squash sample.

## Reproduction

```bash
for m in finetune scratch residual componet kae_moe fame; do
  python scripts/skrl/train.py --tasks circle rectangle figure8 --method $m --headless --run_name setpoints_all
done
# FAME meta learner on all tasks
python scripts/skrl/evaluate.py --headless --checkpoint logs/skrl/crazyflie_tracking/<fame session>/*/meta_policy.pt
# lookahead contrast
python scripts/skrl/train.py --tasks circle rectangle figure8 --method <m> --headless env.observation_mode=lookahead
```

Runs used here, under `logs/skrl/crazyflie_tracking/` (logs are not in git):

| method | setpoints | lookahead |
|---|---|---|
| finetune | `2026-10-22_17-03-28_finetune_circle-rectangle-figure8_setpoints_all` | `2026-10-21_19-05-18_finetune_circle-rectangle-figure8_full` |
| scratch | `2026-10-22_18-39-28_scratch_circle-rectangle-figure8_setpoints_all` | `2026-10-21_19-05-18_scratch_circle-rectangle-figure8_full` |
| residual | `2026-10-22_18-39-28_residual_circle-rectangle-figure8_setpoints_all` | `2026-10-21_19-05-18_residual_circle-rectangle-figure8_full` |
| componet | `2026-10-22_18-39-28_componet_circle-rectangle-figure8_setpoints_all` | `2026-10-21_19-46-43_componet_circle-rectangle-figure8_full` |
| kae_moe | `2026-10-22_17-03-27_kae_moe_circle-rectangle-figure8_setpoints_all` | `2026-10-21_19-46-43_kae_moe_circle-rectangle-figure8_full` |
| fame | `2026-10-22_19-09-55_fame_circle-rectangle-figure8_setpoints_all` | `2026-10-21_19-46-43_fame_circle-rectangle-figure8_full` |
