# Continual learning results: waypoint control, five tasks

Results of seven methods trained on the task sequence circle → rectangle → figure8 → star → vertical_circle with
waypoint control (`CrazySim-TrackingWaypoint-Direct-v0`), single seed, 2026-11-18 to 2026-11-20. Observation
normalization is off for every method. The thrust-control results of the same sequence are in
[continual_learning_results_5tasks.md](continual_learning_results_5tasks.md); the waypoint environment and its
differences from `Drone_Test_TRO` are in [waypoint_control_differences.md](waypoint_control_differences.md). All
numbers come from the `session.json` file of each run unless stated otherwise.

## Setup

- **Control.** The policy observes the positions of the 16 setpoints of the reference curve relative to the drone and
  outputs a position command for a port of the Crazyflie firmware controller (see
  [waypoint_control_differences.md](waypoint_control_differences.md)). The policy is a tanh-squashed Gaussian: the
  network outputs the pre-tanh mean `u`, and the action is `tanh(u)`.
- **Tasks.** The five shapes of the thrust-control sequence, 1.5x larger, flown at 0.25 m/s. The vertical circle has
  radius 0.6 m around 1.0 m height. An episode is two laps (30.2 s for the vertical circle to 48.8 s for the figure 8; 37.7 s for the circle); evaluation episodes start at the first reference
  point with ±0.1 m position noise.
- **Training.** 500 PPO iterations per task (48000 environment steps, 4096 environments), seed 42. Every method keeps
  the best checkpoint of each task: the checkpoint with the highest mean training episode return.
- **Observation normalization: off** (`agent.state_preprocessor: null`, the default since 2026-11-20). `residual`,
  `componet` and `kae_moe` require it off, because they run frozen earlier policies on raw observations. Value
  normalization (of the value network's targets) is on for every method.
- **Metrics.** After each task, the saved policy flies one full episode in each of 4096 environments on every task.
  - **Error**: mean distance to the reference, in cm, over the steps before the episode ends.
  - **Crash rate**: fraction of episodes ended early: a safety limit was exceeded (height, horizontal distance, speed,
    acceleration, tilt, angular velocity) or the drone was more than 1.0 m from the reference. A crashed episode's
    error covers only the steps before the crash.

### Methods and what they carry to the next task

"Initial" means the component starts as on the first task: new random network weights, log standard deviation at
`initial_log_std` (0.0, so the noise standard deviation before the tanh is 1.0), a new Adam optimizer with the
configured learning rate, and a new value normalizer.

| method | policy mean network | log std of the action noise | value network + value normalizer | optimizer |
|---|---|---|---|---|
| `scratch` | initial | initial | initial | initial |
| `finetune` | carried | carried | carried | carried |
| `finetune_reset` | carried | initial | carried | initial |
| `residual`, `componet`, `kae_moe` | frozen earlier policies + initial trainable part | initial | initial | initial |
| `fame` | meta learner's if chosen, else initial | meta learner's if chosen, else initial | initial | initial |

`finetune_reset` is new (2026-11-20). It separates what `finetune` gains from the carried networks from what it gains
or loses from the carried training state (a noise level already shrunk on the earlier task, stale Adam moments).

`fame` keeps two policies: the **fast learner**, a PPO agent trained on each task, and the **meta learner**, distilled
from the fast learners at the end of every task. At each task switch, the deterministic fast and meta learners are
compared over one episode on the new task, and the better one initializes the fast learner. "fame (fast)" below is
the stage's fast learner; "fame (meta)" is the meta learner after that stage, evaluated with
`scripts/skrl/evaluate.py` (1024 environments, `meta_policy_eval.json` in the session directory).

## Main result

### Error on each task right after training on it

Policy saved at that stage, evaluated on the task it was just trained on. Cells give error in cm, then crash rate
where above 0.

| method | circle | rectangle | figure8 | star | vertical_circle |
|---|---|---|---|---|---|
| scratch | 6.9 | 12.4 | 28.0 | 19.3 | 7.7 |
| finetune | 6.9 | 8.6 | 14.9 | 15.6 | 20.7, <1% |
| **finetune_reset** | 6.9 | **7.9** | **13.2** | **8.0** | 8.1 |
| residual | 6.9 | 34.8 | 34.1 | 16.9 | **6.5** |
| componet | 6.9 | 14.8 | 28.1 | 18.4 | 14.5 |
| kae_moe | 6.9 | 24.4 | 32.8 | 14.9 | 9.0 |
| fame (fast) | 6.9 | 45.4 | 22.6 | 23.3 | 37.3 |
| fame (meta) | 7.0 | 46.0 | 27.4 | 23.2 | 37.5 |

The circle column is the same for all methods: nothing is carried yet and the seed is the same.

### Final policy (after training on vertical_circle) on all tasks

Cells give error in cm, then crash rate where above 0.

| method | circle | rectangle | figure8 | star | vertical_circle | mean error |
|---|---|---|---|---|---|---|
| scratch | 57.1, 100% | 59.9, 100% | 50.9, 100% | 50.2, 96% | 7.7 | 45.2 |
| finetune | 48.3, 100% | 48.0, 100% | 47.0, 100% | 40.5, 100% | 20.7, <1% | 40.9 |
| finetune_reset | 57.6, 100% | 60.3, 100% | 49.5, 100% | 47.7, 100% | 8.1 | 44.6 |
| residual | 60.6 | 56.9, 100% | 49.9, 100% | 49.5 | 6.5 | 44.7 |
| componet | 60.9, 100% | 54.2, 100% | 48.8, 100% | 58.6 | 14.5 | 47.4 |
| kae_moe | 57.8, 100% | 60.2, 100% | 51.3, 100% | 51.5 | 9.0 | 46.0 |
| fame (fast) | 46.0, 100% | 41.1, 100% | 44.8, 100% | 47.0, 10% | 37.3 | 43.3 |
| **fame (meta)** | **8.6** | 46.0 | **24.0** | **23.2** | 37.5 | **27.9** |

### Full transfer matrices

Rows: policy after training through that task. Columns: task evaluated on. Cells give error in cm, then crash rate
where above 0.

| method | after | circle | rectangle | figure8 | star | vertical_circle |
|---|---|---|---|---|---|---|
| scratch | circle | 6.9 | 48.3 | 46.3, 100% | 28.8 | 60.0 |
|  | rectangle | 42.5 | 12.4 | 57.6, 100% | 35.8 | 58.5 |
|  | figure8 | 50.6, 100% | 52.2, 100% | 28.0 | 46.7, 100% | 49.5, 100% |
|  | star | 22.7 | 38.8 | 46.5, 100% | 19.3 | 53.7 |
|  | vertical_circle | 57.1, 100% | 59.9, 100% | 50.9, 100% | 50.2, 96% | 7.7 |
| finetune | circle | 6.9 | 48.3 | 46.3, 100% | 28.8 | 60.0 |
|  | rectangle | 39.4 | 8.6 | 60.7, 100% | 36.5 | 62.3 |
|  | figure8 | 50.5, 100% | 54.7, 100% | 14.9 | 51.4, 100% | 47.9, 100% |
|  | star | 28.6 | 42.8 | 47.5, 100% | 15.6 | 49.8, 26% |
|  | vertical_circle | 48.3, 100% | 48.0, 100% | 47.0, 100% | 40.5, 100% | 20.7, <1% |
| finetune_reset | circle | 6.9 | 48.3 | 46.3, 100% | 28.7 | 59.9 |
|  | rectangle | 40.2 | 7.9 | 61.7, 100% | 36.3 | 62.1 |
|  | figure8 | 53.5, 100% | 55.2, 100% | 13.2 | 52.8, 100% | 49.6, 100% |
|  | star | 24.2 | 45.9 | 47.4, 100% | 8.0 | 56.6, 9% |
|  | vertical_circle | 57.6, 100% | 60.3, 100% | 49.5, 100% | 47.7, 100% | 8.1 |
| residual | circle | 6.9 | 48.3 | 46.3, 100% | 28.8 | 60.0 |
|  | rectangle | 32.2 | 34.8 | 49.2, 100% | 40.5 | 59.1, 100% |
|  | figure8 | 54.1, 100% | 64.1, 100% | 34.1 | 52.8, 100% | 52.1, 100% |
|  | star | 35.0 | 37.0 | 49.1, 100% | 16.9 | 54.7, 27% |
|  | vertical_circle | 60.6 | 56.9, 100% | 49.9, 100% | 49.5 | 6.5 |
| componet | circle | 6.9 | 48.3 | 46.3, 100% | 28.8 | 60.0 |
|  | rectangle | 41.7 | 14.8 | 55.2, 100% | 37.9 | 57.8 |
|  | figure8 | 48.2, 100% | 55.2, 100% | 28.1 | 51.0, 100% | 52.4, 100% |
|  | star | 32.3 | 37.8 | 43.9, 100% | 18.4 | 54.8, 36% |
|  | vertical_circle | 60.9, 100% | 54.2, 100% | 48.8, 100% | 58.6 | 14.5 |
| kae_moe | circle | 6.9 | 48.3 | 46.1, 100% | 28.8 | 60.0 |
|  | rectangle | 46.9 | 24.4 | 55.5, 100% | 42.0 | 63.7, 2% |
|  | figure8 | 51.5, 100% | 44.4, 3% | 32.8 | 45.3, 100% | 51.8, 100% |
|  | star | 26.5 | 40.6 | 47.1, 100% | 14.9 | 47.7 |
|  | vertical_circle | 57.8, 100% | 60.2, 100% | 51.3, 100% | 51.5 | 9.0 |
| fame (fast) | circle | 6.9 | 48.3 | 46.3, 100% | 28.8 | 60.0 |
|  | rectangle | 40.6 | 45.4 | 56.7, 100% | 36.9 | 54.5 |
|  | figure8 | 50.9, 100% | 50.7, 100% | 22.6 | 48.7, 100% | 48.8, 100% |
|  | star | 30.7 | 33.7 | 45.3, 100% | 23.3 | 52.4, 100% |
|  | vertical_circle | 46.0, 100% | 41.1, 100% | 44.8, 100% | 47.0, 10% | 37.3 |
| fame (meta) | circle | 7.0 | 47.8 | 46.7, 100% | 29.2 | 59.6 |
|  | rectangle | 8.3 | 46.0 | 55.7, 100% | 28.3 | 56.1 |
|  | figure8 | 8.5 | 45.9 | 27.4 | 27.8 | 50.0, 100% |
|  | star | 8.6 | 45.9 | 24.5 | 23.2 | 59.8, 36% |
|  | vertical_circle | 8.6 | 46.0 | 24.0 | 23.2 | 37.5 |

### Method details

- **Where the best checkpoints are** (timestep of 48000 per task, tasks in order):

  | method | circle | rectangle | figure8 | star | vertical_circle |
  |---|---|---|---|---|---|
  | scratch | 46080 | 38400 | 44160 | 32640 | 42240 |
  | finetune | 46080 | 48000 | 38400 | 40320 | 48000 |
  | finetune_reset | 46080 | 48000 | 48000 | 44160 | 30720 |
  | residual | 46080 | 17280 | 38400 | 42240 | 46080 |
  | componet | 46080 | 46080 | 26880 | 40320 | 38400 |
  | kae_moe | 46080 | 30720 | 48000 | 19200 | 17280 |
  | fame | 46080 | **3840** | 44160 | **3840** | 19200 |

- **fame task switches.** Detection return (mean return of one deterministic episode), fresh vs meta: rectangle
  22.2 vs 69.0 → meta; figure8 17.9 vs 14.2 → fresh; star 27.1 vs 123.0 → meta; vertical_circle 15.6 vs 36.2 → meta.
- **fame noise level.** The log standard deviation of the noise (before the tanh) that each task starts with, and the
  best fast learner's after training:

  | | circle | rectangle | figure8 | star | vertical_circle |
  |---|---|---|---|---|---|
  | start (source) | 0.0 (initial) | −0.70 (meta) | 0.0 (initial) | −1.18 to −1.27 (meta) | −1.24 to −1.36 (meta) |
  | best fast learner | −2.0 to −2.4 | −0.80 to −0.84 | −1.9 to −2.4 | −1.3 to −1.5 | −2.2 to −2.5 |

  After the circle, distillation brought the meta learner's log std only to −0.70, against −2.0 to −2.4 for the fast
  learner it was distilled from. The distillation's step budget limits how far it can move: Adam moves a parameter by
  about its learning rate per step, and with `lr_meta: 1e-3`, decayed by 0.95 every 2 epochs over 200 epochs, at 23
  steps per epoch, the total is at most about 0.9.

## Forward transfer and cost

Computed from the logs and policy files of the same runs with `scripts/skrl/continual_metrics.py` (no new runs).

### Forward transfer

Forward transfer on task i (Wołczyk et al., 2021, Continual World, Eq. 2) compares the area under the learning curve
with that of `scratch`: `FT_i = (AUC_i - AUC_i^b) / (1 - AUC_i^b)`. `AUC_i` is the mean success rate of the training
episodes over the 48000 steps of task i, from TensorBoard (100 points), and `AUC_i^b` is the same for `scratch`.
An episode succeeds if it reaches the end without a crash, with position RMSE ≤ 20 cm, cross-track RMSE ≤ 15 cm and
at least 90% of its steps within 25 cm of the reference (`success_*` in `waypoint_env_cfg.py`). FT_i = 0: as
`scratch`; FT_i = 1: success from the start; FT_i < 0: worse than `scratch`.

| method | circle | rectangle | figure8 | star | vertical_circle | mean FT |
|---|---|---|---|---|---|---|
| finetune | 0.00 | 0.69 | 0.09 | 0.30 | −1.23 | −0.03 |
| **finetune_reset** | 0.00 | **0.95** | 0.00 | **0.74** | **0.67** | **0.47** |
| residual | 0.00 | 0.00 | 0.00 | 0.00 | 0.39 | 0.08 |
| componet | 0.00 | 0.02 | 0.00 | 0.00 | −0.49 | −0.09 |
| kae_moe | 0.00 | 0.01 | 0.00 | 0.08 | 0.61 | 0.14 |
| fame (fast) | 0.00 | 0.00 | 0.00 | 0.00 | −1.23 | −0.25 |
| *scratch AUC* | *0.473* | *0.008* | *0.000* | *0.001* | *0.551* | |

- `finetune_reset` has the highest forward transfer: 0.95 on the rectangle, 0.74 on the star, 0.67 on the vertical
  circle. `finetune` transfers to the rectangle (0.69) and the star (0.30), but succeeds in almost no training
  episode of the vertical circle (AUC 0.001 vs 0.551 for `scratch`).
- The frozen-policy methods and FAME's fast learner have forward transfer near 0 on the rectangle, figure 8 and star.
  On the vertical circle, `residual` (0.39) and `kae_moe` (0.61) are positive, `componet` (−0.49) and FAME (−1.23)
  negative.
- No method transfers on the figure 8: the highest training success rate there is 0.09 (`finetune`).

Limits of this measure:

- **Training episodes use the stochastic policy.** It succeeds far less often than the deterministic policy is
  accurate: `scratch` reaches 12.4 cm on the rectangle in evaluation, but succeeds in 0.8% of its training episodes.
  Methods start tasks with different noise levels (see *Methods and what they carry*), which changes their training
  success.
- **`scratch` almost never succeeds on the rectangle, figure 8 and star** (AUC ≤ 0.008), so FT on those tasks is
  close to the method's own AUC. It does not separate methods that all fail, such as 28 vs 34 cm on the figure 8.
- **Shortened first episodes.** At the start of a stage, every episode gets a random elapsed time, so the episodes
  that end within the first episode length (1508 to 2439 steps depending on the task, the first 4 to 6 points) are
  shorter and pass more easily. Without those points, every FT changes by at most 0.12 and the ranking of the mean
  FT is the same.
- **Success rate is a lower bound.** Its denominator is `crashed + lost + time_out`; an episode that is both crashed
  and lost counts twice.

### Cost

Model memory and inference compute of the policy after each stage (Díaz-Rodríguez et al., 2018). The value network is
not counted (the same size for every method).

| method | parameters after each stage | final memory [KiB] | MS | final kFLOPs per action | final CPU latency [µs] | stored states |
|---|---|---|---|---|---|---|
| scratch, finetune, finetune_reset | 8003 at every stage | 31 | 1.00 | 16 | 20 | 0 |
| residual | 8003 / 16006 / 24009 / 32012 / 40015 | 156 | 0.46 | 79 | 124 | 0 |
| componet | 8003 / 28166 / 48329 / 68492 / 88655 | 347 | 0.33 | 188 | 475 | 0 |
| kae_moe | 8003 / 117829 / 218502 / 319175 / 419848 | 1650 | 0.23 | 836 | 397 | 0 |
| fame (fast, meta) | 8003 + 8006 at every stage | 31 + 31 | 1.00 | 16 | 20 | 500000 (107 MiB) |

- **Memory**: bytes of the parameters and buffers (float32).
- **MS** (model size efficiency): `min(1, mean_i Mem(theta_1) / Mem(theta_i))`, where `Mem(theta_i)` is the memory
  after task i; 1 means no growth.
- **FLOPs per action**: floating-point operations of the matrix products of one forward pass (one multiply-add
  counts as 2); activations are not counted.
- **CPU latency**: median time of one forward pass on one observation, one CPU thread.
- **Stored states**: FAME keeps 100000 states (56 floats each) of every task for distillation, in a FIFO buffer of
  500000 that is allocated at full size from the first stage. No other method stores samples.

The growing methods grow linearly with the number of tasks: `residual` by 8003 parameters per task, `componet` by
20163, `kae_moe` by 100673 from the third task on (one KAE per earlier task; its router, weight, gate and MLP
networks are rebuilt every task). After five tasks, the `kae_moe` policy has 52 times the parameters of an MLP policy
and needs 53 times its FLOPs. FAME's two policies
stay the size of one MLP each, but it stores 107 MiB of states. Training compute is not compared: run times are not
comparable (see *Caveats*), and FAME (detection episodes, state collection) and `kae_moe` (observations for the KAE)
use environment steps outside the 48000 training steps that are not counted here.

## Findings

1. **`finetune_reset` learns each new task best.** It has the lowest error right after training on the rectangle
   (7.9 cm), the figure 8 (13.2 cm) and the star (8.0 cm), and is within 1.6 cm of the best on the vertical circle.
   `finetune`, which also carries its optimizer and noise level, gets worse along the sequence (8.6 → 14.9 → 15.6 →
   20.7 cm); on the vertical circle it is 2.7 times worse than `scratch`. Its log std after the circle is −2.0 to −2.4
   (noise standard deviation about 0.1), so it starts each new task with little exploration.
2. **Forgetting is near total for every single-policy method.** After the last task, `scratch`, `finetune`,
   `finetune_reset`, `residual`, `componet`, `kae_moe` and FAME's fast learner miss the earlier tasks by 40–61 cm,
   mostly with 100% crashes.
3. **FAME's meta learner retains what its fast learners learned, but the fast learners learned poorly.** The final meta
   learner flies the circle at 8.6 cm and the figure 8 and star at 23–24 cm, with no crashes, which no other final
   policy does. But it misses the rectangle (46.0 cm) and the vertical circle (37.5 cm), because the fast learners
   on those tasks did (45.4 and 37.3 cm).
4. **FAME's fast learner does not improve on its starting point after a meta initialization.** On the rectangle and
   the star, the best checkpoint is the first one (timestep 3840 of 48000): training made the policy worse than the
   meta learner it started from. Its errors on the tasks it was trained on are the highest of all methods on the
   rectangle and vertical circle. A possible cause is the meta learner's noise level not matching the policy it was
   distilled from (see *Method details*); this has not been tested.
5. **The frozen-policy methods transfer little.** Compared with `scratch`, `residual`, `componet` and `kae_moe` are
   worse on the rectangle (14.8–34.8 vs 12.4 cm), worse or equal on the figure 8 (28.1–34.1 vs 28.0 cm) and better
   on the star (14.9–18.4 vs 19.3 cm); `residual` is best on the vertical circle (6.5 cm).

## Caveats

- **One seed.** Differences of a few cm between methods are not established.
- **`finetune_reset` was resumed.** Its run stopped during the star stage, when the Claude Code session that had
  launched it ended. Its star and vertical_circle stages continued the session in a new process (`--resume`), which
  sets the seed again, so those two stages are not bit-identical to an uninterrupted run. They also ran with a
  different physics rate: `--resume` re-applies the session's 50 Hz control rate with `set_control_frequency`,
  which chose 200 Hz physics (`sim.dt` 0.005 s, decimation 4) instead of the 500 Hz of the waypoint configuration
  (0.002 s, decimation 10; `params/env.yaml` of the resumed session). The same holds for every evaluation made in
  the resumed process: the star and vertical_circle columns of the earlier rows, and the last two rows.
- **FAME's result is reproducible.** A second run with the same settings (2026-11-20) gave identical numbers.
- **Selection uses the training return.** The best checkpoint is chosen from the mean return of the stochastic
  policy during training, not from an evaluation of the deterministic policy. A method with a larger noise level
  (such as `finetune_reset` after its reset) has a lower training return for the same deterministic policy.
- **Observation normalization changes the results.** A `finetune` run with the normalizer on (2026-11-18, same code)
  had errors on its own tasks of 9.4, 7.3, 8.7, 10.6 and 8.2 cm, against 6.9, 8.6, 14.9, 15.6 and 20.7 cm without it.
  It is not used here because the frozen-policy methods cannot use the normalizer. No FAME run with the normalizer
  and the current policy code exists: the 2026-11-16 FAME run predates commit `4d17470`, in which the mean network
  outputs a bounded action converted back with a clamped `atanh`.
- **Run times are not comparable.** Runs shared the GPU with each other in varying numbers. Total times for the five
  tasks: scratch 404 min, finetune 412, finetune_reset 123 (alone on the GPU), residual 590, componet 595, kae_moe
  618, fame 430.

## Reproduction

```bash
for m in scratch finetune finetune_reset residual componet kae_moe fame; do
  python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 \
    --tasks circle rectangle figure8 star vertical_circle --method $m --headless --run_name waypoint_5tasks_nonorm
done
# FAME meta learner on all tasks
python scripts/skrl/evaluate.py --headless --env_id CrazySim-TrackingWaypoint-Direct-v0 \
  --checkpoint logs/skrl/crazyflie_tracking_waypoint/<fame session>/*/meta_policy.pt \
  --output logs/skrl/crazyflie_tracking_waypoint/<fame session>/meta_policy_eval.json
```

Forward transfer and cost (Isaac Lab Python environment; Isaac Sim is not started):

```bash
python scripts/skrl/continual_metrics.py --sessions logs/skrl/crazyflie_tracking_waypoint/<session> ... \
  --output metrics.json   # one session per method; the scratch session is the baseline
```

The runs before 2026-11-20 were launched with the override `agent.agent.state_preprocessor=null`, which is now the
default.

Runs used here, under `logs/skrl/crazyflie_tracking_waypoint/` (logs are not in git):

| method | session |
|---|---|
| scratch | `2026-11-19_17-38-21_scratch_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks_nonorm` |
| finetune | `2026-11-19_17-38-41_finetune_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks_nonorm` |
| finetune_reset | `2026-11-20_17-50-06_finetune_reset_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks_nonorm_resumed` (circle to figure8 from `2026-11-20_13-57-47_finetune_reset_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks_nonorm`) |
| residual | `2026-11-18_23-38-20_residual_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks` |
| componet | `2026-11-18_23-38-20_componet_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks` |
| kae_moe | `2026-11-18_23-38-20_kae_moe_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks` |
| fame | `2026-11-19_17-39-01_fame_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks_nonorm` (rerun: `2026-11-20_17-55-36_fame_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks_nonorm_rerun`) |
| finetune, normalizer on (caveats only) | `2026-11-18_23-38-20_finetune_circle-rectangle-figure8-star-vertical_circle_waypoint_5tasks` |
