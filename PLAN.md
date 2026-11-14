# Warm-start plan: simulation-assisted real adaptation on waypoint 56/3

Written November 12, 2026. Branch `warm_start_waypoint`, started clean from Yifan's commit
`585e33a` ("Add waypoint control"). Status: implemented the same day and checked without hardware
(see [Implementation status](#implementation-status)). Not yet run in Isaac Sim and never flown.
Usage is in [docs/warm_start.md](docs/warm_start.md).

This replaces the earlier plan, kept as [PLAN_real_test.md](PLAN_real_test.md). That plan and the
launcher on the `real_test` branch are not used here.

## Goal

Compare five continual-learning methods on real Crazyflie flights, each warmed up in simulation
first. For every new shape, each method trains in simulation for the same number of iterations,
then continues training on the real drone in `Drone_Test_TRO`. The real result of one shape is the
starting point for the next shape's simulation warm-up.

Only the waypoint task is used: `CrazySim-TrackingWaypoint-Direct-v0`, 56 observations, 3 actions
(relative waypoint), 50 Hz.

## Methods and shapes

Methods: KAE-MoE (the proposed method), PPO fine-tuning, Residual, CompoNet, FAME.

Shapes, in order, all at 0.75 m height and 0.25 m/s (Yifan's waypoint defaults):

| Shape | Dimensions | Lap time |
| --- | --- | --- |
| Square | 1.0 m side | 16 s |
| Circle | 0.75 m radius | 18.85 s |
| Triangle | equilateral, 1.0 m side | 12 s |
| Figure-8 | 2.0 m × 1.0 m overall | 24.39 s |

## Experiment sequence

**Step 0: Square, simulation only.**

1. Train one shared Square with scratch PPO, using Yifan's command and defaults (500 iterations).
2. Fit its KAE with `fit_kae.py`.
3. Check it with `validate_waypoint.py`. Yifan's own run reached 0.071 m RMSE on Square.
4. FAME trains its own Square with its native first stage, as in Yifan's code.

KAE-MoE, PPO fine-tuning, Residual and CompoNet all start from the shared Square. Their native
first stage is plain PPO anyway. FAME starts from its own Square.

**Then, for each of Circle, Triangle and Figure-8:**

1. **Pilot.** Train KAE-MoE on the new shape from its parent, long enough to see it stabilize.
   The parent is the shared Square for Circle, and KAE-MoE's finalized real result of the previous
   shape afterwards.
2. **Choose the budget.** Woojun shares the pilot results. We review them by hand and pick an
   iteration count slightly before KAE-MoE stabilizes. The budget may differ between shapes.
3. **Warm up.** Train all five methods on the new shape for that same budget, each from its own
   parent. KAE-MoE is rerun with the budget like the others.
4. **Hand over.** Each run hands over its best-return checkpoint within the budget and writes a
   `transfer_bundle.pt`.
5. **Fly.** `Drone_Test_TRO` loads the bundle, continues training on the real drone for the real
   budget, and finalizes the stage.
6. **Carry forward.** The finalized real bundle is that method's parent for the next shape.

Each method keeps its own history. No method ever starts from another method's result.

## Decisions

| Topic | Decision | Reason |
| --- | --- | --- |
| Branch | Clean start from `585e33a`, waypoint 56/3 only | Drop the `real_test` launcher |
| Handoff file | Keep the version-1 `transfer_bundle.pt`; write a new reader and writer in CrazySim | `Drone_Test_TRO` already loads, trains from, finalizes and returns this format, with tests |
| Simulation runner | Edit Yifan's `train.py` | This is a branch; least code |
| Checkpoint handed over | Best training-return checkpoint within the budget (Yifan's rule), then skip the rest of `end_stage` | Protects against stopping in a bad stretch; the stage is only finished after the real flight |
| Real-side checkpoint | Exact final, as `Drone_Test_TRO` does now | A few real laps are too little data to pick a best checkpoint |
| Learning rate on the real drone | The real repository's own rate and schedule (3e-5, exponential decay); changed on 2026-11-13 | Inheriting the simulation's rate (1e-3) degraded KAE-MoE within seven real laps |
| Budget choice | Manual review of the KAE-MoE pilot; no automation | Decided by Woojun |
| Flight gate | None | The real runtime's 1 m abort and geofence are the protection |
| Square | Train fresh with Yifan's code | Gives a full stage folder (critic, optimizer, normalizers), not a deployment-only file |
| FAME start | FAME trains its own Square | Follows Yifan's code |
| Simulation settings | Yifan's defaults, except one lap per episode | Same settings as the Square that reached 0.071 m; one lap matches the real side |
| Command bounds | Yifan's simulation values on both sides (±4.5 m, 0 to 9.85 m) | Decided by Woojun: real and simulation must be identical |
| Observation normalization | As Yifan's code requires (see below) | His methods enforce it |
| New-shape behaviour | Yifan's native behaviour for each method, no extra options | Follows Yifan's code |
| Stage finalization | Stays in `Drone_Test_TRO` (`finalize_stage.py`); the KAE is fitted on real observations | Already implemented and tested there |
| Platform | All simulation runs on Linux with Isaac Lab | Isaac Lab is not on the Mac |

### Yifan's simulation defaults, for reference

4096 environments, 96 steps per iteration, seed 42, two laps per episode (this plan uses one),
random start phase,
±0.1 m spawn noise, learning rate 5e-4 with a KL-adaptive schedule (maximum 1e-3), command bounds
±4.5 m in XY and 0 to 9.85 m in Z, training episodes end when the drone is more than 1 m from the
reference. A checkpoint candidate is saved every 20 iterations, and the best one is the candidate
with the highest mean training return.

### Observation normalization per method

| Method | Observation normalization |
| --- | --- |
| Shared Square (scratch PPO) | On |
| PPO fine-tuning | On; continues updating on each new shape |
| Residual, CompoNet, KAE-MoE | Off for the new trainable parts; frozen earlier policies keep their own |
| FAME | None |

Yifan's method code refuses to run Residual, CompoNet and KAE-MoE with it on
([base.py:138](source/crazysim/crazysim/methods/base.py:138)). `Drone_Test_TRO` already uses the
same arrangement, so nothing changes there.

### What "skip the rest of `end_stage`" means

After a warm-up, `train.py` loads the best checkpoint and exports it. It does not fit a KAE to the
simulation policy, freeze the Residual base, append the CompoNet unit, or run FAME's consolidation.
Those happen in `Drone_Test_TRO` after the real flight.

## What Yifan's code already provides

- Square training, KAE fitting and validation for 56/3 (see commands below).
- `--resume`, which rebuilds each method from a finished stage folder:

| Method | What it reads from the previous stage |
| --- | --- |
| PPO fine-tuning | The selected agent checkpoint (policy, critic, optimizer, normalizers) |
| Residual | `policy.pt` as the frozen base |
| CompoNet | The unit inside each earlier stage's `policy.pt` |
| KAE-MoE | `kae/kae.pt` of every earlier stage |
| FAME | `meta_policy.pt`; past states are regenerated by flying each earlier policy in simulation |

- Best-checkpoint selection as a separate function (`Method.load_best_checkpoint`).

## Work to do in CrazySim

In order. Items 1 and 2 need no new code for the shared Square, so they can start right away.

1. **Shared Square and its KAE.** Run Yifan's commands, validate.
2. **Method settings for 56/3.** The waypoint config has no settings for Residual, CompoNet,
   KAE-MoE or FAME; they exist only in the thrust config. Add them, sized for 56 inputs (KAE input
   57, KAE hidden size 64, as in Yifan's 56/3 KAE settings). Network sizes are the thrust-config
   values, which the real side requires (check 3 below). FAME's Square needs this item first.
3. **Start a method from the shared Square.** `--resume` refuses a session trained with a
   different method. Allow KAE-MoE, Residual, CompoNet and PPO fine-tuning to start from the
   scratch Square stage. CompoNet needs the Square policy wrapped as its first unit, normalizer
   included. As built (November 14, 2026): the start is named by exact files, `--square_checkpoint`
   or `--square_kae`, not by `--resume` of the Square session; FAME takes `--square_policy` and
   `--square_meta_policy` of its own Square.
4. **Warm-up mode in `train.py`.** One flag: train for the budget, load the best checkpoint,
   export, skip the rest of `end_stage`. Record the budget and the selected iteration. Save a
   deterministic `policy.pt` so Yifan's validation script still works. Turn observation
   normalization off for the three methods that require it.
   After items 3 and 4, the KAE-MoE Circle pilot can run; it needs no bundle code.
5. **Bundle writer (simulation to real).** Map each method's native modules and parameter names
   to the version-1 descriptions. Include the critic, optimizer state, normalizers, action noise
   and FAME's memory. Record the parent, the budget and the actually selected iteration. Record
   the KL-adaptive schedule and its settings; the current rate travels in the optimizer state.
6. **Bundle reader (real to simulation).** Turn a finalized real bundle into what each method's
   `resume` needs for the next shape.
7. **Hardware-free checks.** Load every written bundle with `Drone_Test_TRO`'s own loader and
   compare actions and noise on identical observations before and after conversion. Then run the
   real runtime's `--dry-run`.
8. **Short usage doc.**

## Work to do in Drone_Test_TRO

One small exception to leaving that repository unchanged.

1. **Accept the KL-adaptive schedule in bundles.** In `examples/setpoint/transfer_bundle.py`,
   three lines allow only an exponential schedule or none: lines 418 and 421 when a bundle is
   written, and line 483 when one is loaded. Extend them to the KL-adaptive schedule and its
   settings. The real training loop needs no change; the four PPO-type methods use the training
   library's standard update, which already steps this schedule.
2. **Update `TRANSFER_BUNDLE.md`**, which says version 1 accepts only exponential schedules, and
   add a test for the new kind.
3. **Allow Yifan's command bounds.** `examples/setpoint/cf_setpoint_ppo_real_common.py` stops
   unless the bounds equal its four default numbers (lines 103 to 106, checked at lines 1148 to
   1157). Either change those four defaults, which affects every run in that repository, or let
   the checks accept both sets and pass Yifan's values at launch with the existing
   `--position-cmd-*` options. The second keeps other experiments on the current bounds. One test
   fixture (`tests/test_kae_evaluation.py`) pins the current values.

## Check results (November 12, 2026)

Read-only checks of `Drone_Test_TRO` at commit `44d47ba`. Two passed and two need a decision.

1. **Circle parent: passes.** The real import (`load_same_shape_transfer` in
   `examples/setpoint/policy_initialization.py`) requires the method, shape, observation contract
   and task settings to match, and the bundle's phase to be `simulation`. It does not require a
   parent. A Circle bundle built on a simulated Square is accepted.
2. **Learning rate after import: needs a decision.** The real run takes its learning rate,
   schedule and optimizer state from the bundle (`transfer_agent_config` and
   `restore_transfer_agent` in `examples/setpoint/transfer_bundle.py`). The reset to the real
   default of 3e-5 with exponential decay applies only to plain checkpoints, not to bundles. With
   Yifan's KL-adaptive schedule, a bundle would make the real run train at a constant rate equal
   to whatever the simulation had reached (it starts at 5e-4 and is capped at 1e-3). FAME's own
   learning rates are carried the same way.
3. **Network sizes: resolved.** The real side hardcodes the sizes in Yifan's thrust config, so the
   56/3 settings copy them: Residual network 256/128/64; CompoNet internal policy 256/128/64 with
   attention size 256; KAE-MoE MLP path 256/128/64, mode weights 32, router 128, gate 32; KAE
   hidden size 64 with input 57; PPO actor and every PPO critic 64/64. FAME's size is free on the
   real side and defaults to 256/128/64, the same as Yifan's.
4. **Task settings record: needs a decision.** The importer compares every recorded task setting
   with the real run's own. Three of Yifan's simulation defaults differ from the real defaults:

   | Setting | Yifan's simulation | Real default |
   | --- | --- | --- |
   | Laps per episode | 2 | 1 |
   | Command bounds, XY | ±4.5 m | ±4.0 m |
   | Command bounds, Z | 0 to 9.85 m | 0.25 to 2.0 m |

   All other compared settings already match (height, speed, shape sizes, offset limit,
   observation scales, slew limit, 50 Hz). A bundle that records Yifan's three values is rejected
   by a real run launched with its defaults.

### Decisions these checks leave open

- **Learning rate on the real drone: changed on 2026-11-13 to the real repository's own rate and
  schedule.** The first decision was to inherit the simulation's rate and Yifan's KL-adaptive
  schedule (target 0.016, maximum 1e-3). The first real Circle run of KAE-MoE did that, at 1e-3,
  and degraded within seven laps (reward 171 to 110, RMSE 16 cm to 30 cm, policy weights moved by
  about a quarter of their size). The Square real run of 2026-11-12 at about 1e-4 with exponential
  decay stayed flat. The bundle still records the adaptive schedule and the rate simulation had
  reached, but the real stage of the four PPO methods now trains at 3e-5 with exponential decay
  (factor 0.9995 per epoch), keeping the simulation's optimizer moments. The edit is in
  `Drone_Test_TRO` (`real_schedule_replaces` in `examples/setpoint/transfer_bundle.py`); the run
  records it as the reset `learning_rate_schedule`. FAME uses its own fixed rates in both
  repositories and is unaffected. Not yet flown with the new rate.
- **The three task settings: decided, identical on both sides.**
  - **Laps: 1 on both sides.** Simulation is launched with one lap per episode (an override of
    Yifan's default of 2), including the fresh Square and FAME's Square. The real default is
    already 1.
  - **Command bounds: Yifan's simulation values on both sides** (±4.5 m in XY, 0 to 9.85 m in Z).
    Simulation keeps its defaults. `Drone_Test_TRO` must be changed to allow these values (see its
    work list below).

## Implementation status

Implemented on November 12, 2026. Nothing is committed in either repository.

**CrazySim** (all eight work items):

- Method settings for 56/3 in the waypoint config.
- `train.py`: `--warm_start`, `--parent_bundle`, start from the files of a Square (`--square_...`),
  observation normalization turned off for the three methods that require it.
- `crazysim/transfer_bundle.py` (file format) and `crazysim/warm_start.py` (writer and reader).
- `tests/` (28 hardware-free checks) and [docs/warm_start.md](docs/warm_start.md).

**Drone_Test_TRO** (all three work items, on top of its uncommitted changes):

- `transfer_bundle.py` accepts the KL-adaptive schedule in a bundle; since 2026-11-13 the real stage
  does not run it and uses its own rate and schedule instead.
- `cf_setpoint_ppo_real_common.py` accepts either the real default command bounds or the complete
  simulation set, passed at launch with `--position-cmd-*`. The four defaults are unchanged, so other
  experiments keep the current bounds.
- `TRANSFER_BUNDLE.md` updated; new `tests/test_simulation_handoff_settings.py` (6 tests).

**What was checked, all without Isaac Sim and without a drone:**

- CrazySim's 28 checks pass. For all five methods, a learner trained a few steps on the CPU is
  written, loaded by `Drone_Test_TRO`'s loader, gives the same mean action and noise there, and takes
  a training step there. Covered: Square to Circle, a start from a finalized real stage, a two-shape
  chain, and the selection of a best checkpoint that is earlier than the final one.
- `Drone_Test_TRO`'s whole test suite passes with the edits (348 tests before the new file, 354 with it).
- The chain through the real tools, for all five methods: Circle bundle from this repository, real
  `run.py --dry-run` for 800 transitions, real `finalize_stage.py`, Triangle warm-up from that
  finalized bundle, Triangle bundle accepted by the real loader.

**What was not checked:**

- `train.py --warm_start` has run on Isaac Sim once, for the KAE-MoE Circle pilot from the scratch
  Square (see [Circle budget](#circle-budget-240-iterations)). The other four methods, FAME's own
  Square and `--parent_bundle` have not been run there.
- No physics: nothing here says a warmed policy tracks well. The CPU checks use a small tensor
  environment and tiny training settings.
- FAME on the waypoint task.
- Any real flight with the real repository's rate on a simulation bundle, and the wider command
  bounds on the drone beyond one KAE-MoE Circle run.

**Choices made while implementing, to confirm:**

- **FAME's past states when coming back from a real stage** come from the real bundle (the states FAME
  stored during flight, in its consolidated meta buffer), not from regeneration in simulation.
- **FAME's reward scale on 56/3** is 1.0, copied from the thrust config. Yifan's PPO reward scale differs
  between the tasks (0.1 for thrust, 0.01 for waypoint), so FAME's may need the same factor.
- **FAME's optimizers and temperature** are added to its checkpoints during a warm-up, so the handed-over
  best checkpoint has the optimizer state of the same iteration. Yifan's FAME checkpoints hold only the
  actors and the critic.
- **A warm-up session cannot be resumed.** `--resume` refuses it, because the shape is only finished
  after the real flight.
- **The shared Square start is only available with `--warm_start`**, and only from a scratch session
  with exactly one stage.

**Seen in the dry run:** after one real update of 800 transitions, the adaptive schedule had raised the
learning rate from 5e-4 to 7.5e-4 for three methods and to the 1e-3 cap for PPO fine-tuning. The policies
were barely trained, so this says little about a real stage, but it shows the rate can rise quickly.

## Circle budget: 240 iterations

Decided November 12, 2026, by manual review of the KAE-MoE Circle pilot. All five methods run the
Circle warm-up with `--max_iterations 240`.

Pilot: session `2026-11-12_18-27-51_kae_moe_square-circle_pilot700`, 700 iterations, seed 42, one lap,
started from scratch Square session `2026-11-12_17-11-32_scratch_square`. Values are training
values logged every 7 iterations, with exploration noise.

| Iteration | Return | Position error | Episodes that succeed |
| --- | --- | --- | --- |
| 196 | 93 | 36.7 cm | 0% |
| 210 | 178 | 14.7 cm | 55% |
| 224 | 129 | 23.3 cm | 7% |
| 231 | 202 | 10.7 cm | 84% |
| 238 | 199 | 11.4 cm | 99% |
| 252 | 228 | 6.6 cm | 100% |
| 259 | 239 | 4.8 cm | 100% |
| 301 to 560 | 242 (mean) | about 4 to 5 cm | 100% |

- Before about iteration 230 the policy is unstable: success swings between 0% and 84%.
- From 238 it succeeds almost every time, with about 11 cm of error.
- From 259 the return stays within 2% of the plateau mean for the rest of the run.
- No lost episodes at any point and no crashes after the first 70 iterations.

Checkpoint candidates exist every 20 iterations. 220 is in the unstable stretch and 260 is already on
the plateau, so 240 is the last candidate before stabilization. With this budget the handed-over
checkpoint should be iteration 240 itself: its return of about 200 is above every earlier candidate
(the best is about 163).

The pilot's own best checkpoint was iteration 580 (training return 244.3). Its deterministic
validation on Circle: position RMSE 7.7 cm, 100% success, no crashes.

Checks on the 240-iteration reruns:

- **KAE-MoE must reproduce the pilot.** 240 is only about 10 iterations past the jump. Its return near
  iterations 231 to 238 should be about 200 and `selected_iteration` should be 240. If not, stop and
  review before any flight.
- **Validate every method's handed-over policy** with `validate_waypoint.py` on its `policy.pt`. The
  iteration-240 KAE-MoE policy has not been validated; the numbers above are training values.

Seen in the pilot, relevant to the real stage:

- **The learning rate is at its 1e-3 cap** for nearly the whole run, including at iteration 240, about
  33 times the default of `Drone_Test_TRO`. The real stage no longer starts at that rate (see the
  learning-rate decision above).
- **Exploration noise at iteration 240 is about three times that of the converged policy** (standard
  deviation 0.34 against 0.11 at iteration 580).
- **The scratch Square's best checkpoint was its last iteration** (500), so the Square was still
  improving when its training ended.

Limits: one seed; the budget was chosen from KAE-MoE, the proposed method; how the other four methods
look at 240 iterations is not known until they run. Budgets for Triangle and Figure-8 are chosen the
same way from their own pilots.

## Open until flights

These do not block the code.

- **Real budget.** Laps or transitions, and how many. The reader must accept both kinds of parent.
- **Failed real stage.** Rerun, or record the failure and how that method's chain continues.
- **Repeats.** How many real training runs and evaluation flights per method and shape.
- **Square on the real drone.** The fresh Square is never flown in this plan. An evaluation-only
  flight before Circle would confirm it transfers.

## Risks

- **FAME on 56/3 is untested.** Yifan only ran scratch PPO on this task. If FAME does not learn
  Square in its budget, its start needs a new decision.
- **One budget for all methods.** It is chosen from KAE-MoE, the proposed method, by manual review
  of a single seed. Other methods may be further from or closer to convergence at that budget.
- **No flight gate.** A method whose best checkpoint is still poor is flown anyway.
- **The adaptive schedule failed on the drone and is no longer used there.** A real update uses
  about a thousand samples against about 393,000 in simulation. At the simulation's rate the
  KAE-MoE Circle run of 2026-11-13 degraded within seven laps. The replacement, 3e-5 with slow
  decay, is the rate `Drone_Test_TRO` was tuned for, but it has not been flown on a simulation
  bundle yet, and it may be too low to show learning within 25 laps.
- **Wider command bounds on the real drone.** The real 0.25 m lower bound currently keeps
  commands off the floor. With Yifan's bound of 0 m, a policy can command a descent to the floor.
  The remaining protections are the real crash check (height below 0.10 m or above 2.0 m), the
  1.5 m XY geofence and the 1 m tracking abort. The 1 m abort does not help vertically, because
  the floor is 0.75 m below the reference. Simulation has the same arrangement.
- **One lap per episode is a departure from Yifan's default.** His 0.071 m Square used two laps.
  The effect is expected to be small but is not measured.
- **KAE-MoE may not stabilize cleanly.** Earlier Square pilots on `real_test` oscillated. Those
  used a different learning-rate schedule, so this may not repeat; it is not known.

## Reporting

For every method and shape, report:

- The iteration budget and the selected iteration (for example 480 of 500).
- The deterministic simulated evaluation of the handed-over checkpoint.
- Real results as both final error and improvement from the start of the real stage.
- FAME's own Square: its cost and its Square evaluation, next to the shared Square's.
- FAME's fresh-versus-meta test result on each new shape, and its extra interactions.

State once: the budget rule (manual, from KAE-MoE, one seed), the normalization difference between
methods, that the real stage uses the real repository's learning rate (3e-5 with exponential decay)
instead of the simulation's rate and adaptive schedule, and that simulation selects the best
checkpoint while the real side uses the final one.

## Out of scope

- Skewed figure-8.
- A Square real-adaptation stage.
- Automatic plateau detection and a simulated flight gate.
- The thrust task (69 observations, 4 actions).
- Changes to any method's equations or update rule.
- The existing artifacts in `Drone_Test_TRO/checkpoints/` (`starting_square`,
  `square_sim_budget600`, `square_real_finalized`).

## Commands that exist today

Shared Square, its KAE and validation (from Yifan's README):

```bash
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks square --method scratch \
    --headless --no_eval --max_iterations 500 env.trajectory_laps=1
STAGE=logs/skrl/crazyflie_tracking_waypoint/<session>/0_square
python scripts/skrl/fit_kae.py --headless --task square --stage_dir $STAGE
python -u scripts/skrl/validate_waypoint.py --headless --stage_dir $STAGE --tasks square --output validation.json
```

FAME's Square uses `--method fame --tasks square` with the same options. The warm-up commands
(`--warm_start`, `--square_...`, `--parent_bundle`) are in [docs/warm_start.md](docs/warm_start.md).
