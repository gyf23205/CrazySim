# Simulation warm-up for real flights

A warm-up trains one method on one new shape in simulation for a fixed number of iterations and hands the
learner to `Drone_Test_TRO`, which continues training it on the real drone. The finalized real result is
the starting point of the next shape's warm-up. The experiment, its decisions and its open points are in
[PLAN.md](../PLAN.md).

Everything here uses the waypoint task (`CrazySim-TrackingWaypoint-Direct-v0`: 56 observations, 3 actions)
and the shapes `square`, `circle`, `triangle` and `figure8`. Simulation commands need the Isaac Lab
environment. The methods are `kae_moe`, `finetune` (PPO fine-tuning), `residual`, `componet` and `fame`.

## What a warm-up does

`scripts/skrl/train.py --warm_start` trains exactly one task and then, instead of finishing the task:

1. loads the best checkpoint of the run (highest mean training return; a candidate every 20 iterations),
2. writes it as `transfer_bundle.pt`, the file `Drone_Test_TRO` imports,
3. writes `policy.pt`, the deterministic policy of that checkpoint, for the validation script.

It does not fit a KAE, freeze the Residual base, append the CompoNet unit or run FAME's consolidation.
Those steps finish a shape and happen in `Drone_Test_TRO` after the real flight (`finalize_stage.py`).

Use a budget that is a multiple of 20 iterations. A shorter run has no checkpoint candidate and hands over
its final iteration.

## Step 0: the Square, in simulation only

One Square trained with scratch PPO serves `kae_moe`, `finetune`, `residual` and `componet`. FAME trains
its own Square. This plan uses one lap per episode, so every command carries `env.trajectory_laps=1`.

```bash
# shared Square, its KAE and a check
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks square --method scratch \
    --headless --no_eval --max_iterations 500 env.trajectory_laps=1
SQUARE=logs/skrl/crazyflie_tracking_waypoint/<scratch session>
python scripts/skrl/fit_kae.py --headless --task square --stage_dir $SQUARE/0_square
python -u scripts/skrl/validate_waypoint.py --headless --stage_dir $SQUARE/0_square --tasks square \
    --output validation.json env.trajectory_laps=1
# the Square KAE as a file of Drone_Test_TRO: its finalizer asks for it after KAE-MoE's first real stage
python scripts/skrl/export_waypoint_artifacts.py --stage_dir $SQUARE/0_square --drone_test_dir <Drone_Test_TRO>

# FAME's own Square
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks square --method fame \
    --headless --no_eval --max_iterations 500 env.trajectory_laps=1
FAME_SQUARE=logs/skrl/crazyflie_tracking_waypoint/<fame session>
```

## Step 1: warm-up of the first new shape

The `--square_...` options name the exact files of the Square stage a method starts from. The budget is
`--max_iterations`.

| Method | Options | File |
| --- | --- | --- |
| `finetune`, `residual`, `componet` | `--square_checkpoint` | `checkpoints/best_agent.pt` of the shared Square |
| `kae_moe` | `--square_kae` | `kae/kae.pt` of the shared Square, written by `fit_kae.py` |
| `fame` | `--square_policy`, `--square_meta_policy` | `policy.pt` and `meta_policy.pt` of FAME's own Square |

```bash
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks circle --method kae_moe \
    --headless --no_eval --warm_start --max_iterations <budget> env.trajectory_laps=1 \
    --square_kae $SQUARE/0_square/kae/kae.pt
# the same for residual and componet
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks circle --method finetune \
    --headless --no_eval --warm_start --max_iterations <budget> env.trajectory_laps=1 \
    --square_checkpoint $SQUARE/0_square/checkpoints/best_agent.pt
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks circle --method fame \
    --headless --no_eval --warm_start --max_iterations <budget> env.trajectory_laps=1 \
    --square_policy $FAME_SQUARE/0_square/policy.pt --square_meta_policy $FAME_SQUARE/0_square/meta_policy.pt
```

A method refuses the options of another method and a path that is not a file. The files that were read are
recorded as `square` in `session.json` and as `simulation_parent` in the bundle. The control rate is not
checked against the Square's session, as `--resume` would: pass the same `--control_freq` if the Square
used one.

The **pilot** that chooses the budget is the same KAE-MoE command with a long budget. Read its training
curves in TensorBoard (`<session>/<stage>`), choose the budget, and rerun all five methods with it.

The stage directory `<session>/1_circle` then holds:

| File | Content |
| --- | --- |
| `transfer_bundle.pt` | The learner for `Drone_Test_TRO` |
| `policy.pt` | Deterministic policy of the selected checkpoint |
| `checkpoints/best_agent.pt`, `checkpoints/final_agent.pt` | Agent checkpoints of the run |

`<session>/session.json` records `budget_iterations`, `selected_iteration` and
`learning_rate_at_selection` for the stage. Check the handed-over policy with

```bash
python -u scripts/skrl/validate_waypoint.py --headless --stage_dir <session>/1_circle --tasks circle \
    --output validation.json env.trajectory_laps=1
```

## Step 2: the real stage, in `Drone_Test_TRO`

The real run must use the command bounds of the simulation, or it rejects the bundle. Add `--dry-run` for
a software test without the drone. The budget (`--real-lap-budget` or `--real-transition-budget`) is not
fixed yet; see PLAN.md.

```bash
python examples/trajectory_setpoints/run.py --method kae_moe --shape circle \
    --transfer-bundle <session>/1_circle/transfer_bundle.pt \
    --position-cmd-xy-min -4.5 --position-cmd-xy-max 4.5 --position-cmd-z-min 0 --position-cmd-z-max 9.85 \
    --real-lap-budget <laps> --boundary-evaluation-episodes 1

# finalize for the next shape; only KAE-MoE passes KAE files, every earlier one in order
python examples/setpoint/finalize_stage.py --source-run <real run> --output-dir <circle finalized> \
    --kae-checkpoints $SQUARE/0_square/deploy/kae.pth
```

For KAE-MoE's later stages the list grows by one file per finished shape: the Square file first, then
`<circle finalized>/kae/kae.pth`, and so on.

Each run flies one frozen evaluation lap before training and lands, then trains, then evaluates the new
shape and every earlier shape. Add `--skip-initial-evaluation` to start with training; the run then has no
measurement of the handed-over learner on the drone.

With the simulation's command bounds, the real command clamp no longer keeps commands above 0.25 m. The
real height crash check (below 0.10 m) and the XY geofence still apply.

## Step 3: warm-up of the next shape

`--parent_bundle` names the finalized real result of the same method.

```bash
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks triangle --method kae_moe \
    --headless --no_eval --warm_start --max_iterations <budget> env.trajectory_laps=1 \
    --parent_bundle <circle finalized>/transfer_bundle.pt
```

Then repeat steps 2 and 3 for each shape.

## What crosses the handoff

| | Simulation to real | Real to simulation (next shape) |
| --- | --- | --- |
| PPO fine-tuning | Policy, critic, optimizer, both normalizers | The same |
| Residual | Frozen base, residual network, noise, critic, optimizer | The finished controller becomes the frozen base |
| CompoNet | Frozen earlier controllers, attention heads, internal policy, noise, critic, optimizer | The finished controller is appended to the earlier ones |
| KAE-MoE | KAE bank, router, mode weights, gate, MLP path, noise, critic, optimizer | The bank plus the KAE fitted to the real-adapted policy |
| FAME | Fast actor and critic, their optimizers, temperature, meta actor, both state buffers | Consolidated meta actor and its buffer of earlier states |

A new shape in simulation uses each method's own behaviour: Residual, CompoNet and KAE-MoE start with a
new critic and new action noise, PPO fine-tuning continues everything, and FAME tests a fresh actor
against its meta actor.

**Learning rate.** The bundle records the KL-adaptive schedule and the rate the simulation had reached,
but the real stage does not use them. It trains at the rate and schedule of `Drone_Test_TRO` (3e-5 with
exponential decay, factor 0.9995 per epoch) and keeps the simulation's optimizer moments. The first real
Circle run of KAE-MoE used the simulation's rate (1e-3, the schedule's cap) and its tracking degraded
within seven laps: one lap per update is too little data for that step size. PPO fine-tuning brings the
real stage's final rate back to simulation, where the adaptive schedule raises it again within a few
updates; the other methods start each shape with a new optimizer. FAME has fixed rates of its own on both
sides.

**Observation normalization.** PPO fine-tuning normalizes observations and keeps updating the statistics.
Residual, CompoNet and KAE-MoE do not: their frozen earlier policies take raw observations and keep their
own normalizer, so `train.py` turns it off for these methods. FAME has none.

**Task settings.** The bundle records the task settings of the simulation run (shape sizes, speed, laps,
command bounds, observation scales). `Drone_Test_TRO` compares them with its own run and refuses a
mismatch.

## Hardware-free checks

```bash
python -m pytest tests
```

The checks train a few steps on the CPU with the unchanged method classes and a small tensor
environment, write bundles, and load them with the loader of the `Drone_Test_TRO` checkout next to this
repository (`DRONE_TEST_ROOT` if it is elsewhere). They need PyTorch, skrl, Gymnasium, PyYAML and pytest,
not Isaac Sim. For all five methods they compare the mean action and the action noise of the two learners
on the same observations, take a training step on the real side, and cover Square to Circle, a start from
a finalized real stage, a two-shape chain and the selection of an earlier best checkpoint.

They do not run physics and say nothing about tracking quality or flight. `scripts/skrl/train.py` itself
is not covered, since it needs Isaac Sim.

## Files

| File | Role |
| --- | --- |
| [scripts/skrl/train.py](../scripts/skrl/train.py) | `--warm_start`, `--parent_bundle`, `--square_...` |
| [crazysim/warm_start.py](../source/crazysim/crazysim/warm_start.py) | Translation between the methods here and the bundle |
| [crazysim/transfer_bundle.py](../source/crazysim/crazysim/transfer_bundle.py) | Reading, checking and writing the bundle file |
| [skrl_ppo_waypoint_cfg.yaml](../source/crazysim/crazysim/tasks/direct/tracking/agents/skrl_ppo_waypoint_cfg.yaml) | Method settings for the waypoint task |
| [tests/](../tests) | Hardware-free checks |
