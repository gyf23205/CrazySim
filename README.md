# CrazySim

Isaac Lab tasks for training Crazyflie controllers and measuring how methods transfer between tasks.

- **Robot**: Crazyflie 2.x (`CRAZYFLIE_CFG` from `isaaclab_assets`), 100 Hz control.
- **Action**: four per-motor thrusts in `[-1, 1]`, mapped to `[0, f_max]` with `f_max = thrust_to_weight * weight / 4`.
  The thrusts are converted to body force and torque with the motor mixing of the Crazyflie firmware
  (`power_distribution_quadrotor.c`): motor order m1 front-right, m2 back-right, m3 back-left, m4 front-left;
  arm length 0.046 m; thrust-to-torque 0.005964552 m.
- **Tasks**: follow a reference curve in a horizontal plane at 1 m height, centered at the environment origin.

  | name        | curve                                         | default size                       | period |
  |-------------|-----------------------------------------------|------------------------------------|--------|
  | `circle`    | circle                                        | radius 0.5 m                       | 6 s    |
  | `rectangle` | rectangle perimeter at constant speed         | 1.0 m x 0.6 m                      | 6 s    |
  | `figure8`   | lemniscate of Gerono, `x = a sin θ, y = b sin 2θ` | a = 0.6 m, b = 0.3 m           | 6 s    |

All tasks share one gym id (`CrazySim-Tracking-Direct-v0`) and identical observation and action spaces,
so any policy can run on any task. The active task is switched in-process with `env.unwrapped.set_task(name)`.

## Observation

The reference is observed in one of two modes (`observation_mode` in the env configuration), always relative to the
robot position:

| mode | reference part | size | total | tasks look different to the controller? |
|---|---|---|---|---|
| `setpoints` (default) | 16 setpoints spread evenly in time over one whole traversal, fixed in the world (`setpoint_count`), plus the current phase of the traversal as sin and cos | 48 + 2 | 69 | yes: the policy has to find its current target from the shape and the phase |
| `lookahead` | reference points at the current reference time plus `lookahead_times` (0, 0.2, ..., 1.0 s) | 18 | 37 | no: the policy only follows the next 1 s of path, which works on any shape |

The rest of the observation is the same in both modes:

| part | size | frame |
|------|------|-------|
| linear velocity | 3 | world |
| angular velocity | 3 | body |
| rotation matrix (body to world) | 9 | - |
| previous action | 4 | - |

With `lookahead`, a policy trained on one task tracked the others to within 1.8-8.1 cm and never crashed, and
fine-tuning on later tasks did not degrade earlier ones. With `setpoints`, a policy trained on one task misses the
others by 27-55 cm and often crashes, and fine-tuning on the figure 8 degraded the circle from 1.7 cm to 60 cm,
so the tasks interfere as expected in a continual-learning study.

## Reward and termination

Per step (each term is multiplied by the control period): position tracking `10 (1 - tanh(e / 0.25))` with `e` the
distance to the reference, and penalties on velocity error to the reference velocity, angular velocity, yaw angle
and action change. An episode (10 s) ends early when the robot is below 0.1 m or above 2.5 m, upside down, or more
than 1 m from the reference. Every episode starts at a random point of the curve, at rest, within 0.1 m of the reference.

All values are in [tracking_env_cfg.py](source/crazysim/crazysim/tasks/direct/tracking/tracking_env_cfg.py).

## Policy and PPO settings

[skrl_ppo_cfg.yaml](source/crazysim/crazysim/tasks/direct/tracking/agents/skrl_ppo_cfg.yaml): Gaussian MLP policy and
separate MLP value network, both `[256, 128, 64]` with ELU, 4096 environments, 24-step rollouts, 5 epochs x 4 minibatches.
Three settings keep training stable over 1000 iterations (found with an earlier shared policy/value network: without
them the tracking error first dropped to ~2 cm and then drifted up to 10-17 cm; with separate networks the circle
policy reaches 0.7 cm, in the `lookahead` observation mode):

- action noise std bounded below by 0.05 (`min_log_std: -3.0`);
- adaptive learning rate capped at 1e-3 (`max_lr`);
- no running normalization of observations (all observation values are already of order 1).

## Installation

Use the Python environment where Isaac Lab is installed (here: conda env `env_isaaclab`):

```bash
python -m pip install -e source/crazysim
```

## Training

`scripts/skrl/train.py` trains on the task sequence given by `--tasks` with the transfer method given by `--method`:

```bash
# single task
python scripts/skrl/train.py --tasks circle --headless

# task sequence, each task starts from the agent trained on the previous task
python scripts/skrl/train.py --tasks circle rectangle figure8 --method finetune --headless

# transfer methods
python scripts/skrl/train.py --tasks circle rectangle figure8 --method kae_moe --headless
```

### Methods

| `--method` | stage 0 | stage k > 0 | reference code in `less_leg_walking_1` |
|---|---|---|---|
| `finetune` | MLP policy | starts from the best agent (policy, value network, optimizer) of stage k-1: skrl's checkpoint with the highest mean training episode return, checked every 20 iterations; that agent is also the stage's evaluated policy | - |
| `scratch` | MLP policy | new MLP policy; reference for how much the other methods transfer | `train_scratch.py` |
| `residual` | MLP policy | frozen policy of stage k-1 plus a new residual MLP whose output layer starts at zero; base + residual becomes the next base | `res_net.py`, `train_res_CRL.py` |
| `componet` | MLP policy | new CompoNet unit attending over the frozen units of stages 0..k-1 | `baselines/CompoNet.py`, `train_componet_CRL.py` |
| `kae_moe` | MLP policy | gate between a mixture of the eigen-mode experts of the KAEs of stages 0..k-1 and a new MLP; a KAE is fitted to the policy after every stage | `MoE.py`, `agents/KAE_approx.py`, `train_moe_CRL.py` |
| `fame` | fast learner | fresh fast learner, initialized from the meta learner if that scores higher on the new task; the meta learner is distilled from the fast learner after every stage | `baselines/FAME/`, `train_FAME.py` |

- The code is in [crazysim/methods](source/crazysim/crazysim/methods); each file's docstring lists what differs from the
  reference code.
- `finetune`, `scratch`, `residual`, `componet` and `kae_moe` all use the same skrl PPO settings and differ only
  in the policy network. Policy and value networks are separate. The value network and the action noise are new at
  every stage, except for `finetune`.
- `fame` runs its own PPO variant (tanh-squashed actions, state-dependent noise, entropy temperature tuned toward a
  target entropy), as in the reference.
- Method settings are in the `methods:` section of
  [skrl_ppo_cfg.yaml](source/crazysim/crazysim/tasks/direct/tracking/agents/skrl_ppo_cfg.yaml) and can be overridden
  with Hydra, e.g. `agent.methods.kae_moe.kae.epochs=2000`.
- KAE (Koopman autoencoder): an autoencoder whose latent state is mapped by a linear operator `K` from the encoded
  observation to the encoded action of the policy; it is fitted on states drawn from a Gaussian fitted to 30000
  observations visited by the policy. Fitting summary and loss history: `<stage>/kae/`.

Options:

- `--max_iterations N`: iterations per task (default 1000; one iteration = 24 steps of every environment).
- `--checkpoint PATH`: initialize the first task of `finetune` or `scratch` from an agent checkpoint.
- `--num_envs`, `--seed`, `--run_name`, `--no_eval`.
- Hydra overrides for the env and agent configurations, e.g. `env.tasks.circle.radius=0.8 agent.agent.learning_rate=3e-4`.

Each run creates a session directory `logs/skrl/crazyflie_tracking/<time>_<method>_<tasks>[_<run_name>]/`:

```
params/            env and agent configurations
0_circle/          stage 0: TensorBoard logs, policy.pt (deterministic policy), checkpoints/final_agent.pt
1_rectangle/
2_figure8/
transfer/          TensorBoard scalars Transfer/<metric>/<task> per stage
session.json       tasks, method, seed, command, policies, run time and transfer results per stage
```

`policy.pt` is the whole policy module, including frozen parts from earlier stages, so it can be loaded without
knowing the method. `fame` also writes the meta learner as `meta_policy.pt`.

### Transfer evaluation

After each stage, the deterministic policy is evaluated on every task of the sequence for one full episode in every
environment. This gives a matrix with rows "trained through stage i" and columns "evaluated on task j", printed to
the console and saved in `session.json`. Metrics:

- `position_error`: mean distance to the reference [m] until the episode ends.
- `crash_rate`: fraction of episodes that ended by termination instead of time-out.
- `return`: mean undiscounted episode return.

To evaluate saved policies on any task list afterwards:

```bash
python scripts/skrl/evaluate.py --headless --tasks circle rectangle figure8 \
    --checkpoint logs/skrl/crazyflie_tracking/<session>/*/policy.pt --output results.json
```

## Visualization

```bash
python scripts/skrl/play.py --task figure8 --checkpoint logs/skrl/crazyflie_tracking/<session>/2_figure8/policy.pt
# record a video instead (saved in <stage>/videos/play)
python scripts/skrl/play.py --task figure8 --checkpoint <path> --video --headless
```

The red sphere marks the current reference point; blue dots mark the path.

## Adding a task

1. If the shape is new, subclass `Trajectory` in
   [trajectories.py](source/crazysim/crazysim/tasks/direct/tracking/trajectories.py), implement `curve(phase)`
   for `phase` in `[0, 1)`, and add a config class with `class_type` set to it.
2. Add an entry to `TrackingEnvCfg.tasks`, e.g. `"circle_fast": CircleCfg(radius=0.8, period=4.0)`.
3. Use the new name in `--tasks`.

## Adding a method

Subclass `Method` in [crazysim/methods/base.py](source/crazysim/crazysim/methods/base.py): `start_stage` returns the
skrl agent to train on a task (`build_ppo_agent(actor, stage_dir)` builds a PPO agent around any mean-action
network), and `end_stage` returns the deterministic policy and keeps whatever later stages need. Register the class
in `crazysim/methods/__init__.py` and add its name to `METHOD_NAMES` in [train.py](scripts/skrl/train.py).
