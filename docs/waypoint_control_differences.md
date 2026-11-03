# Waypoint control: simulation validation of `Drone_Test_TRO`

This repository duplicates the control scheme of `Drone_Test_TRO/examples/trajectory_setpoints` in Isaac Lab, to
validate it before real flights. This page gives the results and every known difference between the two
implementations. All values were measured on 2026-11-01.

Terms:

- **Waypoint control**: the policy observes the relative positions of all 16 setpoints of the reference curve and
  outputs a position command for the Crazyflie firmware, not motor thrusts.
- **Setpoint**: one of the 16 fixed points on the reference curve. **Waypoint**: the position the policy commands.
- **Firmware cascade**: the Crazyflie PID loops, position to velocity to attitude to body rate to motors.
- **Slew limit**: the command moves at most (0.5, 0.5, 0.25) m/s. It is the real default. **Proximity limit**: the
  alternative, in which the command stays within (0.25, 0.25, 0.125) m of the drone.
- **RMSE**: root-mean-square distance to the reference over an episode. **Cross-track RMSE**: the same for the
  distance to the curve, ignoring timing along it.
- **Success**: the check of the real runtime at the end of an episode: position RMSE at most 0.20 m, cross-track RMSE
  at most 0.15 m, and at least 90% of steps within 0.25 m.
- **KAE**: Koopman autoencoder fitted to imitate the policy. **Mode sum**: the same KAE evaluated as the sum of its
  eigen-modes, which must match the direct evaluation.

## Conclusions

1. The port matches the real code. Equations agree to float32 rounding and reference curves to 1e-13 m.
2. The simulated drone reproduces real flights to about 3 cm on circle and triangle and 8–9 cm on figure-8.
3. Waypoint control can reach the success criteria. A policy trained from scratch on the square and its KAE pass in
   simulation and through the real runtime's dry run.
4. That policy works on the square only. It fails on the other three shapes.
5. Real-flight training from scratch is unlikely to work with the settings of the real repository (see
   [Findings](#findings-for-the-real-experiments)).

## Results

The policy was trained on `square` for 500 iterations, as the first stage of the README chain of the real repository.
Episodes start at the first reference point with ±0.1 m position noise and run two laps. Numbers below use the slew
limit unless stated.

### Square, 1024 episodes from identical starts

| Controller | Successes | Mean RMSE (m) | Median RMSE (m) |
|---|---|---|---|
| Policy | 1024 | 0.071 | 0.071 |
| KAE | 1020 | 0.074 | 0.070 |
| KAE mode sum | 1020 | 0.074 | 0.070 |

- KAE and mode sum differ by at most 4.8e-5 in action. The same 4 episodes fail for both.
- In those 4 episodes the KAE output leaves [-1, 1] at step 16–17 and stays out for 99% of the episode. The real
  runtime holds the last command when that happens, which gives an RMSE of 0.90 m. Clipping the output instead still
  fails those 4 episodes, with an RMSE of 0.55 m. So the hold makes the failure worse but does not cause it. What these
  starts have in common was not investigated.
- For comparison, logged real flights of the older 15-dimensional waypoint policies reach an RMSE of 0.107–0.181 m on
  successful episodes.

### All shapes, 64 episodes per shape

| Shape | Controller | RMSE (m) | Cross-track RMSE (m) | Within 0.25 m | Success | Crash | Output out of bounds |
|---|---|---|---|---|---|---|---|
| Square | Policy | 0.071 | 0.044 | 100% | 100% | 0 | 0% |
| Square | KAE | 0.070 | 0.043 | 100% | 100% | 0 | 0% |
| Circle | Policy | 1.327 | 1.208 | 2% | 0% | 0 | 0% |
| Circle | KAE | 1.372 | 1.111 | 7% | 0% | 0 | 44% |
| Triangle | Policy | 0.214 | 0.195 | 70% | 0% | 0 | 0% |
| Triangle | KAE | 0.228 | 0.204 | 66% | 0% | 0 | 0.1% |
| Figure-8 | Policy | 0.657 | 0.551 | 5% | 0% | 0 | 0% |
| Figure-8 | KAE | 0.976 | 0.769 | 9% | 0% | 0 | 31% |

The plant does not limit the other shapes: the scripted rule "command the reference 0.5 s ahead", with no learning,
tracks all four shapes to under 1 cm. The likely reason is that the setpoints are constant in square-only training, so
the policy has no data on how to respond to them. This was not tested separately. The KAE was fitted on square
observations only, so it extrapolates on the other shapes.

### Sensitivity of the policy, 64 episodes per case

Each case changes one setting. The policy and KAE were not retrained.

| Case | Square RMSE (m) | Square success | Figure-8 RMSE (m) |
|---|---|---|---|
| Baseline | 0.071 | 100% | 0.657 |
| Motor lag 0.03 s | 0.070 | 100% | 0.657 |
| Motor lag 0.15 s | 0.072 | 100% | 0.657 |
| Maximum thrust 0.10 N per motor | 0.071 | 100% | 0.657 |
| Proximity limit | 0.274 | 0% | 0.752 |

Square tracking does not depend on the assumed motor lag or maximum thrust in these ranges. A policy trained with the
slew limit does not carry over to the proximity limit. The KAE reaches an RMSE of 0.85 m on the square there, so a
real run with the proximity limit (`--command-proximity-radius`) needs its own policy and KAE.

### Checks against `Drone_Test_TRO`

| Check | Result |
|---|---|
| Observation, action map, geofence, slew and proximity limits, reward terms, metrics; 2000 random inputs | Largest gap 1.2e-7 (float32 rounding) |
| Reference curves: position, setpoints | At most 1.3e-14 m, 8.9e-16 m; periods equal |
| Reference velocity | At most 1.6e-5 m/s (figure-8) |
| Policy in the real loader | Contract accepted, actions identical (largest gap 0.0) |
| Policy in `run.py --method eval --dry-run` | 2/2 success, RMSE 0.082 m, reward 380 |
| KAE in `run.py --method kae --dry-run` | 2/2 success, RMSE 0.082 m, reward 379.6 |
| Same with `--mode-sum` | 2/2 success, RMSE 0.082 m, reward 379.7 |

The real runtime's dry run uses a kinematic model, so it tests the code path and not the flight physics. Two episodes
are too few to show a failure rate of 0.4%.

### Simulated drone against real flights

The logged commands of real flights were replayed open loop through the simulated firmware cascade.

| Episode | Position gap (cm) |
|---|---|
| Triangle | 3.0–3.6 |
| Circle | 2.6–3.3 |
| Figure-8 | 8.3, 9.1 |
| Figure-8 that crashed in the real flight (538 steps) | 31 |

Over the nine episodes of the second replay batch: mean gap 7.5 cm, velocity RMS error 0.093 m/s (real speed RMS
0.234 m/s, simulated 0.242 m/s), tilt RMS 5.3° real and 6.2° simulated, simulated velocity leads by 7–20 ms. On
figure-8 the real drone lags its command by 0.22 m and the simulated one by 0.175 m. This gap is not explained.

### KAE fit

Input size 57 (observation plus one constant), hidden size 64, 16 modes, 15000 states, 7500 epochs, 932 s. Action MSE
on observed states 0.00099 against an action variance of 0.0303, so about 97% of the variance is explained. Relative
reconstruction error 0.067. Spectral radius of the Koopman matrix 16.56.

## Differences from `Drone_Test_TRO`

### Plant and control loop

1. **Firmware version.** The cascade is ported from bitcraze master as of November 2026. The version flashed on the
   drone is unknown. Fixed-point truncation is not modeled. The gyro filter runs at 500 Hz, not 1 kHz. Compensated
   thrust is linear with the constants of 2.1+ propellers, with no battery sag.
2. **State.** The controller and policy use the true state, not the Kalman estimate or motion capture. There is no
   sensor noise.
3. **Timing.** No radio or estimator latency and no loop jitter. In replay the simulation leads the real drone by
   7–40 ms.
4. **Motor lag.** A first-order lag of 0.072 s, from [Data-Driven System Identification of Quadrotors Subject to Motor
   Delays](https://arxiv.org/pdf/2404.07837). It was not identified on this drone. Without a lag the rate loop is
   unstable at 500 Hz. Tracking is insensitive to it between 0.03 and 0.15 s.
5. **Mass and inertia.** Taken from the USD model: 0.0282 kg and diagonal inertia (1.657e-5, 1.666e-5, 2.926e-5)
   kg m². The real drone with markers was not measured.
6. **Aerodynamics.** No drag, ground effect or disturbances.

### Task and episodes

7. **Episode start.** Training starts at a random lap phase, at rest, with ±0.1 m noise. Validation starts at phase 0
   with the same noise. The real drone starts at the first reference point after takeoff and a 1 s hold within 0.2 m.
   Takeoff, reach-start, handoff and landing are not simulated.
8. **Hover trim.** The velocity integrator is trimmed to hover at reset.
9. **Episode continuity.** Environments reset after two laps. Real episodes are continuous in the air with a
   continuing reference clock.
10. **Shapes.** Only square, circle, triangle and figure-8 are ported. `moving_circle`, `random_height_figure8`,
    `skewed_figure8` and obstacles are not.
11. **Extra termination in training.** Training ends an episode when the drone is more than 1 m from the reference,
    with no penalty. Validation turns this off. The real runtime has no such check. Episode counters are randomized
    at the first reset in training.
12. **Crash checks.** The motion-capture tracking-loss, battery and telemetry checks are omitted. Acceleration comes
    from a 50 Hz velocity difference, without vibration. The real runtime's exclusion of invalid transitions is not
    ported.
13. **Reference velocity.** Circle and figure-8 use a central difference. The gap to the real values is at most
    1.6e-5 m/s.

### Training

14. **PPO settings.**

    | Setting | Here | `Drone_Test_TRO` |
    |---|---|---|
    | Data per update | 96 steps of 4096 drones | 800 steps of 1 drone |
    | Learning rate | 5e-4, KL-adaptive, maximum 1e-3 | 3e-5, exponential decay |
    | Minimum `log_std` | -3 | None |
    | Length | 500 iterations, about 197 million transitions | About 48000 transitions for `--max-episodes 30` |
    | Checkpoint used | Best return | Final |

    Network sizes, squashing, initial `log_std`, discount, λ, epochs, minibatches, reward scale and normalization are
    equal.

### KAE

15. **KAE fit.**

    | Setting | Here | `Drone_Test_TRO` |
    |---|---|---|
    | Training states | Gaussian fitted to 30000 noisy simulated observations | External notebook on a flight dataset |
    | Schedule | 7500 epochs, learning rate 1e-4 | Notebook schedule |
    | Pseudo-inverse `rcond` | 1e-5 | Default |
    | Input normalizer | None (raw observations) | Optional `.state_preprocessor.pt` beside the checkpoint |
    | Modes | 16, as in its README | 8 in the shipped checkpoints, padded to 16 |

## Findings for the real experiments

1. **Sample count.** Training return reached 50% of its best value at 24 million transitions, 80% at 41 million and 95%
   at 157 million. The best value came at the last logged step (197 million), so training had not clearly finished. A
   real `--max-episodes 30` run collects about 48000 transitions, which is 0.2% of the 24 million. Simulation also used
   a higher learning rate and thousands of parallel drones, so the numbers do not transfer directly, but training from
   scratch in flight is unlikely to converge. Each action also moves the slew-limited command by at most 1 cm, which
   may weaken the learning signal. That effect was not measured.
2. **Exploration safety.** With an initial `log_std` of 0, simulated drones wandered about 2.5 m on average before the
   1 m termination was added. The real runtime has no check on distance to the reference, so the 4.5 m geofence is the
   only position limit.
3. **Noise floor.** Every run that succeeded here had a minimum `log_std` of -3. The one run without it failed (RMSE
   0.277 m), but it also used shorter rollouts and no 1 m termination, so the floor alone is not shown to be required.
   The real configuration has no floor.
4. **KAE output bound.** The KAE has no output squashing. It failed 4 of 1024 episodes on the square, and its output
   was out of bounds 31–44% of the time on circle and figure-8. The real runtime then holds the last command, so the
   drone keeps its last position command while the reference moves on.
5. **Shape coverage.** A policy trained on one shape should not be expected to fly another. Only the chain of the real
   README, with one trained stage per shape, can be expected to cover them.

## Reproduce

Run in the `env_isaaclab` conda environment. Only the contract check needs the `FIAD` environment, which has the
dependencies of `Drone_Test_TRO`.

```bash
python scripts/skrl/train.py --env_id CrazySim-TrackingWaypoint-Direct-v0 --tasks square --method scratch \
    --headless --no_eval --max_iterations 500
STAGE=logs/skrl/crazyflie_tracking_waypoint/<session>/0_square
python scripts/skrl/fit_kae.py --headless --task square --stage_dir $STAGE
python scripts/skrl/export_waypoint_artifacts.py --stage_dir $STAGE
python -u scripts/skrl/validate_waypoint.py --headless --num_envs 64 --stage_dir $STAGE \
    --tasks square circle triangle figure8 --output validation.json

# checks against the real implementation
python scripts/skrl/replay_flight.py --headless --csv <flight>/all_episodes.csv --output replay.json
python scripts/skrl/check_waypoint_contract.py --shapes_output waypoint_shapes.npz  # FIAD environment
python -u scripts/skrl/check_waypoint_shapes.py waypoint_shapes.npz
```

Sensitivity cases add a Hydra override to `validate_waypoint.py`, for example `env.motor_time_constant=0.15`,
`env.motor_thrust_max=0.10` or `env.command_constraint=proximity`.
