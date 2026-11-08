# Simulation warm-up between real adaptation stages

`scripts/skrl/warm_start.py` is a dedicated additive launcher for the waypoint
experiment. Existing `train.py`, method implementations, `end_stage`, default
YAML files, task registration, standalone exporters and evaluators retain their
normal behavior. This launcher calls an existing method's new-shape lifecycle
once, trains for an explicit cutoff, exports the current learner, and closes
without calling `end_stage`.

## Experiment order and inputs

Every method keeps an independent chain:

**Square simulation preparation → Square real adaptation → Circle simulation
warm-up → Circle real adaptation → Triangle simulation warm-up → Triangle real
adaptation → Figure-8 simulation warm-up → Figure-8 real adaptation → Skewed
Figure-8 simulation warm-up → Skewed Figure-8 real adaptation.**

The shared Square policy/KAE are source inputs, not complete trained
checkpoints for every method. `--prepare-square` explicitly prepares the
method's Square learner and reports its extra simulation work. Square real is
an adaptation stage, not merely a demonstration. For each later shape,
`--parent-bundle` must identify that method's immediately preceding completed
real result. Do not restart from the original Square input or another method's
history.

The producer/consumer contract is the real repository's version-1,
tensor/state-dictionary `transfer_bundle.pt`. The additive adapter validates
method/equations, previous shape and phase, checksummed content, ordered frozen
dependencies, preprocessing, and required next-stage knowledge. A finalized KAE
parent must contain the KAE fitted to the complete real-adapted policy, including
after Square real. A FAME parent must contain its once-consolidated meta/task
state. `--allow-kinematic-parent` explicitly enables software-only integration
artifacts; their provenance stays labeled software-only.

## Manual cutoff and exploration decisions

Use a separate KAE-MoE pilot from the previous real artifact to locate the
interval immediately before sustained performance flattening. Inspect tracking
error, return, failures and deterministic tracking. Freeze the chosen iteration
budget for all methods with matching environment counts and rollout lengths.
Budgets may differ by transfer. Numerical cutoffs, independent-run counts and
plateau thresholds are experiment choices; no automatic plateau detector or
experiment scheduler is installed.

The launcher requires `--iterations`, `--num-envs`, and
`--checkpoint-selection exact`; `--rollout-steps` defaults to the native waypoint
value of 96 and can be set explicitly for matched runs. One iteration is one
complete update with `num_envs × rollout_steps` transitions. The checkpoint is
the learner at the exact final update. It does not substitute `policy.pt` or
skrl's best-return checkpoint. FAME task-detection interactions and auxiliary
optimizer work are reported separately from this training cutoff.

Every launch also requires a new-shape exploration decision. For KAE-MoE,
Residual, CompoNet and PPO, choose `--new-shape-noise reset` or `preserve`.
Preservation imports the preceding/source actor's learned noise; reset uses the
waypoint policy's log-standard-deviation initialization of zero. FAME requires
`--new-shape-noise original`: the original fresh-versus-meta task detection
selects its complete state-dependent Gaussian. The real backend handoff always
preserves the *warmed* distribution, regardless of the earlier new-shape choice.

## Commands

Run physics commands with the Python environment in which Isaac Lab and this
CrazySim extension are installed. Examples below use explicitly selected
software smoke-test counts; they do not choose the experiment budget.

```bash
python scripts/skrl/warm_start.py --method kae_moe --shape square \
    --prepare-square --square-policy /path/to/starting_square/deployable_policy_checkpoint.pt \
    --square-kae /path/to/starting_square/kae.pth \
    --iterations 2 --num-envs 8 --rollout-steps 96 \
    --checkpoint-selection exact --new-shape-noise preserve \
    --output-dir /path/to/new/square_simulation --headless
```

PPO, Residual, CompoNet and FAME preparation use `--square-policy`; the KAE input
is required for KAE-MoE. FAME preparation preserves the supported Square source's
64/64 actor architecture and frozen source preprocessing. Source-only
initialization cannot recover historical FAME replay memory or absent
critic/optimizer state; those preparations and explicit resets are recorded.

After Square real adaptation and real-stage finalization:

```bash
python scripts/skrl/warm_start.py --method kae_moe --shape circle \
    --parent-bundle /path/to/finalized/square_real/transfer_bundle.pt \
    --iterations 2 --num-envs 8 --rollout-steps 96 \
    --checkpoint-selection exact --new-shape-noise reset \
    --output-dir /path/to/new/circle_simulation --headless
```

Use the same interface for `finetune`, `residual`, `componet`, and `fame`. For
FAME, set `--new-shape-noise original`; `--detection-steps` optionally overrides
the original per-candidate detection count and remains separately reported.
Do not invoke `end_stage` after this warm-up: it would prematurely fit/append a
KAE, finalize a controller, replace the cutoff with a best checkpoint, or
consolidate FAME before real adaptation.

`--validate-only` checks input schema and stage lineage without starting Isaac
Sim or training. It does not establish physics quality or execute the full
native-constructor checks. `--help` also works without Isaac Lab.

## In-memory waypoint configuration

The launcher starts from the registered waypoint configuration, merges missing
`methods` settings from the general tracking YAML in memory, and adapts KAE
padding/fitting width to 57/64. It retains original policy/method equations:

| Method | Original equation or update |
| --- | --- |
| KAE-MoE | gate × weighted mode actions + (1 − gate) × MLP |
| Residual | frozen complete base + current residual |
| CompoNet | original input/output attention, weighted priors + internal policy |
| PPO | waypoint tanh-squashed Gaussian and PPO objective |
| FAME | state-dependent squashed Gaussian, original PPO/temperature and fast/meta updates |

There are 56 observations, 16 relative reference setpoints, and three relative
waypoint actions at 50 Hz (500 Hz physics with decimation 10). Defaults match
the real target: 0.75 m center height, 0.25 m/s reference speed, one lap, square
side 1 m, circle radius 0.75 m, triangle side 1 m, Figure-8 amplitudes (1, 0.5)
m, and skewed Figure-8 amplitudes (0.7, 0.35) m. Geometry overrides use
`--square-side-length`, `--circle-radius`, `--triangle-side-length`,
`--figure8-amp-x`, `--figure8-amp-y`, `--target-height`, `--trajectory-speed`,
`--trajectory-scale`, and `--trajectory-laps`.

Skewed Figure-8 is an additive trajectory factory, rotated +30° about Y so the
positive-X lobe slopes downward. Default altitude is 0.40–1.10 m. Circle and
figure-eight analytic reference tangents match the real runtime. Existing
native trajectory classes, registrations and default tasks are not changed.
Target-specific dimensions reset to that target's defaults unless explicitly
overridden; inactive dimensions do not accidentally inherit the previous
shape's custom settings. Shared observation/command settings and shared
height/speed/scale/laps can inherit the parent contract.

The published observation/action contract currently uses waypoint offsets
(0.5, 0.5, 0.25) m. Other offset-contract identifiers are rejected. Default
commands use XY [−4, 4] m and Z [0.25, 2] m with the existing slew or proximity
constraint. The native interface
supports symmetric XY command bounds; asymmetric source bounds are rejected.
The adapter rejects changed observation widths, non-50-Hz handoffs, unsupported
action guards and references outside the command envelope. Simulation retains
its original random starting phases and spawn noise. Real return-and-settle
lifecycle, telemetry and hardware landing remain real-backend responsibilities.

Compositional methods consume raw observations and retain each frozen prior's
own normalizer. PPO preprocessing remains enabled when its source declares it.
A local, checkpoint-compatible population-variance scaler matches the real
normalization convention and handles a single environment without NaNs.
FAME source preprocessing remains embedded in its fast/meta actor adapter.
No global observation-normalization setting is changed. Parent optimizer,
preprocessor and scheduler topology is checked before constructing the native
method. Policy-only Square preparation uses an explicitly configured
ExponentialLR scheduler with gamma 0.9995 for compatible PPO handoff; FAME keeps
its own original optimizer/temperature behavior. These are dedicated launcher
settings, not edits to the registered YAML defaults.

## Outputs, evaluation and real continuation

The output directory contains the final `transfer_bundle.pt`, a familiar
`checkpoints/final_agent.pt`, `warm_start_config.json`, and
`warm_start_summary.json`, plus a hook-free `policy.pt` for the existing evaluator. The summary reports actual PPO updates and separate actor/critic/temperature optimizer-step counts; FAME task detection stays outside those training counts. They record the source revision, parent identity and
checksums, geometry/commands, architecture, equations, allocated/actual budget,
explicit resets and exploration choice. The portable artifact retains all
frozen dependencies and the currently learned head/residual/attention/router,
gate, MLP, exploration, compatible critic/optimizer/scheduler and FAME method
memory. On-policy rollout and backend-local state are discarded on transfer;
phase counters restart in reality while cumulative provenance remains.

`--evaluate` calls the existing deterministic evaluator after the selected
learner has been exported. Those interactions do not enter training memory or
normalizer updates. Existing mean position error is mean Euclidean distance,
not RMSE. Report it separately from the real backend's RMSE.

Import the warmed **same-shape** artifact in Drone_Test_TRO:

```bash
python examples/trajectory_setpoints/run.py --method kae_moe --shape circle \
    --transfer-bundle /path/to/circle_simulation/transfer_bundle.pt \
    --real-transition-budget 800 --ppo-rollout-steps 800 \
    --boundary-evaluation-episodes 1 --dry-run
```

The numeric real budget is also a software smoke-test example. The real importer
continues the existing warmed learner; it must not add another module merely
because the backend changes. After real adaptation, finalize that method's
completed stage, then use its authoritative real parent for the next shape.
Hardware-free tests validate configuration, reconstruction, policy/distribution
parity and learning continuation. They do not replace an Isaac physics run or
physical-flight validation.

## Hardware-free checks

```bash
python -m pytest -q tests
python scripts/skrl/warm_start.py --help
```

The core tests require PyTorch, Gymnasium, skrl 1.4.x, PyYAML and pytest. They
load the unchanged native method implementations while replacing only the
unavailable simulator type annotation in the test process. They never replace
PPO/FAME updates or claim to exercise physics. Optional roundtrip tests use the
adjacent `Drone_Test_TRO` checkout (or `DRONE_TEST_ROOT`) to check every method
at Circle, Triangle, Figure-8 and Skewed Figure-8 handoffs, including a native
update, preserved real-import behavior/distribution, fresh rollout state and
another real-runtime update. Those tests skip when that companion checkout is
unavailable.

A physical-simulation acceptance run still requires an Isaac Lab environment.
Isaac Lab is not installed on the development host used for the CPU checks;
physics and physical-flight quality must be established separately.
