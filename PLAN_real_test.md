# CrazySim minimal-change plan: simulation warm-start handoffs

Updated November 7, 2026; Square preparation budget added November 12, 2026.
Status: planned; the new handoff path is not yet implemented or validated.

This revision replaces the earlier broad CrazySim implementation plan.
Keep the original equations and existing training framework unchanged.
Hardware adaptation and completed-real-stage exports are owned by
[Drone_Test_TRO](../Drone_Test_TRO/PLAN.md).

## Agreed experiment

All five methods—KAE-MoE, PPO fine-tuning, Residual, CompoNet, and FAME—follow:

**Square pretrained in simulation → Square real adaptation →
Circle simulation warm-up → Circle real adaptation →
Triangle simulation warm-up → Triangle real adaptation →
Figure-8 simulation warm-up → Figure-8 real adaptation →
Skewed Figure-8 simulation warm-up → Skewed Figure-8 real adaptation.**

Each method's previous completed real artifact initializes the next shape in
simulation. Simulation and reality for one shape continue the same learner.
Keep method histories independent. Do not substitute an unrelated target
teacher or restart the chain from the original Square policy.

Use the supplied Square policy and KAE as shared inputs. These artifacts alone
are not complete trained states of every method, so each method first runs
Square preparation in simulation. All methods use the same preparation budget
(see [Square preparation budget](#square-preparation-budget-600-iterations)).
Report the preparation cost.

## Scope: additive launcher and focused adapters

Add a dedicated launcher, provisionally `scripts/skrl/warm_start.py`, and a
small import/export helper, provisionally `scripts/skrl/warm_start_io.py`.
These are proposed files, not existing commands. Reuse the existing environment,
method classes, and trainer from this new entry point.

Leave these existing components unchanged:

- [train.py](scripts/skrl/train.py), including its normal sequential workflow.
- [Method](source/crazysim/crazysim/methods/base.py) and the existing method
  implementations, policy equations, PPO/FAME updates, and stage lifecycle.
- The existing default YAML files and environment registration.
- [export_waypoint_artifacts.py](scripts/skrl/export_waypoint_artifacts.py),
  [evaluate.py](scripts/skrl/evaluate.py), and [play.py](scripts/skrl/play.py).

Do not refactor the framework, introduce a shared algorithm package, or add
anchored KAE-MoE/CompoNet formulations. The dedicated path calls existing
components differently without changing their normal behavior.

## Original equations and launch configuration

| Method | Formulation retained |
| --- | --- |
| KAE-MoE | `mu = gate * sum(mode_weight * mode_action) + (1 - gate) * mlp_action` |
| CompoNet | Original input/output attention and `mu = attention_weighted_prior_actions + internal_policy_action` |
| Residual | `mu = frozen_base + residual` |
| PPO | Original waypoint squashed Gaussian and PPO objective |
| FAME | Original state-dependent squashed Gaussian, PPO/temperature updates, and fast/meta distillation |

Use the waypoint contract: 56 observations and three relative-waypoint actions.
The new launcher supplies missing method settings through launch overrides or
a configuration assembled in memory. A small dedicated configuration is
optional if it makes those overrides easier to reproduce; do not rewrite
the existing defaults.

The registered
[waypoint YAML](source/crazysim/crazysim/tasks/direct/tracking/agents/skrl_ppo_waypoint_cfg.yaml)
does not inherit the `methods` settings from the
[general tracking YAML](source/crazysim/crazysim/tasks/direct/tracking/agents/skrl_ppo_cfg.yaml).
Reuse those settings where compatible, adapting dimensions and architecture to
the waypoint contract and imported artifacts. Its top-level `kae` block
configures standalone fitting, not KAE-MoE training.

Configure preprocessing per method to satisfy existing constructors.
Compositional methods consume raw observations while imported prior actors
retain their normalizers. Preserve compatible FAME source preprocessing in an
adapter. Do not disable PPO normalization globally or bypass compatibility checks.

## Minimal warm-up execution

The new launcher performs one target-shape simulation phase:

1. Read the previous real-stage artifacts and validate method, equation,
   shape lineage, dimensions, preprocessing, and ordered prior dependencies.
2. Supply predecessor knowledge to the existing method before invoking its
   normal new-shape construction. Use the original task-switch behavior.
3. Run the existing trainer for the chosen iteration budget.
4. Export the complete learned current-stage state through the focused adapter.
5. Close the run without invoking `end_stage`.

The normal `train.py` and `end_stage` behavior remain untouched. Avoiding
`end_stage` in this launcher prevents premature best-checkpoint replacement,
KAE fitting/appending, CompoNet finalization, and FAME end-task consolidation.
Those shape-completion operations occur after real adaptation.

| Method | Previous real result used to construct the next shape |
| --- | --- |
| PPO | Actor, normalizers, exploration, and compatible training state |
| Residual | Complete real controller as the next frozen base |
| CompoNet | Ordered completed prior-controller/unit stack |
| FAME | Meta learner and retained task information for original task-switch behavior |
| KAE-MoE | Ordered bank containing KAEs fitted to complete real-adapted policies |

At simulation → real transfer, preserve the current learned heads, gate,
routing/attention, residual/internal policy, frozen priors, and exploration.
The real importer must continue that learner rather than constructing another
new-shape module. A freshly constructed original MoE need not match its
previous controller, but an imported warmed MoE must match its saved behavior.

## Focused artifact conversion

Reuse existing agent checkpoint files, saved configurations, and prior
policy/KAE files. Add only missing method state and a small metadata manifest
needed by the real importer. A universal self-contained bundle system or a
new repository-wide schema is not required. The companion plan's portable
artifact contract can be satisfied by this documented collection of files.

The adapter must preserve or describe:

- Method/equation version and source revision, architecture, observation/action
  contract, normalizers, distribution parameters, and frozen-prior topology.
- Previous real parent identity/checksum, target shape/phase, ordered
  dependencies, consumed budget, and selected checkpoint iteration/rule.
- Current trainable state and compatible critic, optimizer, scheduler, and
  method memory; explicitly list any necessary resets.

Do not assume `agent.save()` includes every required object. In particular,
FAME's current/past-task distillation buffers and task/meta state may need
separate export. Frozen expert architecture and ordering must accompany
weights. Prefer checkpoint state plus configuration over blindly pickling an
actor containing runtime hooks.

Discard on-policy rollout samples and backend-local environment state when
switching domains. Start fresh phase budget counters while retaining cumulative
provenance. FAME distillation memory is method state and must not be discarded
as though it were an on-policy rollout.

## Manual budget selection and evaluation

Use a separate KAE-MoE pilot from the preceding real artifact to locate the
interval just before a sustained performance plateau. Inspect existing tracking
error, return, and failure metrics; confirm useful deterministic tracking.
Freeze that iteration budget and apply it to all methods for the transfer.

### Square preparation budget: 600 iterations

Decided November 12, 2026. All five methods run Square simulation preparation
for **600 iterations** from the shared Square policy (and KAE, for KAE-MoE).
KAE-MoE is the proposed method, so every method gets the same budget.

The plateau rule could not be applied. The KAE-MoE pilots (seed 42, 500 and 700
iterations) never reached a sustained plateau. Training position error cycled
between about 0.09 and 0.58 m, with lows near iterations 260, 413 and 616. The
two runs agreed almost exactly through iteration 500. Their deterministic evals
were 0.465 m at 500 and 0.172 m at 700.

Replacement rule, used when the KAE-MoE pilot has no sustained plateau:

> The budget is the start of the first stretch of at least 50 iterations in
> which at least 95% of finished episodes succeed, rounded to the nearest 50.

Compute success as `Episode_Termination/success` divided by the sum of the
`time_out`, `lost` and `crashed` terminations, per logged window. On the
700-iteration pilot:

| Iterations | Successful episodes | Qualifies |
| --- | --- | --- |
| 252–266 | 99–100% | No: about 21 iterations |
| 595–651 | 99.6–100% | Yes: about 57 iterations |

The qualifying stretch starts at 595, which rounds to 600. Training error there
is about 0.11 m. The rule marks where the task first stays solved, not
KAE-MoE's lowest error (0.089 m at 616).

Run settings, identical for every method: 4096 environments, 96 rollout steps,
seed 42, exact cutoff, square side 1.0 m, height 0.75 m, speed 0.25 m/s, one
lap. Use `--new-shape-noise preserve` for KAE-MoE, PPO, Residual and CompoNet,
and `original` for FAME. The budget applies only to these settings.

Checks and reporting:

- Run KAE-MoE first. Its deterministic eval at 600 should be about
  0.10–0.15 m. If it is far outside that range, the capped run did not
  reproduce the pilot. Revisit the budget before any real adaptation.
- Report that the rule was fixed after viewing the pilot, from one seed.
- Report every method's deterministic eval at 600, its full training curve,
  and the pilot evals at 500 and 700. The FAME pilot flattened by about
  iteration 100–120 (eval 0.055 m at 700).
- Before running the Circle pilot, decide whether this rule is also the
  fallback for transfers.

Pilot runs, under `logs/warm_start/`: `square_kae_moe_pilot_500_seed42`,
`square_kae_moe_pilot_700_seed42`, `square_fame_pilot_700_seed42`.

Keep environment counts, rollout lengths, and sampling conventions equal.
Report auxiliary task-detection interactions and optimizer work separately.
The current waypoint default is 96 rollout steps per iteration; read the run's
saved configuration when interpreting TensorBoard steps.

No automatic plateau detector or new checkpoint browser is needed. Pilot curves
can identify candidate budgets, followed by capped reruns from the same
predecessor. The adapter can export the final learner of a capped run directly,
without replaying an arbitrary intermediate checkpoint.

Exact-cutoff versus best-within-budget selection remains an explicit decision.
Do not silently substitute the existing `policy.pt`, which normally contains
the best training-return policy. If best-within-budget is chosen, reconstruct
consistent selected-checkpoint state in the helper without invoking stage
consolidation. Preserve corresponding method memory or declare its handling.

Reuse existing deterministic evaluation when the helper exports a compatible
policy module. Supply matching waypoint environment settings. Keep evaluation
outside training data and report mean position error as mean distance, not RMSE.

## Remaining shape and optional work

Matching `skewed_figure8` support is required for the full agreed sequence but
can follow the initial transfer validation. Prefer a trajectory/configuration
supplied through the dedicated launcher or a small additive trajectory helper.
Keep existing default tasks and framework registration unchanged.

The following are optional and outside the initial implementation:

- Universal checkpoint packaging or replacing existing exporters.
- Intermediate-checkpoint playback/evaluation support.
- Broad reporting changes, automatic plateau detection, or full scheduling.
- Shared-library extraction and unrelated refactoring.

## Delivery order and acceptance

1. Agree the minimal input/output file contract with the real importer and
   resolve initial Square compatibility.
2. Add the launcher and KAE-MoE adapter; validate
   Square real artifact → Circle simulation → Circle real learner.
3. Add the remaining four method adapters using the same additive path.
4. Add the missing trajectory and validate the remaining shape transitions.

Check matching actions and distribution parameters before/after transfer,
frozen priors, learned components, continued updates, provenance, and absence
of duplicate modules. Verify required FAME memory and documented resets.
Use a short physics-simulation continuation and the real runtime's dry run;
physical validation is separate. Keep existing launch paths unaffected.

The Square preparation budget and its settings are fixed above. Transfer
budgets, run counts, plateau thresholds, new-shape exploration for transfers,
and checkpoint selection for transfers remain open.
This update changes the plan only; it does not implement these capabilities.
