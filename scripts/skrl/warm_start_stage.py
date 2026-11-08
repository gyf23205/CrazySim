"""Additive lifecycle adapter for one simulation warm-up of a new shape.

This module supplies completed-real knowledge to the existing methods before
their normal ``start_stage`` call. It never calls ``end_stage``: the warmed
learner, not a consolidated next-task controller, must cross to reality.
"""

from __future__ import annotations

import copy
from pathlib import Path
import time
from types import MethodType

import torch
from torch import nn


SHAPES = ("square", "circle", "triangle", "figure8", "skewed_figure8")


class PriorControllerUnit(nn.Module):
    """Adapt an independent complete controller to native CompoNet's phi chain.

    A completed controller keeps its own frozen dependencies. The new native
    attention layer sees its action as one prior proposal; no previous unit is
    stripped out of that controller by CompoNet's constructor.
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller.eval().requires_grad_(False)
        self.is_prev = False

    def forward(self, value):
        if isinstance(value, tuple):
            phi, states = value
        else:
            states = value
            phi = None
        with torch.no_grad():
            action = self.controller(states)
        proposal = action[:, None, :]
        phi = proposal if phi is None else torch.cat((phi, proposal), dim=1)
        return (phi, states) if self.is_prev else (action, phi)

    def train(self, mode=True):
        super().train(mode)
        self.controller.eval()
        return self


def configure_for_parent(agent_cfg, parent, method, *, square_policy=None):
    """Delegate topology selection to the dedicated launch configuration helper."""
    from warm_start_config import configure_for_parent as configure

    return configure(agent_cfg, parent, method, square_policy=square_policy)


def deterministic_current_policy(method, agent):
    """Independent frozen evaluator of the current learner, without consolidation."""
    if method.name == "fame":
        from crazysim.methods.fame import SquashedMean

        policy = SquashedMean(copy.deepcopy(agent.fast_actor))
    else:
        from crazysim.methods.base import DeterministicPolicy

        actor = copy.deepcopy(agent.policy.actor)
        normalizer = getattr(agent, "_state_preprocessor", None)
        normalizer = copy.deepcopy(normalizer) if isinstance(normalizer, nn.Module) else None
        policy = DeterministicPolicy(actor, normalizer)
    # The native MoE has a training telemetry hook referring to its agent.
    # Its copied evaluator must not write to that agent's training logs.
    for module in policy.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
    return policy.eval().requires_grad_(False)


def _source_log_std(parent):
    value = parent["agent_modules"]["policy"].get("log_std_parameter")
    if not isinstance(value, torch.Tensor) or value.shape != (3,) or not torch.isfinite(value).all():
        raise ValueError("Source fixed-noise learner must contain three finite learned log standard deviations.")
    return value


def _preserve_fame_input_normalization(agent, parent, io):
    """Keep source preprocessing when native task-switch resets the fast actor.

    The original reset, task detection, PPO and temperature updates still run.
    Only the freshly created actor's observation adapter is replaced, before
    either detection candidate is evaluated.
    """
    factory = io.fame_actor_factory(parent)
    if factory is None:
        return
    native_reset = agent._reset_fast_learner

    def reset_with_source_preprocessing(self):
        native_reset()
        fresh = self.fast_actor
        # Constructing the adapter must not consume another random initial
        # policy: retain the fresh native trunk and its original random stream.
        rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            adapted = factory(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
        finally:
            torch.random.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        adapted.trunk = fresh.trunk
        self.fast_actor = adapted
        self.fast_actor_opt = torch.optim.Adam(self.fast_actor.parameters(), lr=self.cfg["lr_actor"])
        self.checkpoint_modules["fast_actor"] = self.fast_actor

    agent._reset_fast_learner = MethodType(reset_with_source_preprocessing, agent)


def start_warm_stage(
    method, parent, task, stage_dir, *, new_shape_noise,
    prepare_square=False, square_policy=None, square_kae=None,
):
    """Inject predecessor state, then call the unchanged native lifecycle once.

    ``method`` is an existing ``crazysim.methods.Method`` instance. New-shape
    exploration is an explicit decision: fixed-noise methods accept ``reset``
    or ``preserve``; FAME accepts ``original`` for its native fresh/meta choice.
    """
    import warm_start_io as io

    name = method.name
    if task not in SHAPES:
        raise ValueError("Unsupported warm-up shape.")
    if name == "fame":
        if new_shape_noise != "original":
            raise ValueError("FAME requires new_shape_noise='original': use the original fresh/meta Gaussian winner.")
    elif new_shape_noise not in ("reset", "preserve"):
        raise ValueError("Fixed-noise methods require an explicit reset or preserve choice.")
    if prepare_square:
        if parent is not None or task != "square" or square_policy is None:
            raise ValueError("Square preparation requires Square policy input, target square, and no completed-real parent.")
    elif parent is None:
        raise ValueError("A new shape requires its previous completed real artifact.")
    elif square_policy is not None or square_kae is not None:
        raise ValueError("Square source inputs cannot replace a completed-real predecessor.")

    started = time.perf_counter()
    reset_state = ["on_policy_rollout", "backend_environment", "phase_counters"]
    if prepare_square:
        if name == "residual":
            method.base = io.load_square_controller(square_policy, method.device)
        elif name == "componet":
            method.units = [PriorControllerUnit(io.load_square_controller(square_policy, method.device))]
        elif name == "kae_moe":
            if square_kae is None:
                raise ValueError("Square KAE-MoE preparation requires the supplied Square KAE.")
            method.experts = io.build_square_kae_experts(square_kae, method.device)
    else:
        if name == "residual":
            method.base = io.completed_deterministic_controller(parent, method.device)
        elif name == "componet":
            source = parent["architecture"]["policy"]
            priors = [io.decode_actor(item, method.device) for item in source["priors"]]
            priors.append(io.completed_deterministic_controller(parent, method.device))
            method.units = [PriorControllerUnit(controller) for controller in priors]
        elif name == "kae_moe":
            method.experts = io.build_kae_experts(parent, method.device, include_append=True)
        elif name == "fame":
            io.restore_fame_predecessor(method.agent, parent)
            _preserve_fame_input_normalization(method.agent, parent, io)

    # Only FAME distinguishes first/new-task by index. A backend switch is
    # never passed through this function: each invocation constructs one shape.
    index = 0 if prepare_square else SHAPES.index(task)
    agent = method.start_stage(index, task, str(stage_dir))
    if prepare_square and name in ("finetune", "fame"):
        io.initialize_square_agent(agent, name, policy_checkpoint=square_policy, kae_checkpoint=square_kae)
        reset_state.extend(("critic", "optimizer", "scheduler", "value_preprocessor") if name == "finetune" else
                           ("fast_critic", "fast_actor_optimizer", "fast_critic_optimizer", "temperature"))
    elif not prepare_square and name == "finetune":
        io.restore_finetune_agent(agent, parent)
    elif name in ("residual", "componet", "kae_moe"):
        reset_state.extend(("critic", "optimizer", "scheduler", "value_preprocessor"))
    if name == "fame" and not prepare_square:
        reset_state.extend(("fast_learner_original_task_switch", "fast_critic", "fast_actor_optimizer",
                            "fast_critic_optimizer", "temperature", "temperature_optimizer"))
    if name == "fame":
        choice = ("Initial" if prepare_square else
                  {"meta": "Meta", "fresh": "Fresh"}[agent.detection["choice"]])
        agent.games = [*getattr(agent, "games", []), task]
        agent.flag_reg = [*getattr(agent, "flag_reg", []), choice]

    if name != "fame":
        if new_shape_noise == "preserve":
            if prepare_square:
                io.initialize_square_exploration(agent, square_policy)
            else:
                with torch.no_grad():
                    agent.policy.log_std_parameter.copy_(_source_log_std(parent).to(method.device))
        else:
            with torch.no_grad():
                agent.policy.log_std_parameter.zero_()
            reset_state.append("exploration_log_std")

    detection = copy.deepcopy(getattr(agent, "detection", None))
    detection_steps = (2 * int(agent.cfg["detection_steps"]) if name == "fame" and not prepare_square else 0)
    metadata = {
        "entry": "square_method_preparation" if prepare_square else "next_shape_from_completed_real",
        "method": name, "shape": task, "native_stage_index": index,
        "native_start_stage_calls": 1, "native_end_stage_calls": 0,
        "new_shape_exploration": new_shape_noise,
        "current_task_modules_created": {
            "finetune": [], "residual": ["residual"],
            "componet": ["input_attention", "output_attention", "internal_policy"],
            "kae_moe": ["mode_weight_networks", "router", "gate", "mlp"],
            "fame": ["fast_actor", "fast_critic", "temperature"],
        }[name],
        "state_resets": reset_state,
        "preparation_seconds": time.perf_counter() - started,
        "auxiliary_detection_steps_per_environment": detection_steps,
        "auxiliary_detection_transitions": detection_steps * method.env.num_envs,
        "detection": detection,
        "parent_artifact_id": None if parent is None else parent["manifest"]["artifact_id"],
        "square_policy": None if square_policy is None else str(Path(square_policy).expanduser().resolve()),
        "square_kae": None if square_kae is None else str(Path(square_kae).expanduser().resolve()),
    }
    return agent, metadata
