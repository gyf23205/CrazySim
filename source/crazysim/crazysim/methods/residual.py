"""Residual policy learning across a task sequence.

Ported from ``less_leg_walking_1`` (``res_net.py``, ``scripts/rsl_rl/train_res_CRL.py``).

Stage 0 trains a plain MLP policy. Every later stage trains a new residual MLP on top of the frozen
policy of the previous stage: ``action = base(obs) + residual(obs)``. With a squashed Gaussian policy
the residual is added in pre-tanh space, as in ``cf_residual_real.py`` of ``Drone_Test_TRO``:
``action = tanh(atanh(base(obs)) + residual(obs))``. The residual's output layer starts at zero, so the
stage starts by reproducing the base. At the end of the stage, base and
residual are frozen together and become the next stage's base. The value network and the action
noise are new at every stage.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn

from .base import Method, freeze, make_mlp
from .squashed import prior_to_pre_tanh


class ComposedActor(nn.Module):
    """Frozen base policy plus a trainable residual.

    With ``pre_tanh``, the output is the pre-tanh mean ``atanh(base) + residual`` of a squashed Gaussian
    policy. Actors saved before ``pre_tanh`` existed load with ``pre_tanh=False``.
    """

    pre_tanh: bool = False

    def __init__(self, base: nn.Module, residual: nn.Module, pre_tanh: bool = False):
        super().__init__()
        self.base = freeze(base)
        self.residual = residual
        self.pre_tanh = pre_tanh

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_actions = self.base(obs)
            if self.pre_tanh:
                base_actions = prior_to_pre_tanh(base_actions)
        return base_actions + self.residual(obs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self


class Residual(Method):
    name = "residual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base: nn.Module | None = None

    def start_stage(self, index, task, stage_dir):
        if self.base is None:
            actor = self.new_mlp_actor()
        else:
            residual = make_mlp(self.obs_dim, self.method_cfg["hidden_dims"], self.action_dim).to(self.device)
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
            actor = ComposedActor(copy.deepcopy(self.base), residual, pre_tanh=self._squashed_policy)
        return self.build_ppo_agent(actor, stage_dir)

    def end_stage(self, index, task, stage_dir, agent):
        self.load_best_checkpoint(agent, stage_dir)
        policy = self.ppo_policy(agent)
        self.base = freeze(copy.deepcopy(policy))
        return policy

    def resume(self, stages):
        self.base = freeze(self.load_module(stages[-1]["policy"]))
