"""CompoNet (Malagon et al., 2024, "Self-Composing Policies for Scalable Continual Reinforcement Learning").

Ported from ``less_leg_walking_1`` (``baselines/CompoNet.py``, ``scripts/rsl_rl/train_componet_CRL.py``),
which follows the reference implementation (github.com/mikelma/componet) for continuous actions
(``ret_probs=False``). The ablation option ``prevs_to_noise`` and the optional extra return values
of the forward pass are left out.

Stage 0 trains a plain MLP policy (wrapped in :class:`FirstModuleWrapper`). Stage k builds a new
CompoNet unit over the frozen units of stages 0..k-1. The unit's output attention head combines the
earlier units' actions; its input attention head and internal policy add a correction whose last
layer starts near zero, so a new unit starts close to the attention-weighted earlier policies.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Method, freeze


def get_position_encoding(seq_len: int, d: int, n: int = 10_000) -> np.ndarray:
    """Sinusoidal positional encoding matrix of shape ``(seq_len, d)``."""
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


class CompoNet(nn.Module):
    """One CompoNet unit.

    As the last unit, ``forward(s)`` runs the frozen previous units and returns ``(action, phi)``,
    where ``phi`` stacks the actions of all units ``(batch, num_units, out_dim)``. As a previous
    unit, ``forward((phi, s))`` appends its action to ``phi`` and returns ``(phi, s)``.
    """

    def __init__(
        self,
        previous_units: list[nn.Module],
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        internal_policy: nn.Module,
        device: torch.device | str,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.internal_policy = internal_policy
        self.att_temp = np.sqrt(hidden_dim)
        self.is_prev = False

        # linear transformations of the output attention head
        self.headout_wq = nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        self.headout_wk = nn.Linear(out_dim, hidden_dim, bias=proj_bias)
        # linear transformations of the input attention head
        self.headin_wq = nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        self.headin_wk = nn.Linear(out_dim, hidden_dim, bias=proj_bias)
        self.headin_wv = nn.Linear(out_dim, hidden_dim, bias=proj_bias)

        # positional encodings: input head sees the previous units and the output head result,
        # output head sees the previous units only (no encoding needed for a single one)
        n_prev = len(previous_units)
        pe1 = torch.tensor(get_position_encoding(n_prev + 1, out_dim), dtype=torch.float32, device=device)
        self.register_buffer("pe1", pe1[None, :, :])
        self.register_buffer("pe0", self.pe1[:, :-1, :].clone() if n_prev >= 2 else None)

        for unit in previous_units:
            if hasattr(unit, "previous_units"):
                del unit.previous_units
            unit.is_prev = True
            freeze(unit)
        self.previous_units = nn.Sequential(*previous_units)

    def _forward_headout(self, s, phi):
        query = self.headout_wq(s)
        keys = self.headout_wk(phi + self.pe0 if self.pe0 is not None else phi)
        values = phi
        w = torch.matmul(query[:, None, :], keys.permute(0, 2, 1))  # (batch, 1, num_policies)
        att = F.softmax(w / self.att_temp, dim=-1)
        return torch.matmul(att, values), att  # (batch, 1, out_dim)

    def _get_internal_policy(self, s, phi):
        query = self.headin_wq(s)
        values = self.headin_wv(phi)
        keys = self.headin_wk(phi + self.pe1)
        w = torch.matmul(query[:, None, :], keys.permute(0, 2, 1))  # (batch, 1, num_policies)
        att = F.softmax(w / self.att_temp, dim=-1)
        att_dot_val = torch.matmul(att, values)[:, 0, :]
        policy_out = self.internal_policy(torch.hstack([att_dot_val, s]))
        return policy_out, att

    def forward(self, s):
        if not self.is_prev:
            with torch.no_grad():
                phi, _ = self.previous_units(s)
        else:
            phi, s = s

        out_head, _ = self._forward_headout(s, phi)
        logits, _ = self._get_internal_policy(s, torch.cat([phi, out_head], dim=1))
        out = out_head[:, 0, :] + logits

        phi = torch.cat([phi, out[:, None, :]], dim=1)
        if self.is_prev:
            return phi, s
        return out, phi

    def train(self, mode: bool = True):
        super().train(mode)
        # a unit used as a previous unit no longer holds its own previous units
        if hasattr(self, "previous_units"):
            self.previous_units.eval()
        return self


class FirstModuleWrapper(nn.Module):
    """Wraps a plain state-to-action network so that it can be the first CompoNet unit."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.is_prev = False

    def forward(self, x):
        pi = self.model(x)
        phi = pi[:, None, :]  # (batch, 1, out_dim)
        if self.is_prev:
            return phi, x
        return pi, phi


class ActionOnly(nn.Module):
    """Returns only the action of a CompoNet unit."""

    def __init__(self, unit: nn.Module):
        super().__init__()
        self.unit = unit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.unit(obs)[0]


class Componet(Method):
    name = "componet"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units: list[nn.Module] = []
        """Units of the finished stages, in order."""

    def _internal_policy(self) -> nn.Sequential:
        hidden_dims = self.method_cfg["internal_hidden_dims"]
        in_dim = self.obs_dim + hidden_dims[0]
        layers = []
        for h in hidden_dims:
            linear = nn.Linear(in_dim, h)
            nn.init.orthogonal_(linear.weight, np.sqrt(2))
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.ELU()]
            in_dim = h
        # near-zero output layer: the unit starts as its output attention head over earlier units
        final = nn.Linear(in_dim, self.action_dim)
        nn.init.orthogonal_(final.weight, 0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        return nn.Sequential(*layers)

    def start_stage(self, index, task, stage_dir):
        if not self.units:
            unit = FirstModuleWrapper(self.new_mlp_actor())
        else:
            unit = CompoNet(
                previous_units=[copy.deepcopy(u) for u in self.units],
                input_dim=self.obs_dim,
                hidden_dim=self.method_cfg["internal_hidden_dims"][0],
                out_dim=self.action_dim,
                internal_policy=self._internal_policy(),
                device=self.device,
            ).to(self.device)
        return self.build_ppo_agent(ActionOnly(unit), stage_dir)

    def end_stage(self, index, task, stage_dir, agent):
        self.load_best_checkpoint(agent, stage_dir)
        policy = self.ppo_policy(agent)
        self.units.append(copy.deepcopy(agent.policy.actor.unit))
        return policy

    def resume(self, stages):
        # policy.pt of a stage is DeterministicPolicy(ActionOnly(unit))
        self.units = [self.load_module(stage["policy"]).actor.unit for stage in stages]
