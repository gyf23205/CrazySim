"""KAE + mixture of experts (MoE).

Ported from ``less_leg_walking_1`` (``MoE.py``, ``scripts/rsl_rl/train_moe_CRL.py``, ``train_moe.py``).

Stage 0 trains a plain MLP policy. At the end of every stage, a Koopman autoencoder (KAE) is fitted
to that stage's policy (:mod:`.kae`). Stage k trains a policy that blends two pathways:

- KAE pathway: the eigen-modes of the KAEs of stages 0..k-1 are the experts. A router network splits
  a unit L1 budget between the KAEs, and one weight network per KAE weighs its modes.
- MLP pathway: a new MLP on the observation.

A gate network outputs ``g`` in [0, 1] and the action is ``g * kae_action + (1 - g) * mlp_action``.
The KAEs are frozen; all networks above are trained. The value network and the action noise are new
at every stage.
"""

from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn

from .base import Method, make_mlp
from .kae import KAEExperts, pad, train_kae


class MoEActor(nn.Module):
    def __init__(self, experts: list[KAEExperts], obs_dim: int, action_dim: int, padded_dim: int, cfg: dict):
        super().__init__()
        self.kaes = nn.ModuleList(experts)
        self.padded_dim = padded_dim
        self.total_modes = sum(e.num_modes for e in experts)

        # MLP pathway
        self.mlp_network = make_mlp(obs_dim, cfg["residual_hidden_dims"], action_dim)
        # one mode-weight network per KAE, starting at equal weights
        self.expert_weight_networks = nn.ModuleList(
            make_mlp(obs_dim, cfg["weight_hidden_dims"], e.num_modes) for e in experts
        )
        for network in self.expert_weight_networks:
            nn.init.zeros_(network[-1].weight)
            nn.init.constant_(network[-1].bias, 1.0)
        # gate between the two pathways, starting at sigmoid(1) = 0.73
        self.gating_network = make_mlp(obs_dim, cfg["gating_hidden_dims"], 1)
        nn.init.zeros_(self.gating_network[-1].weight)
        nn.init.constant_(self.gating_network[-1].bias, 1.0)
        # router that splits the unit L1 budget between KAEs
        self.kae_router_network = make_mlp(obs_dim, cfg["router_hidden_dims"], len(experts))

        self.last_gate = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            padded_obs = pad(obs, self.padded_dim)
            experts_outputs = torch.cat([kae(padded_obs) for kae in self.kaes], dim=1)  # (batch, modes, action)

        kae_router = torch.softmax(self.kae_router_network(obs), dim=1)
        branch_weights = []
        for index, weight_net in enumerate(self.expert_weight_networks):
            branch_w = weight_net(obs)
            branch_w = branch_w / (branch_w.abs().sum(dim=1, keepdim=True) + 1e-6)
            branch_weights.append(branch_w * kae_router[:, index : index + 1])
        expert_weights = torch.cat(branch_weights, dim=1)
        kae_actions = torch.sum(expert_weights.view(-1, self.total_modes, 1) * experts_outputs, dim=1)

        mlp_actions = self.mlp_network(obs)
        gate = torch.sigmoid(self.gating_network(obs))
        if not torch.is_grad_enabled():
            self.last_gate = gate.detach()
        return gate * kae_actions + (1.0 - gate) * mlp_actions

    def train(self, mode: bool = True):
        super().train(mode)
        self.kaes.eval()
        return self


class KaeMoe(Method):
    name = "kae_moe"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experts: list[KAEExperts] = []
        """KAE experts of the finished stages, in order."""
        self._gate_hook = None
        if self.method_cfg["padded_dim"] < max(self.obs_dim, self.action_dim):
            raise ValueError(
                f"agent.methods.kae_moe.padded_dim={self.method_cfg['padded_dim']} is smaller than the observation "
                f"size {self.obs_dim}."
            )

    def start_stage(self, index, task, stage_dir):
        if not self.experts:
            return self.build_ppo_agent(self.new_mlp_actor(), stage_dir)

        actor = MoEActor(
            [copy.deepcopy(e) for e in self.experts],
            self.obs_dim,
            self.action_dim,
            self.method_cfg["padded_dim"],
            self.method_cfg,
        ).to(self.device)
        agent = self.build_ppo_agent(actor, stage_dir)

        def log_gate(module, inputs, output):
            if module.last_gate is not None and not torch.is_grad_enabled():
                agent.track_data("Policy / MoE gate (mean)", module.last_gate.mean().item())

        self._gate_hook = actor.register_forward_hook(log_gate)
        return agent

    def end_stage(self, index, task, stage_dir, agent):
        if self._gate_hook is not None:
            # hooks are saved with the module; remove it before the policy is copied and saved
            self._gate_hook.remove()
            self._gate_hook = None
        policy = self.ppo_policy(agent)

        cfg = self.method_cfg
        kae_cfg = dict(cfg["kae"])
        observations = self.collect_observations(agent, kae_cfg.pop("observation_samples"))
        print(f"[INFO] Fitting the KAE of stage {index} ({task}) on {observations.shape[0]} observations")
        kae_dir = os.path.join(stage_dir, "kae")
        kae = train_kae(
            policy, observations, cfg["padded_dim"], cfg["observable_dim"], cfg["p"], log_dir=kae_dir, **kae_cfg
        )
        torch.save(kae, os.path.join(kae_dir, "kae.pt"))
        self.experts.append(KAEExperts(kae, self.action_dim, cfg["p"]).to(self.device))
        return policy
