"""Shared pieces of the transfer methods.

A method decides how the agent of each stage (one task of the sequence) is built and what is kept
from earlier stages. ``scripts/skrl/train.py`` drives every method the same way:

    agent = method.start_stage(index, task, stage_dir)   # build or re-initialize the agent
    SequentialTrainer(env, agent).train()                # train on the task
    policy = method.end_stage(index, task, stage_dir, agent)

``policy`` is a plain ``nn.Module`` mapping observations to deterministic actions. It is saved as
``<stage_dir>/policy.pt`` (the whole module, frozen parts included), so evaluation and playback
do not depend on how the method built it.
"""

from __future__ import annotations

import copy
import math
import os
import torch
import torch.nn as nn
from collections.abc import Sequence

from skrl.agents.torch import Agent
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

FINAL_CHECKPOINT = os.path.join("checkpoints", "final_agent.pt")
"""Agent checkpoint written at the end of each stage, relative to the stage directory."""
BEST_CHECKPOINT = os.path.join("checkpoints", "best_agent.pt")
"""skrl's checkpoint with the highest mean training episode return, relative to the stage directory."""
POLICY_FILE = "policy.pt"
"""Deterministic policy module written at the end of each stage, relative to the stage directory."""


def make_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> nn.Sequential:
    """Fully connected network with ELU activations."""
    layers = []
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ELU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


def network_layers(agent_cfg: dict, role: str) -> list[int]:
    """Hidden layer sizes of the ``policy`` or ``value`` network in the skrl agent configuration."""
    return list(agent_cfg["models"][role]["network"][0]["layers"])


class GaussianPolicy(GaussianMixin, Model):
    """Gaussian policy with a state-independent log standard deviation around any mean network."""

    def __init__(self, observation_space, action_space, device, actor: nn.Module, policy_cfg: dict):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=policy_cfg["clip_actions"],
            clip_log_std=policy_cfg["clip_log_std"],
            min_log_std=policy_cfg["min_log_std"],
            max_log_std=policy_cfg["max_log_std"],
        )
        self.actor = actor
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(policy_cfg["initial_log_std"])))

    def compute(self, inputs, role):
        return self.actor(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """State-value network."""

    def __init__(self, observation_space, action_space, device, hidden_dims: Sequence[int]):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = make_mlp(self.num_observations, hidden_dims, 1)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class _ModelRunner(Runner):
    """skrl Runner that uses given models and leaves the trainer to the caller."""

    def __init__(self, env, cfg, models: dict):
        self._given_models = models
        super().__init__(env, cfg)

    def _generate_models(self, env, cfg):
        return {"agent": self._given_models}

    def _generate_trainer(self, env, cfg, agent):
        return None


class DeterministicPolicy(nn.Module):
    """Observation preprocessing followed by the mean action of an actor."""

    def __init__(self, actor: nn.Module, preprocessor: nn.Module | None = None):
        super().__init__()
        self.preprocessor = preprocessor
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.preprocessor is not None:
            obs = self.preprocessor(obs)
        return self.actor(obs)


class Method:
    """Base class of a transfer method. Subclasses override :meth:`start_stage` and :meth:`end_stage`."""

    name: str = ""
    reuses_policies: bool = True
    """True when frozen policies of earlier stages run inside later stages. Those policies take raw
    observations, so the agent must not normalize observations."""

    def __init__(self, env: SkrlVecEnvWrapper, agent_cfg: dict, session_dir: str):
        self.env = env
        self.agent_cfg = agent_cfg
        self.session_dir = session_dir
        self.device = env.device
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.method_cfg = agent_cfg.get("methods", {}).get(self.name, {})
        if self.reuses_policies and agent_cfg["agent"].get("state_preprocessor"):
            raise ValueError(f"Method '{self.name}' requires 'agent.state_preprocessor: null'.")

    stage_info: dict = {}
    """Extra information about the last finished stage, written to session.json by the training script."""

    def start_stage(self, index: int, task: str, stage_dir: str) -> Agent:
        raise NotImplementedError

    def end_stage(self, index: int, task: str, stage_dir: str, agent: Agent) -> nn.Module:
        raise NotImplementedError

    # -- helpers for PPO-based methods --------------------------------------------------------

    def build_ppo_agent(self, actor: nn.Module, stage_dir: str, checkpoint_interval: int | None = None) -> Agent:
        """skrl PPO agent with the configured hyperparameters, ``actor`` as mean network and a new value network.

        ``checkpoint_interval`` (timesteps) overrides the configured interval. skrl also updates its best
        checkpoint (``checkpoints/best_agent.pt``, highest mean training episode return) at that interval.
        """
        cfg = copy.deepcopy(self.agent_cfg)
        cfg.pop("methods", None)
        cfg["agent"]["experiment"]["directory"] = stage_dir
        cfg["agent"]["experiment"]["experiment_name"] = ""
        if checkpoint_interval is not None:
            cfg["agent"]["experiment"]["checkpoint_interval"] = checkpoint_interval
        models = {
            "policy": GaussianPolicy(
                self.env.observation_space, self.env.action_space, self.device, actor, cfg["models"]["policy"]
            ),
            "value": Value(
                self.env.observation_space, self.env.action_space, self.device, network_layers(cfg, "value")
            ),
        }
        agent = _ModelRunner(self.env, cfg, models).agent
        # skrl appends "<time-stamp>_<agent>" to an empty experiment name; log directly into the stage directory
        agent.experiment_dir = stage_dir
        return agent

    def new_mlp_actor(self) -> nn.Module:
        """Plain MLP mean network with the configured policy layers."""
        return make_mlp(self.obs_dim, network_layers(self.agent_cfg, "policy"), self.action_dim).to(self.device)

    @staticmethod
    def ppo_policy(agent: Agent) -> DeterministicPolicy:
        """Deterministic policy of a PPO agent built by :meth:`build_ppo_agent`."""
        preprocessor = agent._state_preprocessor if isinstance(agent._state_preprocessor, nn.Module) else None
        return DeterministicPolicy(agent.policy.actor, copy.deepcopy(preprocessor))

    def collect_observations(self, agent: Agent, num_samples: int) -> torch.Tensor:
        """Observations visited by the agent's stochastic policy, continuing from the current env state."""
        obs, _ = self.env.reset()  # the skrl wrapper returns the current observation after its first reset
        observations = []
        with torch.no_grad():
            for _ in range(math.ceil(num_samples / self.env.num_envs)):
                observations.append(obs.clone())
                actions = agent.act(obs, timestep=0, timesteps=0)[0]
                obs, _, _, _, _ = self.env.step(actions)
        return torch.cat(observations)[-num_samples:]


def freeze(module: nn.Module) -> nn.Module:
    """Disable gradients of every parameter and switch to evaluation mode."""
    for param in module.parameters():
        param.requires_grad = False
    return module.eval()
