"""FAME (Anand and Precup, 2023, "Prediction and Control in Continual Reinforcement Learning";
fast and meta learners), adapted to continuous actions with PPO.

Ported from ``less_leg_walking_1`` (``baselines/FAME/``, ``scripts/skrl/train_FAME.py``).

- Fast learner: the same skrl PPO agent as the other methods (``build_ppo_agent`` with a new MLP
  policy and value network, settings in ``models:`` and ``agent:`` of the agent configuration). A
  new one is built at every task.
- Meta learner (:class:`MetaPolicy`): a mean network of the same shape as the fast learner's, with a
  state-independent log standard deviation. It is distilled from the fast learner at the end of every
  task with a Wasserstein-like loss on the Gaussian ``(mean, std)``. Earlier knowledge is kept by also
  matching the previous meta learner on states stored from earlier tasks.
- Task switch: the deterministic actions of the new fast learner and of the meta learner are both
  evaluated on the new task; the better one initializes the fast learner's policy (mean network and
  log standard deviation; the value network stays new).

Changes from the reference:

- The fast learner is the PPO agent of the other methods, not the reference's PPO variant
  (tanh-squashed actions, state-dependent noise, entropy temperature tuned toward a target
  entropy, own learning rates and initialization). Only the continual-learning parts of FAME
  differ between ``fame`` and ``scratch``.
- The states used for the distillation are the last ``size_fast2meta`` states visited by the
  stage's best fast learner (stochastic actions) on its task, collected after training. The reference
  keeps the most recent training states, which were visited by the last actor, not the best one.
- Buffers are tensors filled for all environments at once.

Resuming a session (:meth:`Fame.resume`) loads the last meta learner. The buffer of earlier-task
states is not saved, so it is refilled with states visited by each stage's best fast learner on its
task, as many per task as during training.
"""

from __future__ import annotations

import copy
import math
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from skrl.agents.torch import Agent

from .base import Method


class MetaPolicy(nn.Module):
    """Meta learner: mean network and state-independent log standard deviation.

    Calling it returns the deterministic action, like a stage's ``policy.pt``.
    """

    def __init__(self, actor: nn.Module, log_std: torch.Tensor):
        super().__init__()
        self.actor = actor
        self.log_std = nn.Parameter(log_std.detach().clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class StateBuffer:
    """FIFO buffer of states on the GPU; keeps the most recent ``capacity`` rows."""

    def __init__(self, capacity: int, dim: int, device):
        self.data = torch.zeros(capacity, dim, device=device)
        self.capacity = capacity
        self.index = 0
        self.size = 0

    def add(self, states: torch.Tensor):
        states = states[-self.capacity :]
        n = states.shape[0]
        idx = (self.index + torch.arange(n, device=states.device)) % self.capacity
        self.data[idx] = states
        self.index = (self.index + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        return self.data[torch.randint(self.size, (batch_size,), device=self.data.device)]

    def clear(self):
        self.index = self.size = 0


class Fame(Method):
    name = "fame"
    reuses_policies = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.agent_cfg["agent"].get("state_preprocessor"):
            # the meta learner is distilled on raw observations and copied into new agents
            raise ValueError("Method 'fame' requires 'agent.state_preprocessor: null'.")
        self.meta: MetaPolicy | None = None
        """Built at the end of the first stage, so that the first stage's fast learner starts from the same
        random state as the first stage of the other methods."""
        self.meta_buffer = StateBuffer(self.method_cfg["size_meta"], self.obs_dim, self.device)

    # -- stages ----------------------------------------------------------------------------------

    def start_stage(self, index, task, stage_dir):
        agent = self.build_ppo_agent(self.new_mlp_actor(), stage_dir)
        agent.detection = None
        if self.meta is not None:
            steps = self.method_cfg["detection_steps"]
            avg_fresh = self._evaluate(agent.policy.actor, steps)
            avg_meta = self._evaluate(self.meta.actor, steps)
            choice = "meta" if avg_meta > avg_fresh else "fresh"
            if choice == "meta":
                with torch.no_grad():
                    agent.policy.actor.load_state_dict(self.meta.actor.state_dict())
                    agent.policy.log_std_parameter.copy_(self.meta.log_std)
            print(f"[FAME] Detection on {task}: fresh return {avg_fresh:.3f}, meta return {avg_meta:.3f} -> {choice}")
            agent.detection = {"fresh_return": avg_fresh, "meta_return": avg_meta, "choice": choice}
        return agent

    def end_stage(self, index, task, stage_dir, agent):
        self.load_best_checkpoint(agent, stage_dir)
        states = self._visited_states(agent, task)
        print(f"[FAME] Collected {states.shape[0]} states of {task} with the best fast learner")
        start = time.time()
        self._train_meta(states, agent.policy)
        print(f"[FAME] Meta distillation done in {time.time() - start:.1f} s")
        self.meta_buffer.add(states)
        torch.save(copy.deepcopy(self.meta).eval(), os.path.join(stage_dir, "meta_policy.pt"))
        return self.ppo_policy(agent)

    def resume(self, stages):
        meta = self.load_module(os.path.join(os.path.dirname(stages[-1]["policy"]), "meta_policy.pt"))
        if not isinstance(meta, MetaPolicy):
            raise ValueError("The session was made with FAME's earlier PPO variant; its meta learner cannot be resumed.")
        self.meta = meta.train()
        self.meta_buffer.clear()
        for stage in stages:
            agent = self.build_ppo_agent(self.new_mlp_actor(), os.path.dirname(stage["policy"]))
            agent.load(stage["selected_checkpoint"])
            states = self._visited_states(agent, stage["task"])
            self.meta_buffer.add(states)
            print(f"[FAME] Refilled the meta buffer with {states.shape[0]} states of {stage['task']}")

    # -- internals ---------------------------------------------------------------------------------

    def _std(self, log_std: torch.Tensor) -> torch.Tensor:
        """Standard deviation of the action noise, with the log clamped as in the PPO policy."""
        cfg = self.agent_cfg["models"]["policy"]
        if self._squashed_policy:
            return log_std.clamp(min=float(cfg.get("min_log_std", -20.0))).exp()
        if cfg["clip_log_std"]:
            log_std = log_std.clamp(cfg["min_log_std"], cfg["max_log_std"])
        return log_std.exp()

    def _train_meta(self, new_states: torch.Tensor, fast_policy: nn.Module):
        """Distill the fast learner on this task's states and the previous meta learner on earlier states."""
        cfg = self.method_cfg
        batch_size = cfg["meta_batch_size"]
        previous_meta = None
        if self.meta is None:
            policy_cfg = self.agent_cfg["models"]["policy"]
            self.meta = MetaPolicy(
                self.new_mlp_actor(), torch.full((self.action_dim,), float(policy_cfg["initial_log_std"]))
            ).to(self.device)
        elif self.meta_buffer.size >= batch_size:
            previous_meta = copy.deepcopy(self.meta)
        new_buffer = StateBuffer(new_states.shape[0], self.obs_dim, self.device)
        new_buffer.add(new_states)

        optimizer = Adam(self.meta.parameters(), lr=cfg["lr_meta"])
        scheduler = ExponentialLR(optimizer, gamma=cfg["meta_lr_scheduler_gamma"])
        steps = max(1, max(self.meta_buffer.size, new_buffer.size) // batch_size - 1)

        def wasserstein(states, teacher_mean, teacher_log_std):
            with torch.no_grad():
                mean, std = teacher_mean(states), self._std(teacher_log_std)
            m_mean, m_std = self.meta.actor(states), self._std(self.meta.log_std)
            return torch.mean(torch.square(m_mean - mean).sum(-1)) + torch.square(m_std - std).sum(-1)

        for epoch in range(cfg["epoch_meta"]):
            for _ in range(steps):
                # new knowledge: match the fast learner on this task's states
                loss = wasserstein(new_buffer.sample(batch_size), fast_policy.actor, fast_policy.log_std_parameter)
                if previous_meta is not None:  # old knowledge: match the previous meta learner on past states
                    loss = loss + wasserstein(self.meta_buffer.sample(batch_size), previous_meta.actor,
                                              previous_meta.log_std)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch == 0 or (epoch + 1) % 50 == 0:
                print(f"[FAME] Meta epoch {epoch + 1}/{cfg['epoch_meta']} loss={loss.item():.3e}")
            if (epoch + 1) % 2 == 0:
                scheduler.step()

    def _evaluate(self, actor: nn.Module, num_steps: int) -> float:
        """Mean return of the episodes that end within ``num_steps`` steps of the deterministic actor."""
        self.env._reset_once = True
        obs, _ = self.env.reset()
        returns = torch.zeros(self.env.num_envs, device=self.device)
        completed = []
        with torch.no_grad():
            for _ in range(num_steps):
                obs, rewards, terminated, truncated, _ = self.env.step(actor(obs))
                returns += rewards.view(-1)
                done = (terminated | truncated).view(-1)
                completed.append(returns[done])
                returns[done] = 0.0
        completed = torch.cat(completed)
        # the trainer's first reset must reset the environment again
        self.env._reset_once = True
        return completed.mean().item() if completed.numel() else 0.0

    def _visited_states(self, agent: Agent, task: str) -> torch.Tensor:
        """The last ``size_fast2meta`` states visited by the agent's stochastic policy on ``task``.

        One episode length of warm-up comes first, so that the episodes are spread over their length
        as in training.
        """
        capacity = self.method_cfg["size_fast2meta"]
        self.env.unwrapped.set_task(task)
        self.env._reset_once = True
        obs, _ = self.env.reset()
        warmup = int(self.env.unwrapped.max_episode_length)
        states = []
        with torch.no_grad():
            for step in range(warmup + math.ceil(capacity / self.env.num_envs)):
                if step >= warmup:
                    states.append(obs.clone())
                obs, _, _, _, _ = self.env.step(agent.act(obs, timestep=0, timesteps=0)[0])
        self.env._reset_once = True
        return torch.cat(states)[-capacity:]
