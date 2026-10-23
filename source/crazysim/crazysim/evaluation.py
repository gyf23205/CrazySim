"""Evaluate a policy on a list of tracking tasks in one running environment."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import yaml
from collections.abc import Callable

from isaaclab_rl.skrl import SkrlVecEnvWrapper

METRICS = ("position_error", "crash_rate", "return")
"""Keys of the per-task result dictionary returned by :func:`evaluate_tasks`."""


def switch_task(env: SkrlVecEnvWrapper, task: str):
    """Set the active task. The next ``env.reset()`` then resets every environment.

    The skrl wrapper only forwards its first ``reset()`` call to the environment, so it is re-armed here.
    """
    env.unwrapped.set_task(task)
    env._reset_once = True


@torch.no_grad()
def evaluate_task(env: SkrlVecEnvWrapper, policy: Callable[[torch.Tensor], torch.Tensor], task: str) -> dict:
    """Run one full-length episode in every environment and summarize the tracking quality.

    Uses ``no_grad`` rather than ``inference_mode``: tensors the environment creates here are reused in
    later training, where in-place updates on inference tensors are not allowed.

    Returns:
        ``position_error``: mean distance to the reference [m] over the steps before each episode ends.
        ``crash_rate``: fraction of episodes that ended by termination instead of time-out.
        ``return``: mean undiscounted episode return.
    """
    base_env = env.unwrapped
    switch_task(env, task)
    base_env.spread_episode_starts = False
    obs, _ = env.reset()

    active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    crashed = torch.zeros_like(active)
    error_sum = torch.zeros(base_env.num_envs, device=base_env.device)
    reward_sum = torch.zeros_like(error_sum)
    steps = torch.zeros_like(error_sum)
    for _ in range(int(base_env.max_episode_length)):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        terminated, truncated = terminated.view(-1), truncated.view(-1)
        error_sum += base_env.position_error * active
        reward_sum += reward.view(-1) * active
        steps += active
        crashed |= terminated & active
        active &= ~(terminated | truncated)
        if not active.any():
            break

    base_env.spread_episode_starts = True
    return {
        "position_error": (error_sum / steps.clamp(min=1)).mean().item(),
        "crash_rate": crashed.float().mean().item(),
        "return": reward_sum.mean().item(),
    }


def evaluate_tasks(
    env: SkrlVecEnvWrapper, policy: Callable[[torch.Tensor], torch.Tensor], tasks: list[str]
) -> dict[str, dict]:
    """Evaluate ``policy`` on each task. The environment is left on the last evaluated task."""
    return {task: evaluate_task(env, policy, task) for task in tasks}


def load_policy(path: str, env: SkrlVecEnvWrapper) -> Callable[[torch.Tensor], torch.Tensor]:
    """Deterministic policy from a file written by ``scripts/skrl/train.py``.

    ``path`` is either a stage's ``policy.pt`` (a whole ``nn.Module``, any method), or a skrl agent
    checkpoint of a run whose agent configuration has ``models: separate: False`` (runs made before
    the transfer methods were added). For the latter, ``<session>/params/agent.yaml`` must exist.
    """
    data = torch.load(path, map_location=env.device, weights_only=False)
    if isinstance(data, nn.Module):
        return data.eval()

    from skrl.utils.runner.torch import Runner

    session_cfg = os.path.join(os.path.dirname(path), "..", "..", "params", "agent.yaml")
    with open(session_cfg) as f:
        agent_cfg = yaml.safe_load(f)
    if agent_cfg["models"].get("separate", False):
        raise ValueError(f"Load the stage's policy.pt instead of the agent checkpoint {path}.")
    # no logging or checkpoint files for inference
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent = Runner(env, agent_cfg).agent
    agent.load(path)
    agent.set_running_mode("eval")

    def policy(obs: torch.Tensor) -> torch.Tensor:
        outputs = agent.act(obs, timestep=0, timesteps=0)
        return outputs[-1].get("mean_actions", outputs[0])

    return policy


def stage_dir_of(path: str) -> str:
    """Stage directory of a ``policy.pt`` (``<stage>/policy.pt``) or agent checkpoint (``<stage>/checkpoints/*.pt``)."""
    parent = os.path.dirname(os.path.abspath(path))
    return os.path.dirname(parent) if os.path.basename(parent) == "checkpoints" else parent


def format_table(results: dict[str, dict[str, dict]], metric: str) -> str:
    """Format ``results[trained_through][evaluated_on][metric]`` as a text table."""
    rows = list(results.keys())
    cols = list(next(iter(results.values())).keys()) if rows else []
    width = max([len(metric) + 2] + [len(name) + 2 for name in rows + cols])
    lines = [f"{metric:<{width}}" + "".join(f"{c:>{width}}" for c in cols)]
    for r in rows:
        lines.append(f"{r:<{width}}" + "".join(f"{results[r][c][metric]:>{width}.4f}" for c in cols))
    return "\n".join(lines)
