"""Simulation runs with trained policies (no training) for the analysis figures of ``plot_analysis.py``.

Parts (``--parts``, default all):

- ``traces``: every method's policy of stage i flies task i once in every environment, from the first point of the
  reference. Writes the flight path of environment 0 and, for KAE-MoE, the gate, router and mode weights at every
  step (environment 0) and their means over all environments and steps. Also checks how well each stage's KAE
  reproduces its policy's actions on the states the policy visits.
- ``ablation``: KAE-MoE policies of stages 1+ on their own task with the gate as learned, fixed to 0 (MLP pathway
  only) or fixed to 1 (KAE pathway only).
- ``jumpstart``: the policy each method starts stage i with (before any training), rebuilt with the method's own
  ``resume`` and ``start_stage``, on task i. New networks get fresh random weights (seed ``--seed`` + stage), so for
  them this estimates, not reproduces, the starting policy of the training run.

``ablation`` and ``jumpstart`` use the protocol of the transfer matrix (``crazysim.evaluation.evaluate_task``):
one episode per environment from random points of the reference.

Example:
    python scripts/skrl/analysis_rollouts.py --headless --metrics docs/figures/waypoint_5tasks_1/metrics.json \
        --output_dir docs/figures/waypoint_5tasks_1/analysis
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

PARTS = ("traces", "ablation", "jumpstart")
parser = argparse.ArgumentParser(description="Simulation runs with trained policies for the analysis figures.")
parser.add_argument("--metrics", required=True, help="JSON written by scripts/skrl/continual_metrics.py (lists the sessions).")
parser.add_argument("--output_dir", required=True, help="Directory for rollouts.json and traces.npz.")
parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
parser.add_argument("--env_id", type=str, default="CrazySim-TrackingWaypoint-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import os
import pickle
import tempfile
import torch
import torch.nn as nn

from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.evaluation import evaluate_task, switch_task
from crazysim.methods import METHODS, Method
from crazysim.methods.kae import KAEExperts, pad
from crazysim.tasks.direct.tracking.waypoint_env_cfg import WaypointTrackingEnvCfg

OBS_SAMPLES = 50000
"""Visited states kept per KAE-MoE stage for the KAE fidelity check."""
FIDELITY_PAIRS = 3000
"""(policy action, KAE action) pairs written for the scatter plot, per stage."""


class FixedGate(nn.Module):
    """KAE-MoE policy with the gate replaced by a constant (``None``: the learned gate)."""

    def __init__(self, policy: nn.Module, gate: float | None):
        super().__init__()
        if policy.preprocessor is not None:
            raise ValueError("FixedGate expects a policy without observation normalization.")
        self.policy = policy
        self.gate = gate

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        parts = self.policy.actor.components(obs)
        gate = parts["gate"] if self.gate is None else torch.full_like(parts["gate"], self.gate)
        action = gate * parts["kae_actions"] + (1.0 - gate) * parts["mlp_actions"]
        return torch.tanh(action) if self.policy.squash else action


def load_session(path: str) -> dict:
    with open(os.path.join(path, "session.json")) as f:
        session = json.load(f)
    with open(os.path.join(path, "params", "agent.pkl"), "rb") as f:
        session["agent_cfg"] = pickle.load(f)
    return session


def load(path: str, device) -> nn.Module:
    return torch.load(path, map_location=device, weights_only=False).eval()


@torch.no_grad()
def record(env, policy: nn.Module, task: str, moe: bool) -> dict[str, np.ndarray]:
    """One episode per environment from the first point of the reference (positions are spawn-noised)."""
    base = env.unwrapped
    switch_task(env, task)
    base.spread_episode_starts = False
    random_start_phase = base.cfg.random_start_phase
    base.cfg.random_start_phase = False
    obs, _ = env.reset()
    origin = base._terrain.env_origins

    active = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    crashed = torch.zeros_like(active)
    trace = {"position": [], "reference": [], "gate": [], "router": []}
    sums = {"gate": 0.0, "router": 0.0, "weights": 0.0, "abs_weights": 0.0, "count": 0}
    observations = []
    # keep the observations of every environment at every ``every``-th step: about OBS_SAMPLES in all
    every = max(1, int(base.max_episode_length) * base.num_envs // OBS_SAMPLES)
    error_sum = torch.zeros(base.num_envs, device=base.device)
    steps = torch.zeros_like(error_sum)
    for step in range(int(base.max_episode_length)):
        if active[0]:
            trace["position"].append(base._position()[0].cpu().numpy())
            trace["reference"].append((base._reference_pos[0] - origin[0]).cpu().numpy())
        if moe:
            parts = policy.actor.components(obs)
            action = parts["gate"] * parts["kae_actions"] + (1.0 - parts["gate"]) * parts["mlp_actions"]
            action = torch.tanh(action) if policy.squash else action
            if active[0]:
                trace["gate"].append(parts["gate"][0, 0].item())
                trace["router"].append(parts["router"][0].cpu().numpy())
            mask = active.float().unsqueeze(1)
            sums["gate"] += (parts["gate"] * mask).sum(0)
            sums["router"] += (parts["router"] * mask).sum(0)
            sums["weights"] += (parts["expert_weights"] * mask).sum(0)
            sums["abs_weights"] += (parts["expert_weights"].abs() * mask).sum(0)
            sums["count"] += int(active.sum())
        else:
            action = policy(obs)
        if step % every == 0:
            observations.append(obs[active].clone())
        obs, _, terminated, truncated, _ = env.step(action)
        terminated, truncated = terminated.view(-1), truncated.view(-1)
        error_sum += base.position_error * active
        steps += active
        crashed |= terminated & active
        active &= ~(terminated | truncated)
        if not active.any():
            break

    base.cfg.random_start_phase = random_start_phase
    base.spread_episode_starts = True
    result = {key: np.array(value) for key, value in trace.items() if value}
    result["crashed_env0"] = np.array(bool(crashed[0]))
    result["position_error"] = np.array((error_sum / steps.clamp(min=1)).mean().item())
    result["crash_rate"] = np.array(crashed.float().mean().item())
    if moe:
        for key in ("gate", "router", "weights", "abs_weights"):
            result[f"mean_{key}"] = (sums[key] / sums["count"]).cpu().numpy()
    result["observations"] = torch.cat(observations)[:OBS_SAMPLES]
    return result


@torch.no_grad()
def kae_fidelity(policy: nn.Module, kae_path: str, observations: torch.Tensor, cfg: dict, action_dim: int) -> dict:
    """How well the KAE fitted to ``policy`` reproduces its actions on the states the policy visited."""
    kae = KAEExperts(torch.load(kae_path, map_location=observations.device, weights_only=False), action_dim, cfg["p"])
    kae = kae.to(observations.device)
    target = policy(observations)
    prediction = kae(pad(observations, cfg["padded_dim"])).sum(dim=1)
    mse = ((prediction - target) ** 2).mean(dim=0)
    variance = target.var(dim=0)
    eigvals = kae.eigvals_p  # eigenvalues of K to the power p
    pick = torch.randperm(observations.shape[0], device=observations.device)[:FIDELITY_PAIRS]
    return {
        "r2_per_action": (1.0 - mse / variance).tolist(),
        "mse_per_action": mse.tolist(),
        "variance_per_action": variance.tolist(),
        "eigvals_p_real": eigvals.real.tolist(),
        "eigvals_p_imag": eigvals.imag.tolist(),
        "samples": observations.shape[0],
        "_pairs": (target[pick].cpu().numpy(), prediction[pick].cpu().numpy()),
    }


def starting_policy(env, session: dict, index: int, work_dir: str) -> tuple[nn.Module, dict]:
    """The policy ``session``'s method starts stage ``index`` with, before any training."""
    method: Method = METHODS[session["method"]](env, session["agent_cfg"], work_dir)
    stages = session["stages"][:index]
    if index > 0:
        if session["method"] == "fame":
            # resume() also refills the distillation buffer, which only the next distillation needs
            meta = method.load_module(os.path.join(os.path.dirname(stages[-1]["policy"]), "meta_policy.pt"))
            method.meta = meta.train()
        else:
            method.resume(stages)
    task = session["stages"][index]["task"]
    switch_task(env, task)
    set_seed(args_cli.seed + index)
    agent = method.start_stage(index, task, os.path.join(work_dir, f"{index}_{task}"))
    if getattr(method, "_gate_hook", None) is not None:
        method._gate_hook.remove()
    info = {"detection": getattr(agent, "detection", None)}
    return method.ppo_policy(agent).eval(), info


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: WaypointTrackingEnvCfg, agent_cfg: dict):
    with open(args_cli.metrics) as f:
        metrics = json.load(f)
    sessions = {method: load_session(result["session"]) for method, result in metrics["methods"].items()}
    tasks = next(iter(sessions.values()))["tasks"]

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = False
    env = SkrlVecEnvWrapper(gym.make(args_cli.env_id, cfg=env_cfg), ml_framework="torch")
    device = env.device
    action_dim = env.action_space.shape[0]

    os.makedirs(args_cli.output_dir, exist_ok=True)
    json_path = os.path.join(args_cli.output_dir, "rollouts.json")
    results = {"num_envs": args_cli.num_envs, "seed": args_cli.seed, "tasks": tasks}
    if os.path.isfile(json_path):
        with open(json_path) as f:
            results.update(json.load(f))

    def write():
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    if "traces" in args_cli.parts:
        arrays = {}
        results["traces"] = {}
        results["kae_fidelity"] = {}
        for method, session in sessions.items():
            for index, stage in enumerate(session["stages"]):
                policy = load(stage["policy"], device)
                moe = method == "kae_moe" and index > 0
                print(f"[INFO] traces: {method} stage {index} on {stage['task']}")
                torch.manual_seed(args_cli.seed)
                trace = record(env, policy, stage["task"], moe)
                observations = trace.pop("observations")
                results["traces"].setdefault(method, {})[stage["task"]] = {
                    "position_error": float(trace["position_error"]), "crash_rate": float(trace["crash_rate"]),
                    "crashed_env0": bool(trace["crashed_env0"]),
                }
                for key, value in trace.items():
                    arrays[f"{method}/{stage['task']}/{key}"] = value
                if method == "kae_moe":
                    kae_path = os.path.join(os.path.dirname(stage["policy"]), "kae", "kae.pt")
                    fidelity = kae_fidelity(policy, kae_path, observations, session["agent_cfg"]["methods"]["kae_moe"],
                                            action_dim)
                    target, prediction = fidelity.pop("_pairs")
                    arrays[f"kae_fidelity/{stage['task']}/policy"] = target
                    arrays[f"kae_fidelity/{stage['task']}/kae"] = prediction
                    results["kae_fidelity"][stage["task"]] = fidelity
                    print(f"[INFO] KAE of stage {index}: R^2 per action {np.round(fidelity['r2_per_action'], 3)}")
        np.savez_compressed(os.path.join(args_cli.output_dir, "traces.npz"), **arrays)
        write()

    if "ablation" in args_cli.parts and "kae_moe" in sessions:
        results["ablation"] = {}
        for index, stage in enumerate(sessions["kae_moe"]["stages"][1:], start=1):
            policy = load(stage["policy"], device)
            for name, gate in (("learned", None), ("mlp_only", 0.0), ("kae_only", 1.0)):
                torch.manual_seed(args_cli.seed)
                result = evaluate_task(env, FixedGate(policy, gate), stage["task"])
                results["ablation"].setdefault(stage["task"], {})[name] = result
                print(f"[INFO] ablation: stage {index} {stage['task']} gate {name}: {result}")
            write()

    if "jumpstart" in args_cli.parts:
        results["jumpstart"] = {}
        with tempfile.TemporaryDirectory() as work_dir:
            for method, session in sessions.items():
                for index, stage in enumerate(session["stages"]):
                    policy, info = starting_policy(env, session, index, work_dir)
                    torch.manual_seed(args_cli.seed)
                    result = evaluate_task(env, policy, stage["task"])
                    result.update(info)
                    results["jumpstart"].setdefault(method, {})[stage["task"]] = result
                    print(f"[INFO] jumpstart: {method} stage {index} on {stage['task']}: {result}")
                    del policy
                    torch.cuda.empty_cache()
                write()

    write()
    print(f"[INFO] Results written to {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
