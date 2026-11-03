"""Fly a policy and its KAE as ``Drone_Test_TRO`` flies them and report the same tracking metrics.

Episodes start at the first point of the reference, at rest, and run two laps. The KAE flies directly (no mixture
of experts); ``kae_modes`` is the same KAE as the sum of its eigen-modes, which should match it.

Example:
    python scripts/skrl/validate_waypoint.py --headless --tasks square circle triangle figure8 \
        --stage_dir logs/skrl/crazyflie_tracking_waypoint/<session>/0_square --output validation.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate a policy and its KAE on the trajectory-setpoint controller.")
parser.add_argument("--stage_dir", type=str, required=True, help="Stage directory with policy.pt and kae/kae.pt.")
parser.add_argument("--tasks", type=str, nargs="+", default=["square"], help="Tasks to fly.")
parser.add_argument("--env_id", type=str, default="CrazySim-TrackingWaypoint-Direct-v0", help="Name of the gym environment.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments (one episode each).")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment.")
parser.add_argument("--output", type=str, default=None, help="Path of a JSON file to write the results to.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.evaluation import load_policy, switch_task
from crazysim.methods.base import POLICY_FILE
from crazysim.methods.kae import KAEExperts, pad
from crazysim.tasks.direct.tracking.waypoint_env_cfg import WaypointTrackingEnvCfg


@torch.no_grad()
def fly(env, policy, task: str) -> dict:
    """One episode in every environment; the metrics of the quality check of Drone_Test_TRO and more."""
    base_env = env.unwrapped
    switch_task(env, task)
    base_env.spread_episode_starts = False
    obs, _ = env.reset()
    active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    crashed = torch.zeros_like(active)
    success = torch.zeros_like(active)
    causes = {}
    keys = ("steps", "error", "error_squared", "cross_squared", "within", "reward", "out_of_bounds")
    sums = {key: torch.zeros(base_env.num_envs, device=base_env.device) for key in keys}
    for _ in range(int(base_env.max_episode_length)):
        action = policy(obs)
        sums["out_of_bounds"] += (action.abs() > 1.0).float().mean(dim=1) * active
        # finished episodes are reset within step(); these buffers still describe the step
        obs, reward, terminated, truncated, _ = env.step(action)
        terminated, truncated = terminated.view(-1), truncated.view(-1)
        error = base_env.position_error
        sums["steps"] += active
        sums["error"] += error * active
        sums["error_squared"] += torch.square(error) * active
        sums["cross_squared"] += torch.square(base_env.cross_error) * active
        sums["within"] += (error <= base_env.cfg.success_distance_threshold).float() * active
        sums["reward"] += reward.view(-1) * active
        success |= base_env.episode_success & active
        for key, value in base_env.crash_causes.items():
            causes[key] = causes.get(key, torch.zeros_like(active)) | (value & active & terminated)
        crashed |= terminated & active
        active &= ~(terminated | truncated)
        if not active.any():
            break
    base_env.spread_episode_starts = True
    steps = sums["steps"].clamp(min=1)
    return {
        "position_error_m": (sums["error"] / steps).mean().item(),
        "position_rmse_m": torch.sqrt(sums["error_squared"] / steps).mean().item(),
        "cross_track_rmse_m": torch.sqrt(sums["cross_squared"] / steps).mean().item(),
        "within_0.25m": (sums["within"] / steps).mean().item(),
        "success_rate": success.float().mean().item(),
        "crash_rate": crashed.float().mean().item(),
        "crash_causes": {key: value.float().mean().item() for key, value in causes.items() if value.any()},
        "return": sums["reward"].mean().item(),
        "output_out_of_bounds_fraction": (sums["out_of_bounds"] / steps).mean().item(),
        "mean_steps": sums["steps"].mean().item(),
    }


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: WaypointTrackingEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = False
    env_cfg.random_start_phase = False
    env_cfg.max_position_error = 1.0e9
    env_cfg.task = args_cli.tasks[0]
    env = SkrlVecEnvWrapper(gym.make(args_cli.env_id, cfg=env_cfg), ml_framework="torch")

    controllers = {"policy": load_policy(os.path.join(args_cli.stage_dir, POLICY_FILE), env)}
    kae_path = os.path.join(args_cli.stage_dir, "kae", "kae.pt")
    if os.path.isfile(kae_path):
        kae = torch.load(kae_path, map_location=env.device, weights_only=False)
        padded_dim, action_dim = kae.state_dim, env.action_space.shape[0]
        controllers["kae"] = lambda obs: kae(pad(obs, padded_dim))[2][:, :action_dim]
        experts = KAEExperts(kae, action_dim).to(env.device)
        controllers["kae_modes"] = lambda obs: experts(pad(obs, padded_dim)).sum(dim=1)

    results = {
        name: {task: fly(env, controller, task) for task in args_cli.tasks} for name, controller in controllers.items()
    }
    print(json.dumps(results, indent=2))
    if args_cli.output:
        with open(args_cli.output, "w") as f:
            json.dump(results, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
