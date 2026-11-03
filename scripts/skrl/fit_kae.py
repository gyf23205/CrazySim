"""Fit a Koopman autoencoder (KAE) to a trained policy of the trajectory-setpoint controller.

States are drawn from a Gaussian fitted to observations visited by the policy with its training noise over whole
episodes (a random subset, so the data covers the full reference and not only the time right after a reset).
Settings are in the ``kae:`` section of ``skrl_ppo_waypoint_cfg.yaml``. Writes ``<stage>/kae/``:

    kae.pt             whole module (crazysim.methods.kae.KoopmanAutoencoder)
    kae_state.pt       encoder and decoder state dictionaries and K, read by export_waypoint_artifacts.py
    kae_summary.json, kae_history.csv

Example:
    python scripts/skrl/fit_kae.py --headless --task square --stage_dir logs/skrl/crazyflie_tracking_waypoint/<session>/0_square
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fit a KAE to a trained policy.")
parser.add_argument("--stage_dir", type=str, required=True, help="Stage directory with policy.pt and checkpoints/best_agent.pt.")
parser.add_argument("--task", type=str, required=True, help="Task the observations are collected on.")
parser.add_argument("--env_id", type=str, default="CrazySim-TrackingWaypoint-Direct-v0", help="Name of the gym environment.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--seed", type=int, default=0, help="Seed of the observation rollout.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.evaluation import load_policy
from crazysim.methods.base import BEST_CHECKPOINT, POLICY_FILE
from crazysim.methods.kae import train_kae
from crazysim.methods.squashed import SquashedNormal
from crazysim.tasks.direct.tracking.waypoint_env_cfg import WaypointTrackingEnvCfg


@torch.no_grad()
def collect_observations(env, policy, log_std: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Random subset of the observations of one full-length rollout with the policy's training noise."""
    obs, _ = env.reset()
    observations = []
    for _ in range(int(env.unwrapped.max_episode_length)):
        observations.append(obs.clone())
        obs, _, _, _, _ = env.step(SquashedNormal(policy(obs), log_std).sample())
    observations = torch.cat(observations)
    indices = torch.randperm(observations.shape[0], device=observations.device)[:num_samples]
    return observations[indices]


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: WaypointTrackingEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = False
    env_cfg.task = args_cli.task
    env = SkrlVecEnvWrapper(gym.make(args_cli.env_id, cfg=env_cfg), ml_framework="torch")

    policy = load_policy(os.path.join(args_cli.stage_dir, POLICY_FILE), env)
    checkpoint = torch.load(os.path.join(args_cli.stage_dir, BEST_CHECKPOINT), map_location=env.device, weights_only=False)
    log_std = checkpoint["policy"]["log_std_parameter"]
    print(f"[INFO] Action noise std before the tanh: {log_std.exp().tolist()}")

    cfg = dict(agent_cfg["kae"])
    padded_dim, observable_dim, p = cfg.pop("padded_dim"), cfg.pop("observable_dim"), cfg.pop("p")
    observations = collect_observations(env, policy, log_std, cfg.pop("observation_samples"))
    print(f"[INFO] Fitting the KAE on {observations.shape[0]} observations ({args_cli.task})")

    kae_dir = os.path.join(args_cli.stage_dir, "kae")
    with torch.enable_grad():
        kae = train_kae(policy, observations, padded_dim, observable_dim, p, log_dir=kae_dir, **cfg)
    torch.save(kae, os.path.join(kae_dir, "kae.pt"))
    torch.save(
        {
            "encoder": kae.encoder.state_dict(),
            "decoder": kae.decoder.state_dict(),
            "K": kae.K.detach(),
            "state_dim": padded_dim,
            "hidden_dim": cfg["hidden_dim"],
            "observable_dim": observable_dim,
        },
        os.path.join(kae_dir, "kae_state.pt"),
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
