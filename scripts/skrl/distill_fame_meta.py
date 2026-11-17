"""Distill FAME's meta learner from the shared Square's PPO agent, without training a new shape.

FAME's fast learner is the PPO agent of ``--method scratch``, so the shared Square's best checkpoint is a
finished FAME Square fast learner. This runs what ``Fame.end_stage`` runs after FAME's own first stage, through
``crazysim.warm_start.seed_from_square``: the checkpoint's stochastic policy flies the Square to collect
``size_fast2meta`` states, and the meta learner is distilled from it (settings in ``methods: fame:`` of
``skrl_ppo_waypoint_cfg.yaml``). Writes ``<output_dir>/meta_policy.pt``; nothing in the Square stage is changed.

A FAME warm-up then starts from both files:

    python scripts/skrl/train.py ... --method fame --warm_start \
        --square_checkpoint <stage>/checkpoints/best_agent.pt --square_meta_policy <stage>/fame_meta/meta_policy.pt

Example:
    python scripts/skrl/distill_fame_meta.py --headless \
        --stage_dir logs/skrl/crazyflie_tracking_waypoint/<session>/0_square env.trajectory_laps=1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distill FAME's meta learner from the shared Square's PPO agent.")
parser.add_argument("--stage_dir", type=str, required=True, help="Square stage with checkpoints/best_agent.pt.")
parser.add_argument(
    "--output_dir", type=str, default=None, help="Directory of meta_policy.pt. Default: <stage_dir>/fame_meta."
)
parser.add_argument("--env_id", type=str, default="CrazySim-TrackingWaypoint-Direct-v0", help="Name of the gym environment.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments. Default: as in training.")
parser.add_argument("--seed", type=int, default=None, help="Seed. Default: the agent configuration's (42).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import gymnasium as gym
import json
import os

from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.methods import METHODS
from crazysim.methods.base import BEST_CHECKPOINT
from crazysim.tasks.direct.tracking.waypoint_env_cfg import WaypointTrackingEnvCfg
from crazysim.warm_start import seed_from_square


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: WaypointTrackingEnvCfg, agent_cfg: dict):
    checkpoint = os.path.join(args_cli.stage_dir, BEST_CHECKPOINT)
    output_dir = os.path.abspath(args_cli.output_dir or os.path.join(args_cli.stage_dir, "fame_meta"))
    if os.path.exists(os.path.join(output_dir, "meta_policy.pt")):
        raise FileExistsError(f"{output_dir}/meta_policy.pt exists; remove it or pass another --output_dir.")

    # as train.py with --method fame on the Square
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.task = "square"
    env_cfg.debug_vis = False
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env = SkrlVecEnvWrapper(gym.make(args_cli.env_id, cfg=env_cfg), ml_framework="torch")

    set_seed(agent_cfg["seed"])
    method = METHODS["fame"](env, agent_cfg, output_dir)
    seed_from_square(method, checkpoint=checkpoint, work_dir=output_dir)

    summary = {
        "square_checkpoint": os.path.abspath(checkpoint),
        "meta_policy": os.path.join(output_dir, "meta_policy.pt"),
        "seed": agent_cfg["seed"],
        "num_envs": env.num_envs,
        "trajectory_laps": env_cfg.trajectory_laps,
        "distillation_states": method.meta_buffer.size,
        "fame": method.method_cfg,
        "command": " ".join(sys.orig_argv),
    }
    with open(os.path.join(output_dir, "distill_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Meta learner: {summary['meta_policy']}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
