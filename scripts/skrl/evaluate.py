"""Evaluate trained policies on a list of tracking tasks.

Prints one table per metric (rows: policies, columns: tasks) and optionally writes them to JSON.

Example:
    python scripts/skrl/evaluate.py --headless --tasks circle rectangle figure8 \
        --checkpoint logs/skrl/crazyflie_tracking/<session>/*/policy.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate trained policies on tracking tasks.")
parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Tasks to evaluate on. Defaults to all tasks.")
parser.add_argument("--checkpoint", type=str, nargs="+", required=True, help="One or more policy.pt files (or agent checkpoints of older runs).")
parser.add_argument("--env_id", type=str, default="CrazySim-Tracking-Direct-v0", help="Name of the gym environment.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments (one episode each).")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment.")
parser.add_argument("--output", type=str, default=None, help="Path of a JSON file to write the results to.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.evaluation import METRICS, evaluate_tasks, format_table, load_policy, stage_dir_of
from crazysim.tasks.direct.tracking.tracking_env_cfg import TrackingEnvCfg


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: TrackingEnvCfg, agent_cfg: dict):
    """Evaluate every checkpoint on every task."""
    tasks = args_cli.tasks or list(env_cfg.tasks.keys())
    unknown = [task for task in tasks if task not in env_cfg.tasks]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Available tasks: {list(env_cfg.tasks.keys())}")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = False

    env = gym.make(args_cli.env_id, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    checkpoints = [os.path.abspath(retrieve_file_path(path)) for path in args_cli.checkpoint]
    # label: path relative to the inputs' common directory, at least <stage>/<file> (e.g. 0_circle/policy.pt)
    label_root = os.path.commonpath([os.path.dirname(stage_dir_of(c)) for c in checkpoints])
    results = {}
    for checkpoint in checkpoints:
        label = os.path.relpath(checkpoint, label_root)
        print(f"[INFO] Evaluating {checkpoint}")
        results[label] = evaluate_tasks(env, load_policy(checkpoint, env), tasks)

    for metric in METRICS:
        print(format_table(results, metric))
    if args_cli.output:
        with open(args_cli.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Results written to {args_cli.output}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
