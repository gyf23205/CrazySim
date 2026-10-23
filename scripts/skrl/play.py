"""Visualize a trained policy on one tracking task.

Example:
    python scripts/skrl/play.py --task figure8 --checkpoint logs/skrl/crazyflie_tracking/<session>/2_figure8/policy.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained policy on a tracking task.")
parser.add_argument("--task", type=str, default="circle", help="Task to fly, e.g. circle, rectangle, figure8.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a policy.pt file (or an agent checkpoint of an older run).")
parser.add_argument("--env_id", type=str, default="CrazySim-Tracking-Direct-v0", help="Name of the gym environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video instead of running forever.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.evaluation import load_policy, stage_dir_of
from crazysim.tasks.direct.tracking.tracking_env_cfg import TrackingEnvCfg


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: TrackingEnvCfg, agent_cfg: dict):
    """Play with skrl agent."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.task = args_cli.task
    checkpoint = os.path.abspath(retrieve_file_path(args_cli.checkpoint))

    env = gym.make(args_cli.env_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(stage_dir_of(checkpoint), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": args_cli.task,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    print(f"[INFO] Loading policy from: {checkpoint}")
    policy = load_policy(checkpoint, env)
    dt = env.unwrapped.step_dt

    obs, _ = env.reset()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(policy(obs))
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
