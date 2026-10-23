"""Train policies on a sequence of Crazyflie tracking tasks with a transfer method.

All tasks run in one Isaac Sim process: the environment switches its reference trajectory between
tasks. The method (``--method``) decides how each task's agent is built from earlier tasks; see
``crazysim/methods``. After each task, the policy is evaluated on every task in the sequence, which
gives a transfer matrix (rows: trained through task i, columns: evaluated on task j).

Examples:
    # single task
    python scripts/skrl/train.py --tasks circle --headless
    # sequence; each task starts from the policy of the previous one
    python scripts/skrl/train.py --tasks circle rectangle figure8 --method finetune --headless
    # reference: every task trained from a fresh policy
    python scripts/skrl/train.py --tasks circle rectangle figure8 --method scratch --headless
    # transfer methods
    python scripts/skrl/train.py --tasks circle rectangle figure8 --method kae_moe --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# names of crazysim.methods.METHODS (that module can only be imported after the app is launched)
METHOD_NAMES = ["finetune", "scratch", "residual", "componet", "kae_moe", "fame"]

# add argparse arguments
parser = argparse.ArgumentParser(description="Train policies with skrl on a sequence of tracking tasks.")
parser.add_argument(
    "--tasks", type=str, nargs="+", default=["circle"], help="Task sequence, e.g. --tasks circle rectangle figure8."
)
parser.add_argument(
    "--method",
    type=str,
    default="finetune",
    choices=METHOD_NAMES,
    help=(
        "finetune: each task starts from the previous agent. scratch: each task starts from a new agent. "
        "residual, componet, kae_moe, fame: transfer methods (see crazysim/methods)."
    ),
)
parser.add_argument("--env_id", type=str, default="CrazySim-Tracking-Direct-v0", help="Name of the gym environment.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and the agent.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations per task.")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Agent checkpoint to initialize the first task (finetune, scratch)."
)
parser.add_argument("--run_name", type=str, default=None, help="Suffix of the session directory name.")
parser.add_argument("--no_eval", action="store_true", default=False, help="Skip the evaluation after each task.")

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
import random
import time
import torch
from datetime import datetime

import skrl
from packaging import version
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from torch.utils.tensorboard import SummaryWriter

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.evaluation import METRICS, evaluate_tasks, format_table, switch_task
from crazysim.methods import FINAL_CHECKPOINT, METHODS, POLICY_FILE
from crazysim.tasks.direct.tracking.tracking_env_cfg import TrackingEnvCfg

assert sorted(METHOD_NAMES) == sorted(METHODS), "METHOD_NAMES is out of sync with crazysim.methods.METHODS"


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: TrackingEnvCfg, agent_cfg: dict):
    """Train on every task of the sequence with the selected method."""
    tasks = args_cli.tasks
    unknown = [task for task in tasks if task not in env_cfg.tasks]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Available tasks: {list(env_cfg.tasks.keys())}")
    if args_cli.checkpoint and args_cli.method not in ("finetune", "scratch"):
        raise ValueError("--checkpoint is only supported by the finetune and scratch methods.")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.task = tasks[0]
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # session directory: logs/skrl/<experiment>/<time-stamp>_<method>_<tasks>[_<run_name>]/<index>_<task>/
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.method}_{'-'.join(tasks)}"
    if args_cli.run_name:
        session_name += f"_{args_cli.run_name}"
    session_dir = os.path.join(log_root_path, session_name)
    print(f"[INFO] Logging session in directory: {session_dir}")

    session = {
        "tasks": tasks,
        "method": args_cli.method,
        "seed": agent_cfg["seed"],
        "command": " ".join(sys.orig_argv),
        "stages": [],
    }

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = session_dir

    # create isaac environment and wrap it for skrl
    env = gym.make(args_cli.env_id, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # dump the configuration into log-directory (after the environment has set the observation size)
    dump_yaml(os.path.join(session_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(session_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(session_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(session_dir, "params", "agent.pkl"), agent_cfg)

    set_seed(agent_cfg["seed"])
    method_kwargs = {}
    if args_cli.method in ("finetune", "scratch") and args_cli.checkpoint:
        method_kwargs["checkpoint"] = retrieve_file_path(args_cli.checkpoint)
    method = METHODS[args_cli.method](env, agent_cfg, session_dir, **method_kwargs)
    trainer_cfg = {k: v for k, v in agent_cfg["trainer"].items() if k != "class"}

    transfer_writer = SummaryWriter(log_dir=os.path.join(session_dir, "transfer"))
    results = {}
    for index, task in enumerate(tasks):
        stage = f"{index}_{task}"
        stage_dir = os.path.join(session_dir, stage)
        print(f"\n[INFO] Stage {index + 1}/{len(tasks)}: training on '{task}' ({args_cli.method})")
        start_time = time.time()
        switch_task(env, task)

        agent = method.start_stage(index, task, stage_dir)
        SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg).train()
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        agent.save(os.path.join(stage_dir, FINAL_CHECKPOINT))
        if getattr(agent, "writer", None) is not None:
            agent.writer.close()

        policy = method.end_stage(index, task, stage_dir, agent).eval()
        torch.save(policy, os.path.join(stage_dir, POLICY_FILE))
        stage_info = {
            "stage": stage,
            "task": task,
            "policy": os.path.join(stage_dir, POLICY_FILE),
            "checkpoint": os.path.join(stage_dir, FINAL_CHECKPOINT),
            "minutes": (time.time() - start_time) / 60.0,
        }
        if getattr(agent, "detection", None):
            stage_info["detection"] = agent.detection
        stage_info.update(method.stage_info)
        session["stages"].append(stage_info)

        if not args_cli.no_eval:
            results[stage] = evaluate_tasks(env, policy, tasks)
            for evaluated_task, metrics in results[stage].items():
                for metric, value in metrics.items():
                    transfer_writer.add_scalar(f"Transfer/{metric}/{evaluated_task}", value, index)
            transfer_writer.flush()
            session["transfer"] = results
            for metric in METRICS:
                print(format_table(results, metric))

        with open(os.path.join(session_dir, "session.json"), "w") as f:
            json.dump(session, f, indent=2)
        del agent

    transfer_writer.close()
    print(f"\n[INFO] Session finished: {session_dir}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
