"""Fly logged position setpoints of a real flight through the simulated firmware controller and compare the motion.

Validates the simulated plant and ``firmware_controller.py`` against a flight of ``Drone_Test_TRO``: the commands of
a flight (``commanded_*`` in ``all_episodes.csv``) are replayed open loop, without a policy, from the logged state.

Example:
    python scripts/skrl/replay_flight.py --headless --csv <flight>/all_episodes.csv --output replay.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay the position setpoints of a real flight in simulation.")
parser.add_argument("--csv", type=str, nargs="+", required=True, help="all_episodes.csv files of real flights.")
parser.add_argument("--env_id", type=str, default="CrazySim-TrackingWaypoint-Direct-v0")
parser.add_argument("--max_episodes", type=int, default=5, help="Episodes replayed per file.")
parser.add_argument("--output", type=str, default=None, help="Path of a JSON file to write the results to.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import csv
import gymnasium as gym
import json
import numpy as np
import torch

from isaaclab_tasks.utils.hydra import hydra_task_config

import crazysim  # noqa: F401
from crazysim.tasks.direct.tracking.waypoint_env_cfg import WaypointTrackingEnvCfg

AXES = ("x", "y", "z")


def read_episodes(path: str, max_episodes: int) -> dict[str, dict[str, np.ndarray]]:
    """Tracking rows of the first episodes of a flight log, as arrays per column."""
    episodes: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["tracking_active"].strip().lower() != "true":
                continue
            if row["episode_index"] not in episodes and len(episodes) == max_episodes:
                break
            episodes.setdefault(row["episode_index"], []).append(row)
    columns = {
        "actual": [f"actual_{a}" for a in AXES],
        "commanded": [f"commanded_{a}" for a in AXES],
        "velocity": [f"telemetry_world_v{a}" for a in AXES],
        "attitude": ["telemetry_roll_deg", "telemetry_pitch_deg", "telemetry_yaw_deg"],
    }
    result = {}
    for name, rows in episodes.items():
        if len(rows) < 50:
            continue
        result[name] = {key: np.array([[float(r[c]) for c in cols] for r in rows]) for key, cols in columns.items()}
    return result


def replay(env, episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fly the commands from the logged initial state. Returns the simulated position, velocity and attitude."""
    device = env.device
    origin = env._terrain.env_origins
    env.reset()
    state = env._robot.data.default_root_state.clone()
    state[:, :3] = torch.tensor(episode["actual"][0], device=device, dtype=torch.float32) + origin
    state[:, 7:10] = torch.tensor(episode["velocity"][0], device=device, dtype=torch.float32)
    env._robot.write_root_pose_to_sim(state[:, :7])
    env._robot.write_root_velocity_to_sim(state[:, 7:])
    env._firmware.reset(env._robot._ALL_INDICES, env._hover_motor_thrust, env._position())
    env.episode_length_buf[:] = 0

    commands = torch.tensor(episode["commanded"], device=device, dtype=torch.float32)
    position, velocity, attitude = [], [], []
    for command in commands:
        env._command_from_action = lambda _actions, c=command: c.unsqueeze(0)
        env.step(torch.zeros(1, env.cfg.action_space, device=device))
        position.append(env._position()[0].cpu().numpy())
        velocity.append(env._robot.data.root_lin_vel_w[0].cpu().numpy())
        attitude.append(env._attitude_deg()[0].cpu().numpy())
    return {"actual": np.array(position), "velocity": np.array(velocity), "attitude": np.array(attitude)}


def compare(real: dict[str, np.ndarray], sim: dict[str, np.ndarray], dt: float) -> dict:
    position_gap = np.linalg.norm(real["actual"] - sim["actual"], axis=1)

    def lag_s(signal_real, signal_sim):
        """Time shift of the simulated horizontal velocity that best matches the flight, by cross-correlation."""
        best = max(
            range(-50, 51),
            key=lambda k: float(np.sum(signal_real[max(0, -k) : len(signal_real) - max(0, k)]
                                       * signal_sim[max(0, k) : len(signal_sim) - max(0, -k)])),
        )
        return best * dt

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values))))

    return {
        "steps": len(position_gap),
        "position_gap_mean_m": float(position_gap.mean()),
        "position_gap_max_m": float(position_gap.max()),
        "real_command_gap_m": float(np.linalg.norm(real["commanded"] - real["actual"], axis=1).mean()),
        "sim_command_gap_m": float(np.linalg.norm(real["commanded"] - sim["actual"], axis=1).mean()),
        "velocity_rms_error_ms": rms(real["velocity"] - sim["velocity"]),
        "velocity_rms_real_ms": rms(real["velocity"]),
        "velocity_rms_sim_ms": rms(sim["velocity"]),
        "sim_velocity_delay_vx_s": lag_s(real["velocity"][:, 0], sim["velocity"][:, 0]),
        "sim_velocity_delay_vy_s": lag_s(real["velocity"][:, 1], sim["velocity"][:, 1]),
        "tilt_rms_real_deg": rms(real["attitude"][:, :2]),
        "tilt_rms_sim_deg": rms(sim["attitude"][:, :2]),
    }


@hydra_task_config(args_cli.env_id, "skrl_cfg_entry_point")
def main(env_cfg: WaypointTrackingEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = False
    env_cfg.random_start_phase = False
    # the replay ends with the flight; the crash checks of the environment would reset it
    env_cfg.tasks[env_cfg.task].period = 100000.0
    for name in ("max_acceleration_g", "max_tilt_deg", "max_angular_velocity_deg", "speed_limit"):
        setattr(env_cfg, name, 1.0e6)
    env_cfg.min_height, env_cfg.max_height, env_cfg.max_horizontal_distance = -1.0e6, 1.0e6, 1.0e6
    env = gym.make(args_cli.env_id, cfg=env_cfg).unwrapped
    print(f"[INFO] mass {env._robot_mass.item():.5f} kg, weight {env._robot_weight:.4f} N")
    print(f"[INFO] inertia {env._inertia.diagonal().tolist()} kg m^2")
    print(f"[INFO] hover command {env._hover_motor_thrust / env_cfg.motor_thrust_max * 65535:.0f}")

    results = {}
    for path in args_cli.csv:
        for name, episode in read_episodes(path, args_cli.max_episodes).items():
            with torch.no_grad():
                sim = replay(env, episode)
            result = compare(episode, sim, env.step_dt)
            results[f"{path}:{name}"] = result
            print(f"[INFO] {path} episode {name}: {json.dumps(result)}")

    keys = next(iter(results.values())).keys()
    results["mean"] = {k: float(np.mean([r[k] for r in results.values()])) for k in keys}
    print(f"[INFO] mean: {json.dumps(results['mean'], indent=2)}")
    if args_cli.output:
        with open(args_cli.output, "w") as f:
            json.dump(results, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
