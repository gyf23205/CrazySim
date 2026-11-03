"""Compare the simulated control scheme with the code of ``Drone_Test_TRO`` on random inputs.

Needs ``qfly`` and that repository's dependencies, not Isaac Sim (conda env ``FIAD`` here). Compares the observation,
the action decoding, the command constraints, the reward and the shapes:

    python scripts/skrl/check_waypoint_contract.py --drone_test_dir ~/git/Drone_Test_TRO
"""

import argparse
import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--drone_test_dir", type=str, default=os.path.expanduser("~/git/Drone_Test_TRO"))
parser.add_argument("--samples", type=int, default=2000)
parser.add_argument("--shapes_output", type=str, default="waypoint_shapes.npz", help="Reference table of the shapes.")
args = parser.parse_args()

setpoint_dir = os.path.join(args.drone_test_dir, "examples", "setpoint")
sys.path[:0] = [args.drone_test_dir, setpoint_dir]

import cf_setpoint_ppo_real_common as real  # noqa: E402
from examples.trajectory_setpoints import observation as real_observation  # noqa: E402
from qfly import Pose  # noqa: E402

tracking_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "crazysim", "crazysim", "tasks", "direct", "tracking"
)
spec = importlib.util.spec_from_file_location("waypoint_control", os.path.join(tracking_dir, "waypoint_control.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

rng = np.random.default_rng(0)
config = real.SetpointRuntimeConfig(observation_mode="trajectory_setpoints", control_period_s=0.02)
limit = torch.tensor(config.waypoint_offset_limit, dtype=torch.float64)
position_scale = torch.tensor(config.observation_position_error_scale, dtype=torch.float64)
worst = {}


def record(name: str, sim_value, real_value):
    difference = float(np.max(np.abs(np.asarray(sim_value, dtype=np.float64) - np.asarray(real_value, dtype=np.float64))))
    worst[name] = max(worst.get(name, 0.0), difference)


world = real.build_world()
setpoints = rng.uniform(-1.0, 1.0, (config.trajectory_setpoint_count, 3)) + np.array([0.0, 0.0, 0.75])
for _ in range(args.samples):
    pose = Pose(*rng.uniform([-1.5, -1.5, 0.2], [1.5, 1.5, 1.8]), yaw=0.0)
    position = torch.tensor([[pose.x, pose.y, pose.z]], dtype=torch.float64)
    velocity = rng.uniform(-1.5, 1.5, 3)
    previous = Pose(*(np.array([pose.x, pose.y, pose.z]) + rng.uniform(-0.3, 0.3, 3)), yaw=0.0)
    previous_t = torch.tensor([[previous.x, previous.y, previous.z]], dtype=torch.float64)
    action = rng.uniform(-1.0, 1.0, 3)
    action_t = torch.tensor(action, dtype=torch.float64).unsqueeze(0)
    phase = float(rng.uniform(0.0, 1.0))

    record(
        "observation",
        sim.build_observation(
            position, torch.tensor(velocity, dtype=torch.float64).unsqueeze(0), torch.tensor(setpoints), torch.tensor([phase], dtype=torch.float64),
            previous_t, position_scale, config.observation_velocity_scale, limit,
        )[0].numpy(),
        real_observation.build_observation(
            pose, velocity, setpoints, phase, previous,
            position_scale=config.observation_position_error_scale,
            velocity_scale=config.observation_velocity_scale,
            waypoint_offset_limit=config.waypoint_offset_limit,
        ),
    )

    target = real_observation.action_to_waypoint(action, pose, config.waypoint_offset_limit)
    target_t = sim.action_to_waypoint(action_t, position, limit)
    record("action_to_waypoint", target_t[0].numpy(), [target.x, target.y, target.z])

    # command constraints, called on a stand-in for the runner that holds only what they read
    runner = SimpleNamespace(
        config=config, world=world, backend=SimpleNamespace(pose=pose), _last_command_target=real.copy_pose(previous)
    )
    runner._rate_limit_command = lambda t, r=runner: real.SetpointPolicyRunner._rate_limit_command(r, t)
    clamped = real.clamp_pose_to_test_geofence(real.copy_pose(target), world, config.xy_geofence_limit)
    clamped.clamp(world)
    clamped_t = sim.clamp_to_geofence(target_t, config.xy_geofence_limit, 0.0, 9.85)
    record("geofence", clamped_t[0].numpy(), [clamped.x, clamped.y, clamped.z])

    slew = real.SetpointPolicyRunner._rate_limit_command(runner, real.copy_pose(clamped))
    slew_t = sim.slew_limit(clamped_t, previous_t, torch.tensor(config.command_rate_limit, dtype=torch.float64), config.control_period_s)
    slew_t = sim.clamp_to_geofence(slew_t, config.xy_geofence_limit, 0.0, 9.85)
    record("slew_limit", slew_t[0].numpy(), [slew.x, slew.y, slew.z])

    runner._last_command_target = real.copy_pose(previous)
    prox = real.SetpointPolicyRunner._proximity_limit_command(runner, real.copy_pose(clamped))
    prox_t = sim.proximity_limit(clamped_t, position, torch.tensor(config.command_proximity_radius, dtype=torch.float64))
    prox_t = sim.clamp_to_geofence(prox_t, config.xy_geofence_limit, 0.0, 9.85)
    record("proximity_limit", prox_t[0].numpy(), [prox.x, prox.y, prox.z])

    # reward at the pose after the step, against the reference at the end of the step
    next_pose = Pose(*(np.array([pose.x, pose.y, pose.z]) + rng.uniform(-0.05, 0.05, 3)), yaw=0.0)
    next_t = torch.tensor([[next_pose.x, next_pose.y, next_pose.z]], dtype=torch.float64)
    reference = Pose(*(np.array([pose.x, pose.y, pose.z]) + rng.uniform(-1.0, 1.0, 3)), yaw=0.0)
    reference_t = torch.tensor([[reference.x, reference.y, reference.z]], dtype=torch.float64)
    reference_velocity = rng.uniform(-0.3, 0.3, 3) * float(rng.uniform() > 0.1)
    desired_action = real_observation.waypoint_to_action(reference, pose, config.waypoint_offset_limit)
    desired_action_t = sim.waypoint_to_action(reference_t, position, limit)
    record("desired_action", desired_action_t[0].numpy(), desired_action)
    report = {
        "values": {"pose_ready": True, "telemetry_ready": True, "velocity": dict(zip(("vx", "vy", "vz"), velocity))},
        "overall_safe": True,
    }
    components = real.compute_reward_components(
        next_pose, report, reference, tuple(reference_velocity), action, target, slew, True, config.control_period_s,
        config, real.TrackingEpisodeStats(), desired_action=desired_action,
    )
    terms, metrics = sim.reward_components(
        next_t, torch.tensor(velocity, dtype=torch.float64).unsqueeze(0), reference_t, torch.tensor(reference_velocity, dtype=torch.float64).unsqueeze(0),
        action_t, desired_action_t, config.control_period_s,
        torch.tensor(config.position_precision_scale, dtype=torch.float64), torch.tensor(config.velocity_precision_scale, dtype=torch.float64),
        config.coarse_position_distance, config.velocity_gate_distance, config.alive_reward_scale,
        config.coarse_position_reward_scale, config.precise_position_reward_scale, config.velocity_reward_scale,
        config.waypoint_error_reward_scale,
    )
    for key, value in terms.items():
        record(f"reward_{key}", value[0].item(), components[key])
    record("metric_position_error", metrics["position_error"][0].item(), components["metrics"]["position_error"])
    record("metric_cross_error", metrics["cross_error"][0].item(), math.sqrt(components["metrics"]["cross_error_squared"]))

print(f"{'quantity':<28}largest absolute difference over {args.samples} random inputs")
for name, value in worst.items():
    print(f"{name:<28}{value:.3e}")

# reference table of the shapes, read by scripts/skrl/check_waypoint_shapes.py
from cf_circle_setpoint_ppo_real import SPEC as circle  # noqa: E402
from cf_figure8_setpoint_ppo_real import SPEC as figure8  # noqa: E402
from cf_square_setpoint_ppo_real import SPEC as square  # noqa: E402
from cf_triangle_setpoint_ppo_real import SPEC as triangle  # noqa: E402

tables = {}
for name, shape in (("square", square), ("circle", circle), ("triangle", triangle), ("figure8", figure8)):
    period = shape.period_fn(config)
    times = np.linspace(0.0, 2.0 * period, 4001)[:-1] + 0.00123
    tables[f"{name}_period"] = period
    tables[f"{name}_times"] = times
    tables[f"{name}_position"] = np.array([shape.offset_fn(t, config) for t in times]) + np.array([0.0, 0.0, config.target_height])
    tables[f"{name}_velocity"] = np.array([shape.velocity_fn(t, config) for t in times])
    tables[f"{name}_setpoints"] = real_observation.lap_shape_offsets(
        shape.offset_fn, shape.center_fn, period, config.trajectory_setpoint_count, config
    ) + np.array([0.0, 0.0, config.target_height])
np.savez(args.shapes_output, **tables)
