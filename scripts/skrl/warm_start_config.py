"""In-memory waypoint configuration for the dedicated warm-start launcher.

Importing this module needs no simulator. Native trajectory classes are loaded
only while building a configuration, after AppLauncher has started Isaac Sim.
Existing registrations, YAML files, methods and normal launchers are untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path

import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler


class WarmStartRunningStandardScaler(RunningStandardScaler):
    """Real-compatible population variance, retaining SKRL's checkpoint layout."""

    def _compute(self, x, train=False, inverse=False):
        if train:
            axes = (0, 1) if x.ndim == 3 else (0,)
            count = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
            if count:
                self._parallel_variance(torch.mean(x, dim=axes), torch.var(x, dim=axes, unbiased=False), count)
        return super()._compute(x, train=False, inverse=inverse)


METHODS = ("kae_moe", "finetune", "residual", "componet", "fame")
SHAPES = ("square", "circle", "triangle", "figure8", "skewed_figure8")
GEOMETRY_FIELDS = (
    "target_height", "trajectory_scale", "trajectory_speed", "trajectory_laps",
    "square_side_length", "circle_radius", "triangle_side_length", "figure8_amp_x", "figure8_amp_y",
)
DEFAULT_CONTRACT = {
    "control_period_s": .02, "target_height": .75, "trajectory_scale": 1.0,
    "trajectory_speed": .25, "trajectory_laps": 1,
    "square_side_length": 1.0, "circle_radius": .75, "triangle_side_length": 1.0,
    "figure8_amp_x": 1.0, "figure8_amp_y": .5, "figure8_random_height": False,
    "observation_mode": "trajectory_setpoints", "trajectory_setpoint_count": 16,
    "waypoint_offset_limit": [.5, .5, .25], "observation_position_error_scale": [2.0, 2.0, 1.0],
    "observation_velocity_scale": .5, "action_clip_min": -1.0, "action_clip_max": 1.0,
    "command_constraint": "slew", "command_rate_limit": [.5, .5, .25],
    "command_proximity_radius": [.25, .25, .125], "position_cmd_xy_min": -4.0,
    "position_cmd_xy_max": 4.0, "position_cmd_z_min": .25, "position_cmd_z_max": 2.0,
    "command_semantics": "world_position_setpoint_relative_waypoint_v1", "yaw_radians": 0.0,
    "reference_geometry_version": "fixed_center_five_shapes_v1", "shape_rotation_y_degrees": 0.0,
}


def _merge(defaults, overrides):
    result = deepcopy(defaults)
    for key, value in overrides.items():
        result[key] = (_merge(result[key], value) if isinstance(value, Mapping)
                       and isinstance(result.get(key), Mapping) else deepcopy(value))
    return result


def _positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be a positive integer.")


def _contract(shape, parent_environment, geometry):
    result = deepcopy(DEFAULT_CONTRACT)
    if parent_environment:
        # Parent-specific shape dimensions are not target defaults. Inactive
        # dimensions must also match the real target parser's ordinary values.
        inherited = set(result) - {"square_side_length", "circle_radius", "triangle_side_length",
                                   "figure8_amp_x", "figure8_amp_y", "shape", "shape_rotation_y_degrees"}
        for key in inherited.intersection(parent_environment):
            result[key] = deepcopy(parent_environment[key])
    if shape == "skewed_figure8":
        result.update(figure8_amp_x=.7, figure8_amp_y=.35, shape_rotation_y_degrees=30.0)
    aliases = {"speed": "trajectory_speed", "laps": "trajectory_laps", "scale": "trajectory_scale",
               "height": "target_height"}
    for key, value in (geometry or {}).items():
        key = aliases.get(key, key)
        if key not in GEOMETRY_FIELDS:
            raise ValueError(f"Unknown warm-start geometry field: {key}.")
        if value is not None:
            result[key] = value
    for key in GEOMETRY_FIELDS:
        _positive(result[key], key, integer=key == "trajectory_laps")
    for key in ("waypoint_offset_limit", "observation_position_error_scale", "command_rate_limit",
                "command_proximity_radius"):
        value = result[key]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{key} must contain three positive values.")
        for item in value:
            _positive(item, key)
        result[key] = list(value)
    _positive(result["observation_velocity_scale"], "observation_velocity_scale")
    fixed = {"control_period_s": .02, "trajectory_setpoint_count": 16,
             "observation_mode": "trajectory_setpoints", "action_clip_min": -1.0,
             "action_clip_max": 1.0, "figure8_random_height": False,
             "yaw_radians": 0.0, "command_semantics": DEFAULT_CONTRACT["command_semantics"]}
    fixed["reference_geometry_version"] = "fixed_center_five_shapes_v1"
    if any(result[key] != expected for key, expected in fixed.items()):
        raise ValueError("Warm starts require the original 50 Hz, 56-input, three-waypoint contract.")
    if result["command_constraint"] not in ("slew", "proximity"):
        raise ValueError("Unsupported waypoint command constraint.")
    for key in ("position_cmd_xy_min", "position_cmd_xy_max", "position_cmd_z_min", "position_cmd_z_max"):
        if not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise ValueError("Command bounds must be finite.")
    if (result["position_cmd_xy_max"] <= 0
            or result["position_cmd_xy_min"] != -result["position_cmd_xy_max"]):
        raise ValueError("The native waypoint environment requires symmetric XY command bounds.")
    if result["position_cmd_z_min"] >= result["position_cmd_z_max"]:
        raise ValueError("Command Z bounds must be ordered.")
    result["shape"] = shape
    return result


def configure_for_parent(agent_cfg, parent, method, *, square_policy=None):
    """Pin constructor topology before an existing native Method is created."""
    if method not in METHODS:
        raise ValueError("Unsupported warm-start method.")
    result = deepcopy(agent_cfg)
    if parent is None:
        result["agent"]["learning_rate_scheduler"] = None if method == "fame" else torch.optim.lr_scheduler.ExponentialLR
        result["agent"]["learning_rate_scheduler_kwargs"] = {} if method == "fame" else {"gamma": .9995}
        if method == "fame" and square_policy is not None:
            payload = torch.load(Path(square_policy).expanduser(), map_location="cpu", weights_only=True)
            state = payload.get("policy", {})
            try:
                dims = [int(state["net_container.0.weight"].shape[0]),
                        int(state["net_container.2.weight"].shape[0])]
                if dims != [64, 64] or state["net_container.0.weight"].shape[1] != 56:
                    raise ValueError("Square FAME preparation needs a 56-input, 64/64 source actor.")
            except (KeyError, AttributeError, IndexError) as error:
                raise ValueError("Square FAME preparation needs the source waypoint policy state dictionary.") from error
            result.setdefault("methods", {}).setdefault("fame", {})["hidden_dims"] = dims
        return result
    saved = parent.get("agent_configuration", {})
    agent = result["agent"]
    scheduler = saved.get("scheduler")
    if scheduler is None:
        agent["learning_rate_scheduler"], agent["learning_rate_scheduler_kwargs"] = None, {}
    elif scheduler.get("kind") == "ExponentialLR":
        agent["learning_rate_scheduler"] = torch.optim.lr_scheduler.ExponentialLR
        agent["learning_rate_scheduler_kwargs"] = {"gamma": scheduler["gamma"]}
    else:
        raise ValueError("Unsupported parent scheduler topology.")
    for role in ("state_preprocessor", "value_preprocessor"):
        descriptor = saved.get("preprocessors", {}).get(role)
        if method in ("residual", "componet", "kae_moe") and role == "state_preprocessor":
            if descriptor is not None:
                raise ValueError("Compositional current learners must consume raw observations.")
        if method == "fame" and descriptor is not None:
            raise ValueError("FAME preprocessing belongs inside the actor, not an outer scaler.")
        if descriptor is None:
            agent[role], agent[role + "_kwargs"] = None, {}
        else:
            if descriptor.get("kind") not in ("SingleDroneRunningStandardScaler", "RunningStandardScaler"):
                raise ValueError("Unsupported parent preprocessor topology.")
            agent[role] = (WarmStartRunningStandardScaler if descriptor["kind"] == "SingleDroneRunningStandardScaler"
                           else "RunningStandardScaler")
            agent[role + "_kwargs"] = {key: descriptor[key] for key in ("size", "epsilon", "clip_threshold")
                                      if key in descriptor}
    if method == "fame":
        current = parent["architecture"]["policy"]["configuration"]
        consolidated = (parent["manifest"].get("next_stage") or {}).get("consolidated_fame_state")
        if consolidated is not None:
            meta = consolidated["actor_configuration"]
            if meta["hidden_dims"] != current["hidden_dims"]:
                raise ValueError("FAME current and consolidated meta architectures disagree.")
        result.setdefault("methods", {}).setdefault("fame", {})["hidden_dims"] = list(current["hidden_dims"])
    return result


def build_warm_start_configuration(
    env_cfg, agent_cfg, method, shape, *, general_agent_cfg=None, parent_environment=None,
    num_envs, rollout_steps, iterations, geometry=None, parent_bundle=None, square_preparation=False,
):
    """Return isolated (native env config, native agent config, real contract)."""
    if method not in METHODS or shape not in SHAPES:
        raise ValueError("Unsupported warm-start method or shape.")
    for value, name in ((num_envs, "num_envs"), (rollout_steps, "rollout_steps"), (iterations, "iterations")):
        _positive(value, name, integer=True)
    if general_agent_cfg is None:
        import yaml
        path = Path(__file__).resolve().parents[2] / "source/crazysim/crazysim/tasks/direct/tracking/agents/skrl_ppo_cfg.yaml"
        general_agent_cfg = yaml.safe_load(path.read_text())
    if parent_bundle is not None and parent_environment is None:
        parent_environment = parent_bundle["manifest"]["environment"]
    contract = _contract(shape, parent_environment, geometry)
    env, cfg = deepcopy(env_cfg), deepcopy(agent_cfg)
    cfg["methods"] = _merge(general_agent_cfg.get("methods", {}), cfg.get("methods", {}))
    moe = cfg["methods"].setdefault("kae_moe", {})
    moe["padded_dim"] = 57
    moe["kae"] = _merge(moe.get("kae", {}), cfg.get("kae", {}))
    moe["kae"]["hidden_dim"] = 64
    for key in ("observable_dim", "padded_dim", "p"):
        moe["kae"].pop(key, None)
    cfg["agent"]["rollouts"] = rollout_steps
    cfg["memory"]["memory_size"] = rollout_steps
    cfg["trainer"]["timesteps"] = iterations * rollout_steps
    if cfg["models"]["policy"].get("class") != "SquashedGaussian":
        raise ValueError("Warm-start launch requires the registered waypoint SquashedGaussian configuration.")
    if method in ("kae_moe", "residual", "componet", "fame"):
        cfg["agent"].update(state_preprocessor=None, state_preprocessor_kwargs={})
    if method == "fame":
        cfg["agent"].update(value_preprocessor=None, value_preprocessor_kwargs={})
        if square_preparation:
            cfg["methods"]["fame"]["hidden_dims"] = [64, 64]
    for role in ("state_preprocessor", "value_preprocessor"):
        if cfg["agent"].get(role) == "RunningStandardScaler":
            cfg["agent"][role] = WarmStartRunningStandardScaler
    if parent_bundle is not None:
        cfg = configure_for_parent(cfg, parent_bundle, method)

    from crazysim.tasks.direct.tracking.trajectories import PolygonCfg
    from warm_start_trajectories import figure8_path_length, matching_circle_config, matching_figure8_config

    scale, speed = contract["trajectory_scale"], contract["trajectory_speed"]
    center = (0.0, 0.0, contract["target_height"])
    if shape == "square":
        half = contract["square_side_length"] * scale / 2.0
        trajectory = PolygonCfg(vertices=((-half, -half), (half, -half), (half, half), (-half, half)),
                                center=center, period=8.0 * half / speed)
    elif shape == "triangle":
        side = contract["triangle_side_length"] * scale
        height = math.sqrt(3.0) * side / 2.0
        trajectory = PolygonCfg(vertices=((-side / 2.0, -height / 3.0), (side / 2.0, -height / 3.0),
                                          (0.0, 2.0 * height / 3.0)), center=center, period=3.0 * side / speed)
    elif shape == "circle":
        radius = contract["circle_radius"] * scale
        trajectory = matching_circle_config(radius=radius, center=center, period=2.0 * math.pi * radius / speed)
    else:
        size = (contract["figure8_amp_x"] * scale, contract["figure8_amp_y"] * scale)
        trajectory = matching_figure8_config(size=size, center=center,
                                             period=figure8_path_length(*size) / speed,
                                             skewed=shape == "skewed_figure8")
    env.task = shape
    env.tasks = dict(env.tasks)
    env.tasks[shape] = trajectory
    env.trajectory_laps = contract["trajectory_laps"]
    env.episode_length_s = env.trajectory_laps * trajectory.period
    env.decimation, env.sim.dt, env.sim.render_interval = 10, .002, 10
    env.scene.num_envs = num_envs
    env.action_mode, env.action_space = "relative_waypoint", 3
    env.observation_mode, env.setpoint_count, env.observation_space = "trajectory_setpoints", 16, 56
    env.waypoint_offset_limit = tuple(contract["waypoint_offset_limit"])
    env.observation_position_scale = tuple(contract["observation_position_error_scale"])
    env.observation_velocity_scale = contract["observation_velocity_scale"]
    env.command_constraint = contract["command_constraint"]
    env.command_rate_limit = tuple(contract["command_rate_limit"])
    env.command_proximity_radius = tuple(contract["command_proximity_radius"])
    env.command_xy_limit = contract["position_cmd_xy_max"]
    env.command_z_range = (contract["position_cmd_z_min"], contract["position_cmd_z_max"])
    reference = trajectory.class_type(trajectory).position(
        torch.linspace(0.0, trajectory.period, 4097, dtype=torch.float64),
    )
    if (not torch.isfinite(reference).all()
            or (reference[:, :2].abs() > env.command_xy_limit + 1e-10).any()
            or (reference[:, 2] < env.command_z_range[0] - 1e-10).any()
            or (reference[:, 2] > env.command_z_range[1] + 1e-10).any()):
        raise ValueError("Target reference leaves the agreed waypoint command envelope.")
    return env, cfg, contract
