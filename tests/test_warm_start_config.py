"""Waypoint warm-start configuration is additive and testable without Isaac Sim."""

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("CRAZYSIM_SOURCE_ROOT", ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "skrl"))
from warm_start_config import WarmStartRunningStandardScaler, build_warm_start_configuration, configure_for_parent


@pytest.fixture
def native_trajectories(monkeypatch):
    # Only the native trajectory/config module is loaded. No Isaac application
    # or mocked physics is started or claimed by these configuration checks.
    for name in ("isaaclab", "isaaclab.utils", "crazysim", "crazysim.tasks",
                 "crazysim.tasks.direct", "crazysim.tasks.direct.tracking"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["isaaclab.utils"].configclass = lambda cls: dataclass(cls, kw_only=True)
    name = "crazysim.tasks.direct.tracking.trajectories"
    spec = importlib.util.spec_from_file_location(
        name, SOURCE_ROOT / "source/crazysim/crazysim/tasks/direct/tracking/trajectories.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.delitem(sys.modules, "warm_start_trajectories", raising=False)
    yield module


def _configs():
    directory = SOURCE_ROOT / "source/crazysim/crazysim/tasks/direct/tracking/agents"
    waypoint = yaml.safe_load((directory / "skrl_ppo_waypoint_cfg.yaml").read_text())
    general = yaml.safe_load((directory / "skrl_ppo_cfg.yaml").read_text())
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=4096),
                          sim=SimpleNamespace(dt=.002, render_interval=10),
                          task="square", tasks={}, random_start_phase=True)
    return env, waypoint, general


def _build(method, shape, **kwargs):
    env, agent, general = _configs()
    return build_warm_start_configuration(env, agent, method, shape, general_agent_cfg=general,
                                         num_envs=8, rollout_steps=96, iterations=3, **kwargs)


@pytest.mark.parametrize("method", ["kae_moe", "finetune", "residual", "componet", "fame"])
def test_original_waypoint_contract_and_method_defaults_are_assembled_in_memory(native_trajectories, method):
    env, original, general = _configs()
    before = deepcopy(original)
    configured, agent, contract = build_warm_start_configuration(
        env, original, method, "circle", general_agent_cfg=general,
        num_envs=8, rollout_steps=96, iterations=3,
    )
    assert original == before and env.scene.num_envs == 4096 and not env.tasks
    assert configured.sim.dt * configured.decimation == .02
    assert (configured.observation_space, configured.action_space, configured.setpoint_count) == (56, 3, 16)
    assert configured.command_xy_limit == 4.0 and configured.command_z_range == (.25, 2.0)
    assert configured.trajectory_laps == 1 and configured.random_start_phase is True
    assert agent["agent"]["rollouts"] == agent["memory"]["memory_size"] == 96
    assert agent["trainer"]["timesteps"] == 288
    assert agent["methods"]["kae_moe"]["padded_dim"] == 57
    assert agent["methods"]["kae_moe"]["kae"]["hidden_dim"] == 64
    assert agent["methods"]["componet"]["internal_hidden_dims"] == [256, 128, 64]
    assert agent["models"]["policy"]["network"][0]["layers"] == [64, 64]
    assert agent["agent"]["state_preprocessor"] == (WarmStartRunningStandardScaler if method == "finetune" else None)
    if method == "fame":
        assert agent["agent"]["value_preprocessor"] is None
    assert contract["shape"] == "circle"
    assert contract["command_semantics"] == "world_position_setpoint_relative_waypoint_v1"
    assert contract["reference_geometry_version"] == "fixed_center_five_shapes_v1"
    assert contract["shape_rotation_y_degrees"] == 0


@pytest.mark.parametrize("shape,period", [("square", 16), ("circle", 6 * math.pi), ("triangle", 12)])
def test_real_shape_geometry_and_phase_start(native_trajectories, shape, period):
    env, _, _ = _build("finetune", shape)
    cfg = env.tasks[shape]
    assert cfg.period == pytest.approx(period)
    trajectory = cfg.class_type(cfg)
    time = torch.tensor([0.0, period], dtype=torch.float64)
    positions = trajectory.position(time)
    torch.testing.assert_close(positions[0], positions[1], rtol=0, atol=1e-12)
    expected = {"square": [-.5, -.5, .75], "circle": [.75, 0, .75],
                "triangle": [-.5, -math.sqrt(3) / 6, .75]}[shape]
    torch.testing.assert_close(positions[0], torch.tensor(expected, dtype=torch.float64))
    torch.testing.assert_close(trajectory.velocity(time).norm(dim=-1), torch.full((2,), .25, dtype=torch.float64))


def test_skewed_shape_is_lengthwise_tilted_and_retains_arc_length_timing(native_trajectories):
    env, _, contract = _build("kae_moe", "skewed_figure8",
                              parent_environment={"figure8_amp_x": 1.0, "figure8_amp_y": .5})
    assert (contract["figure8_amp_x"], contract["figure8_amp_y"]) == (.7, .35)
    assert contract["shape_rotation_y_degrees"] == 30
    cfg = env.tasks["skewed_figure8"]
    trajectory = cfg.class_type(cfg)
    time = torch.linspace(0.0, cfg.period, 2001, dtype=torch.float64)
    position = trajectory.position(time)
    velocity = trajectory.velocity(time)
    # Rotation +30° about Y gives z-center = -tan(30°)*x.
    torch.testing.assert_close(position[:, 2] - .75, -math.tan(math.pi / 6) * position[:, 0],
                               rtol=1e-12, atol=1e-12)
    assert position[:, 2].min() == pytest.approx(.4, abs=1e-5)
    assert position[:, 2].max() == pytest.approx(1.1, abs=1e-5)
    torch.testing.assert_close(velocity.norm(dim=-1), torch.full_like(time, .25), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(velocity[:, 2], -math.tan(math.pi / 6) * velocity[:, 0],
                               rtol=1e-12, atol=1e-12)


def test_explicit_geometry_and_shared_parent_contract_survive_target_selection(native_trajectories):
    env, _, contract = _build("residual", "circle", parent_environment={
        "square_side_length": 2.0, "target_height": .8, "trajectory_speed": .2,
        "command_constraint": "proximity", "waypoint_offset_limit": [.4, .4, .2],
    }, geometry={"circle_radius": 1.0, "trajectory_laps": 2})
    assert contract["square_side_length"] == 1.0
    assert contract["circle_radius"] == 1.0 and contract["trajectory_speed"] == .2
    assert env.episode_length_s == pytest.approx(20 * math.pi)
    assert env.command_constraint == "proximity" and env.waypoint_offset_limit == (.4, .4, .2)


@pytest.mark.parametrize("parent", [
    {"control_period_s": .01}, {"trajectory_setpoint_count": 15}, {"action_clip_max": 2},
    {"position_cmd_xy_min": -3}, {"figure8_random_height": True}, {"command_constraint": "unknown"},
])
def test_unsupported_parent_contract_is_rejected_before_environment_creation(native_trajectories, parent):
    with pytest.raises(ValueError):
        _build("finetune", "square", parent_environment=parent)


def test_parent_fame_architecture_and_scheduler_are_configured_before_native_constructor():
    _, agent, general = _configs()
    agent["methods"] = general["methods"]
    parent = {"agent_configuration": {"scheduler": {"kind": "ExponentialLR", "gamma": .99},
                                       "preprocessors": {"state_preprocessor": None, "value_preprocessor": None}},
              "architecture": {"policy": {"configuration": {"hidden_dims": [64, 64]}}},
              "manifest": {"next_stage": {"consolidated_fame_state": {
                  "actor_configuration": {"hidden_dims": [64, 64]}}}}}
    configured = configure_for_parent(agent, parent, "fame")
    assert configured["methods"]["fame"]["hidden_dims"] == [64, 64]
    assert agent["methods"]["fame"]["hidden_dims"] == [256, 128, 64]
    assert configured["agent"]["learning_rate_scheduler"] is torch.optim.lr_scheduler.ExponentialLR
    assert configured["agent"]["learning_rate_scheduler_kwargs"] == {"gamma": .99}
    parent["manifest"]["next_stage"]["consolidated_fame_state"]["actor_configuration"]["hidden_dims"] = [32]
    with pytest.raises(ValueError, match="architectures disagree"):
        configure_for_parent(agent, parent, "fame")


def test_square_preparation_uses_source_fame_architecture(tmp_path):
    _, agent, general = _configs()
    agent["methods"] = general["methods"]
    source = tmp_path / "policy.pt"
    torch.save({"policy": {"net_container.0.weight": torch.zeros(64, 56),
                           "net_container.2.weight": torch.zeros(64, 64)}}, source)
    configured = configure_for_parent(agent, None, "fame", square_policy=source)
    assert configured["methods"]["fame"]["hidden_dims"] == [64, 64]


def test_population_scaler_accepts_one_environment_and_preserves_checkpoint_statistics():
    scaler = WarmStartRunningStandardScaler(size=3, device="cpu")
    for _ in range(3):
        assert torch.isfinite(scaler(torch.tensor([[1.0, 2.0, 3.0]]), train=True)).all()
    restored = WarmStartRunningStandardScaler(size=3, device="cpu")
    restored.load_state_dict(scaler.state_dict())
    for key, value in scaler.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_preparation_scheduler_is_explicitly_compatible_with_the_real_handoff():
    from skrl.utils.runner.torch import Runner

    _, agent, _ = _configs()
    configured = configure_for_parent(agent, None, "finetune")
    assert configured["agent"]["learning_rate_scheduler"] is torch.optim.lr_scheduler.ExponentialLR
    assert configured["agent"]["learning_rate_scheduler_kwargs"] == {"gamma": .9995}
    # Native Runner merges PPO defaults (including this slot) before processing.
    configured["agent"]["rewards_shaper"] = None
    processed = Runner.__new__(Runner)._process_cfg(configured)
    assert processed["agent"]["learning_rate_scheduler"] is torch.optim.lr_scheduler.ExponentialLR


def test_unreachable_target_reference_is_rejected_without_opening_physics(native_trajectories):
    with pytest.raises(ValueError, match="command envelope"):
        _build("kae_moe", "skewed_figure8", geometry={"target_height": .4})
