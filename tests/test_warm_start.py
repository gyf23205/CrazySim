"""Warm-up handoff: simulation learner -> bundle -> ``Drone_Test_TRO`` learner, and the way back.

Each test trains a few steps with the unchanged method classes, writes the bundle, loads it with the
loader of ``Drone_Test_TRO`` and compares the two learners on the same observations.
"""

import copy
import importlib.util
import math
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from skrl.trainers.torch import SequentialTrainer

from conftest import DRONE_TEST_ROOT, ROOT
from crazysim import transfer_bundle as tb
from crazysim import warm_start as ws
from crazysim.methods import FINAL_CHECKPOINT, METHODS, POLICY_FILE
from crazysim.methods.kae import KoopmanAutoencoder

ROLLOUTS, NUM_ENVS = 4, 2
FAME_HIDDEN = [16, 8]
SHARED = ("kae_moe", "finetune", "residual", "componet")
ALL = (*SHARED, "fame")


@pytest.fixture(scope="module")
def real():
    """Test helpers of ``Drone_Test_TRO`` (agents of the real runtime without a drone)."""
    fixtures = DRONE_TEST_ROOT / "tests" / "test_transfer_bundle.py"
    if not fixtures.is_file():
        pytest.skip("These checks need the Drone_Test_TRO checkout next to this repository (or DRONE_TEST_ROOT).")
    sys.path.insert(0, str(DRONE_TEST_ROOT))
    spec = importlib.util.spec_from_file_location("drone_test_fixtures", fixtures)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorEnv:
    """Stand-in for the wrapped Isaac environment: random observations, every episode ends after two steps."""

    device = "cpu"
    num_envs = NUM_ENVS
    num_agents = 1
    max_episode_length = 2
    observation_space = gym.spaces.Box(-np.inf, np.inf, (56,), dtype=np.float32)
    state_space = observation_space
    action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def __init__(self, penalty_after=None):
        self.steps, self.task, self._reset_once = 0, "square", True
        self.generator = torch.Generator().manual_seed(7)
        self.penalty_after = penalty_after
        """Step after which every reward is much lower, so that a later checkpoint is worse."""

    @property
    def unwrapped(self):
        return self

    def set_task(self, task):
        self.task = task

    def _obs(self):
        return torch.randn(self.num_envs, 56, generator=self.generator)

    def reset(self):
        return self._obs(), {}

    def step(self, actions):
        self.steps += 1
        done = torch.full((self.num_envs, 1), self.steps % 2 == 0)
        reward = -actions.square().sum(-1, keepdim=True)
        if self.penalty_after is not None and self.steps > self.penalty_after:
            reward = reward - 100.0
        return self._obs(), reward, done, torch.zeros_like(done), {}

    def render(self):
        return None

    def close(self):
        return None


def agent_cfg(method: str) -> dict:
    """The waypoint agent configuration with small training settings for the CPU."""
    path = ROOT / "source/crazysim/crazysim/tasks/direct/tracking/agents/skrl_ppo_waypoint_cfg.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["agent"].update(rollouts=ROLLOUTS, learning_epochs=1, mini_batches=1)
    cfg["agent"]["experiment"].update(write_interval=ROLLOUTS, checkpoint_interval=ROLLOUTS)
    cfg["trainer"].update(
        timesteps=2 * ROLLOUTS, disable_progressbar=True, close_environment_at_exit=False, headless=True
    )
    cfg["methods"]["fame"].update(
        hidden_dims=FAME_HIDDEN, ppo_epochs=1, num_mini_batches=1, mini_batch_size=4, size_fast2meta=32,
        size_meta=64, detection_steps=4, epoch_meta=1,
    )  # fmt: skip
    if METHODS[method].reuses_policies:  # as scripts/skrl/train.py
        cfg["agent"].update(state_preprocessor=None, state_preprocessor_kwargs=None)
    return cfg


def env_cfg(laps=1, xy=4.0, z=(0.25, 2.0)):
    """Values of ``WaypointTrackingEnvCfg`` that enter the bundle; laps and command bounds as given."""
    height, speed, tri = 0.75, 0.25, 0.5 * math.sqrt(3.0)
    center = (0.0, 0.0, height)
    figure8_length = sum(
        math.hypot(math.cos((i + 0.5) * 2 * math.pi / 4096), math.cos(2 * (i + 0.5) * 2 * math.pi / 4096))
        * 2 * math.pi / 4096
        for i in range(4096)
    )  # fmt: skip
    tasks = {
        "square": SimpleNamespace(
            vertices=((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)), center=center, period=4.0 / speed
        ),
        "circle": SimpleNamespace(radius=0.75, center=center, period=2.0 * math.pi * 0.75 / speed),
        "triangle": SimpleNamespace(
            vertices=((-0.5, -tri / 3.0), (0.5, -tri / 3.0), (0.0, 2.0 * tri / 3.0)), center=center, period=3.0 / speed
        ),
        "figure8": SimpleNamespace(size=(1.0, 0.5), center=center, period=figure8_length / speed),
    }
    return SimpleNamespace(
        tasks=tasks, sim=SimpleNamespace(dt=1.0 / 500.0), decimation=10, trajectory_laps=laps,
        observation_mode="trajectory_setpoints", setpoint_count=16, waypoint_offset_limit=(0.5, 0.5, 0.25),
        observation_position_scale=(2.0, 2.0, 1.0), observation_velocity_scale=0.5, command_constraint="slew",
        command_rate_limit=(0.5, 0.5, 0.25), command_proximity_radius=(0.25, 0.25, 0.125), command_xy_limit=xy,
        command_z_range=z,
    )  # fmt: skip


def train(env, agent, timesteps=2 * ROLLOUTS):
    cfg = {"timesteps": timesteps, "disable_progressbar": True, "close_environment_at_exit": False, "headless": True}
    SequentialTrainer(env=env, agents=agent, cfg=cfg).train()


def finished_stage(method, env, tmp_path, name, task="square") -> dict:
    """Train ``method`` on its first task with the normal ``end_stage``; returns the stage entry of session.json."""
    stage_dir = tmp_path / name / f"0_{task}"
    agent = method.start_stage(0, task, str(stage_dir))
    train(env, agent)
    (stage_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    agent.save(str(stage_dir / FINAL_CHECKPOINT))
    policy = method.end_stage(0, task, str(stage_dir), agent)
    torch.save(policy, stage_dir / POLICY_FILE)
    return {"stage": f"0_{task}", "task": task, "policy": str(stage_dir / POLICY_FILE),
            "checkpoint": str(stage_dir / FINAL_CHECKPOINT), **method.stage_info}  # fmt: skip


def warm_up(method, env, tmp_path, task, index, **parents):
    """One warm-up stage as scripts/skrl/train.py --warm_start runs it."""
    stage_dir = tmp_path / f"warm_{task}" / f"{index}_{task}"
    agent = method.start_stage(index, task, str(stage_dir))
    if method.name == "fame":
        ws.track_fame_training_state(agent)
    train(env, agent)
    (stage_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    agent.save(str(stage_dir / FINAL_CHECKPOINT))
    policy, info = ws.finish_warm_stage(method, agent, str(stage_dir), task, env_cfg(), budget_iterations=2, **parents)
    torch.save(policy, stage_dir / POLICY_FILE)
    return agent, policy, info


def simulation_distribution(method_name, agent, states):
    """Mean action and log standard deviation of the learner in this repository."""
    with torch.no_grad():
        if method_name == "fame":
            mean, std = agent.fast_actor.dist_params(states)
            return mean.tanh(), std.log()
        return agent.policy.actor(agent._state_preprocessor(states, train=False)), agent.policy.log_std_parameter


def check_real_import(real, method_name, agent, policy, info, task):
    """Load the bundle in ``Drone_Test_TRO``, compare the learners, and take one training step there."""
    config = real.runtime.SetpointRuntimeConfig(pattern=task, ppo_rollout_steps=ROLLOUTS)
    spec = real.runtime.setpoint_spec_for_shape(task)
    bundle = real.load_transfer_bundle(
        info["transfer_bundle"], expected_method=method_name, expected_shape=task,
        expected_observation_contract=real.runtime.policy_observation_contract(config),
        expected_environment=real.environment_contract(config, spec),
    )  # fmt: skip
    assert bundle["manifest"]["phase"] == "simulation"
    assert bundle["manifest"]["checkpoint_selection"]["actual_iteration"] == info["selected_iteration"] <= 2
    target = real._agent(method_name, bundle)
    real.restore_transfer_agent(target, bundle)

    states = torch.randn(9, 56)
    ours = simulation_distribution(method_name, agent, states)
    for mine, theirs in zip(ours, real._params(target, states)):
        torch.testing.assert_close(mine.expand_as(theirs), theirs, rtol=2e-5, atol=3e-6)
    # policy.pt, which the validation script flies, is the same controller
    with torch.no_grad():
        torch.testing.assert_close(policy(states), ours[0])
    if method_name == "fame":  # method memory and temperature arrive unchanged
        assert (
            target.memory_fast2meta.size == agent.fast2meta.size and target.memory_meta.size == agent.meta_buffer.size
        )
        torch.testing.assert_close(target.memory_meta.all(), agent.meta_buffer.all())
        torch.testing.assert_close(target.log_alpha, agent.log_alpha)
    else:
        optimizer = target.optimizer
        assert optimizer.param_groups[0]["lr"] == pytest.approx(info["learning_rate_at_selection"])
        assert type(target.scheduler).__name__ == "KLAdaptiveLR"
        assert target.scheduler.kl_threshold == pytest.approx(0.016) and target.scheduler.max_lr == pytest.approx(1e-3)

    before = copy.deepcopy(target.models["policy"].state_dict())
    for step in range(ROLLOUTS):
        real._step(target, step)
    assert target._ppo_update_count == 1 and target._accepted_transition_count == ROLLOUTS
    assert any(not torch.equal(before[key], value) for key, value in target.models["policy"].state_dict().items())
    return bundle, target


@pytest.fixture
def square_stage(tmp_path):
    """A Square trained with ``--method scratch`` and a KAE file next to it, as ``fit_kae.py`` leaves them."""
    torch.manual_seed(3)
    env = TensorEnv()
    stage = finished_stage(
        METHODS["scratch"](env, agent_cfg("scratch"), str(tmp_path)), env, tmp_path, "square_scratch"
    )
    kae_dir = tmp_path / "square_scratch" / "0_square" / "kae"
    kae_dir.mkdir()
    torch.save(KoopmanAutoencoder(57, 64, 16, "cpu"), kae_dir / "kae.pt")
    return stage


@pytest.mark.parametrize("method_name", SHARED)
def test_circle_warm_up_from_shared_square_continues_on_the_real_side(real, square_stage, tmp_path, method_name):
    torch.manual_seed(11)
    env = TensorEnv()
    method = METHODS[method_name](env, agent_cfg(method_name), str(tmp_path))
    ws.seed_from_square(method, square_stage)
    agent, policy, info = warm_up(method, env, tmp_path, "circle", 1, parent_square="square_scratch/0_square")
    bundle, target = check_real_import(real, method_name, agent, policy, info, "circle")

    assert bundle["manifest"]["parent"] is None and bundle["manifest"]["simulation_parent"]["shape"] == "square"
    states = torch.randn(5, 56)
    square = ws.controller_from_node(
        ws.ppo_controller_node(torch.load(square_stage["selected_checkpoint"], weights_only=False))
    )
    with torch.no_grad():
        if method_name == "residual":
            assert bundle["architecture"]["policy"]["chain_depth"] == 1
            torch.testing.assert_close(agent.policy.actor.base(states), square(states))
        if method_name == "componet":
            assert len(bundle["architecture"]["policy"]["priors"]) == 1
            torch.testing.assert_close(agent.policy.actor.unit.previous_units[0].controller(states), square(states))
        if method_name == "kae_moe":
            assert bundle["architecture"]["policy"]["kae"]["origins"][0].startswith("simulation-square:")
            assert len(target.policy.moe_core.kae.blocks) == 1


def test_fame_trains_its_own_square_and_its_circle_warm_up_continues_on_the_real_side(real, tmp_path):
    torch.manual_seed(13)
    env = TensorEnv()
    method = METHODS["fame"](env, agent_cfg("fame"), str(tmp_path))
    stage = finished_stage(method, env, tmp_path, "square_fame")
    # a new process, as train.py --resume <FAME Square session> --warm_start
    method = METHODS["fame"](env, agent_cfg("fame"), str(tmp_path))
    method.resume([stage])
    agent, policy, info = warm_up(method, env, tmp_path, "circle", 1)
    assert agent.detection["choice"] in ("meta", "fresh")
    check_real_import(real, "fame", agent, policy, info, "circle")


def real_parent(real, method_name, shape, next_shape, tmp_path):
    """A finalized real bundle of ``shape`` for ``next_shape``, made with the code of ``Drone_Test_TRO``."""
    if method_name == "fame":
        # the test agent of Drone_Test_TRO has an input normalizer; FAME here uses raw observations
        from skrl.memories.torch import RandomMemory

        models = {"policy": real.FameActor(real.OBS, real.ACT, "cpu", hidden_dims=FAME_HIDDEN),
                  "value": real.FameValue(real.OBS, real.ACT, "cpu", hidden_dims=FAME_HIDDEN)}  # fmt: skip
        cfg = {"rollouts": ROLLOUTS, "learning_epochs": 1, "mini_batches": 1, "state_preprocessor": None,
               "value_preprocessor": None, "learning_rate": 3e-4, "learning_rate_scheduler": None,
               "fame_size_fast2meta": 32, "fame_size_meta": 64,
               "fame_core": {"ppo_epochs": 1, "num_mini_batches": 1, "mini_batch_size": 4, "epoch_meta": 1},
               "experiment": {"write_interval": 0, "checkpoint_interval": 0}}  # fmt: skip
        source = real.RealFameAgent(models=models, memory=RandomMemory(memory_size=ROLLOUTS, num_envs=1, device="cpu"),
                                    observation_space=real.OBS, action_space=real.ACT, device="cpu", cfg=cfg)  # fmt: skip
        source.init()
        source.games, source.flag_reg = [shape], ["Initial"]
    else:
        source = real._agent(method_name)
    for step in range(ROLLOUTS):
        real._step(source, step)
    return finalize_real_stage(real, source, method_name, shape, next_shape, tmp_path), source


def finalize_real_stage(real, source, method_name, shape, next_shape, tmp_path):
    """Export the real learner ``source`` after its stage on ``shape`` and finalize it for ``next_shape``."""
    config = real.runtime.SetpointRuntimeConfig(
        pattern=shape, ppo_rollout_steps=ROLLOUTS, real_transition_budget=ROLLOUTS
    )
    path = real.export_transfer_bundle(
        source,
        config,
        real.runtime.setpoint_spec_for_shape(shape),
        tmp_path / f"real_{shape}",
        method_name,
        phase="real",
    )
    original = real.load_transfer_bundle(path)
    next_stage = {
        "shape": next_shape, "phase": "simulation", "strategy": ws.NEXT_STAGE_STRATEGY[method_name],
        "completed_real_stage": {"artifact_id": original["manifest"]["artifact_id"], "path": str(path),
                                 "sha256": original["_source"]["sha256"]},
    }  # fmt: skip
    if method_name == "kae_moe":
        next_stage["appended_kae"] = real.describe_actor(source.policy.moe_core.kae.blocks[-1])
        next_stage["appended_kae_record"] = {"path": "fitted.pth", "sha256": real._digest(next_stage["appended_kae"])}
    if method_name == "fame":
        source.end_of_shape_segment()
        next_stage["consolidated_fame_state"] = source.fame_state(shape)
        next_stage["consolidation_count"] = 1
    finalized = real.update_transfer_bundle(
        path, next_stage=next_stage, output_path=tmp_path / f"real_{shape}_finalized.pt"
    )
    return tb.read_bundle(finalized)


@pytest.mark.parametrize("method_name", ALL)
@pytest.mark.parametrize("shapes", [("circle", "triangle"), ("triangle", "figure8")])
def test_warm_up_from_a_finalized_real_parent_continues_on_the_real_side(real, tmp_path, method_name, shapes):
    torch.manual_seed(17)
    shape, next_shape = shapes
    parent, source = real_parent(real, method_name, shape, next_shape, tmp_path)
    env = TensorEnv()
    method = METHODS[method_name](env, agent_cfg(method_name), str(tmp_path))
    ws.seed_from_bundle(method, parent, next_shape, str(tmp_path / "parent"))

    states = torch.randn(6, 56)
    with torch.no_grad():
        completed = source.models["policy"]
        if method_name == "residual":
            torch.testing.assert_close(method.base(states), completed.residual_core(states), rtol=2e-5, atol=3e-6)
        if method_name == "componet":
            assert len(method.units) == completed.componet_core.n_prev + 1
            torch.testing.assert_close(
                method.units[-1].controller(states), completed.componet_core(states), rtol=2e-5, atol=3e-6
            )
        if method_name == "kae_moe":
            assert len(method.experts) == len(completed.moe_core.kae.blocks) + 1

    index = tb.SHAPES.index(shape) + 1
    agent, policy, info = warm_up(method, env, tmp_path, next_shape, index, parent=parent)
    bundle, target = check_real_import(real, method_name, agent, policy, info, next_shape)

    manifest = bundle["manifest"]
    assert (
        manifest["parent"]["artifact_id"] == parent["manifest"]["artifact_id"] and manifest["parent"]["phase"] == "real"
    )
    spent = info["selected_iteration"] * ROLLOUTS * NUM_ENVS
    assert manifest["cumulative_counters"]["accepted_transitions"] == ROLLOUTS + spent
    if method_name == "kae_moe":
        assert len(target.policy.moe_core.kae.blocks) == len(source.policy.moe_core.kae.blocks) + 1
    if method_name == "componet":
        assert target.policy.componet_core.n_prev == source.policy.componet_core.n_prev + 1
    if method_name == "residual":
        assert bundle["architecture"]["policy"]["chain_depth"] == source.policy.residual_core.chain_depth + 1
    if method_name == "fame":
        assert agent.detection is not None


@pytest.mark.parametrize("task", ["square", "circle", "triangle", "figure8"])
def test_task_settings_match_what_the_real_side_compares(real, task):
    config = real.runtime.SetpointRuntimeConfig(pattern=task)
    expected = real.environment_contract(config, real.runtime.setpoint_spec_for_shape(task))
    assert ws.environment_contract(env_cfg(), task) == tb.plain(expected)
    assert ws.observation_contract(expected) == real.runtime.policy_observation_contract(config)
    # the command envelope of the simulation, passed to the real run with --position-cmd-*
    spec = real.runtime.setpoint_spec_for_shape(task)
    wide = real.runtime.SetpointRuntimeConfig(
        pattern=task, position_cmd_xy_min=-4.5, position_cmd_xy_max=4.5, position_cmd_z_min=0.0, position_cmd_z_max=9.85
    )
    real.runtime.validate_setpoint_runtime_config(wide, spec)
    assert ws.environment_contract(env_cfg(xy=4.5, z=(0.0, 9.85)), task) == tb.plain(
        real.environment_contract(wide, spec)
    )
    with pytest.raises(ValueError):
        mixed = real.runtime.SetpointRuntimeConfig(pattern=task, position_cmd_z_min=0.0)
        real.runtime.validate_setpoint_runtime_config(mixed, spec)


def test_bundle_checksum_and_parent_checks(real, square_stage, tmp_path):
    torch.manual_seed(19)
    env = TensorEnv()
    method = METHODS["residual"](env, agent_cfg("residual"), str(tmp_path))
    ws.seed_from_square(method, square_stage)
    _, _, info = warm_up(method, env, tmp_path, "circle", 1)
    bundle = tb.read_bundle(info["transfer_bundle"])
    unsealed = {key: value for key, value in bundle.items() if key != "_source"}
    assert tb.digest(unsealed) == real._digest(unsealed)
    with pytest.raises(ValueError, match="finished real stage"):
        ws.check_parent_bundle(bundle, "residual", "triangle")  # a warm-up is not a parent
    with pytest.raises(ValueError, match="trains its own first stage"):
        ws.seed_from_square(METHODS["fame"](env, agent_cfg("fame"), str(tmp_path)), square_stage)
    bundle["manifest"]["shape"] = "triangle"
    torch.save({key: value for key, value in bundle.items() if key != "_source"}, tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="checksum"):
        tb.read_bundle(tmp_path / "changed.pt")


@pytest.mark.parametrize("method_name", ALL)
def test_chain_square_circle_real_triangle_real(real, square_stage, tmp_path, method_name):
    """Simulation Circle -> real Circle (trained and finalized there) -> simulation Triangle -> real Triangle."""
    torch.manual_seed(23)
    env = TensorEnv()
    method = METHODS[method_name](env, agent_cfg(method_name), str(tmp_path))
    if method_name == "fame":
        method.resume([finished_stage(method, env, tmp_path, "square_fame")])
    else:
        ws.seed_from_square(method, square_stage)
    agent, policy, info = warm_up(method, env, tmp_path, "circle", 1)
    circle, target = check_real_import(real, method_name, agent, policy, info, "circle")  # one real update included

    if method_name == "fame":
        target.games, target.flag_reg = ["square", "circle"], ["Initial", agent.detection["choice"]]
    parent = finalize_real_stage(real, target, method_name, "circle", "triangle", tmp_path)
    assert parent["manifest"]["parent"]["artifact_id"] == circle["manifest"]["artifact_id"]
    if method_name != "fame":  # the real side writes the adaptive schedule back
        assert parent["agent_configuration"]["scheduler"]["kind"] == "KLAdaptiveLR"

    method = METHODS[method_name](env, agent_cfg(method_name), str(tmp_path))
    ws.seed_from_bundle(method, parent, "triangle", str(tmp_path / "parent"))
    if method_name == "finetune":  # the rate that the real stage ended with carries over
        real_rate = target.optimizer.param_groups[0]["lr"]
        assert torch.load(method.checkpoint, weights_only=False)["optimizer"]["param_groups"][0]["lr"] == real_rate
    agent, policy, info = warm_up(method, env, tmp_path, "triangle", 2, parent=parent)
    triangle, target = check_real_import(real, method_name, agent, policy, info, "triangle")

    assert triangle["manifest"]["parent"]["artifact_id"] == parent["manifest"]["artifact_id"]
    policy_node = triangle["architecture"]["policy"]
    if method_name == "residual":
        assert policy_node["chain_depth"] == 2 and policy_node["base"]["chain_depth"] == 1
    if method_name == "componet":
        assert [prior["kind"] for prior in policy_node["priors"]] == ["DeployablePolicy", "CompoNetCore"]
    if method_name == "kae_moe":
        assert len(policy_node["kae"]["blocks"]) == 2 and policy_node["kae"]["origins"][0].startswith(
            "simulation-square:"
        )


@pytest.mark.parametrize("method_name", ["finetune", "kae_moe", "fame"])
def test_the_best_checkpoint_within_the_budget_is_handed_over(real, square_stage, tmp_path, method_name):
    """Rewards drop after the first iteration, so the checkpoint of iteration 1 of 2 must be selected."""
    torch.manual_seed(29)
    env = TensorEnv()
    method = METHODS[method_name](env, agent_cfg(method_name), str(tmp_path))
    if method_name == "fame":
        method.resume([finished_stage(method, env, tmp_path, "square_fame")])
    else:
        ws.seed_from_square(method, square_stage)
    env.penalty_after = env.steps + ROLLOUTS  # counted from here: detection steps of FAME come first
    if method_name == "fame":
        env.penalty_after += 2 * 4  # two candidates, detection_steps each
    agent, policy, info = warm_up(method, env, tmp_path, "circle", 1)
    assert info["selected_iteration"] == 1 and info["budget_iterations"] == 2
    assert info["selected_checkpoint"].endswith("best_agent.pt")

    best = torch.load(info["selected_checkpoint"], weights_only=False)
    bundle = tb.read_bundle(info["transfer_bundle"])
    manifest = bundle["manifest"]
    assert manifest["checkpoint_selection"] == {
        "rule": "best_training_return_within_budget", "actual_update": 1, "actual_iteration": 1}  # fmt: skip
    assert manifest["phase_counters"]["accepted_transitions"] == ROLLOUTS * NUM_ENVS
    assert manifest["budget"]["allocated_iterations"] == 2
    if method_name == "fame":
        torch.testing.assert_close(
            bundle["agent_modules"]["policy"]["trunk.0.weight"], best["fast_actor"]["trunk.0.weight"]
        )
        # optimizer state and temperature of the same iteration, not of the final one
        saved = best["fast_actor_optimizer"]["state"][0]["exp_avg"]
        torch.testing.assert_close(bundle["agent_modules"]["optimizer"]["state"][0]["exp_avg"], saved)
        torch.testing.assert_close(
            bundle["agent_modules"]["temperature"]["log_alpha"], best["temperature"]["log_alpha"]
        )
    else:
        torch.testing.assert_close(
            bundle["agent_modules"]["policy"]["log_std_parameter"], best["policy"]["log_std_parameter"]
        )
        assert manifest["learning_rate_at_selection"] == best["optimizer"]["param_groups"][0]["lr"]
    check_real_import(real, method_name, agent, policy, info, "circle")
