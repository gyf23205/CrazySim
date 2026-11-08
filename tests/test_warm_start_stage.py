"""CPU checks use the original native Method classes, PPO and FAME updates."""

import copy
import importlib
import os
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(os.environ.get("CRAZYSIM_ROOT", os.environ.get("CRAZYSIM_SOURCE_ROOT", Path(__file__).resolve().parents[1])))
STAGED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGED_ROOT / "scripts" / "skrl"))
from warm_start_stage import PriorControllerUnit, configure_for_parent, deterministic_current_policy, start_warm_stage


@pytest.fixture
def native(monkeypatch):
    # Import the actual method package without importing Isaac Sim's app-level
    # registration. The only external import inside these methods is a type.
    package = types.ModuleType("crazysim")
    package.__path__ = [str(ROOT / "source" / "crazysim" / "crazysim")]
    monkeypatch.setitem(sys.modules, "crazysim", package)
    external = types.ModuleType("isaaclab_rl.skrl")
    external.SkrlVecEnvWrapper = object
    monkeypatch.setitem(sys.modules, "isaaclab_rl.skrl", external)
    return importlib.import_module("crazysim.methods")


class _Env:
    device = "cpu"
    num_envs = 2
    num_agents = 1
    observation_space = gym.spaces.Box(-np.inf, np.inf, (56,), dtype=np.float32)
    state_space = observation_space
    action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)

    def __init__(self):
        self.steps = 0

    def reset(self):
        return torch.zeros(self.num_envs, 56), {}

    def step(self, actions):
        self.steps += 1
        return (torch.full((self.num_envs, 56), self.steps / 100),
                -actions.square().sum(-1, keepdim=True),
                torch.ones(self.num_envs, 1, dtype=torch.bool),
                torch.zeros(self.num_envs, 1, dtype=torch.bool), {})

    def render(self):
        return None

    def close(self):
        return None


def _cfg():
    network = [{"layers": [64, 64]}]
    return {
        "seed": 42,
        "models": {"policy": {"class": "SquashedGaussian", "initial_log_std": 0., "min_log_std": -3.,
                                "network": copy.deepcopy(network)},
                   "value": {"network": copy.deepcopy(network)}},
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {"class": "PPO", "rollouts": 4, "learning_epochs": 1, "mini_batches": 1,
                  "state_preprocessor": None, "value_preprocessor": None,
                  "learning_rate_scheduler": None,
                  "experiment": {"directory": "unused", "experiment_name": "", "write_interval": 0,
                                 "checkpoint_interval": 0}},
        "trainer": {"class": "SequentialTrainer", "timesteps": 4},
        "methods": {
            "residual": {"hidden_dims": [8, 8]},
            "componet": {"internal_hidden_dims": [8, 8]},
            "kae_moe": {"padded_dim": 57, "residual_hidden_dims": [8, 8], "weight_hidden_dims": [8],
                        "gating_hidden_dims": [8], "router_hidden_dims": [8]},
            "fame": {"hidden_dims": [8, 8], "lr_actor": 3e-4, "lr_critic": 1e-3, "lr_meta": 1e-3,
                     "gamma": .99, "gae_lambda": .95, "clip_ratio": .2, "value_coef": 1., "grad_norm_clip": 1.,
                     "init_temperature": .1, "alpha_lr": 1e-4, "target_entropy": None,
                     "ppo_epochs": 1, "num_mini_batches": 1, "mini_batch_size": 4,
                     "rewards_shaper_scale": 1., "size_fast2meta": 16, "size_meta": 32,
                     "detection_steps": 2, "epoch_meta": 1, "meta_lr_scheduler_gamma": .95},
        },
    }


@pytest.fixture
def io_stub(native, monkeypatch):
    from crazysim.methods.kae import KAEExperts, KoopmanAutoencoder
    from crazysim.methods.fame import SquashedActor

    io = types.ModuleType("warm_start_io")
    controller = nn.Sequential(nn.Linear(56, 3))
    io.completed_deterministic_controller = lambda parent, device: copy.deepcopy(controller).to(device)
    io.load_square_controller = lambda path, device: copy.deepcopy(controller).to(device)
    io.decode_actor = lambda descriptor, device: copy.deepcopy(controller).to(device)

    def experts():
        return [KAEExperts(KoopmanAutoencoder(57, 2, n, "cpu"), 3) for n in (2, 3)]
    io.build_kae_experts = lambda parent, device, include_append: experts()
    io.build_square_kae_experts = lambda path, device: experts()[:1]
    io.restores = []
    io.restore_finetune_agent = lambda agent, parent: io.restores.append("finetune")
    io.initialize_square_agent = lambda agent, name, **kwargs: io.restores.append(name)
    io.initialize_square_exploration = lambda agent, path: agent.policy.log_std_parameter.data.fill_(-1.)

    class NormalizedActor(SquashedActor):
        def __init__(self, obs_dim, act_dim, hidden_dims):
            super().__init__(obs_dim, act_dim, hidden_dims)
            self.register_buffer("source_offset", torch.full((obs_dim,), .25))

        def dist_params(self, observations):
            return super().dist_params(observations - self.source_offset)

    def restore_fame(agent, parent):
        normalized = NormalizedActor(agent.obs_dim, agent.action_dim, agent.hidden_dims)
        normalized.trunk.load_state_dict(agent.meta_actor.trunk.state_dict())
        agent.meta_actor = normalized
        agent.checkpoint_modules["meta_actor"] = normalized
        agent._meta_trained = True
        agent.meta_buffer.add(torch.ones(3, 56))
        io.restores.append("fame")
    io.restore_fame_predecessor = restore_fame
    io.fame_actor_factory = lambda parent: NormalizedActor
    monkeypatch.setitem(sys.modules, "warm_start_io", io)
    return io


def _parent():
    return {"manifest": {"artifact_id": "completed-square"},
            "architecture": {"policy": {"priors": [{}, {}]}},
            "agent_modules": {"policy": {"log_std_parameter": torch.tensor([-2., -1., -.5])}}}


@pytest.mark.parametrize("name", ["kae_moe", "finetune", "residual", "componet", "fame"])
def test_original_new_shape_constructor_runs_once_and_current_learner_can_update(native, io_stub, name, tmp_path):
    env = _Env()
    method = native.METHODS[name](env, _cfg(), str(tmp_path))
    method.start_stage = Mock(wraps=method.start_stage)
    method.end_stage = Mock(side_effect=AssertionError("Warm-up must not finish a shape"))
    noise = "original" if name == "fame" else "preserve"
    agent, metadata = start_warm_stage(method, _parent(), "circle", tmp_path, new_shape_noise=noise)
    assert method.start_stage.call_count == 1 and not method.end_stage.called
    assert metadata["native_end_stage_calls"] == 0 and metadata["native_stage_index"] == 1
    if name == "kae_moe":
        assert len(method.experts) == len(agent.policy.actor.kaes) == 2
    elif name == "componet":
        assert len(method.units) == len(agent.policy.actor.unit.previous_units) == 3
    elif name == "fame":
        assert agent.meta_buffer.size == 3
        assert agent.fast2meta.size == 0
        assert hasattr(agent.fast_actor, "source_offset")
        assert metadata["auxiliary_detection_transitions"] == 8
        assert env.steps == 4  # fresh and meta, each evaluated for two steps
    if name != "fame":
        torch.testing.assert_close(agent.policy.log_std_parameter, _parent()["agent_modules"]["policy"]["log_std_parameter"])
    actor = agent.fast_actor if name == "fame" else agent.policy
    before = {key: value.detach().clone() for key, value in actor.named_parameters() if value.requires_grad}
    frozen = {key: value.detach().clone() for key, value in actor.named_parameters() if not value.requires_grad}
    agent.init()
    for i in range(4):
        states = torch.randn(2, 56)
        with torch.no_grad():
            actions = agent.act(states, i, 4)[0]
            agent.record_transition(states, actions, torch.full((2, 1), float(i)), states + .1,
                                    torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool), {}, i, 4)
        agent.post_interaction(i, 4)
    assert any(not torch.equal(value, before[key]) for key, value in actor.named_parameters() if key in before)
    for key, value in actor.named_parameters():
        if key in frozen:
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    policy = deterministic_current_policy(method, agent)
    assert policy(torch.zeros(2, 56)).shape == (2, 3)
    assert not any(parameter.requires_grad for parameter in policy.parameters())
    assert not method.end_stage.called


@pytest.mark.parametrize("name", ["kae_moe", "finetune", "residual", "componet", "fame"])
def test_square_preparation_is_explicit_and_not_claimed_to_be_a_trained_method(native, io_stub, name, tmp_path):
    method = native.METHODS[name](_Env(), _cfg(), str(tmp_path))
    noise = "original" if name == "fame" else "reset"
    agent, metadata = start_warm_stage(method, None, "square", tmp_path, new_shape_noise=noise,
                                     prepare_square=True, square_policy="square_policy.pt", square_kae="square_kae.pth")
    assert metadata["entry"] == "square_method_preparation" and metadata["parent_artifact_id"] is None
    assert metadata["auxiliary_detection_transitions"] == 0
    if name == "fame":
        assert agent.detection is None and io_stub.restores == ["fame"]
    elif name == "finetune":
        assert io_stub.restores == ["finetune"]
    else:
        assert {"critic", "optimizer", "scheduler"} <= set(metadata["state_resets"])


def test_fame_requires_original_state_dependent_noise_decision(native, io_stub, tmp_path):
    method = native.METHODS["fame"](_Env(), _cfg(), str(tmp_path))
    with pytest.raises(ValueError, match="original"):
        start_warm_stage(method, _parent(), "circle", tmp_path, new_shape_noise="preserve")


def test_compose_prior_wrapper_keeps_full_controllers_frozen_and_ordered():
    controllers = [nn.Linear(56, 3) for _ in range(2)]
    units = [PriorControllerUnit(controller) for controller in controllers]
    states = torch.randn(4, 56)
    for unit in units:
        unit.is_prev = True
    phi, actual_states = nn.Sequential(*units)(states)
    torch.testing.assert_close(phi, torch.stack([controller(states) for controller in controllers], dim=1))
    torch.testing.assert_close(actual_states, states)
    assert not any(parameter.requires_grad for unit in units for parameter in unit.parameters())


@pytest.mark.parametrize("name", ["kae_moe", "finetune", "residual", "componet", "fame"])
def test_actual_square_artifact_decoders_initialize_native_methods_and_train(native, name, tmp_path):
    """Use tensor-only Square policy input and a full native KAE, without IO stubs."""
    import warm_start_io
    from warm_start_wire import OBSERVATION_CONTRACT
    from crazysim.methods.base import make_mlp
    from crazysim.methods.kae import KoopmanAutoencoder

    source_actor = make_mlp(56, [64, 64], 3)
    policy = {"log_std_parameter": torch.tensor([-2., -1.5, -1.])}
    for source, target in (("0.", "net_container.0."), ("2.", "net_container.2."), ("4.", "policy_layer.0.")):
        policy.update({target + suffix: source_actor.state_dict()[source + suffix] for suffix in ("weight", "bias")})
    checkpoint = tmp_path / "square_policy.pt"
    torch.save({"policy": policy, "state_preprocessor": {
        "running_mean": torch.linspace(-.3, .3, 56, dtype=torch.float64),
        "running_variance": torch.linspace(.2, 1.8, 56, dtype=torch.float64),
        "current_count": torch.tensor(10001., dtype=torch.float64),
    }, "observation_contract": OBSERVATION_CONTRACT}, checkpoint)
    kae_path = tmp_path / "square_kae.pt"
    kae = KoopmanAutoencoder(57, 64, 3, "cpu")
    kae.K = torch.diag(torch.tensor([.7, .8, .9]))
    torch.save(kae, kae_path)

    cfg = _cfg()
    if name == "finetune":
        cfg["agent"].update(state_preprocessor="RunningStandardScaler", state_preprocessor_kwargs={})
    cfg = configure_for_parent(cfg, None, name, square_policy=checkpoint)
    method = native.METHODS[name](_Env(), cfg, str(tmp_path))
    noise = "original" if name == "fame" else "preserve"
    agent, metadata = start_warm_stage(method, None, "square", tmp_path,
                                     new_shape_noise=noise, prepare_square=True,
                                     square_policy=checkpoint, square_kae=kae_path)
    assert metadata["parent_artifact_id"] is None
    agent.init()
    for i in range(4):
        states = torch.randn(2, 56)
        with torch.no_grad():
            actions = agent.act(states, i, 4)[0]
            agent.record_transition(states, actions, torch.full((2, 1), float(i)), states + .1,
                                    torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool), {}, i, 4)
        agent.post_interaction(i, 4)
    evaluator = deterministic_current_policy(method, agent)
    assert torch.isfinite(evaluator(torch.randn(2, 56))).all()
    output = tmp_path / "evaluation_policy.pt"
    torch.save(evaluator, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=False)
    states = torch.randn(2, 56)
    torch.testing.assert_close(evaluator(states), reloaded(states), rtol=0, atol=0)
