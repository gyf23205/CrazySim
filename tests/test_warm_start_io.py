"""Native model/state conversion tests require only PyTorch, Gymnasium and skrl."""
from __future__ import annotations

import copy
import hashlib
import io

import pytest
import torch

from test_warm_start_stage import _Env, _cfg
from warm_start_config import _contract, WarmStartRunningStandardScaler
from warm_start_io import (build_square_kae_experts, completed_deterministic_controller, decode_actor,
                           export_warmed_learner, load_real_parent, validate_square_inputs)
from warm_start_stage import deterministic_current_policy, start_warm_stage
from warm_start_wire import OBSERVATION_CONTRACT, digest, read_bundle, write_bundle


def _inputs(tmp_path):
    from crazysim.methods.base import make_mlp
    from crazysim.methods.kae import KoopmanAutoencoder
    actor = make_mlp(56, [64, 64], 3)
    state = {"log_std_parameter": torch.tensor([-1.3, -1.6, -.9])}
    for native, canonical in (("0.", "net_container.0."), ("2.", "net_container.2."), ("4.", "policy_layer.0.")):
        for suffix in ("weight", "bias"):
            state[canonical + suffix] = actor.state_dict()[native + suffix]
    payload = {"policy": state, "observation_contract": OBSERVATION_CONTRACT,
               "state_preprocessor": {"running_mean": torch.linspace(-.3, .3, 56, dtype=torch.float64),
                                      "running_variance": torch.linspace(.2, 2., 56, dtype=torch.float64),
                                      "current_count": torch.tensor(12345., dtype=torch.float64)}}
    policy = tmp_path / "square_policy.pt"
    torch.save(payload, policy)
    kae = KoopmanAutoencoder(57, 64, 4, "cpu")
    kae.K = torch.diag(torch.tensor([.6, .7, .8, .9]))
    kae_path = tmp_path / "square_kae.pt"
    torch.save(kae, kae_path)
    return policy, kae_path


def _source(method_name, tmp_path):
    from crazysim.methods import METHODS
    policy, kae = _inputs(tmp_path)
    cfg = _cfg()
    cfg["methods"]["residual"]["hidden_dims"] = [256, 128, 64]
    cfg["methods"]["componet"]["internal_hidden_dims"] = [256, 128, 64]
    cfg["methods"]["kae_moe"].update(residual_hidden_dims=[256, 128, 64], weight_hidden_dims=[32],
                                      gating_hidden_dims=[32], router_hidden_dims=[128])
    cfg["agent"].update(learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
                         learning_rate_scheduler_kwargs={"gamma": .9995})
    if method_name == "finetune":
        cfg["agent"].update(state_preprocessor=WarmStartRunningStandardScaler,
                             state_preprocessor_kwargs={"size": 56, "device": "cpu"})
    if method_name != "fame":
        cfg["agent"].update(value_preprocessor=WarmStartRunningStandardScaler,
                             value_preprocessor_kwargs={"size": 1, "device": "cpu"})
    method = METHODS[method_name](_Env(), cfg, str(tmp_path))
    agent, metadata = start_warm_stage(method, None, "square", tmp_path,
                                      new_shape_noise="original" if method_name == "fame" else "preserve",
                                      prepare_square=True, square_policy=policy, square_kae=kae)
    agent.init()
    return method, agent, metadata


def _update(agent):
    for i in range(4):
        states = torch.randn(2, 56)
        with torch.no_grad():
            actions = agent.act(states, i, 4)[0]
            agent.record_transition(states, actions, torch.full((2, 1), float(i)), states + .01,
                                    torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool), {}, i, 4)
        agent.post_interaction(i, 4)


@pytest.mark.parametrize("method_name", ["kae_moe", "finetune", "residual", "componet", "fame"])
def test_native_export_preserves_current_mapping_noise_optimizer_and_memory(method_name, tmp_path):
    torch.manual_seed(77)
    method, agent, metadata = _source(method_name, tmp_path)
    _update(agent)
    path = export_warmed_learner(method, agent, tmp_path / "transfer_bundle.pt", parent=None, shape="square",
                                 environment=_contract("square", None, {}), iterations=1,
                                 counters={"accepted_transitions": 8, "complete_updates": 1},
                                 source_revision="native-test", initialization=metadata)
    bundle = read_bundle(path)
    assert bundle["manifest"]["phase"] == "simulation"
    assert bundle["manifest"]["checkpoint_selection"]["actual_iteration"] == 1
    assert "memory" not in bundle and "rollout" not in bundle
    states = torch.randn(7, 56)
    source = deterministic_current_policy(method, agent)
    reconstructed = completed_deterministic_controller(bundle)
    with torch.no_grad():
        torch.testing.assert_close(source(states), reconstructed(states), rtol=2e-5, atol=1e-5)
    assert bundle["agent_configuration"]["optimizer_types"]["optimizer"] == "Adam"
    assert bundle["agent_modules"]["optimizer"]["state"]
    if method_name == "fame":
        buffers = bundle["agent_modules"]["fame_buffers"]
        assert buffers["fast2meta"]["size"] == agent.fast2meta.size == 8
        torch.testing.assert_close(buffers["fast2meta"]["data"], agent.fast2meta.data)
        torch.testing.assert_close(bundle["agent_modules"]["temperature"]["log_alpha"], agent.log_alpha)
    else:
        torch.testing.assert_close(bundle["agent_modules"]["policy"]["log_std_parameter"], agent.policy.log_std_parameter)
    assert "critic" not in bundle["manifest"]["resets"]
    assert "on_policy_rollout" in bundle["manifest"]["resets"]
    stream = io.BytesIO()
    torch.save(source, stream)
    stream.seek(0)
    reloaded = torch.load(stream, weights_only=False)
    with torch.no_grad():
        torch.testing.assert_close(source(states), reloaded(states), rtol=0, atol=0)


def test_hash_is_typed_order_independent_and_unambiguous():
    assert digest({"a": {}, "b": 3}) != digest({"a": {"b": 3}})
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest(3) != digest(3.0)
    assert digest(3) == hashlib.sha256(b"int:3;").hexdigest()


def test_square_policy_normalizer_and_kae_validation_rejects_incompatible_inputs(tmp_path):
    policy, kae = _inputs(tmp_path)
    checked = validate_square_inputs(policy, kae)
    assert checked["normalizer_count"] == 12345.
    assert checked["kae"]["padded_dim"] == 57
    payload = torch.load(policy, weights_only=True)
    payload["state_preprocessor"]["running_variance"][0] = -1
    torch.save(payload, policy)
    with pytest.raises(ValueError, match="normalizer"):
        validate_square_inputs(policy)


def test_exact_real_budget_finalization_and_lineage_are_required(tmp_path):
    method, agent, metadata = _source("residual", tmp_path)
    _update(agent)
    path = export_warmed_learner(method, agent, tmp_path / "simulation.pt", parent=None, shape="square",
                                 environment=_contract("square", None, {}), iterations=1,
                                 counters={"accepted_transitions": 8, "complete_updates": 1})
    bundle = read_bundle(path)
    with pytest.raises(ValueError, match="completed real"):
        load_real_parent(path, "residual", "circle")
    bundle["manifest"]["phase"] = "real"
    bundle["manifest"]["phase_counters"]["accepted_transitions"] = 4
    bundle["manifest"]["counters"]["accepted_transitions"] = 4
    bundle["manifest"]["budget"]["accepted_transitions"] = 4
    bundle["manifest"]["next_stage"] = {"shape": "circle", "phase": "simulation",
        "strategy": "freeze_completed_current_controller_as_base", "software_validation_only": False,
        "completed_real_stage": {"artifact_id": "a" * 64, "sha256": "b" * 64}}
    write_bundle(bundle, path)
    assert load_real_parent(path, "residual", "circle")["manifest"]["phase"] == "real"
    with pytest.raises(ValueError, match="preceding"):
        load_real_parent(path, "residual", "triangle")
    with pytest.raises(ValueError, match="another method"):
        load_real_parent(path, "componet", "circle")
    bundle["manifest"]["budget"]["accepted_transitions"] = 8
    write_bundle(bundle, path)
    with pytest.raises(ValueError, match="exact complete-update"):
        load_real_parent(path, "residual", "circle")


def test_nonfinite_or_checksum_corruption_is_rejected(tmp_path):
    method, agent, _ = _source("finetune", tmp_path)
    _update(agent)
    path = export_warmed_learner(method, agent, tmp_path / "bundle.pt", parent=None, shape="square",
                                 environment=_contract("square", None, {}), iterations=1,
                                 counters={"accepted_transitions": 8, "complete_updates": 1})
    raw = torch.load(path, weights_only=True)
    raw["manifest"]["equations"] = "anchored_ppo"
    torch.save(raw, path)
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(path)
    raw["manifest"]["equations"] = "crazysim_waypoint_ppo_v1"
    raw["agent_modules"]["policy"]["log_std_parameter"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        write_bundle(raw, path)


def test_exact_cutoff_cannot_claim_an_unfinished_warmup(tmp_path):
    method, agent, _ = _source("finetune", tmp_path)
    with pytest.raises(ValueError, match="observed complete-update"):
        export_warmed_learner(method, agent, tmp_path / "bundle.pt", parent=None, shape="square",
                              environment=_contract("square", None, {}), iterations=1,
                              counters={"accepted_transitions": 2, "complete_updates": 0})
