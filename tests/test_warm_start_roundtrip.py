"""Optional cross-checkout acceptance: real parent -> native next shape -> real learner.

Only tensor environments are exercised here. This is not an Isaac physics or
flight test. Set DRONE_TEST_ROOT when the companion checkout is elsewhere.
"""

import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch
from skrl.trainers.torch import SequentialTrainer

from test_warm_start_stage import native, _Env
from test_warm_start_config import native_trajectories, _configs
from warm_start import count_training_interactions
from warm_start_config import build_warm_start_configuration, configure_for_parent
from warm_start_stage import start_warm_stage
from warm_start_io import export_warmed_learner, load_real_parent

ROOT = Path(os.environ.get('CRAZYSIM_SOURCE_ROOT', Path(__file__).resolve().parents[1]))


@pytest.fixture
def real_runtime():
    companion = Path(os.environ.get('DRONE_TEST_ROOT', ROOT.parent / 'Drone_Test_TRO'))
    fixtures = companion / 'tests/test_transfer_bundle.py'
    if not fixtures.is_file():
        pytest.skip('Optional cross-repository acceptance needs the Drone_Test_TRO checkout.')
    sys.path.insert(0, str(companion))
    spec = importlib.util.spec_from_file_location('warm_start_real_fixture', fixtures)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorEnvironment(_Env):
    max_episode_length = 4
    spread_episode_starts = True

    @property
    def unwrapped(self):
        return self

    def set_task(self, name):
        self.task = name


def _parent(real, method, target_shape, tmp_path):
    shapes = ('square', 'circle', 'triangle', 'figure8', 'skewed_figure8')
    source_shape = shapes[shapes.index(target_shape) - 1]
    source = real._agent(method)
    if method == 'fame':
        source.games, source.flag_reg = [source_shape], ['Initial']
    for i in range(4):
        real._step(source, i)
    config = real.runtime.SetpointRuntimeConfig(pattern=source_shape, ppo_rollout_steps=4, real_transition_budget=4)
    spec = real.runtime.setpoint_spec_for_shape(source_shape)
    path = real.export_transfer_bundle(source, config, spec, tmp_path / 'parent_real', method, phase='real')
    original = real.load_transfer_bundle(path)
    strategies = {'finetune': 'continue_actor_and_compatible_algorithm_state',
                  'residual': 'freeze_completed_current_controller_as_base',
                  'componet': 'append_completed_current_controller_to_prior_stack',
                  'kae_moe': 'append_full_real_policy_kae',
                  'fame': 'retain_once_consolidated_meta_state'}
    next_stage = {'shape': target_shape, 'phase': 'simulation', 'strategy': strategies[method],
                  'completed_real_stage': {'artifact_id': original['manifest']['artifact_id'],
                                           'path': str(path), 'sha256': original['_source']['sha256']},
                  'software_validation_only': False}
    if method == 'kae_moe':
        # A schema-valid synthetic fitted-KAE fixture; no fitting fidelity claim.
        next_stage['appended_kae'] = real.describe_actor(source.policy.moe_core.kae.blocks[-1])
        next_stage['appended_kae_record'] = {'path': 'synthetic_square_full_policy_kae.pth',
                                             'sha256': real._digest(next_stage['appended_kae'])}
    if method == 'fame':
        source.end_of_shape_segment()
        next_stage['consolidated_fame_state'] = source.fame_state(source_shape)
        next_stage['consolidation_count'] = 1
    finalized = real.update_transfer_bundle(path, next_stage=next_stage, output_path=tmp_path / 'parent_finalized.pt')
    return load_real_parent(finalized, method, target_shape)


@pytest.mark.parametrize('method', ['kae_moe', 'finetune', 'residual', 'componet', 'fame'])
@pytest.mark.parametrize('target_shape', ['circle', 'triangle', 'figure8', 'skewed_figure8'])
def test_completed_real_parent_native_next_shape_then_real_import_and_update(
    native_trajectories, native, real_runtime, method, target_shape, tmp_path,
):
    torch.manual_seed(317)
    real = real_runtime
    parent = _parent(real, method, target_shape, tmp_path)
    env_cfg, base_cfg, general_cfg = _configs()
    _, cfg, contract = build_warm_start_configuration(
        env_cfg, base_cfg, method, target_shape, general_agent_cfg=general_cfg,
        parent_environment=parent['manifest']['environment'], num_envs=2, rollout_steps=4, iterations=1,
    )
    cfg = configure_for_parent(cfg, parent, method)
    cfg['agent']['learning_epochs'] = cfg['agent']['mini_batches'] = 1
    cfg['agent']['experiment'].update(write_interval=0, checkpoint_interval=0)
    cfg['methods']['fame'].update(detection_steps=4, ppo_epochs=1, num_mini_batches=1,
                                mini_batch_size=4, epoch_meta=1, size_fast2meta=32, size_meta=64)
    env = TensorEnvironment()
    selected = native.METHODS[method](env, cfg, str(tmp_path / 'next_simulation'))
    agent, metadata = start_warm_stage(
        selected, parent, target_shape, str(tmp_path / 'next_simulation'),
        new_shape_noise='original' if method == 'fame' else 'preserve',
    )
    trainer = SequentialTrainer(env=env, agents=agent, cfg={'timesteps': 4, 'disable_progressbar': True,
                                                         'close_environment_at_exit': False, 'headless': True})
    with count_training_interactions(agent) as counters:
        trainer.train()
    assert counters == {'accepted_transitions': 8, 'complete_updates': 1}
    path = export_warmed_learner(
        selected, agent, tmp_path / 'next_simulation' / 'transfer_bundle.pt', parent=parent,
        shape=target_shape, environment=contract, iterations=1, counters=counters,
        checkpoint_selection='exact_cutoff', source_revision='hardware-free-fixture', initialization=metadata,
    )
    target_config = real.runtime.SetpointRuntimeConfig(pattern=target_shape, ppo_rollout_steps=4,
        figure8_amp_x=.7 if target_shape == 'skewed_figure8' else 1.,
        figure8_amp_y=.35 if target_shape == 'skewed_figure8' else .5)
    bundle = real.load_transfer_bundle(path, expected_method=method, expected_shape=target_shape,
                                      expected_observation_contract=real.runtime.policy_observation_contract(target_config),
                                      expected_environment=real.environment_contract(target_config, real.runtime.setpoint_spec_for_shape(target_shape)))
    target = real._agent(method, bundle)
    real.restore_transfer_agent(target, bundle)
    states = torch.randn(5, 56)
    with torch.no_grad():
        if method == 'fame':
            mu, scale = agent.fast_actor.dist_params(states)
            expected = (mu.tanh(), scale.log())
        else:
            normalized = agent._state_preprocessor(states, train=False)
            expected = (agent.policy.actor(normalized), agent.policy.log_std_parameter)
        actual = real._params(target, states)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, rtol=2e-5, atol=3e-6)
    assert target.memory.memory_index == 0 and target._accepted_transition_count == 0
    assert bundle['manifest']['parent']['artifact_id'] == parent['manifest']['artifact_id']
    assert bundle['manifest']['cumulative_counters']['accepted_transitions'] == 12
    if method == 'kae_moe':
        assert len(target.policy.moe_core.kae.blocks) == len(agent.policy.actor.kaes) == 3
    if method == 'componet':
        assert target.policy.componet_core.n_prev == len(agent.policy.actor.unit.previous_units) == 3
    if method == 'fame':
        assert target.memory_fast2meta.size == agent.fast2meta.size
        assert target.memory_meta.size == agent.meta_buffer.size
    before = copy.deepcopy(target.models['policy'].state_dict())
    for i in range(4):
        real._step(target, i)
    assert target._ppo_update_count == 1 and target._accepted_transition_count == 4
    assert any(not torch.equal(before[key], value) for key, value in target.models['policy'].state_dict().items())
