"""The additive CLI makes budgets and task-boundary choices explicit."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'skrl'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from warm_start import build_parser, count_training_interactions, validate_arguments
from test_warm_start_stage import native


def test_help_needs_no_simulator_or_companion_checkout():
    result = subprocess.run([sys.executable, str(SCRIPTS / 'warm_start.py'), '--help'],
                            text=True, capture_output=True)
    assert result.returncode == 0
    assert '--parent-bundle' in result.stdout
    assert '--checkpoint-selection' in result.stdout
    assert '--new-shape-noise' in result.stdout


def _args(tmp_path, *, method='finetune', shape='square', noise='preserve'):
    source = tmp_path / 'source.pt'
    source.write_bytes(b'only_argument_validation')
    return ['--method', method, '--shape', shape, '--prepare-square', '--square-policy', str(source),
            '--iterations', '2', '--num-envs', '2', '--rollout-steps', '4',
            '--checkpoint-selection', 'exact', '--new-shape-noise', noise]


@pytest.mark.parametrize('flag', ['--iterations', '--num-envs', '--rollout-steps'])
def test_nonpositive_budgets_are_rejected(tmp_path, flag):
    argv = _args(tmp_path)
    argv[argv.index(flag) + 1] = '0'
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_square_preparation_cannot_silently_initialize_another_shape(tmp_path):
    parser = build_parser()
    args = parser.parse_args(_args(tmp_path, shape='circle'))
    with pytest.raises(SystemExit):
        validate_arguments(args, parser)


def test_fame_uses_the_original_detection_winners_distribution(tmp_path):
    parser = build_parser()
    invalid = parser.parse_args(_args(tmp_path, method='fame', noise='preserve'))
    with pytest.raises(SystemExit):
        validate_arguments(invalid, parser)
    valid = parser.parse_args(_args(tmp_path, method='fame', noise='original'))
    validate_arguments(valid, parser)


def test_kae_square_preparation_requires_the_shared_kae(tmp_path):
    parser = build_parser()
    args = parser.parse_args(_args(tmp_path, method='kae_moe'))
    with pytest.raises(SystemExit):
        validate_arguments(args, parser)


def test_budget_observer_counts_successful_transitions_and_updates_without_changing_calls():
    calls = []
    record = lambda *values, **kw: calls.append(('record', values, kw))
    update = lambda *values, **kw: calls.append(('update', values, kw))
    agent = SimpleNamespace(record_transition=record, _update=update)
    states = torch.zeros(2, 56)
    with count_training_interactions(agent) as counters:
        agent.record_transition(states, 'actions', timestep=0)
        agent.record_transition(states=states, timestep=1)
        agent._update(1, 2)
    assert counters == {'accepted_transitions': 4, 'complete_updates': 1}
    assert len(calls) == 3
    assert calls[0][1][0] is states and calls[0][2] == {'timestep': 0}
    assert agent.record_transition is record and agent._update is update


def test_failed_update_does_not_claim_budget_completion_and_observer_restores_original():
    def failed_update(*args, **kwargs):
        raise RuntimeError('optimizer failed')
    agent = SimpleNamespace(record_transition=lambda *args, **kw: None, _update=failed_update)
    with pytest.raises(RuntimeError, match='optimizer failed'):
        with count_training_interactions(agent) as counts:
            agent._update()
    assert counts['complete_updates'] == 0
    assert agent._update is failed_update


@pytest.mark.parametrize('method_name', ['kae_moe', 'finetune', 'residual', 'componet', 'fame'])
def test_native_launcher_orchestration_counts_real_updates_exports_and_never_finalizes(native, method_name, tmp_path, monkeypatch):
    from unittest.mock import Mock
    import warm_start
    from test_warm_start_stage import _Env, _cfg
    from test_warm_start_io import _inputs
    from warm_start_config import _contract, configure_for_parent, WarmStartRunningStandardScaler
    from warm_start_wire import read_bundle

    policy, kae = _inputs(tmp_path)
    cfg = _cfg()
    cfg['methods']['residual']['hidden_dims'] = [256, 128, 64]
    cfg['methods']['componet']['internal_hidden_dims'] = [256, 128, 64]
    cfg['methods']['kae_moe'].update(residual_hidden_dims=[256, 128, 64], weight_hidden_dims=[32],
                                   gating_hidden_dims=[32], router_hidden_dims=[128])
    if method_name == 'finetune':
        cfg['agent'].update(state_preprocessor=WarmStartRunningStandardScaler,
                            state_preprocessor_kwargs={'size': 56, 'device': 'cpu'})
    cfg = configure_for_parent(cfg, None, method_name, square_policy=policy)
    method = native.METHODS[method_name](_Env(), cfg, str(tmp_path))
    method.end_stage = Mock(side_effect=AssertionError('warm-up must not finalize the stage'))
    monkeypatch.setattr(warm_start, 'repository_revision', lambda root: 'orchestration-fixture')
    args = SimpleNamespace(method=method_name, shape='square', prepare_square=True, square_policy=policy,
                           square_kae=kae, new_shape_noise='original' if method_name == 'fame' else 'preserve',
                           iterations=1, rollout_steps=4, num_envs=2)
    _, artifact, record = warm_start.run_native_phase(method, None, args, _contract('square', None, {}), tmp_path)
    bundle = read_bundle(artifact)
    assert bundle['manifest']['phase_counters'] == {'accepted_transitions': 8, 'complete_updates': 1}
    assert record['training']['actual_iteration'] == 1
    steps = record['training']['optimizer_steps']
    assert set(steps) == ({'actor', 'critic', 'temperature'} if method_name == 'fame' else {'ppo'})
    assert all(count >= 1 for count in steps.values())
    assert (tmp_path / 'policy.pt').is_file() and (tmp_path / 'checkpoints/final_agent.pt').is_file()
    assert (tmp_path / 'warm_start_summary.json').is_file()
    method.end_stage.assert_not_called()


def test_native_launcher_closes_writer_when_training_fails(native, tmp_path, monkeypatch):
    from unittest.mock import Mock
    import warm_start
    from test_warm_start_stage import _Env, _cfg
    from test_warm_start_io import _inputs
    from warm_start_config import _contract, configure_for_parent, WarmStartRunningStandardScaler

    policy, kae = _inputs(tmp_path)
    cfg = _cfg()
    cfg['agent'].update(state_preprocessor=WarmStartRunningStandardScaler,
                        state_preprocessor_kwargs={'size': 56, 'device': 'cpu'})
    cfg = configure_for_parent(cfg, None, 'finetune', square_policy=policy)
    method = native.METHODS['finetune'](_Env(), cfg, str(tmp_path))
    writer = Mock()

    class FailedTrainer:
        def __init__(self, env, agents, cfg):
            agents.writer = writer

        def train(self):
            raise RuntimeError('training failed')

    args = SimpleNamespace(method='finetune', shape='square', prepare_square=True, square_policy=policy,
                           square_kae=kae, new_shape_noise='preserve', iterations=1, rollout_steps=4, num_envs=2)
    with pytest.raises(RuntimeError, match='training failed'):
        warm_start.run_native_phase(method, None, args, _contract('square', None, {}), tmp_path,
                                    trainer_class=FailedTrainer)
    writer.flush.assert_called_once()
    writer.close.assert_called_once()
    assert not (tmp_path / 'transfer_bundle.pt').exists()
