"""Isolated synthetic replay and mock lifecycle; never use production games."""
import copy
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_torus9_five_channel_training import parent, game
from gocube_golden.b64_perf_audit import exact_equal
from gocube_golden.neural import model_hash
from gocube_golden.orchestrator_v2 import learning_curve as curve, operator_job as job
from gocube_golden.orchestrator_v2.execution_permit import _test_authority
from gocube_golden.provenance import file_sha256
from gocube_golden.torus9_adaptation import AdaptationTrainer, FINGERPRINT
from gocube_golden.torus9_five_channel_training import OrdinaryTrainer, OrdinaryReplayWindow


class Sink:
    def __init__(self):
        self.events = []
    def publish(self, event):
        self.events.append(event)
    def flush(self, **kwargs):
        pass


def params(tmp_path, **kwargs):
    return curve.parse_parameters({'source': 'source/M2', 'output': str(tmp_path / 'output'), **kwargs})


def input_config(tmp_path, parent):
    actor = torch.load(parent, weights_only=False)['metadata']['model_hash']
    buckets = []
    for generation in (1, 2):
        path = tmp_path / ('shard-%d.pt' % generation)
        games = [game('game-%d-%d' % (generation, i), 'train', actor) for i in range(generation)]
        torch.save({'contract': FINGERPRINT, 'actor_hash': actor, 'games': games}, path)
        buckets.append({'generation': generation, 'shards': [{'path': str(path), 'sha': file_sha256(path)}]})
    fresh = tmp_path / 'fresh.json'
    rolling = tmp_path / 'rolling.jsonl'
    fresh.write_text(json.dumps(buckets[-1]))
    rolling.write_text('\n'.join(json.dumps(b) for b in buckets))
    return {'schema': curve.SCHEMA, 'run_id': 'synthetic', 'parent_path': str(parent),
            'parent': {'sha256': file_sha256(parent)}, 'training_seed': 91,
            'device': 'cpu', 'runs_root': str(tmp_path / 'runs'),
            'parameters': params(tmp_path, snapshots=[0, 1, 3], full_positions=9, fresh_positions=6, old_positions=3),
            'replay': {'generation': 2, 'buckets': buckets, 'fresh_bucket': buckets[-1],
                       'fresh_replay': {'path': str(fresh), 'sha256': file_sha256(fresh)},
                       'rolling_replay': {'path': str(rolling), 'sha256': file_sha256(rolling)}}}


@pytest.mark.parametrize('change', [
    {'snapshots': [0, 64, 64]}, {'snapshots': [0, 2561]}, {'snapshots': [False, 64]},
    {'arms': {'A': 1e-4}}, {'arms': {'A': float('nan'), 'B': 1e-4}},
    {'full_positions': True}, {'batch_size': 128}, {'gradient_clip': 0},
    {'output': 'relative'}, {'source': '/path/checkpoint.pt'}, {'mcts': 0},
])
def test_strict_parameters(tmp_path, change):
    with pytest.raises(ValueError):
        params(tmp_path, **change)


def test_exclusive_job_and_no_game_actions(tmp_path, monkeypatch):
    raw = {'schema': job.SCHEMA, 'run_id': 'curve', 'parent': 'source/M1', 'learning_curve': params(tmp_path)}
    assert job.parse_job(job.parse_job(raw)) == job.parse_job(raw)
    for field in ('training', 'arena', 'execution', 'ab_tests', 'arenas', 'self_play', 'winner_selection'):
        with pytest.raises(ValueError, match='exclusive'):
            job.parse_job({**raw, field: {}})
    monkeypatch.setattr(curve, 'compile_parameters', lambda *a, **kw: {'verified': True})
    result = job.compile_job(raw, resolver=object())
    assert [s['action'] for s in result['workflow']['steps']] == ['learning_curve']


def test_uniform_panel_and_whole_fresh_exclusions(parent, tmp_path):
    config = input_config(tmp_path, parent)
    window, train, provenance = curve.load_inputs(config)
    panels = curve.make_panels(window, provenance, config['parameters'], 2)
    assert len(panels['panels']['full']) == 9
    assert sum(generation == 2 for g, _ in panels['panels']['full'] for generation in [provenance[g]['generation']]) == 6
    assert panels['counts'] == {'1': 3, '2': 6}
    exclusion = tmp_path / 'excluded.json'
    exclusion.write_text(json.dumps({'schema': 'gocube-learning-curve-exclusions-v1', 'game_ids': ['game-2-0']}))
    config['parameters']['exclude_games'] = {'path': str(exclusion), 'sha256': file_sha256(exclusion)}
    window, train, provenance = curve.load_inputs(config)
    panels = curve.make_panels(window, provenance, config['parameters'], 2)
    assert train.positions == 6
    assert all(g['game_id'] != 'game-2-0' for g in train)
    assert {window[g]['game_id'] for g, _ in panels['panels']['heldout-fresh']} == {'game-2-0'}
    assert all(window[g]['game_id'] != 'game-2-0' for g, _ in panels['panels']['full'])
    exclusion.write_text(exclusion.read_text() + '\n')
    with pytest.raises(ValueError, match='identity mismatch'):
        curve.load_inputs(config)


def test_source_sha_fails_before_training(parent, tmp_path):
    config = input_config(tmp_path, parent)
    Path(config['replay']['rolling_replay']['path']).write_text('{}')
    with _test_authority(), pytest.raises(ValueError, match='identity mismatch'):
        curve.run(config, dispatcher=Sink(), trainer_factory=lambda *a, **k: pytest.fail('training'))


def test_original_loss_and_nonmutating_gradient(parent):
    trainer = OrdinaryTrainer(parent, learning_rate=1e-4, gradient_clip=8, seed=91)
    window = OrdinaryReplayWindow.from_games([game('example', 'train', model_hash(trainer.model))])
    rows = [[0, 0], [0, 1], [0, 2]]
    batch = next(curve.batches(window, rows, 'cpu'))
    original, _ = trainer.losses(batch)
    calculated = curve.row_metrics(trainer, batch)
    for loss, metric in [('policy', 'policy_ce'), ('wdl', 'wdl_ce'), ('ownership', 'ownership'), ('score', 'score_mse')]:
        assert float(original[loss]) == pytest.approx(float(calculated[metric].mean()), abs=1e-6)
    assert float(calculated['total_loss'].mean()) == pytest.approx(float(sum(original.values())), abs=1e-6)
    trainer.model.train()
    for p in trainer.model.parameters():
        p.grad = torch.ones_like(p)
    old_grads = [p.grad for p in trainer.model.parameters()]
    adam = copy.deepcopy(trainer.optimizer.state_dict())
    model = copy.deepcopy(trainer.model.state_dict())
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    gradient = curve.fixed_gradient(trainer, window, rows)
    curve.evaluate_panel(trainer, window, rows)
    assert trainer.model.training
    assert all(p.grad is old for p, old in zip(trainer.model.parameters(), old_grads))
    assert exact_equal(adam, trainer.optimizer.state_dict())
    assert exact_equal(model, trainer.model.state_dict())
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert trainer.update == 0
    # Known analytical direction, including zero-norm handling.
    stats = curve.step_statistics(torch.tensor([1., 2.]), torch.tensor([.9, 1.8]), torch.tensor([1., 2.]))
    assert stats['gradient_dot_step'] < 0
    assert stats['gradient_step_cosine'] == pytest.approx(-1)
    assert curve.step_statistics(torch.zeros(2), torch.zeros(2), torch.ones(2))['gradient_step_cosine'] is None


def test_analytical_losses():
    class Model:
        def forward_auxiliary(self, observation):
            n = len(observation)
            return torch.zeros(n, 82), torch.zeros(n, 3), torch.zeros(n, 81, 3), torch.ones(n)
    batch = {'observation': torch.zeros(2, 5, 81), 'pi': torch.full((2, 82), 1 / 82),
             'z': torch.tensor([[1., 0., 0.], [0., 1., 0.]]), 'ownership': torch.zeros(2, 81, dtype=torch.long), 'score': torch.zeros(2)}
    metrics = curve.row_metrics(SimpleNamespace(model=Model()), batch)
    assert metrics['policy_ce'].mean() == pytest.approx(math.log(82), abs=1e-6)
    assert metrics['policy_kl'].mean() == pytest.approx(0, abs=1e-6)
    assert metrics['wdl_ce'].mean() == pytest.approx(math.log(3), abs=1e-6)
    assert metrics['ownership'].mean() == pytest.approx(math.log(3), abs=1e-6)
    assert metrics['score_mse'].mean() == 1
    assert metrics['score_mae_points'].mean() == 81.5
    assert metrics['brier'].mean() == pytest.approx(2 / 3)
    assert metrics['total_loss'].mean() == pytest.approx(math.log(82) + 2 * math.log(3) + 1, abs=1e-6)


def test_continuous_snapshots_exact_parity_and_resume(parent, tmp_path, monkeypatch):
    config = input_config(tmp_path, parent)
    window, train, _ = curve.load_inputs(config)
    baseline = OrdinaryTrainer(parent, learning_rate=1e-4, gradient_clip=8, seed=91)
    with _test_authority():
        for _ in range(3):
            baseline.step(train)
        sink = Sink()
        result = curve.run(config, dispatcher=sink)
    output = Path(config['parameters']['output'])
    saved = OrdinaryTrainer(output / 'A/update-0003/checkpoint.pt', learning_rate=1e-4, gradient_clip=8, seed=91)
    assert saved.update == 3
    assert exact_equal(saved.model.state_dict(), baseline.model.state_dict())
    assert exact_equal(saved.optimizer.state_dict(), baseline.optimizer.state_dict())
    assert result['same_ordered_samples']
    assert [e.event_type for e in sink.events] == ['EXPERIMENT_STARTED', 'EXPERIMENT_COMPLETED']
    report = json.loads((output / 'report.json').read_text())
    assert [s['updates'] for s in report['summaries']['A']] == [0, 1, 3]
    assert report['sample_draws_per_arm'] == 192
    # Re-entry must read sealed endpoints and perform no more optimizer steps.
    monkeypatch.setattr(OrdinaryTrainer, 'step', lambda *a: pytest.fail('completed resume stepped'))
    with _test_authority():
        assert curve.run(config, dispatcher=Sink()) == result
    (output / 'B/update-0001/metrics.json').write_text('{}')
    with _test_authority(), pytest.raises(ValueError, match='snapshot changed'):
        curve.run(config, dispatcher=Sink())


def test_interrupted_trajectory_resumes_from_sealed_snapshot(parent, tmp_path, monkeypatch):
    config = input_config(tmp_path, parent)
    original = OrdinaryTrainer.step
    calls = 0
    def interrupt(self, games):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('synthetic interruption')
        return original(self, games)
    monkeypatch.setattr(OrdinaryTrainer, 'step', interrupt)
    with _test_authority(), pytest.raises(RuntimeError, match='interruption'):
        curve.run(config, dispatcher=Sink())
    assert (Path(config['parameters']['output']) / 'A/update-0001/seal.json').exists()
    monkeypatch.setattr(OrdinaryTrainer, 'step', original)
    with _test_authority():
        result = curve.run(config, dispatcher=Sink())
    assert result['same_ordered_samples']
    _, train, _ = curve.load_inputs(config)
    baseline = OrdinaryTrainer(parent, learning_rate=1e-4, gradient_clip=8, seed=91)
    with _test_authority():
        for _ in range(3):
            baseline.step(train)
    restored = OrdinaryTrainer(Path(config['parameters']['output']) / 'A/update-0003/checkpoint.pt', learning_rate=1e-4, gradient_clip=8, seed=91)
    assert exact_equal(baseline.model.state_dict(), restored.model.state_dict())
    assert exact_equal(baseline.optimizer.state_dict(), restored.optimizer.state_dict())


def test_job_check_is_read_only_and_validates_adam(parent, tmp_path, monkeypatch):
    from gocube_golden.orchestrator_v2 import production_entrypoint as entry, offline_replay
    config = input_config(tmp_path, parent)
    parameters = config['parameters']
    node = SimpleNamespace(path=parent,
        effective_config=SimpleNamespace(config=SimpleNamespace(topology='torus9', compatibility={'input_channels': 5})),
        ref=SimpleNamespace(to_dict=lambda: config['parent']))
    monkeypatch.setattr(job, 'resolve_parent', lambda *a, **kw: node)
    monkeypatch.setattr(job, '_base_config', lambda *a: {'execution': {'training_master_seed': 91}})
    monkeypatch.setattr(offline_replay, 'resolve_offline_replay', lambda *a, **kw: [config['replay']])
    monkeypatch.setattr(entry, 'ArtifactResolver', lambda *a: SimpleNamespace(runs_root=tmp_path / 'runs'))
    # compile_job constructs its own resolver, so replace it at its owning module.
    monkeypatch.setattr(job, 'ArtifactResolver', lambda *a: SimpleNamespace(runs_root=tmp_path / 'runs'))
    monkeypatch.setattr(entry, '_launch_durable_workflow_controller', lambda *a, **kw: pytest.fail('controller started'))
    monkeypatch.setattr(OrdinaryTrainer, 'step', lambda *a: pytest.fail('optimizer stepped'))
    monkeypatch.setattr(curve, 'create_telegram_dispatcher', lambda *a: pytest.fail('transport created'))
    raw = {'schema': job.SCHEMA, 'run_id': 'synthetic', 'parent': 'source/M1', 'learning_curve': parameters}
    compiled = entry.launch_operator_job(raw, runs_root=tmp_path / 'runs', check_only=True)
    assert compiled['state'] == 'VALIDATED'
    assert not (tmp_path / 'output').exists() and not (tmp_path / 'runs').exists()
    broken = torch.load(parent, weights_only=False)
    first = next(iter(broken['optimizer_state_dict']['state'].values()))
    first['step'] += 1
    torch.save(broken, parent)
    config['parent']['sha256'] = file_sha256(parent)
    with pytest.raises(ValueError, match='clock'):
        entry.launch_operator_job(raw, runs_root=tmp_path / 'runs', check_only=True)
