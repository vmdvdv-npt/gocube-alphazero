"""Opt-in frozen replay diagnostic, owned by the standard durable V2 workflow.

Calls the ordinary trainer unchanged. All snapshots/evidence live outside runs;
no production registration, games, search or winner selection occurs here.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

import torch
import torch.nn.functional as F

from ..notifications import create_telegram_dispatcher, operator_event
from ..process_supervision import atomic_write_json
from ..provenance import file_sha256, sha256_fingerprint
from ..torus9_five_channel_training import OrdinaryReplayWindow, OrdinaryTrainer, TrainingHeartbeat, load_replay
from ..torus9_adaptation import save_torch
from ..training_profile import Collector
from .execution_permit import active_authority, require_engine_execution

SCHEMA = 'gocube-learning-curve-v1'
FIELDS = ('observation', 'pi', 'z', 'ownership', 'score')
DEFAULTS = {'batch_size': 64, 'gradient_clip': 8.0,
            'snapshots': [0, 64, 256, 640, 1280, 2560],
            'arms': {'A': 0.0001, 'B': 0.000025}, 'evaluation_seed': 2026101101,
            'full_positions': 16384, 'fresh_positions': 8192, 'old_positions': 4096}


def parse_parameters(value):
    from .operator_job import _object, _selector, _integer, _name
    raw = _object(value, {*DEFAULTS, 'source', 'output', 'exclude_games'}, 'learning_curve')
    result = {**copy.deepcopy(DEFAULTS), **raw}
    result['source'] = _selector(raw.get('source'), 'learning_curve.source')
    if not isinstance(raw.get('output'), str) or not Path(raw['output']).is_absolute():
        raise ValueError('learning_curve.output must be an absolute external directory')
    result['output'] = str(Path(raw['output']).resolve())
    if result['batch_size'] != 64 or type(result['batch_size']) is not int:
        raise ValueError('learning_curve requires batch_size 64')
    clip = result['gradient_clip']
    if type(clip) not in (int, float) or not math.isfinite(clip) or clip <= 0:
        raise ValueError('learning_curve.gradient_clip must be finite and positive')
    steps = result['snapshots']
    if (not isinstance(steps, list) or any(type(s) is not int for s in steps)
            or len(steps) < 2 or steps[0] != 0 or steps != sorted(set(steps))
            or steps[-1] > 2560):
        raise ValueError('snapshots must be increasing unique integers from 0 to at most 2560')
    arms = _object(result['arms'], {'A', 'B'}, 'learning_curve.arms')
    if set(arms) != {'A', 'B'}:
        raise ValueError('learning_curve requires exactly A and B')
    for name, lr in arms.items():
        _name(name, 'arm')
        if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr < 1:
            raise ValueError('arm learning rates must be finite and between 0 and 1')
    for key in ('evaluation_seed', 'full_positions', 'fresh_positions', 'old_positions'):
        _integer(result[key], 'learning_curve.' + key, minimum=0 if key == 'evaluation_seed' else 1)
        if key != 'evaluation_seed' and result[key] > 65536:
            raise ValueError('evaluation panel limit is 65536 positions')
    if 'exclude_games' in raw:
        ref = _object(raw['exclude_games'], {'path', 'sha256'}, 'exclude_games')
        if (set(ref) != {'path', 'sha256'} or not isinstance(ref['path'], str)
                or not Path(ref['path']).is_absolute() or not isinstance(ref['sha256'], str)
                or re.fullmatch(r'sha256:[0-9a-f]{64}', ref['sha256']) is None):
            raise ValueError('exclude_games requires absolute path and sha256: identity')
        result['exclude_games'] = ref
    return result


def verify_sources(config):
    identities = {config['parent_path']: config['parent']['sha256']}
    row = config['replay']
    for key in ('fresh_replay', 'rolling_replay'):
        identities[row[key]['path']] = row[key]['sha256']
    for bucket in row['buckets']:
        for shard in bucket['shards']:
            identities[shard['path']] = shard['sha']
    exclusion = config['parameters'].get('exclude_games')
    if exclusion:
        identities[exclusion['path']] = exclusion['sha256']
    for path, digest in identities.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError('Learning curve source identity mismatch: ' + path)
    if json.loads(Path(row['fresh_replay']['path']).read_text()) != row['fresh_bucket']:
        raise ValueError('Learning curve fresh manifest changed')
    if [json.loads(line) for line in Path(row['rolling_replay']['path']).read_text().splitlines()] != row['buckets']:
        raise ValueError('Learning curve rolling manifest changed')
    return identities


def excluded_ids(parameters):
    ref = parameters.get('exclude_games')
    if ref is None:
        return set()
    if file_sha256(Path(ref['path'])) != ref['sha256']:
        raise ValueError('Exclusion manifest SHA mismatch')
    raw = json.loads(Path(ref['path']).read_text())
    if (set(raw) != {'schema', 'game_ids'} or raw['schema'] != 'gocube-learning-curve-exclusions-v1'
            or not isinstance(raw['game_ids'], list) or not raw['game_ids']
            or any(not isinstance(g, str) or not g for g in raw['game_ids'])
            or len(set(raw['game_ids'])) != len(raw['game_ids'])):
        raise ValueError('Expected a nonempty unique game_ids exclusion manifest')
    return set(raw['game_ids'])


def load_inputs(config):
    """Preserve historical order and every training row; attach evidence only."""
    verify_sources(config)
    games, provenance = [], []
    for bucket in config['replay']['buckets']:
        for shard in bucket['shards']:
            window = load_replay([{'shards': [shard]}], split='train')
            for game in window:
                games.append(game)
                provenance.append({'game_id': game['game_id'], 'generation': bucket['generation'],
                                   'shard_sha256': shard['sha'], 'positions': len(game['score'])})
    if len({g['game_id'] for g in games}) != len(games):
        raise ValueError('Duplicate games across rolling replay')
    excluded = excluded_ids(config['parameters'])
    fresh_generation = config['replay']['generation']
    fresh_ids = {p['game_id'] for p in provenance if p['generation'] == fresh_generation}
    if not excluded <= fresh_ids:
        raise ValueError('Exclusions must identify whole fresh training games of the source generation')
    keep = [g for g in games if g['game_id'] not in excluded]
    if not keep:
        raise ValueError('Exclusions removed all training positions')
    return OrdinaryReplayWindow.from_games(games), OrdinaryReplayWindow.from_games(keep), provenance


def compile_parameters(job, *, resolver):
    from .operator_job import resolve_parent, _require_five_channel_checkpoint, _base_config
    from .offline_replay import resolve_offline_replay
    parent = resolve_parent(job['parent'], resolver=resolver)
    _require_five_channel_checkpoint(parent, 'Learning curve parent')
    base = _base_config(parent)
    parameters = job['learning_curve']
    output = Path(parameters['output']).resolve()
    runs = resolver.runs_root.resolve()
    if output == runs or runs in output.parents or output in runs.parents:
        raise ValueError('Learning curve output must be outside and not contain runs_root')
    row = resolve_offline_replay([parameters['source']], parent=parent, resolver=resolver)[0]
    config = {'schema': SCHEMA, 'run_id': job['run_id'], 'parameters': parameters,
              'parent': parent.ref.to_dict(), 'parent_path': str(parent.path.resolve()),
              'training_seed': base['execution']['training_master_seed'], 'replay': row,
              'device': 'cuda', 'runs_root': str(runs)}
    verify_sources(config)
    # Validate the actual inherited model, complete Adam and unequal named clocks.
    # CPU preflight performs no step, writes nothing and sends no event.
    OrdinaryTrainer(parent.path, learning_rate=parameters['arms']['A'],
                    gradient_clip=parameters['gradient_clip'], batch_size=64,
                    seed=config['training_seed'], device='cpu')
    if parameters.get('exclude_games'):
        window, _, provenance = load_inputs(config)
        make_panels(window, provenance, parameters, row['generation'])
    if output.exists():
        saved = output / 'config.json'
        if not saved.is_file() or json.loads(saved.read_text()) != config:
            raise ValueError('Diagnostic output already exists with a different identity')
        verify_existing_evidence(config)
    return config


def position_rows(window, indices):
    rows = []
    for index in indices:
        game = bisect_right(window.cumulative, index)
        row = index - (window.cumulative[game - 1] if game else 0)
        rows.append((game, row))
    return rows


def make_panels(window, provenance, parameters, fresh_generation):
    excluded = excluded_ids(parameters)
    def choose(game_indices, count, seed):
        subset = OrdinaryReplayWindow.from_games(window[g] for g in game_indices)
        if not subset.positions:
            raise ValueError('Empty evaluation panel')
        draws = sorted(random.Random(seed).sample(range(subset.positions), min(count, subset.positions)))
        return [[game_indices[g], row] for g, row in position_rows(subset, draws)]
    eligible = [i for i, g in enumerate(window) if g['game_id'] not in excluded]
    seed = parameters['evaluation_seed']
    panels = {'full': choose(eligible, parameters['full_positions'], seed)}
    for generation in sorted({p['generation'] for p in provenance}):
        indices = [i for i in eligible if provenance[i]['generation'] == generation]
        key = 'fresh' if generation == fresh_generation else 'old-' + str(generation)
        panels[key] = choose(indices, parameters['fresh_positions'] if key == 'fresh' else parameters['old_positions'], seed + generation)
    if excluded:
        panels['heldout-fresh'] = choose([i for i, g in enumerate(window) if g['game_id'] in excluded], parameters['fresh_positions'], seed + 1000000)
    return {'sampling': 'uniform positions without replacement; equal weights; historical train split',
            'seed': seed, 'games': provenance, 'panels': panels,
            'excluded_game_ids': sorted(excluded),
            'counts': {str(gen): sum(p['positions'] for p in provenance if p['generation'] == gen and p['game_id'] not in excluded)
                       for gen in sorted({p['generation'] for p in provenance})}}


def batches(window, rows, device, size=256):
    for start in range(0, len(rows), size):
        part = rows[start:start + size]
        yield {key: torch.stack([window[g][key][r] for g, r in part]).to(device) for key in FIELDS}


@contextmanager
def evaluation_state(trainer):
    """Diagnostics never consume stochastic state or replace training gradients."""
    mode = trainer.model.training
    grads = [p.grad for p in trainer.model.parameters()]
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        trainer.model.eval()
        yield
    finally:
        trainer.model.train(mode)
        for p, grad in zip(trainer.model.parameters(), grads):
            p.grad = grad
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def row_metrics(trainer, batch):
    policy, value, ownership, score = trainer.model.forward_auxiliary(batch['observation'])
    log_policy = policy.log_softmax(1)
    pi = batch['pi']
    entropy = -(torch.where(pi > 0, pi * pi.clamp_min(torch.finfo(pi.dtype).tiny).log(), 0)).sum(1)
    metrics = {'policy_ce': -(pi * log_policy).sum(1),
               'target_entropy': entropy,
               'wdl_ce': -(batch['z'] * value.log_softmax(1)).sum(1),
               'ownership': F.cross_entropy(ownership.reshape(-1, 3), batch['ownership'].reshape(-1), reduction='none').reshape(len(pi), -1).mean(1),
               'score_mse': (score - batch['score']).square().reshape(len(pi), -1).mean(1),
               'brier': (value.softmax(1) - batch['z']).square().sum(1),
               'score_mae_points': ((score - batch['score']).abs() * 81.5).reshape(len(pi), -1).mean(1)}
    metrics['policy_kl'] = metrics['policy_ce'] - entropy
    metrics['total_loss'] = sum(metrics[k] for k in ('policy_ce', 'wdl_ce', 'ownership', 'score_mse'))
    return metrics


def evaluate_panel(trainer, window, rows):
    device = next(trainer.model.parameters()).device
    collected = {}
    with evaluation_state(trainer), torch.no_grad():
        for batch in batches(window, rows, device):
            for key, values in row_metrics(trainer, batch).items():
                collected.setdefault(key, []).append(values.detach().cpu())
    values = {key: torch.cat(parts).double() for key, parts in collected.items()}
    if any(not bool(torch.isfinite(v).all()) for v in values.values()):
        raise FloatingPointError('Nonfinite fixed-panel diagnostic metric')
    return {'means': {key: float(v.mean()) for key, v in values.items()}, 'rows': values}


def fixed_gradient(trainer, window, rows):
    device = next(trainer.model.parameters()).device
    with evaluation_state(trainer):
        trainer.optimizer.zero_grad(set_to_none=True)
        for batch in batches(window, rows, device, size=64):
            # Call the original loss function, including its original reductions.
            (sum(trainer.losses(batch)[0].values()) * len(batch['pi']) / len(rows)).backward()
        return torch.cat([p.grad.detach().reshape(-1).clone() for p in trainer.model.parameters()])


def vector(trainer):
    return torch.cat([p.detach().reshape(-1) for p in trainer.model.parameters()])


def step_statistics(before, after, gradient=None):
    delta = (after - before).double()
    norm = float(delta.norm())
    base = float(before.double().norm())
    result = {'step_l2': norm, 'relative_step_l2': norm / base if base else None}
    if gradient is not None:
        gradient = gradient.double()
        dot = float(torch.dot(gradient, delta))
        grad_norm = float(gradient.norm())
        result.update(fixed_gradient_l2=grad_norm, gradient_dot_step=dot,
                      gradient_step_cosine=dot / (grad_norm * norm) if grad_norm and norm else None)
    return result


def parameter_change(trainer, origin):
    result, offset = {}, 0
    total = (vector(trainer) - origin).double()
    result['all'] = {'change_l2': float(total.norm()), 'relative_change_l2': float(total.norm() / origin.double().norm())}
    groups = {}
    for name, p in trainer.model.named_parameters():
        group = next((head for head in ('policy', 'wdl', 'value', 'ownership', 'score') if head in name), 'trunk')
        groups.setdefault(group, []).append((offset, offset + p.numel()))
        offset += p.numel()
    for group, slices in groups.items():
        change = torch.cat([total[a:b] for a, b in slices])
        base = torch.cat([origin[a:b] for a, b in slices]).double().norm()
        result[group] = {'change_l2': float(change.norm()), 'relative_change_l2': float(change.norm() / base) if base else None}
    return result


def seal_directory(path):
    identities = {str(p.relative_to(path)): file_sha256(p) for p in path.rglob('*') if p.is_file()}
    atomic_write_json(path / 'seal.json', identities)
    return identities


def verify_snapshot(path):
    identities = json.loads((path / 'seal.json').read_text())
    required = {'checkpoint.pt', 'checkpoint.metadata.json', 'metrics.json', 'row-metrics.pt', 'steps.json'}
    if set(identities) != required:
        raise ValueError('Snapshot required file inventory mismatch')
    if set(identities) != {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name != 'seal.json'}:
        raise ValueError('Snapshot file inventory mismatch')
    for name, sha in identities.items():
        if file_sha256(path / name) != sha:
            raise ValueError('Immutable diagnostic snapshot changed: ' + str(path / name))


def verify_existing_evidence(config):
    """Preflight checks evidence even when a completed workflow skips its action."""
    output = Path(config['parameters']['output'])
    updates = config['parameters']['snapshots']
    for arm in config['parameters']['arms']:
        present = [u for u in updates if (output / arm / ('update-%04d' % u)).exists()]
        if present != updates[:len(present)]:
            raise ValueError('Snapshot sequence contains a gap')
        for update in present:
            verify_snapshot(output / arm / ('update-%04d' % update))
    report_path = output / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['config_sha256'] != sha256_fingerprint(config):
            raise ValueError('Diagnostic report config identity mismatch')
        if file_sha256(output / 'panels.json') != report['panel_sha256']:
            raise ValueError('Diagnostic panel integrity mismatch')
        for arm in config['parameters']['arms']:
            if not all((output / arm / ('update-%04d' % u)).is_dir() for u in updates):
                raise ValueError('Completed report has missing snapshots')
    result_path = output / 'runtime/results/learning-curve.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        evidence = result['result']
        if (result.get('schema') != SCHEMA or not report_path.exists()
                or Path(evidence['report']).resolve() != report_path
                or file_sha256(report_path) != evidence['sha256']
                or result['execution_code_commit'] != report['execution_code_commit']):
            raise ValueError('Diagnostic result/report integrity mismatch')


def save_snapshot(trainer, root, update, config, window, panels, origin, trace):
    final = root / ('update-%04d' % update)
    pending = root / '.pending-snapshot'
    if final.exists():
        raise ValueError('Refusing to overwrite diagnostic snapshot')
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    trainer.save(pending / 'checkpoint.pt', config_hash=sha256_fingerprint(config),
                 parent=config['parent'], replay_buckets=config['replay']['buckets'])
    evaluations = {name: evaluate_panel(trainer, window, rows) for name, rows in panels.items()}
    save_torch(pending / 'row-metrics.pt', {name: result['rows'] for name, result in evaluations.items()})
    summary = {'updates': update, 'ordinary_update': trainer.update,
               'evaluations': {name: result['means'] for name, result in evaluations.items()},
               'parameter_change': parameter_change(trainer, origin)}
    atomic_write_json(pending / 'metrics.json', summary)
    atomic_write_json(pending / 'steps.json', trace)
    seal_directory(pending)
    os.rename(pending, final)
    from ..process_supervision import _fsync_dir
    _fsync_dir(root)
    return summary


def paired_changes(root, panels, window, updates, seed):
    """Game-cluster ratio bootstrap on fixed sampled rows, including multiplicity."""
    import numpy as np
    baseline = torch.load(root / 'update-0000/row-metrics.pt', weights_only=True)
    result = {}
    for update in updates:
        current = torch.load(root / ('update-%04d/row-metrics.pt' % update), weights_only=True)
        per_panel = {}
        for name, rows in panels.items():
            game_ids = [window[g]['game_id'] for g, _ in rows]
            games = sorted(set(game_ids))
            mapping = {g: i for i, g in enumerate(games)}
            labels = np.array([mapping[g] for g in game_ids])
            counts = np.bincount(labels, minlength=len(games))
            rng = np.random.default_rng(seed)
            # Bounded chunks avoid a bootstrap-replicates x games allocation.
            estimates = {key: [] for key in current[name]}
            differences = {key: (current[name][key] - baseline[name][key]).numpy() for key in current[name]}
            sums = {key: np.bincount(labels, weights=d, minlength=len(games)) for key, d in differences.items()}
            for _ in range(20):
                draws = rng.integers(0, len(games), size=(50, len(games)))
                denominator = counts[draws].sum(1)
                for key in estimates:
                    estimates[key].extend((sums[key][draws].sum(1) / denominator).tolist())
            per_panel[name] = {key: {'delta_from_zero': float(d.mean()),
                                    'game_cluster_ci95': np.quantile(estimates[key], [.025, .975]).tolist(),
                                    'sampled_games': len(games)} for key, d in differences.items()}
        result[str(update)] = per_panel
    return result


def execution_commit():
    authority = active_authority()
    return authority.code_identity if authority else os.environ.get('AZ_ORCHESTRATOR_RUNTIME_COMMIT')


def notify(dispatcher, config, kind, action, message, payload=None):
    dispatcher.publish(operator_event(kind, topology='torus9', owner_type='experiment',
        owner_id=config['run_id'], action_id=action, message=message,
        execution_code_commit=execution_commit(),
        payload={'experiment_id': config['run_id'], **(payload or {})}))
    dispatcher.flush(timeout=7)


def run(config, *, dispatcher=None, trainer_factory=OrdinaryTrainer):
    require_engine_execution(__name__, action='training', topology='torus9')
    if config.get('schema') != SCHEMA:
        raise ValueError('Expected compiled learning curve config')
    output = Path(config['parameters']['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = output / 'config.json'
        if saved.exists():
            if json.loads(saved.read_text()) != config:
                raise ValueError('Diagnostic config identity changed')
        else:
            if set(p.name for p in output.iterdir()) != {'.run.lock'}:
                raise ValueError('Output is not an exclusive diagnostic directory')
            atomic_write_json(saved, config)
        identities = verify_sources(config)
        dispatcher = dispatcher or create_telegram_dispatcher(output)
        notify(dispatcher, config, 'EXPERIMENT_STARTED', 'start',
               'Frozen replay learning curve started: two continuous Adam trajectories; no games or search.')
        try:
            return _run_locked(config, output, identities, dispatcher, trainer_factory)
        except Exception as exc:
            notify(dispatcher, config, 'RUN_FAILED', 'failed-' + sha256_fingerprint(str(exc))[-16:], str(exc))
            raise


def _run_locked(config, output, identities, dispatcher, trainer_factory):
    parameters = config['parameters']
    torch.set_num_threads(1)
    window, training, provenance = load_inputs(config)
    panels = make_panels(window, provenance, parameters, config['replay']['generation'])
    panel_path = output / 'panels.json'
    if panel_path.exists():
        if json.loads(panel_path.read_text()) != panels:
            raise ValueError('Frozen evaluation panel changed')
    else:
        atomic_write_json(panel_path, panels)
    updates = parameters['snapshots']
    generation = config['replay']['generation']
    heartbeat = TrainingHeartbeat(output / ('runtime/heartbeats/generation-%04d.json' % generation), generation)
    heartbeat.start()
    summaries = {}
    try:
        for arm, lr in parameters['arms'].items():
            root = output / arm
            root.mkdir(exist_ok=True)
            snapshots = []
            for update in updates:
                path = root / ('update-%04d' % update)
                if path.exists():
                    verify_snapshot(path)
                    snapshots.append(update)
            if snapshots != updates[:len(snapshots)]:
                raise ValueError('Snapshot sequence contains a gap')
            original = trainer_factory(Path(config['parent_path']), learning_rate=lr,
                gradient_clip=parameters['gradient_clip'], batch_size=64,
                seed=config['training_seed'], device=config['device'])
            origin = vector(original).clone()
            parent_update = original.update
            trainer = original
            start = 0
            if snapshots:
                start = snapshots[-1]
                trainer = trainer_factory(root / ('update-%04d/checkpoint.pt' % start), learning_rate=lr,
                    gradient_clip=parameters['gradient_clip'], batch_size=64,
                    seed=config['training_seed'], device=config['device'])
                if trainer.update != parent_update + start:
                    raise ValueError('Snapshot ordinary clock disagrees with trajectory')
            if not snapshots:
                save_snapshot(trainer, root, 0, config, window, panels['panels'], origin, [])
            del original
            trace = []
            observed = Collector(timings=False, capture_positions=True)
            gradient_steps = {1, updates[-1], *(u + 1 for u in updates[1:-1])}
            for update in range(start + 1, updates[-1] + 1):
                heartbeat.mark('arm-' + arm, update - 1, updates[-1])
                gradient = fixed_gradient(trainer, window, panels['panels']['full']) if update in gradient_steps else None
                before = vector(trainer).clone()
                begin = time.perf_counter()
                with observed.activate():
                    metrics = trainer.step(training)
                positions = observed.positions.pop()
                sample_hash = sha256_fingerprint(positions)
                row = {'update': update, 'ordinary_update': trainer.update,
                       'sample_sha256': sample_hash, 'training': metrics,
                       'step_wall_sec': time.perf_counter() - begin,
                       **step_statistics(before, vector(trainer), gradient)}
                if gradient is not None:
                    row['fixed_gradient_at_update'] = update - 1
                norm = metrics['grad_norm_before_clip']
                row['clip_factor'] = min(1.0, parameters['gradient_clip'] / (norm + 1e-6))
                row['grad_norm_after_clip'] = norm * row['clip_factor']
                trace.append(row)
                if update in updates:
                    save_snapshot(trainer, root, update, config, window, panels['panels'], origin, trace)
                    trace = []
            summaries[arm] = [json.loads((root / ('update-%04d/metrics.json' % u)).read_text()) for u in updates]
            del trainer, origin
        traces = {arm: [row for u in updates for row in json.loads((output / arm / ('update-%04d/steps.json' % u)).read_text())]
                  for arm in parameters['arms']}
        if [r['sample_sha256'] for r in traces['A']] != [r['sample_sha256'] for r in traces['B']]:
            raise ValueError('A/B actual ordered samples differ')
        verify_sources(config)
        report = {'schema': SCHEMA, 'config_sha256': sha256_fingerprint(config),
                  'execution_code_commit': execution_commit(),
                  'source_sha256': identities, 'panel_sha256': file_sha256(panel_path),
                  'same_ordered_samples': True, 'sample_draws_per_arm': 64 * updates[-1],
                  'training_positions': training.positions, 'summaries': summaries,
                  'paired_changes': {arm: paired_changes(output / arm, panels['panels'], window, updates, parameters['evaluation_seed']) for arm in parameters['arms']},
                  'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                  'device': config['device'], 'threads': torch.get_num_threads(),
                                  'device_name': torch.cuda.get_device_name(0) if config['device'] == 'cuda' else 'CPU',
                                  'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                                  'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                                  'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                  'cudnn_deterministic': torch.backends.cudnn.deterministic,
                                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
                  'interpretation': 'Training-data diagnostic, not independent holdout or playing strength.'}
        report_path = output / 'report.json'
        if report_path.exists() and json.loads(report_path.read_text()) != report:
            raise ValueError('Refusing to replace a completed diagnostic report')
        if not report_path.exists():
            atomic_write_json(report_path, report)
        notify(dispatcher, config, 'EXPERIMENT_COMPLETED', 'complete',
               'Frozen replay learning curve completed. No playing-strength conclusion.', {'report': str(report_path)})
        return {'report': str(report_path), 'sha256': file_sha256(report_path), 'same_ordered_samples': True}
    finally:
        heartbeat.close()


def run_supervised(config):
    """Controller adapter: use the existing process supervisor and child permit."""
    import subprocess
    import sys
    from ..process_supervision import start_owned_child
    from .execution_permit import _child_execution_permit
    from .immutable_runtime import ImmutableRuntimeManager
    from .supervisor import SupervisorV2

    require_engine_execution(__name__, action='training', topology='torus9')
    authority = active_authority()
    if authority is None:
        raise RuntimeError('Learning curve supervision requires the V2 controller authority')
    output = Path(config['parameters']['output'])
    verify_sources(config)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Diagnostic config identity changed')
    else:
        if list(output.iterdir()):
            raise ValueError('Output is not an exclusive diagnostic directory')
        atomic_write_json(config_path, config)
    runtime = ImmutableRuntimeManager(Path(__file__).resolve().parents[2]).ensure(authority.code_identity)
    request = {'schema': SCHEMA, 'config': config, 'execution_code_commit': runtime.commit}
    request_path = output / 'runtime/requests/learning-curve.json'
    result_path = output / 'runtime/results/learning-curve.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Learning curve worker request changed')
    else:
        atomic_write_json(request_path, request)
    command = [sys.executable, '-m', 'gocube_golden.orchestrator_v2.learning_curve',
               '--request', str(request_path), '--result', str(result_path)]
    heartbeat = output / ('runtime/heartbeats/generation-%04d.json' % config['replay']['generation'])

    def launch(child_request):
        with _child_execution_permit(action_type='training', topology='torus9', run_id=config['run_id'],
                                     code_identity=runtime.commit, attempt=int(child_request.attempt)):
            env = runtime.environment(os.environ)
            with (output / 'worker.log').open('ab') as log:
                def popen(*args, **kwargs):
                    return subprocess.Popen(*args, stdout=log, stderr=subprocess.STDOUT, **kwargs)
                return start_owned_child(command, cwd=runtime.path, env=env, popen=popen)

    supervisor = SupervisorV2(output, execution_id=config['run_id'] + ':learning-curve',
        liveness_path=heartbeat, progress_path=heartbeat, launcher=launch,
        command=command, cwd=runtime.path)

    def verified_result():
        result = json.loads(result_path.read_text())
        if result.get('schema') != SCHEMA or result.get('execution_code_commit') != runtime.commit:
            raise ValueError('Learning curve worker result identity mismatch')
        evidence = result['result']
        if Path(evidence['report']).resolve() != output / 'report.json' or file_sha256(output / 'report.json') != evidence['sha256']:
            raise ValueError('Learning curve report integrity mismatch')
        report = json.loads((output / 'report.json').read_text())
        if report['config_sha256'] != sha256_fingerprint(config) or report['execution_code_commit'] != runtime.commit:
            raise ValueError('Learning curve report identity mismatch')
        if file_sha256(output / 'panels.json') != report['panel_sha256']:
            raise ValueError('Learning curve panel integrity mismatch')
        for arm in config['parameters']['arms']:
            for update in config['parameters']['snapshots']:
                verify_snapshot(output / arm / ('update-%04d' % update))
        verify_sources(config)
        return evidence

    if result_path.exists():
        result = verified_result()
        supervised = supervisor.reconcile_completed_execution()
    else:
        supervised = supervisor.run_once()
        if not supervised.success:
            raise RuntimeError('Learning curve worker stopped: ' + str(supervised.reason))
        result = verified_result()
    if not supervised.success:
        raise RuntimeError('Learning curve reconciliation failed: ' + str(supervised.reason))
    return result


def run_worker(request_path, result_path):
    from .execution_permit import require_child_execution_permit
    from .immutable_runtime import validate_runtime_head
    request = json.loads(Path(request_path).read_text())
    if request.get('schema') != SCHEMA:
        raise ValueError('Expected learning curve worker request')
    require_child_execution_permit(__name__, action_type='training', topology='torus9',
        run_id=request['config']['run_id'], code_identity=request['execution_code_commit'])
    validate_runtime_head(Path.cwd(), request['execution_code_commit'])
    result = run(request['config'])
    atomic_write_json(Path(result_path), {'schema': SCHEMA, 'execution_code_commit': request['execution_code_commit'], 'result': result})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Internal signed learning curve worker; launch with operator job.')
    parser.add_argument('--request', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    run_worker(args.request, args.result)
