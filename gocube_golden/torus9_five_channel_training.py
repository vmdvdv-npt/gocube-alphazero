"""Ordinary 5CH generations behind the existing V2 production boundary.

The adaptation checkpoint is an immutable parent. Named Adam states (including
unequal per-parameter clocks) survive the transition; all parameters now train
at the run LR, without the adaptation anchor penalty. Replay contains only the
komi=1.5 target contract and retains whole-generation buckets.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time

import torch

from .training_profile import span, measured, sampled, transfer

from .artifact_graph import ArtifactRef, CheckpointRef, publish_checkpoint_graph
from .neural import model_hash
from .process_supervision import atomic_write_json, atomic_write_text
from .provenance import file_sha256
from .torus9_adaptation import (
    AdaptationModel, AdaptationTrainer, FINGERPRINT, activation,
    game_targets, save_torch, selfplay, validate_game,
)
from .torus9_pcr import position_telemetry, resolve_search_mode, search_description
from .torus9_new_komi_guard import assert_new_komi_training_checkpoint_metadata

SCHEMA = 'torus9-five-channel-ordinary-training-v1'


class OrdinaryTrainer(AdaptationTrainer):
    def __init__(self, checkpoint, *, learning_rate, seed, gradient_clip=None, batch_size=None, device='cpu'):
        raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
        meta = raw['metadata']
        assert_new_komi_training_checkpoint_metadata(meta)
        if meta.get('target_fingerprint') != FINGERPRINT or meta.get('komi') != 1.5:
            raise ValueError('Ordinary 5CH target/komi mismatch')
        self.model = AdaptationModel().to(device)
        self.model.load_state_dict(raw['model_state_dict'], strict=True)
        if model_hash(self.model) != meta['model_hash']:
            raise ValueError('Parent model hash mismatch')
        names = list(dict(self.model.named_parameters()))
        groups = raw['optimizer_state_dict']['param_groups']
        if [g.get('name') for g in groups] != names or any(len(g['params']) != 1 for g in groups):
            raise ValueError('Named Adam group order mismatch')
        self.optimizer = torch.optim.Adam([
            {'params': [p], 'name': n} for n, p in self.model.named_parameters()])
        self.optimizer.load_state_dict(raw['optimizer_state_dict'])
        self.learning_rate = float(learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Invalid learning rate')
        configured_clip = meta.get('gradient_clip', 1.0) if gradient_clip is None else gradient_clip
        if (type(configured_clip) not in (float, int) or
                not math.isfinite(configured_clip) or configured_clip <= 0):
            raise ValueError('Invalid gradient clip')
        self.gradient_clip = float(configured_clip)
        configured_batch = meta.get('batch_size', 64) if batch_size is None else batch_size
        if type(configured_batch) is not int or configured_batch <= 0:
            raise ValueError('batch_size must be a positive integer')
        self.batch_size = configured_batch
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate
            if group['weight_decay'] != 0:
                raise ValueError('Expected unregularized Adam')
            group['params'][0].requires_grad_(True)
        self.update = int(raw.get('ordinary_update', 0))
        if meta.get('ordinary_schema') == SCHEMA:
            self.clock_origin = raw['clock_origin']
            if int(raw['seed']) != int(seed):
                raise ValueError('Training seed changed across resume')
        else:
            if int(raw.get('update', -1)) != 2400:
                raise ValueError('Ordinary training requires completed adaptation')
            self.clock_origin = {
                n: (0 if n == 'input_projection.bias' else 15520) + 2400 - activation(n)
                for n in names}
        self.seed = int(seed)
        self.validate_clocks()

    def validate_clocks(self):
        # Adam clocks remain CPU scalars; inspect all structural constraints before
        # packing the read-only moment tensors. The detailed path preserves the
        # original first-error order and messages if anything is invalid.
        moments, variances = [], []
        device = next(self.model.parameters()).device
        for n, p in self.model.named_parameters():
            state = self.optimizer.state[p]
            with span('validate.step'):
                try:
                    clock = int(state['step'])
                except (KeyError, TypeError, ValueError, OverflowError, RuntimeError):
                    return self._validate_clocks_detailed()
                if clock != self.clock_origin[n] + self.update:
                    return self._validate_clocks_detailed()
            for key, collection in (('exp_avg', moments), ('exp_avg_sq', variances)):
                if key not in state or not isinstance(state[key], torch.Tensor):
                    return self._validate_clocks_detailed()
                value = state[key]
                if value.shape != p.shape or value.device != device or value.layout != torch.strided or value.is_complex():
                    return self._validate_clocks_detailed()
                collection.append(value)
        with torch.no_grad():
            with span('validate.pack'):
                average = torch.cat([value.reshape(-1) for value in moments])
                variance = torch.cat([value.reshape(-1) for value in variances])
            with span('validate.exp_avg'):
                average_finite = torch.isfinite(average).all()
            with span('validate.exp_avg_sq'):
                variance_finite = torch.isfinite(variance).all()
            with span('validate.negative'):
                variance_negative = (variance < 0).any()
            with span('validate.scalar_sync'):
                valid_average, valid_variance, negative = torch.stack(
                    (average_finite, variance_finite, variance_negative)).cpu().tolist()
        if not valid_average or not valid_variance or negative:
            self._validate_clocks_detailed()

    def _validate_clocks_detailed(self):
        for n, p in self.model.named_parameters():
            state = self.optimizer.state[p]
            if measured('validate.step', lambda: int(state['step']) != self.clock_origin[n] + self.update):
                raise ValueError('Adam clock mismatch: ' + n)
            for key in ('exp_avg', 'exp_avg_sq'):
                if measured('validate.' + key, lambda: state[key].shape != p.shape or not torch.isfinite(state[key]).all()):
                    raise ValueError('Invalid Adam state: ' + n)
            if measured('validate.negative', lambda: bool((state['exp_avg_sq'] < 0).any())):
                raise ValueError('Negative Adam variance')

    def batch(self, games, update):
        # Ordinary replay sampling is uniform over positions, without the
        # adaptation-only game/phase stratification.
        with span('batch.sampling'):
            import bisect
            if isinstance(games, OrdinaryReplayWindow):
                cumulative, count = games.cumulative, games.positions
            else:
                # Mutable caller-owned sequences retain the uncached sampler.
                cumulative, count = [], 0
                for g in games:
                    count += len(g['score'])
                    cumulative.append(count)
            rng = random.Random(self.seed + update)
            indices = []
            for _ in range(self.batch_size):
                index = rng.randrange(count)
                game = bisect.bisect_right(cumulative, index)
                indices.append((games[game], index - (cumulative[game - 1] if game else 0)))
        sampled(indices, games)
        device = next(self.model.parameters()).device
        return {k: transfer(k, measured('stack.' + k, lambda: torch.stack([g[k][i] for g, i in indices])), device)
                for k in ('observation', 'pi', 'z', 'ownership', 'score')}

    def evaluate(self, games, batches=32):
        # Keep the original fixed validation sampler for comparability.
        with torch.no_grad():
            self.model.eval()
            totals = {}
            for i in range(batches):
                losses, metrics = self.losses(AdaptationTrainer.batch(self, games, -10000-i))
                for k, value in {**losses, **metrics}.items():
                    totals[k] = totals.get(k, 0.) + float(value) / batches
            return totals

    def step(self, games):
        from .orchestrator_v2.execution_permit import require_engine_execution
        with span('execution_permit'):
            require_engine_execution(__name__, action='training', topology='torus9')
        with span('validate.pre'):
            self.validate_clocks()
        with span('model.train'):
            self.model.train()
        with span('zero_grad'):
            self.optimizer.zero_grad(set_to_none=True)
        with span('batch_and_losses'):
            losses, _ = self.losses(self.batch(games, self.update + 1))
        with span('loss.sum'):
            total = sum(losses.values())
        with span('loss.finite'):
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite ordinary loss')
        with span('backward'):
            total.backward()
        with span('grad_clip'):
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip, error_if_nonfinite=True)
        with span('adam'):
            self.optimizer.step()
        self.update += 1
        with span('validate.post'):
            self.validate_clocks()
        with span('weights.finite'):
            with torch.no_grad():
                weights = torch.cat([p.reshape(-1) for p in self.model.parameters()])
                if not torch.isfinite(weights).all():
                    raise FloatingPointError('Nonfinite ordinary weights')
        with span('scalar.telemetry'):
            scalars = torch.stack([v.detach() for v in losses.values()] + [grad.detach()]).cpu().tolist()
            return {'update': self.update, 'losses': dict(zip(losses, map(float, scalars[:-1]))),
                    'grad_norm_before_clip': float(scalars[-1]), 'learning_rate': self.learning_rate,
                    'gradient_clip': self.gradient_clip, 'batch_size': self.batch_size,
                    'l2_sp_coefficient': 0.}

    def save(self, path, *, config_hash, parent, replay_buckets):
        self.validate_clocks()
        meta = {'architecture_id': self.model.architecture_id,
                'architecture_config': self.model.architecture_config, 'observation_shape': [5,81],
                'model_hash': model_hash(self.model), 'komi': 1.5, 'training_ready': True,
                'target_fingerprint': FINGERPRINT, 'ordinary_schema': SCHEMA,
                'ordinary_update': self.update, 'config_hash': config_hash,
                'parent_checkpoint': parent, 'learning_rate': self.learning_rate,
                'gradient_clip': self.gradient_clip, 'batch_size': self.batch_size}
        save_torch(path, {'metadata': meta, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'ordinary_update': self.update,
            'clock_origin': self.clock_origin, 'seed': self.seed, 'replay_buckets': replay_buckets})
        atomic_write_json(path.with_suffix('.metadata.json'), {**meta, 'artifact_sha256': file_sha256(path)})


class TrainingHeartbeat:
    """The ordinary driver's durable heartbeat, shared with the diagnostic loop."""
    def __init__(self, path, generation, collector=None):
        self.path, self.generation = Path(path), generation
        self.progress = {'phase': 'prepare', 'done': 0, 'total': 1, 'progress_at': time.time()}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.collector = collector
        self.pulse_writes = 0

    def beat(self):
        with span('heartbeat'):
            with self.lock:
                payload = dict(self.progress)
            payload.update(liveness_at=time.time(), generation=self.generation,
                           progress_token=f"{payload['phase']}:{payload['done']}")
            atomic_write_json(self.path, payload)

    def mark(self, phase, done, total):
        with span('mark'):
            with self.lock:
                self.progress.update(phase=phase, done=done, total=total, progress_at=time.time())
            self.beat()

    def pulse(self):
        from contextlib import nullcontext
        with self.collector.activate() if self.collector is not None else nullcontext():
            while not self.stop.wait(10):
                self.beat()
                self.pulse_writes += 1

    def start(self):
        self.beat()
        self.thread = threading.Thread(target=self.pulse, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)


@dataclass(frozen=True)
class OrdinaryReplayWindow(Sequence):
    """Read-only generation window; cache lengths without copying replay tensors.

    Loaded games and their targets are immutable for the lifetime of a window.
    Changing a window requires constructing a new one. Mutable sequences passed
    directly to batch() continue to rebuild offsets on every call.
    """
    games: tuple
    cumulative: tuple
    positions: int

    @classmethod
    def from_games(cls, games):
        games = tuple(games)
        cumulative, count = [], 0
        for game in games:
            count += len(game['score'])
            cumulative.append(count)
        return cls(games, tuple(cumulative), count)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, index):
        return self.games[index]

    def __deepcopy__(self, memo):
        # Immutable window, shared target references; never duplicate replay data.
        return self


def load_replay(buckets, *, split):
    games, seen = [], set()
    for bucket in buckets:
        for ref in bucket['shards']:
            path = Path(ref['path'])
            if file_sha256(path) != ref['sha']:
                raise ValueError('Replay SHA mismatch')
            raw = torch.load(path, map_location='cpu', weights_only=False)
            if raw['contract'] != FINGERPRINT:
                raise ValueError('Replay target contract mismatch')
            for game in raw['games']:
                if game.get('split') != split:
                    continue
                validate_game(game)
                if game['actor_hash'] != raw['actor_hash']:
                    raise ValueError('Replay actor mismatch')
                if game['game_id'] in seen:
                    raise ValueError('Duplicate replay game')
                seen.add(game['game_id'])
                games.append(game)
    if not games:
        raise ValueError('Empty replay split: ' + split)
    return OrdinaryReplayWindow.from_games(games)


def validate_config(config):
    if config.compatibility.get('input_channels') != 5:
        raise ValueError('Legacy 6-channel Torus9 training is retired; use a 5CH checkpoint')
    gradient_clip = config.training.get('gradient_clip', 1.0)
    if (type(gradient_clip) not in (float, int) or
            not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError('gradient_clip must be a finite positive number')
    batch_size = config.training.get('batch_size')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    expected = {'optimizer': 'Adam', 'weight_decay': 0.0,
                'l2_sp': False}
    if any(config.training.get(k) != v for k,v in expected.items()):
        raise ValueError('Unsupported ordinary training contract')
    if config.self_play.get('komi') != 1.5 or config.replay.get('cap') is not None:
        raise ValueError('Ordinary training requires komi=1.5 and uncapped replay')
    if int(config.replay['generations']) <= 0:
        raise ValueError('Replay window must be positive')
    caps = resolve_search_mode(config.self_play)
    if type(config.self_play.get('tree_reuse', False)) is not bool:
        raise ValueError('self_play.tree_reuse must be a boolean')
    if caps is None:
        value = config.self_play.get('mcts_simulations')
        if type(value) is not int or value <= 0:
            raise ValueError('Generation budgets must be positive integers')
    for value in (config.training['optimizer_steps_per_iteration'],
                  config.self_play['games_per_iteration'],
                  config.execution['workers']):
        if type(value) is not int or value <= 0:
            raise ValueError('Generation budgets must be positive integers')
    if config.extensions.get('training_driver') != SCHEMA:
        raise ValueError('Missing ordinary 5CH driver selection')
    # The shared 5CH adapter owns these fixed search/execution semantics;
    # refuse unsupported overrides instead of silently ignoring a run setting.
    search = {'cpuct': 1.25, 'fpu': 0., 'dirichlet_alpha': .11,
              'dirichlet_epsilon': .25, 'watchdog': 500, 'root_noise': True,
              'resign': False, 'temperature_until_ply': 8, 'temperature_after': 0.}
    execution = {'active_games_per_worker': 4, 'inference_batch_cap': 64,
                 'inference_batch_wait_ms': 1.}
    for values, expected_values in ((config.self_play, search), (config.execution, execution)):
        if any(values.get(k, v) != v for k, v in expected_values.items()):
            raise ValueError('Unsupported 5CH search/execution override')
    if config.training.get('lr_scheduler') is not None:
        raise ValueError('Ordinary block requires a fixed LR')


def run_generation(resolved):
    from .orchestrator_v2.execution_permit import require_engine_execution
    from .orchestrator_v2.generation_runner import GenerationExecutionResult
    require_engine_execution(__name__, action='training', topology='torus9')
    cfg = resolved.effective_config.config
    validate_config(cfg)
    if resolved.execution_overrides:
        raise ValueError('5CH execution overrides are not supported')
    caps = resolve_search_mode(cfg.self_play)
    from .orchestrator_v2.offline_replay import iteration_input
    offline = iteration_input(cfg, resolved.generation)
    root, generation = resolved.output_lineage.root, resolved.generation
    parent = resolved.parent_checkpoint
    if file_sha256(parent.path) != parent.ref.sha256:
        raise ValueError('Parent checkpoint SHA mismatch')
    seed = int(cfg.execution['training_master_seed'])
    device = str(cfg.execution['device'])
    torch.set_num_threads(1)
    trainer = OrdinaryTrainer(parent.path, learning_rate=cfg.training['learning_rate'],
                               gradient_clip=cfg.training.get('gradient_clip', 1.0),
                               batch_size=cfg.training['batch_size'],
                               seed=seed, device=device)
    raw_parent = torch.load(parent.path, map_location='cpu', weights_only=False)
    if raw_parent['metadata'].get('ordinary_schema') == SCHEMA:
        buckets = raw_parent['replay_buckets']
    else:
        if parent.ref.to_dict() != cfg.extensions['adaptation_parent']:
            raise ValueError('Adaptation parent identity mismatch')
        buckets = cfg.to_dict()['extensions']['initial_replay_buckets']
        if {s['sha'] for b in buckets for s in b['shards']} != set(raw_parent['metadata']['replay_ids']):
            raise ValueError('Adaptation replay differs from parent checkpoint')
    validation = load_replay(cfg.extensions['validation_buckets'], split='validation')
    # Validate inherited data before allocating self-play workers.
    inherited = load_replay(buckets, split='train')
    if {g['game_id'] for g in inherited} & {g['game_id'] for g in validation}:
        raise ValueError('Validation leaked into replay')
    del inherited, raw_parent
    baseline = trainer.evaluate(validation)
    heartbeat = Path(os.environ.get('AZ_DRIVER_HEARTBEAT_PATH', root / 'runtime' / 'heartbeats' / f'generation-{generation:04d}.json'))
    heartbeat_writer = TrainingHeartbeat(heartbeat, generation)
    mark = heartbeat_writer.mark
    heartbeat_writer.start()
    try:
        fresh_shards = []
        shard_telemetry = []
        games_count = int(cfg.self_play['games_per_iteration'])
        if offline is None:
            for offset in range(0, games_count, 128):
                number = min(128, games_count-offset)
                path = root / 'replay' / f'g{generation:04d}-{offset:04d}.pt'
                identity_path = path.with_suffix('.identity.json')
                expected = {'parent': parent.ref.to_dict(), 'config': cfg.fingerprint,
                            'generation': generation, 'offset': offset, 'games': number}
                if identity_path.exists():
                    saved = json.loads(identity_path.read_text())
                    if saved['request'] != expected or file_sha256(path) != saved['shard']['sha']:
                        raise ValueError('Saved self-play shard identity mismatch')
                    fresh_shards.append(saved['shard'])
                    if caps is not None:
                        shard_telemetry.append(saved['pcr_telemetry'])
                    continue
                mark('selfplay', offset, games_count)
                ids = [f'{root.name}-g{generation:04d}-game-{i:04d}' for i in range(offset,offset+number)]
                result = selfplay(trainer.model, checkpoint=parent.path, run_id=root.name, ids=ids,
                    seed=int(cfg.execution['selfplay_master_seed']), device=device,
                    workers=int(cfg.execution['workers']), simulations=caps.full_simulations if caps else int(cfg.self_play['mcts_simulations']),
                    **({'search_mode': 'pcr', 'pcr': dict(cfg.self_play['pcr'])} if caps else {}),
                    **({'tree_reuse': True} if cfg.self_play.get('tree_reuse', False) else {}),
                    progress=lambda d,n: mark('selfplay',offset+d,games_count))
                raw_path = path.with_suffix('.games.jsonl.gz')
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = raw_path.with_suffix('.tmp')
                with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                    for game in result.records:
                        stream.write(json.dumps(game.to_dict())+'\n')
                os.replace(temporary,raw_path)
                games = []
                for i, record in enumerate(result.records):
                    mark('target-build', offset+i,games_count)
                    game = game_targets(record)
                    if game is None:
                        continue
                    game['split'] = 'train'
                    validate_game(game)
                    games.append(game)
                if len(result.records) != number:
                    raise ValueError('Incomplete self-play batch')
                save_torch(path, {'contract': FINGERPRINT, 'actor_hash': model_hash(trainer.model),
                    'games': games, 'raw_games_sha': file_sha256(raw_path), 'generation': generation,
                    'selfplay_simulations': None if caps else cfg.self_play['mcts_simulations'],
                    **({'search_mode': 'pcr', 'pcr': dict(cfg.self_play['pcr'])} if caps else {})})
                shard = {'path': str(path), 'sha': file_sha256(path), 'games': number}
                telemetry = dict(result.telemetry)
                pcr_telemetry = position_telemetry(result.records, caps) if caps else {}
                if caps:
                    if pcr_telemetry['training_positions'] != sum(len(g['score']) for g in games):
                        raise ValueError('PCR learner position count drift')
                    shard_telemetry.append(pcr_telemetry)
                telemetry.update(pcr_telemetry)
                atomic_write_json(path.with_suffix('.telemetry.json'), telemetry)
                atomic_write_json(identity_path, {'request': expected, 'shard': shard,
                    **({'pcr_telemetry': pcr_telemetry} if caps else {})})
                fresh_shards.append(shard)
                del result, games
        else:
            # Offline experiment inputs are immutable external references. No
            # self-play adapter is called and no source artifact is rewritten.
            fresh_shards = list(offline['fresh_bucket']['shards'])
            games_count = 0
        fresh_path = root / 'replay' / f'iter-{generation:02d}-fresh.json'
        fresh_bucket = {'generation': generation, 'shards': fresh_shards}
        atomic_write_json(fresh_path, fresh_bucket)
        buckets = (offline['buckets'] if offline is not None else
                   (list(buckets)+[fresh_bucket])[-int(cfg.replay['generations']):])
        games = load_replay(buckets,split='train')
        if {g['game_id'] for g in games} & {g['game_id'] for g in validation}:
            raise ValueError('Validation leaked into training')
        rolling_path = root / 'replay' / f'rolling-after-{generation:02d}.jsonl'
        atomic_write_text(rolling_path, ''.join(json.dumps(b,sort_keys=True)+'\n' for b in buckets))
        updates = int(cfg.training['optimizer_steps_per_iteration'])
        metrics = []
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        training_started = time.perf_counter()
        for i in range(updates):
            metrics.append(trainer.step(games))
            mark('training',i+1,updates)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        training_wall = time.perf_counter() - training_started
        performance = {'training_wall_time_sec': training_wall,
                       'training_minutes': training_wall / 60,
                       'optimizer_updates_per_sec': updates / training_wall,
                       'samples_per_sec': updates * trainer.batch_size / training_wall,
                       'processed_samples': updates * trainer.batch_size,
                       'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                       'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved() if device.startswith('cuda') else None}
        result_validation = trainer.evaluate(validation)
        if not all(math.isfinite(v) for v in result_validation.values()):
            raise FloatingPointError('Nonfinite validation')
        checkpoint_path = root / 'checkpoints' / f'M{generation}.pt'
        trainer.save(checkpoint_path, config_hash=cfg.fingerprint,
                     parent=parent.ref.to_dict(), replay_buckets=buckets)
        reloaded = OrdinaryTrainer(checkpoint_path, learning_rate=cfg.training['learning_rate'],
                                   gradient_clip=cfg.training.get('gradient_clip', 1.0),
                                   batch_size=cfg.training['batch_size'],
                                   seed=seed, device='cpu')
        if model_hash(reloaded.model) != model_hash(trainer.model):
            raise ValueError('Checkpoint reload changed model')
        for name,p in trainer.model.named_parameters():
            q = dict(reloaded.model.named_parameters())[name]
            for key in ('step','exp_avg','exp_avg_sq'):
                if not torch.equal(trainer.optimizer.state[p][key].cpu(),reloaded.optimizer.state[q][key].cpu()):
                    raise ValueError('Checkpoint reload changed Adam state')
        training_path = root / 'training' / f'iter-{generation:02d}.json'
        mean_losses = {key: sum(row['losses'][key] for row in metrics) / updates
                       for key in metrics[0]['losses']}
        performance['mean_losses'] = mean_losses
        performance['mean_total_loss'] = sum(mean_losses.values())
        atomic_write_json(training_path, {'updates': metrics,'baseline_validation': baseline,'validation': result_validation, 'performance': performance})
        summary_path = root / f'iter-{generation:02d}-summary.json'
        summary = {'status':'COMPLETED','generation':generation,'games':games_count,'updates':updates,
                   'learning_rate':cfg.training['learning_rate'],'replay_generations':len(buckets),
                   'replay_positions':sum(len(g['score']) for g in games),
                   'validation':result_validation,'checkpoint_reload_verified':True}
        summary.update(performance, offline_replay=offline is not None)
        if caps and offline is None:
            counts = {key: sum(t[key] for t in shard_telemetry) for key in (
                'pcr_full_positions', 'pcr_cheap_positions', 'training_positions', 'raw_positions')}
            summary.update(counts, search_mode='pcr', pcr=dict(cfg.self_play['pcr']),
                pcr_full_fraction=counts['pcr_full_positions'] / counts['raw_positions'] if counts['raw_positions'] else 0.,
                pcr_cheap_fraction=counts['pcr_cheap_positions'] / counts['raw_positions'] if counts['raw_positions'] else 0.,
                pcr_full_simulations=caps.full_simulations, pcr_cheap_simulations=caps.cheap_simulations,
                nominal_mean_simulations=caps.nominal_mean_simulations)
        atomic_write_json(summary_path,summary)
        paths = {'checkpoint':checkpoint_path, 'checkpoint_metadata':checkpoint_path.with_suffix('.metadata.json'),
                 'fresh_replay':fresh_path,'rolling_replay':rolling_path,'training_metrics':training_path,'summary':summary_path}
        identities = {k:{'path':p.relative_to(root).as_posix(),'sha256':file_sha256(p),'size_bytes':p.stat().st_size}
                      for k,p in paths.items()}
        # Publish every payload used by the fresh replay ledger in the catalog.
        for index, shard in enumerate(fresh_shards if offline is None else []):
            p = Path(shard['path'])
            identities[f'replay_shard_{index}'] = {'path':p.relative_to(root).as_posix(),'sha256':shard['sha'],'size_bytes':p.stat().st_size}
        marker = {'generation':generation,'lineage_id':root.name,
                  **{k+'_sha256':v['sha256'] for k,v in identities.items()},
                  'rolling_replay_size_bytes':rolling_path.stat().st_size}
        marker_path = root / f'generation-{generation:02d}.complete.json'
        marker_text = json.dumps(marker,indent=2,sort_keys=True)+'\n'
        marker_ref = ArtifactRef(marker_path.name,'sha256:'+hashlib.sha256(marker_text.encode()).hexdigest())
        checkpoint_ref = CheckpointRef('torus9',root.name,f'M{generation}',generation,
                                      identities['checkpoint']['path'],identities['checkpoint']['sha256'])
        fresh_ref = ArtifactRef(identities['fresh_replay']['path'],identities['fresh_replay']['sha256'])
        publish_checkpoint_graph(root=root,parent=parent.ref,checkpoint=checkpoint_ref,fresh_replay=fresh_ref,
            effective_config=resolved.effective_config.ref,generation_commit=marker_ref,
            artifact_identities=identities,checkpoint_reload_verified=True)
        atomic_write_text(marker_path,marker_text)
        mark('committed',updates,updates)
        return GenerationExecutionResult(generation,True,checkpoint_ref,marker_ref,fresh_ref)
    finally:
        heartbeat_writer.close()


def write_block_report(result):
    """Persist the bounded coordinator result without promoting a candidate."""
    if result.state != 'COMPLETED':
        return
    root = result.lineage_root
    rows = []
    for generation in range(result.original_parent.generation + 1, result.final_checkpoint.generation + 1):
        rows.append(json.loads((root / f'iter-{generation:02d}-summary.json').read_text()))
    arenas = [{'output_dir': str(a.output_dir), 'validity': a.validity,
               'summary': {k: a.summary[k] for k in ('games','wins','losses','draws','W/L/D',
                   'win_rate','technical_games','95_percent_hoeffding_interval') if k in a.summary}}
              for a in result.arenas]
    payload = {'state': result.state, 'parent': result.original_parent.ref.to_dict(),
               'checkpoint': result.final_checkpoint.ref.to_dict(), 'iterations': rows, 'arenas': arenas}
    atomic_write_json(root / 'reports' / 'block-report.json', payload)
    lines = ['# Обычное обучение Torus9 5CH', '',
             f'Статус: {result.state}. Итераций: {len(rows)}.',
             f"LR: {rows[-1]['learning_rate']}. Новых игр: {sum(r['games'] for r in rows)}. "
             f"Updates: {sum(r['updates'] for r in rows)}.", '',
             f'Checkpoint: {result.final_checkpoint.path}',
             f'SHA: {result.final_checkpoint.ref.sha256}', '',
             '## Арена', '', '```json', json.dumps(arenas,ensure_ascii=False,indent=2), '```', '',
             'Блок завершён. Следующий блок автоматически не запускается.']
    if rows[-1].get('search_mode') == 'pcr':
        lines[2:2] = [search_description(rows[-1]), '']
    atomic_write_text(root / 'reports' / 'block-report.md', '\n'.join(lines)+'\n')
