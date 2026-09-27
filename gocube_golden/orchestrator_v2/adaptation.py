"""Durable, human-gated M137-5CH adaptation workflow.

One process owns a run lock. A completed phase always enters NEEDS_REVIEW;
Telegram delivery and retries do not authorize another phase.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback

import numpy as np
import torch

from gocube_golden.torus9_adaptation import (
    AdaptationTrainer, AdaptationModel, BOOTSTRAP_SHA, CONTRACT, FINGERPRINT,
    STAGES, ENDS, game_targets, save_torch, selfplay, validate_game,
)
from gocube_golden.process_supervision import atomic_write_json, atomic_write_text
from gocube_golden.provenance import capture_code_identity, file_sha256, sha256_fingerprint
from gocube_golden.notifications import create_telegram_dispatcher, operator_event
from gocube_golden.notifications.telegram import load_config

REPO = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPO / 'runs/torus9/active/new_komi/checkpoints/M137-5CH-bootstrap.pt'
DEFAULTS = {'schema': 'torus9-adaptation-run-v1', 'komi': 1.5, 'seed': 2026092701,
    'bootstrap_sha': BOOTSTRAP_SHA, 'target': FINGERPRINT,
    'device': 'cuda', 'workers': 16, 'selfplay_simulations': 200,
    'arena_simulations': 256, 'batch_games': 128, 'bootstrap_games': 512,
    'bootstrap_min_train_positions': 50000, 'bootstrap_max_games': 2048,
    'fresh_games_per_160_updates': 384, 'batch_size': 64,
    'phase_ends': list(ENDS), 'arena_games': [0, 256, 256, 256, 1024],
    'replay_generations': 6, 'gradient_clip': 1.0,
    'lr_policy': 'staged-v1-hold-5e-6', 'human_review_every_stage': True,
    'l2_sp': '5-percent-shared-gradient-calibration-at-update-200',
}


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


@contextmanager
def exclusive(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def init_run(root, *, smoke=False):
    with exclusive(root):
        if (root / 'state.json').exists():
            raise FileExistsError('Run already exists')
        config = dict(DEFAULTS)
        if smoke:
            config.update(workers=2, selfplay_simulations=8, arena_simulations=8,
                batch_games=4, bootstrap_games=4, bootstrap_min_train_positions=1,
                bootstrap_max_games=8, arena_games=[0, 4, 4, 4, 4], smoke=True)
        config['code'] = asdict(capture_code_identity())
        config['run_id'] = root.name
        config['created_at'] = utc()
        config['fingerprint'] = sha256_fingerprint(config)
        atomic_write_json(root / 'config.json', config)
        trainer = AdaptationTrainer(BOOTSTRAP, device='cpu', seed=config['seed'])
        cp = root / 'checkpoints/update-0000.pt'
        trainer.save(cp, replay_ids=[], config_hash=config['fingerprint'])
        atomic_write_json(root / 'state.json', {
            'schema': 'torus9-adaptation-state-v1', 'status': 'READY', 'stage': 0,
            'update': 0, 'checkpoint': str(cp), 'checkpoint_sha': file_sha256(cp),
            'champion': str(cp), 'champion_sha': file_sha256(cp),
            'checkpoint_replay': [], 'shards': [], 'created_at': utc(),
            'config_hash': config['fingerprint'], 'arena_attempt': 0})
        return config


def arena_statistics(path, *, seed=93271):
    pairs = defaultdict(list)
    counts = {'A_WIN': 0, 'B_WIN': 0, 'DRAW': 0}
    for line in (path / 'games.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('technical_termination') or row.get('mapped_result') not in counts:
            raise ValueError('Technical or invalid arena game')
        result = row['mapped_result']
        counts[result] += 1
        pairs[row['pair_id']].append((bool(row['candidate_black']),
                                     {'A_WIN': 1., 'B_WIN': 0., 'DRAW': .5}[result]))
    if not pairs or any(len(v) != 2 or {x[0] for x in v} != {False, True} for v in pairs.values()):
        raise ValueError('Arena must contain complete color-swapped pairs')
    values = np.array([sum(x[1] for x in v) / 2 for v in pairs.values()])
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    lo, hi = np.quantile(means, [.025, .975]).tolist()
    return {'wins': counts['A_WIN'], 'losses': counts['B_WIN'], 'draws': counts['DRAW'],
            'games': sum(counts.values()), 'pairs': len(values), 'score': float(values.mean()),
            'ci95': [lo, hi], 'gate': 'PASS' if lo >= .47 else 'FAIL' if hi < .47 else 'INCONCLUSIVE',
            'ci_method': 'paired-bootstrap-10000', 'improved': lo > .5}


class Workflow:
    def __init__(self, root):
        self.root = root.resolve()
        self.config = read(root / 'config.json')
        check = dict(self.config)
        fp = check.pop('fingerprint')
        if sha256_fingerprint(check) != fp:
            raise ValueError('Run config was modified')
        if capture_code_identity().git_commit_sha != self.config['code']['git_commit_sha']:
            raise ValueError('Run is pinned to another code commit; use its checkout')
        self.state = read(root / 'state.json')
        if self.state['config_hash'] != fp:
            raise ValueError('State/config mismatch')
        self.notifications = create_telegram_dispatcher(root)
        self.stop = threading.Event()
        self.progress = {}
        self.last_progress = time.monotonic()
        self.started = time.monotonic()
        self.progress_started = self.started

    def persist(self):
        self.state['updated_at'] = utc()
        atomic_write_json(self.root / 'state.json', self.state)

    def heartbeat(self):
        while not self.stop.is_set():
            atomic_write_json(self.root / 'heartbeat.json', {
                'pid': os.getpid(), 'time': utc(), 'status': self.state['status'],
                'stage': STAGES[self.state['stage']], 'update': self.state['update'],
                'progress': dict(self.progress),
                'seconds_since_progress': round(time.monotonic() - self.last_progress),
                'elapsed_process_sec': round(time.monotonic() - self.started)})
            self.stop.wait(15)

    def mark_progress(self, phase, done, total):
        if self.progress.get('phase') != phase or done < self.progress.get('completed', 0):
            self.progress_started = time.monotonic()
        elapsed = time.monotonic() - self.progress_started
        self.progress = {'phase': phase, 'completed': done, 'total': total,
                         'eta_seconds_estimate': round(elapsed * (total - done) / done) if done else None}
        self.last_progress = time.monotonic()

    def notify(self, kind, action, payload, report=None):
        event = operator_event(kind, topology='torus9', owner_type='workflow',
            owner_id=self.root.name, action_id=action, payload=payload,
            execution_code_commit=self.config['code']['git_commit_sha'],
            evidence_refs=[] if report is None else [{'ref': str(report)}])
        self.notifications.publish(event)
        self.notifications.flush(timeout=7)
        return event.event_id

    def verify_checkpoint(self, key='checkpoint'):
        path = Path(self.state[key])
        if file_sha256(path) != self.state[key + '_sha']:
            raise ValueError('Checkpoint integrity failure')
        return path

    def trainer(self):
        t = AdaptationTrainer(BOOTSTRAP, device=self.config['device'], seed=self.config['seed'])
        t.restore(self.verify_checkpoint(), config_hash=self.config['fingerprint'],
                  replay_ids=self.state['checkpoint_replay'])
        if t.update != self.state['update']:
            raise ValueError('Trainer update/state mismatch')
        return t

    def load_games(self):
        train, validation, ids = [], [], []
        max_generation = max((s['generation'] for s in self.state['shards']), default=0)
        for s in self.state['shards']:
            path = Path(s['path'])
            if file_sha256(path) != s['sha']:
                raise ValueError('Replay artifact integrity failure')
            raw = torch.load(path, map_location='cpu', weights_only=False)
            if raw['contract'] != FINGERPRINT or raw['actor_hash'] != s['actor_hash']:
                raise ValueError('Replay provenance failure')
            active = s['generation'] >= max(0, max_generation - 5)
            if active:
                ids.append(s['sha'])
            for g in raw['games']:
                validate_game(g)
                if s['generation'] == 0 and g['split'] == 'validation':
                    validation.append(g)
                elif active and g['split'] == 'train':
                    train.append(g)
        if {g['game_id'] for g in train} & {g['game_id'] for g in validation}:
            raise ValueError('Validation leaked into training')
        return train, validation, ids

    def collect_batch(self, generation, number):
        cfg = self.config
        if capture_code_identity().git_commit_sha != cfg['code']['git_commit_sha']:
            raise ValueError('Code changed before spawning self-play workers')
        offset = sum(s['games'] for s in self.state['shards'])
        batch_path = self.root / 'replay' / f'g{generation:02d}-{offset:06d}.pt'
        checkpoint = self.verify_checkpoint('champion')
        model = AdaptationModel().to(cfg['device'])
        raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(raw['model_state_dict'])
        model.eval()
        ids = [f'{self.root.name}-game-{i:06d}' for i in range(offset, offset + number)]
        self.mark_progress('selfplay', 0, number)
        result = selfplay(model, checkpoint=checkpoint, run_id=self.root.name, ids=ids,
            seed=cfg['seed'], device=cfg['device'], workers=cfg['workers'],
            simulations=cfg['selfplay_simulations'],
            progress=lambda d, n: self.mark_progress('selfplay', d, n))
        raw_path = batch_path.with_suffix('.games.jsonl.gz')
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temp = raw_path.with_suffix('.tmp')
        with gzip.open(temp, 'wt', encoding='utf-8') as f:
            for game in result.records:
                f.write(json.dumps(game.to_dict(), separators=(',', ':')) + '\n')
        os.replace(temp, raw_path)
        games = []
        for i, game in enumerate(result.records):
            self.mark_progress('target-build', i, number)
            g = game_targets(game)
            g['split'] = ('validation' if generation == 0 and
                           int(sha256_fingerprint(g['game_id'])[-8:], 16) % 10 == 0 else 'train')
            validate_game(g)
            games.append(g)
        actor_hash = raw['metadata']['model_hash']
        save_torch(batch_path, {'contract': FINGERPRINT, 'actor_hash': actor_hash, 'games': games,
            'raw_games_sha': file_sha256(raw_path), 'generation': generation,
            'selfplay_simulations': cfg['selfplay_simulations']})
        atomic_write_json(batch_path.with_suffix('.telemetry.json'), result.telemetry)
        shard = {'path': str(batch_path), 'sha': file_sha256(batch_path), 'games': len(games),
            'generation': generation, 'actor_hash': actor_hash,
            'train_positions': sum(len(g['score']) for g in games if g['split'] == 'train'),
            'validation_positions': sum(len(g['score']) for g in games if g['split'] == 'validation')}
        self.state['shards'].append(shard)
        self.persist()
        del model, raw, result, games
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def collect_initial(self):
        cfg = self.config
        while True:
            shards = self.state['shards']
            games = sum(s['games'] for s in shards)
            positions = sum(s['train_positions'] for s in shards)
            valid = sum(s['validation_positions'] for s in shards)
            if games >= cfg['bootstrap_games'] and positions >= cfg['bootstrap_min_train_positions'] and valid > 0:
                break
            if games >= cfg['bootstrap_max_games']:
                raise RuntimeError('Replay budget exhausted before quality/size gate')
            self.collect_batch(0, cfg['batch_games'])
        train, valid, _ = self.load_games()
        trainer = self.trainer()
        baseline = trainer.evaluate(valid)
        self.state['baseline_validation'] = baseline
        self.state['best_validation'] = baseline
        self.state['validation_history'] = []
        self.persist()
        quality = {'training_games': len(train), 'validation_games': len(valid),
            'positions_per_game_min': min(len(g['score']) for g in train + valid),
            'positions_per_game_max': max(len(g['score']) for g in train + valid),
            'training_wdl_counts': torch.cat([g['z'] for g in train]).sum(0).tolist(),
            'validation_wdl_counts': torch.cat([g['z'] for g in valid]).sum(0).tolist(),
            'technical_games': 0, 'contract': FINGERPRINT}
        self.finish({'gate': 'PASS', 'validation': baseline, 'data_quality': quality,
            'train_positions': sum(len(g['score']) for g in train),
            'validation_positions': sum(len(g['score']) for g in valid),
            'games': games, 'arena': None}, 'Изучить качество replay и разрешить прогрев WDL/score.')

    def save_trainer(self, trainer, replay_ids):
        path = self.root / 'checkpoints' / f'update-{trainer.update:04d}.pt'
        trainer.save(path, replay_ids=replay_ids, config_hash=self.config['fingerprint'])
        self.state.update(checkpoint=str(path), checkpoint_sha=file_sha256(path),
                          checkpoint_replay=replay_ids, update=trainer.update)
        self.persist()

    def train_phase(self):
        cfg = self.config
        end = ENDS[self.state['stage']]
        trainer = self.trainer()
        while trainer.update < end:
            generation = trainer.update // 160 + 1
            if generation > 1:
                target = cfg['fresh_games_per_160_updates']
                have = sum(s['games'] for s in self.state['shards'] if s['generation'] == generation)
                while have < target:
                    n = min(cfg['batch_games'], target - have)
                    self.collect_batch(generation, n)
                    have += n
            train, valid, replay_ids = self.load_games()
            block_end = min(end, (trainer.update // 160 + 1) * 160)
            if (block_end - trainer.update) * 64 / sum(len(g['score']) for g in train) > (0.5 if trainer.update < 480 else 1.):
                raise RuntimeError('Sample reuse budget exceeded; collect additional reviewed data')
            history = []
            for u in range(trainer.update + 1, block_end + 1):
                self.mark_progress('training', u - 1, end)
                row = trainer.step(train)
                atomic_write_json(self.root / 'training' / f'update-{u:04d}.json', row)
                # Compare like parameters against an established active segment.
                if len(history) >= 20:
                    for name, group in row['groups'].items():
                        base = float(np.median([r['groups'][name]['delta'] for r in history[-20:]]))
                        if base > 1e-8 and group['delta'] > max(5 * base, 1e-5):
                            raise RuntimeError('Update spike: ' + name)
                history.append(row)
                if u % 40 == 0:
                    self.save_trainer(trainer, replay_ids)
            metrics = trainer.evaluate(valid)
            self.state['validation_history'].append({'update': trainer.update, 'metrics': metrics})
            best = self.state['best_validation']
            bad = sum(metrics[k] > best[k] * 1.10 for k in ('wdl', 'score')) > 0
            self.state['consecutive_bad_validation'] = self.state.get('consecutive_bad_validation', 0) + 1 if bad else 0
            self.save_trainer(trainer, replay_ids)
            if self.state['consecutive_bad_validation'] >= 2:
                raise RuntimeError('Two validation regressions >10%; review required')
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.evaluate_phase()

    def evaluate_phase(self):
        from tools.arena_engine import ArenaExecutionConfig, run_arena
        from tools.arena_profiles import get_profile
        cfg, stage = self.config, self.state['stage']
        attempt = self.state['arena_attempt']
        count = cfg['arena_games'][stage] * (2 ** attempt)
        out = self.root / 'arena' / f'stage-{stage + 1:02d}-attempt-{attempt}'
        # Complete arena outputs can be reused only when bound to the same candidate.
        identity = {'candidate': self.state['checkpoint_sha'], 'reference': BOOTSTRAP_SHA,
                    'games': count, 'config': cfg['fingerprint']}
        marker = out / 'adaptation-complete.json'
        if marker.exists():
            if read(marker) != identity:
                raise ValueError('Arena identity drift')
        else:
            if out.exists():
                # Never overwrite evidence from an interrupted arena.
                out.rename(out.with_name(out.name + '-interrupted-' + str(time.time_ns())))
            self.mark_progress('arena', 0, count)
            run_arena(profile=get_profile(f"torus9|komi=1.5|simulations={cfg['arena_simulations']}|watchdog=1000|5ch"),
                candidate_path=self.verify_checkpoint(), reference_path=BOOTSTRAP,
                output_dir=out, master_seed=cfg['seed'] + 100000 + stage * 10000 + attempt,
                config=ArenaExecutionConfig(games=count, workers=cfg['workers'],
                    games_per_worker=12 if not cfg.get('smoke') else 2,
                    inference_batch_rows=192 if not cfg.get('smoke') else 4,
                    device=cfg['device'], strict_production=not cfg.get('smoke'),
                    early_gate_enabled=not cfg.get('smoke')),
                progress_callback=lambda d, n: self.mark_progress('arena', d, n))
            atomic_write_json(marker, identity)
        stats = arena_statistics(out)
        self.finish({'gate': stats['gate'], 'arena': stats,
            'validation': self.state['validation_history'][-1]['metrics'],
            'baseline_validation': self.state['baseline_validation'],
            'arena_report': str(out / 'summary.json')},
            'Изучить отчёт; выбрать продолжение, расширение арены или остановку. '
            'При FAIL продолжение запрещено.')

    def finish(self, result, recommendation):
        stage = self.state['stage']
        report_dir = self.root / 'reports' / f'stage-{stage + 1:02d}-attempt-{self.state["arena_attempt"]}'
        start = self.state.get('stage_start', {})
        before_games = sum(s['games'] for s in start.get('shards', []))
        all_games = sum(s['games'] for s in self.state['shards'])
        last_metrics = self.root / 'training' / f'update-{self.state["update"]:04d}.json'
        training = read(last_metrics) if last_metrics.exists() else None
        active = {} if training is None else {n: g['lr'] for n, g in training['groups'].items() if g['lr'] > 0}
        report = {**result, 'run_id': self.root.name, 'stage': STAGES[stage],
            'stage_index': stage, 'update': self.state['update'], 'status': 'NEEDS_REVIEW',
            'checkpoint': self.state['checkpoint'], 'checkpoint_sha': self.state['checkpoint_sha'],
            'config_hash': self.config['fingerprint'], 'completed_at': utc(),
            'selfplay_games_total': all_games, 'selfplay_games_stage': all_games - before_games,
            'updates_stage': self.state['update'] - start.get('update', 0),
            'stage_wall_sec': time.time() - self.state.get('stage_started_at', time.time()),
            'active_parameter_lrs': active,
            'replay': self.state['shards'], 'recommendation': recommendation,
            'next_stage': STAGES[stage + 1] if stage < 4 else 'ordinary-training-separate-decision'}
        atomic_write_json(report_dir / 'report.json', report)
        lines = ['# Отчёт адаптации M137-5CH', '',
            f'Этап {stage + 1}/5: {STAGES[stage]}. Статус: NEEDS_REVIEW. Проверки: {report["gate"]}.', '',
            f'На этапе: {report["selfplay_games_stage"]} новых игр, {report["updates_stage"]} updates; '
            f'всего: {all_games} игр, {report["update"]} updates.',
            f'Время этапа: {report["stage_wall_sec"] / 3600:.2f} ч.', '',
            '## Оценка', '', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '',
            '## Следующее действие', '', recommendation,
            f'Следующий этап: {report["next_stage"]}. Автоматический переход запрещён.', '',
            '## Воспроизводимость', '', f'Checkpoint: {report["checkpoint"]}',
            f'SHA: {report["checkpoint_sha"]}', f'Config: {report["config_hash"]}',
            f'Полные метаданные: {report_dir / "report.json"}', '',
            '## Размороженные параметры и LR', '', '```json', json.dumps(active, indent=2), '```']
        atomic_write_text(report_dir / 'report.md', '\n'.join(lines) + '\n')
        self.state.update(status='NEEDS_REVIEW', report=str(report_dir / 'report.json'),
                          report_sha=file_sha256(report_dir / 'report.json'))
        self.persist()
        self.publish_review()

    def publish_review(self):
        r = read(self.state['report'])
        md = str(Path(self.state['report']).with_suffix('.md'))
        self.notify('EXPERIMENT_STAGE_DECIDED', f'stage-{self.state["stage"]}-arena-{self.state["arena_attempt"]}', {
            'Этап': f'{self.state["stage"] + 1}/5 — {r["stage"]}',
            'Статус': 'NEEDS_REVIEW — обучение приостановлено',
            'Сделано': f'{r["selfplay_games_stage"]} новых игр, {r["updates_stage"]} updates на этапе; '
                f'всего {r["selfplay_games_total"]} игр / {r["update"]} updates',
            'Время этапа': f'{r["stage_wall_sec"] / 3600:.2f} ч',
            'LR': ', '.join(map(str, sorted(set(r['active_parameter_lrs'].values())))) or 'веса заморожены',
            'Проверки': r['gate'], 'Арена': json.dumps(r.get('arena'), ensure_ascii=False),
            'Validation': json.dumps(r.get('validation'), ensure_ascii=False),
            'Дальше': r['recommendation'], 'Следующий этап': r['next_stage'], 'Отчёт': md,
            'Запрос для Codex': f'Изучи отчёт {md} для run {self.root.name}. '
                'Проверь метрики, предложи продолжение, расширение оценки или повтор с меньшим LR. '
                'До моего решения обучение не возобновляй.'}, report=md)

    def consume_decision(self):
        path = self.root / 'decision.json'
        if not path.exists():
            return False
        decision = read(path)
        if decision == self.state.get('last_decision'):
            path.rename(self.root / f'decision-{time.time_ns()}.json')
            return False
        if decision['report_sha'] != self.state['report_sha']:
            raise ValueError('Stale review decision')
        report = read(self.state['report'])
        action = decision['action']
        if action == 'continue':
            if report['gate'] != 'PASS' or self.state['stage'] >= 4:
                raise ValueError('Continuation requires PASS and a next adaptation stage')
            self.state['champion'] = self.state['checkpoint']
            self.state['champion_sha'] = self.state['checkpoint_sha']
            if 'validation' in report:
                self.state['best_validation'] = report['validation']
            self.state['stage'] += 1
            self.state['arena_attempt'] = 0
            self.state['status'] = 'READY'
            self.state.pop('stage_start', None)
        elif action == 'expand':
            if self.state['stage'] == 0 or self.state['arena_attempt'] >= 2:
                raise ValueError('Arena expansion unavailable; manual investigation required')
            self.state['arena_attempt'] += 1
            self.state['status'] = 'EVALUATE'
        elif action == 'stop':
            self.state['status'] = 'STOPPED'
        else:
            raise ValueError('Unknown decision')
        # State is authoritative; archived decisions cannot be consumed twice.
        self.state['last_decision'] = decision
        self.persist()
        path.rename(self.root / f'decision-{time.time_ns()}.json')
        return True

    def run(self):
        if not self.config.get('smoke') and load_config() is None:
            raise RuntimeError('Telegram configuration required before production launch')
        if not torch.cuda.is_available() and self.config['device'] == 'cuda':
            raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(1)
        thread = threading.Thread(target=self.heartbeat, daemon=True)
        thread.start()
        try:
            self.notify('TRAINING_STARTED', 'workflow-start', {
                'Запуск': self.root.name, 'Режим': 'M137-5CH, komi=1.5, этапы с ручным подтверждением',
                'Первый этап': 'Свежий replay: минимум 512 игр и 50000 training-позиций',
                'Остановка': 'После каждого этапа — отчёт в Telegram и NEEDS_REVIEW',
                'Статус': str(self.root / 'heartbeat.json')})
            while self.state['status'] not in ('STOPPED', 'FAILED'):
                if self.state['status'] == 'NEEDS_REVIEW':
                    # Also reconciles a crash between durable report commit and enqueue.
                    self.publish_review()
                    if not self.consume_decision():
                        self.notifications.flush(timeout=2)
                        self.stop.wait(30)
                    continue
                if self.state['status'] == 'EVALUATE':
                    self.evaluate_phase()
                    continue
                # Bounded rollback starts at the phase boundary, including replay.
                if 'stage_start' not in self.state:
                    self.state['stage_start'] = copy.deepcopy(self.state)
                    self.state['stage_started_at'] = time.time()
                pending = self.root / 'decision.json'
                if pending.exists() and read(pending) == self.state.get('last_decision'):
                    pending.rename(self.root / f'decision-{time.time_ns()}.json')
                self.state['status'] = 'RUNNING'
                self.persist()
                if self.state['stage'] == 0:
                    self.collect_initial()
                else:
                    self.train_phase()
        except BaseException as exc:
            self.state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            self.persist()
            atomic_write_text(self.root / 'failure.txt', traceback.format_exc())
            self.notify('RUN_FAILED', f'stage-{self.state["stage"]}-update-{self.state["update"]}', {
                'Статус': 'Остановлено; автоматического продолжения нет',
                'Ошибка': self.state['error'], 'Checkpoint': self.state['checkpoint'],
                'Действие': 'Вызвать Codex для разбора; не менять LR и не возобновлять вслепую',
                'Отчёт': str(self.root / 'failure.txt')})
            # Keep only delivery alive after a scientific failure. Training is not retried.
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                while not self.stop.wait(30):
                    self.notifications.flush(timeout=2)
            raise
        finally:
            self.stop.set()
            thread.join(timeout=2)
            self.notifications.close(timeout=7)


def write_decision(root, action, expected):
    state = read(root / 'state.json')
    if state['status'] != 'NEEDS_REVIEW' or state['report_sha'] != expected:
        raise ValueError('Review state or report hash changed')
    report = read(state['report'])
    if file_sha256(state['report']) != expected:
        raise ValueError('Report integrity failure')
    if action == 'continue' and (report['gate'] != 'PASS' or state['stage'] >= 4):
        raise ValueError('Next phase requires PASS; final pilot needs separate training decision')
    if action == 'expand' and (state['stage'] == 0 or state['arena_attempt'] >= 2):
        raise ValueError('Expansion unavailable')
    with (root / 'decision.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'decision.json').exists():
            raise ValueError('Decision already pending')
        atomic_write_json(root / 'decision.json', {'action': action, 'report_sha': expected, 'at': utc()})


def retry_phase(root, destination, scale):
    """Explicit full rollback into a new run; retain the failed run as evidence."""
    state = read(root / 'state.json')
    if state['status'] not in ('NEEDS_REVIEW', 'FAILED') or not 0 < scale < 1:
        raise ValueError('Retry requires a paused/failed run and a smaller LR multiplier')
    old_config = read(root / 'config.json')
    start = copy.deepcopy(state['stage_start'])
    with exclusive(destination):
        if (destination / 'state.json').exists():
            raise FileExistsError('Retry destination already exists')
        cfg = {**old_config, 'run_id': destination.name, 'created_at': utc(),
               'code': asdict(capture_code_identity()),
               'retry': {'parent': str(root), 'stage': state['stage'],
                         'checkpoint_sha': start['checkpoint_sha'], 'lr_multiplier': scale}}
        cfg.pop('fingerprint')
        cfg['fingerprint'] = sha256_fingerprint(cfg)
        cp = Path(start['checkpoint'])
        if file_sha256(cp) != start['checkpoint_sha']:
            raise ValueError('Rollback checkpoint hash changed')
        trainer = AdaptationTrainer(BOOTSTRAP)
        trainer.restore(cp, config_hash=old_config['fingerprint'], replay_ids=start['checkpoint_replay'])
        trainer.lr_scale *= scale
        target = destination / 'checkpoints' / f'update-{trainer.update:04d}.pt'
        trainer.save(target, config_hash=cfg['fingerprint'], replay_ids=start['checkpoint_replay'])
        # Champion can refer to the immutable accepted checkpoint in the parent.
        start.update(status='READY', checkpoint=str(target), checkpoint_sha=file_sha256(target),
                     config_hash=cfg['fingerprint'], arena_attempt=0)
        for key in ('report', 'report_sha', 'last_decision', 'error'):
            start.pop(key, None)
        atomic_write_json(destination / 'config.json', cfg)
        atomic_write_json(destination / 'state.json', start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'run', 'status', 'decide', 'retry'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--action', choices=['continue', 'expand', 'stop'])
    p.add_argument('--expected-report-sha')
    p.add_argument('--retry-root', type=Path)
    p.add_argument('--lr-scale', type=float, default=.5)
    a = p.parse_args()
    root = a.root.resolve()
    if a.command == 'init':
        print(json.dumps(init_run(root, smoke=a.smoke), indent=2))
    elif a.command == 'status':
        print(json.dumps({'state': read(root / 'state.json'),
            'heartbeat': read(root / 'heartbeat.json') if (root / 'heartbeat.json').exists() else None}, indent=2))
    elif a.command == 'decide':
        write_decision(root, a.action, a.expected_report_sha)
    elif a.command == 'retry':
        if a.retry_root is None:
            p.error('--retry-root is required')
        retry_phase(root, a.retry_root.resolve(), a.lr_scale)
    else:
        with exclusive(root):
            workflow = Workflow(root)
            def terminate(*_):
                raise KeyboardInterrupt('Supervisor stop requested')
            signal.signal(signal.SIGTERM, terminate)
            workflow.run()


if __name__ == '__main__':
    main()
