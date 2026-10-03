"""Manual, bounded M233 comparison; writes counters/traces only to temporary storage.

Run the same harness in separate Python processes with --repo selecting before
and after worktrees. It uses the existing test-only authority and production
cooperative adapter/engine, never an orchestrator job or training/replay writer.
Use nsys externally for CUDA API counts and kernel+copy active fractions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

# Spawned workers inherit the selected source through sys.path from __main__.
parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seconds', type=float, default=120)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
sys.dont_write_bytecode = True

import torch
from gocube_golden.inference import BatchedPolicyWDLInferenceOwner
from gocube_golden.neural import model_hash
from gocube_golden.orchestrator_v2.execution_permit import _test_authority
from gocube_golden.rules import apply_action, prepare_legal_actions
from gocube_golden.selfplay_engine import SelfPlayEngineConfig, run_cooperative_selfplay
from gocube_golden.state import initial_state
from gocube_golden.topology import TORUS_9X9
from gocube_golden.torus9_adaptation import AdaptationModel, AdaptationSelfPlayAdapter
from gocube_golden.torus9_m137_5ch import build_m137_five_channel_observation
from gocube_golden.torus9_selfplay import _make_torus9_cooperative_game


def save(name, value):
    (args.output / name).write_text(json.dumps(value, indent=2) + '\n')


def stats(values):
    values = sorted(values)
    return {'count': len(values), 'mean': statistics.mean(values),
            'p50': statistics.median(values), 'p95': values[min(len(values) - 1, int(.95 * len(values)))]}


def fixtures():
    state = initial_state(topology=TORUS_9X9, komi=1.5)
    observations = []
    for ply in range(65):
        legal = prepare_legal_actions(state)
        if ply in (0, 8, 24, 40, 64):
            observations.append(build_m137_five_channel_observation(state, legal_context=legal))
        actions = [i for i, yes in enumerate(legal.action_mask[:81]) if yes]
        if not actions:
            break
        state = apply_action(state, actions[(ply * 17 + 3) % len(actions)]).after
    return torch.stack(observations)


_games = {}
_observing = False


def observed_factory(context, game_id, client):
    global _observing
    game = _make_torus9_cooperative_game(context, game_id, client)
    _games[game_id] = game
    if not _observing:
        _observing = True
        threading.Thread(target=observe_moves, daemon=True, name='benchmark-move-observer').start()
    return game


def observe_moves():
    with (args.output / f'worker-{os.getpid()}.jsonl').open('w') as log:
        while True:
            games = tuple(_games.values())
            log.write(json.dumps({'time': time.perf_counter(),
                'moves': sum(len(game.trace) for game in games),
                'technical': sum(game.technical is not None for game in games)}) + '\n')
            log.flush()
            time.sleep(.5)


class TimedAdapter:
    def __init__(self, base):
        self.worker_context = base.worker_context
        self.shared_memory = base.shared_memory
        self.worker_game_factory = observed_factory
        self.record_metrics = base.record_metrics
        self.callback = base.infer_shared_batch
        self.timings = []
        self.broker = None

    def infer_shared_batch(self, observations):
        # Read caller timing boundaries; the callback result is returned intact.
        frame = sys._getframe(1)
        self.broker = frame.f_locals['self']
        start = time.perf_counter()
        result = self.callback(observations)
        end = time.perf_counter()
        self.timings.append({'dispatch': frame.f_locals['dispatch_started'],
            'start': start, 'end': end, 'rows': len(observations),
            'envelopes': len(frame.f_locals['batch']),
            'forward_end': result.forward_finished_at})
        return result


def observe_broker(adapter, stop, samples):
    while not stop.wait(.5):
        if adapter.broker:
            now = time.perf_counter()
            samples.append({'time': now, 'telemetry': adapter.broker.telemetry(now)})


def micro(model):
    owner = BatchedPolicyWDLInferenceOwner(model, device='cuda',
        expected_observation_shape=(5, 81), expected_policy_size=82, forward_policy_wdl_logits=model)
    observations = fixtures()
    results = []
    for rows in (1, 22, 64):
        batch = observations[torch.arange(rows) % len(observations)].pin_memory()
        policy, wdl = torch.empty(rows, 82), torch.empty(rows, 3)
        with torch.inference_mode():
            p, w = model(batch.cuda())
            reference_policy, reference_wdl = p.softmax(1).cpu(), w.softmax(1).cpu()
        def call():
            result = owner.evaluate_shared_batch(batch)
            for i in range(rows):
                policy[i].copy_(result.policy[i])
                wdl[i].copy_(result.wdl[i])
        for _ in range(25):
            call()
        torch.testing.assert_close(policy, reference_policy, rtol=0, atol=0)
        torch.testing.assert_close(wdl, reference_wdl, rtol=0, atol=0)
        walls = []
        for _ in range(150):
            start = time.perf_counter()
            call()  # All outputs are on the CPU when dispatch completes.
            walls.append((time.perf_counter() - start) * 1000)
        results.append({'batch': rows, 'owner_and_output_ms': stats(walls),
                        'rows_s': rows / (statistics.mean(walls) / 1000), 'parity': 'bitwise PASS'})
    save('micro.json', results)


def main():
    output = args.output.resolve()
    if not output.is_relative_to(Path('/tmp')):
        raise ValueError('Benchmark output must be under /tmp, outside canonical storage')
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint_sha == '15aedb102c386837227daa660d412eab7e937f96e022dc93c1d4503675d81cfd'
    model = AdaptationModel()
    model.load_state_dict(raw['model_state_dict'], strict=True)
    assert model_hash(model) == raw['metadata']['model_hash']
    model.cuda().eval()
    save('identity.json', {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha,
        'model_hash': model_hash(model), 'metadata': raw['metadata'], 'repo': str(args.repo),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip(),
        'inference_source_sha256': hashlib.sha256((args.repo / 'gocube_golden/inference.py').read_bytes()).hexdigest(),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'cpu_threads': torch.get_num_threads(), 'seconds': args.seconds})
    micro(model)
    adapter = TimedAdapter(AdaptationSelfPlayAdapter(model, run_id='central-batch-output-benchmark',
        checkpoint=args.checkpoint, seed=20261003, device='cuda', simulations=200))
    stop = threading.Event()
    samples = []
    observer = threading.Thread(target=observe_broker, args=(adapter, stop, samples), daemon=True)
    observer.start()
    def finish_window(*_):
        raise KeyboardInterrupt('bounded benchmark complete')
    signal.signal(signal.SIGALRM, finish_window)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    torch.cuda.nvtx.range_push('CENTRAL_SELFPLAY')
    try:
        with _test_authority(topology='torus9', run_id='central-batch-output-benchmark'):
            run_cooperative_selfplay([f'benchmark-{i:04}' for i in range(128)], adapter=adapter,
                engine_config=SelfPlayEngineConfig(workers=16, inference_batch_cap=64,
                    inference_batch_wait_ms=1., device='cuda', process_start_method='spawn',
                    lanes_per_worker=1, active_games_per_worker=4), active_games_per_worker=4)
    except KeyboardInterrupt:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        torch.cuda.nvtx.range_pop()
        stop.set()
        observer.join(5)
    save('samples.json', samples)
    save('owner-timings.json', adapter.timings)
    workers = [[json.loads(line) for line in path.read_text().splitlines()]
               for path in output.glob('worker-*.jsonl')]
    assert len(workers) == 16
    assert all(not row['technical'] for worker in workers for row in worker)
    left = max(worker[0]['time'] for worker in workers) + 10
    right = min(worker[-1]['time'] for worker in workers) - 2
    a = min(samples, key=lambda s: abs(s['time'] - left))
    b = min(samples, key=lambda s: abs(s['time'] - right))
    left, right = a['time'], b['time']
    ta, tb = a['telemetry'], b['telemetry']
    rows = tb['inference_rows'] - ta['inference_rows']
    batches = tb['batch_rows'][len(ta['batch_rows']):]
    timings = [t for t in adapter.timings if left <= t['dispatch'] <= right]
    envelopes = sum(t['envelopes'] for t in timings)
    def latency_mean(key):
        lo, hi = ta[key], tb[key]
        return (hi['mean'] * hi['count'] - lo['mean'] * lo['count']) / (hi['count'] - lo['count'])
    owner = sum((t['end'] - t['start']) * t['envelopes'] for t in timings) / envelopes * 1000
    staging = sum((t['start'] - t['dispatch']) * t['envelopes'] for t in timings) / envelopes * 1000
    moves = sum(min(w, key=lambda s: abs(s['time'] - right))['moves'] -
                min(w, key=lambda s: abs(s['time'] - left))['moves'] for w in workers)
    report = {'window_s': right - left, 'rows': rows, 'rows_s': rows / (right - left),
        'moves': moves, 'moves_s': moves / (right - left), 'batch': stats(batches),
        'worker_blocked_mean_ms': latency_mean('worker_blocked_inference_ms'),
        'full_service_mean_ms': latency_mean('gpu_service_latency_ms'),
        'output_dispatch_residual_mean_ms': latency_mean('gpu_service_latency_ms') - staging - owner,
        'telemetry_count': tb['gpu_service_latency_ms']['count'] - ta['gpu_service_latency_ms']['count'],
        'callback_envelopes': envelopes, 'technical_games': 0,
        'method': 'Bounded 128 requested games, 64 contexts; 200 sims, cap64/wait1ms. No games/replay published. Moves: 0.5s read-only snapshots; latencies: request-weighted differenced steady window. Output residual: service minus staging and owner, small boundary count mismatch.',
        'central_fatal': adapter.broker.fatal,
        'checkpoint_unchanged': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_sha}
    assert not report['central_fatal'] and report['checkpoint_unchanged']
    save('summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
