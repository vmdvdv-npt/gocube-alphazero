"""Passive, external monitor for one Orchestrator V2 training generation.

Uses only standard library, existing heartbeat/artifacts and nvidia-smi.
Never imports or controls training code and never writes inside the run.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

GPU_FIELDS = ['index', 'uuid', 'name', 'utilization.gpu', 'utilization.memory',
              'memory.used', 'memory.total', 'power.draw', 'temperature.gpu',
              'clocks.sm', 'clocks.mem', 'pstate']


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def gpu_sample(binary):
    if not binary:
        return {'status': 'unavailable', 'error': 'nvidia-smi not found'}
    try:
        result = subprocess.run([binary, '--query-gpu=' + ','.join(GPU_FIELDS),
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=3)
        if result.returncode:
            return {'status': 'error', 'error': result.stderr.strip()[:500]}
        rows = []
        for fields in csv.reader(result.stdout.splitlines()):
            if len(fields) != len(GPU_FIELDS):
                raise ValueError('Unexpected GPU column count')
            row = {}
            for key, raw in zip(GPU_FIELDS, fields):
                raw = raw.strip()
                if key in ('uuid', 'name', 'pstate'):
                    row[key] = raw
                else:
                    try:
                        value = float(raw)
                        row[key] = value if math.isfinite(value) else None
                    except ValueError:
                        row[key] = None
            rows.append(row)
        return {'status': 'ok' if rows else 'unavailable', 'gpus': rows}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {'status': 'error', 'error': str(exc)[:500]}


def process_sample(pid):
    if not pid:
        return None
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'cpu_seconds': (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK'),
                'rss_bytes': int(stat[21]) * os.sysconf('SC_PAGE_SIZE'), 'threads': int(stat[17])}
    except (OSError, ValueError, IndexError):
        return {'pid': pid, 'status': 'unavailable'}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def summarize(samples, run, generation, parameters):
    rolling = run / 'replay' / f'rolling-after-{generation:02d}.jsonl'
    metrics = read_json(run / 'training' / f'iter-{generation:02d}.json')
    summary = read_json(run / f'iter-{generation:02d}-summary.json')
    complete = run / f'generation-{generation}.complete.json'
    training = [s for s in samples if (s.get('heartbeat') or {}).get('phase') == 'training']
    start = rolling.stat().st_mtime if rolling.exists() else None
    end = max((s['heartbeat'].get('progress_at', 0) for s in training), default=None)
    updates = len(metrics['updates']) if metrics and isinstance(metrics.get('updates'), list) else None
    config = (parameters or {}).get('training', {})
    batch = config.get('batch_size')
    if metrics and metrics.get('updates'):
        recorded = {u.get('batch_size') for u in metrics['updates'] if u.get('batch_size') is not None}
        if len(recorded) == 1:
            batch = recorded.pop()
    # The final heartbeat may become 'done' between polls. Do not pretend the
    # last observed intermediate update was the final update.
    observed = max((s['heartbeat'].get('done', 0) for s in training), default=0)
    full_loop_observed = updates is not None and observed == updates and start is not None
    duration = end - start if full_loop_observed and end and end >= start else None
    training_rows = [s for s in samples if start is not None and end is not None
                     and start <= s['sample_started_at'] <= end]
    gpu_rows = {}
    for s in training_rows:
        for gpu in s['gpu'].get('gpus', []):
            gpu_rows.setdefault(gpu['uuid'], []).append(gpu)
    gpu_stats = {}
    for uuid, rows in gpu_rows.items():
        stats = {'name': rows[0]['name'], 'samples': len(rows)}
        for key in GPU_FIELDS:
            values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if values:
                stats[key] = {'sample_mean': sum(values) / len(values),
                              'min': min(values), 'max': max(values)}
        gpu_stats[uuid] = stats
    complete_at = complete.stat().st_mtime if complete.exists() else None
    child_starts = [s.get('child', {}).get('started_at') for s in samples if s.get('child')]
    child_start = next((t for t in child_starts if t is not None), None)
    return {
        'schema': 'gocube-external-training-efficiency-v1', 'run': str(run), 'generation': generation,
        'completed': complete.exists(), 'training_batch_size': batch,
        'optimizer_updates': updates, 'last_observed_update': observed,
        'training_examples_processed': updates * batch if updates is not None and batch else None,
        'optimizer_loop_start_proxy_at': start, 'last_observed_update_completed_at': end,
        'full_optimizer_loop_observed': full_loop_observed,
        'optimizer_loop_duration_proxy_seconds': duration,
        'updates_per_minute': updates * 60 / duration if duration and updates else None,
        'examples_per_minute': updates * batch * 60 / duration if duration and updates and batch else None,
        'generation_started_at': child_start, 'generation_committed_at': complete_at,
        'whole_generation_seconds': complete_at - child_start if complete_at and child_start else None,
        'training_through_commit_proxy_seconds': complete_at - start if complete_at and start else None,
        'post_updates_through_commit_seconds': complete_at - end if full_loop_observed and complete_at else None,
        'gpu_during_optimizer_loop': gpu_stats,
        'generation_summary': summary,
        'gpu_capture_started_before_optimizer_loop': bool(samples and start and samples[0]['sample_started_at'] <= start),
        'measurement_notes': [
            'Start proxy is rolling replay manifest publication, immediately before optimizer loop in current driver. It includes a small setup gap; no exact start event exists.',
            'End uses final update heartbeat progress_at only when all updates were observed. If final heartbeat is missed, exact loop throughput remains null.',
            'GPU statistics are device-wide sampled values, not kernel occupancy or per-process attribution. Other GPU jobs can affect them.',
            'memory.used max is a sampled device memory peak, not a CUDA allocator peak. GPU means are sample means.',
            'Examples processed counts batch draws, including repeated examples, not unique replay positions or model parameters.',
            'Training through commit includes validation, checkpoint save/reload verification and publication. Whole generation also includes preparation, self-play and replay loading; Arena is excluded.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--parameters', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--generation', type=int)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--max-hours', type=float, default=24.0)
    parser.add_argument('--nvidia-smi')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0.2:
        parser.error('--interval must be finite and >= 0.2 seconds')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error('--max-hours must be finite and positive')
    run, output = args.run_root.resolve(), args.output.resolve()
    if not run.is_dir():
        parser.error('--run-root must be an existing run directory')
    if output == run or run in output.parents:
        parser.error('--output must be outside the monitored run')
    parameters = read_json(args.parameters)
    if not parameters:
        parser.error('Cannot read operator parameters')
    if parameters.get('run_id') != run.name:
        parser.error('Operator parameters run_id does not match run directory')
    state = read_json(run / 'runtime' / 'state.json') or {}
    generation = args.generation or state.get('active_generation')
    if not isinstance(generation, int) or generation < 1:
        parser.error('No active generation; provide --generation')
    heartbeat_path = run / 'runtime' / 'heartbeats' / f'generation-{generation:04d}.json'
    if not heartbeat_path.is_file():
        parser.error('Target generation has no heartbeat yet')
    if (run / f'generation-{generation}.complete.json').exists():
        parser.error('Generation already completed; cannot capture it retrospectively')
    binary = args.nvidia_smi or shutil.which('nvidia-smi')
    if not binary and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        binary = '/usr/lib/wsl/lib/nvidia-smi'
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental duplicate monitors overwriting logs.
    log = (output / 'samples.jsonl').open('x', buffering=1)
    write_json(output / 'monitor.json', {'pid': os.getpid(), 'run': str(run),
               'generation': generation, 'interval_seconds': args.interval,
               'started_at': time.time(), 'nvidia_smi': binary,
               'source': str(Path(__file__).resolve()),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'parameters': parameters})
    samples, previous, deadline = [], None, time.monotonic() + args.max_hours * 3600
    next_report = 0
    reason = 'time_limit'
    try:
        while time.monotonic() < deadline:
            tick = time.monotonic()
            heartbeat = read_json(heartbeat_path)
            child = read_json(run / 'runtime' / 'active-child.json')
            row = {'sample_started_at': time.time(), 'heartbeat': heartbeat,
                   'child': child, 'process': process_sample((child or {}).get('pid')),
                   'gpu': gpu_sample(binary)}
            row['sample_finished_at'] = time.time()
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            samples.append(row)
            signature = ((heartbeat or {}).get('phase'), (heartbeat or {}).get('done'))
            if signature != previous:
                print(datetime.now(timezone.utc).isoformat(), generation, signature, flush=True)
                previous = signature
            completed = (run / f'generation-{generation}.complete.json').exists()
            if time.monotonic() >= next_report or completed:
                report = summarize(samples, run, generation, parameters)
                write_json(output / 'report.json', report)
                next_report = time.monotonic() + 15
            if completed:
                reason = 'generation_completed'
                break
            pid = (child or {}).get('pid')
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    reason = 'generation_process_exited_before_commit'
                    break
            time.sleep(max(0, args.interval - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        reason = 'monitor_interrupted'
    finally:
        log.close()
        report = summarize(samples, run, generation, parameters)
        metrics = read_json(run / 'training' / f'iter-{generation:02d}.json')
        if metrics:
            write_json(output / 'training-metrics.json', metrics)
        report.update(monitor_exit_reason=reason, monitor_finished_at=time.time(), sample_count=len(samples))
        write_json(output / 'report.json', report)
        print('Monitor finished:', reason, 'report:', output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
