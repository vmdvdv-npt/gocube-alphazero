"""Paired immutable-artifact comparison, called inside the authorized V2 audit.

The reference is an existing, SHA-pinned trainer module, imported under its own
name. No trainer method is replaced and no training loop is duplicated.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from .provenance import file_sha256
from .training_profile import Collector
from .torus9_five_channel_training import TrainingHeartbeat


def load_reference(path, expected):
    path = Path(path).resolve()
    if file_sha256(path) != expected:
        raise ValueError('Reference trainer SHA mismatch')
    name = 'gocube_golden._b64_reference_' + expected.split(':')[-1][:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module.OrdinaryTrainer


def state_hash(value):
    """Hash exact tensor bytes (including signed zero), shapes and scalar metadata."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(json.dumps([str(item.dtype), list(item.shape), str(item.device), list(item.stride())]).encode()+b'\0')
            digest.update(item.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict\0')
            for key, child in item.items():
                visit(key); visit(child)
            digest.update(b'end-dict\0')
        elif isinstance(item, (list,tuple)):
            digest.update((type(item).__name__+'\0').encode())
            for child in item:
                visit(child)
            digest.update(b'end-sequence\0')
        else:
            digest.update(json.dumps([type(item).__name__,item],sort_keys=True).encode()+b'\0')
    visit(value)
    return 'sha256:'+digest.hexdigest()


def fingerprint(trainer):
    return {'model':state_hash(trainer.model.state_dict()),
            'adam':state_hash(trainer.optimizer.state_dict()), 'update':trainer.update}


def compare(payload, report, trainer, checkpoint, games, *, training, execution, output, warmup, updates):
    from .b64_perf_audit import stats
    reference_type = load_reference(payload['reference_trainer_source'],payload['reference_trainer_sha256'])
    reference = reference_type(checkpoint,learning_rate=training['learning_rate'],
        gradient_clip=training['gradient_clip'],batch_size=64,
        seed=execution['training_master_seed'],device='cuda')
    # This copies only the container of game references, never targets/replay tensors.
    window_type = getattr(sys.modules[reference_type.__module__], 'OrdinaryReplayWindow', None)
    reference_games = window_type.from_games(games) if window_type else list(games)
    left, right = copy.deepcopy(reference), copy.deepcopy(trainer)
    a, b = Collector(timings=False,capture_positions=True), Collector(timings=False,capture_positions=True)
    first_metrics, second_metrics, update_hashes = [], [], []
    for _ in range(8):
        with a.activate():
            first_metrics.append(left.step(reference_games))
        with b.activate():
            second_metrics.append(right.step(games))
        left_hash, right_hash = fingerprint(left), fingerprint(right)
        update_hashes.append({'reference': left_hash, 'optimized': right_hash})
        if left_hash != right_hash or state_hash(first_metrics[-1]) != state_hash(second_metrics[-1]):
            raise RuntimeError('B64 optimization violates per-update exact trajectory parity')
    before, after = fingerprint(left), fingerprint(right)
    parity={'updates':8,'sampled_positions_equal':a.positions==b.positions,
        'telemetry_equal':first_metrics==second_metrics,'model_bytes_equal':before['model']==after['model'],
        'adam_bytes_equal':before['adam']==after['adam'],'update_equal':left.update==right.update,
        'reference':before,'optimized':after,'per_update_fingerprints':update_hashes}
    (output/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    if not all(parity[key] for key in ('sampled_positions_equal','telemetry_equal','model_bytes_equal','adam_bytes_equal','update_equal')):
        raise RuntimeError('B64 optimization violates exact trajectory parity')
    del left,right
    for _ in range(warmup):
        reference.step(reference_games)
        trainer.step(games)
    reference_warm, optimized_warm=fingerprint(reference),fingerprint(trainer)
    if reference_warm != optimized_warm:
        raise RuntimeError('Warm-up trajectory differs')
    report['comparison']={'reference_source':payload['reference_trainer_source'],
        'reference_sha256':payload['reference_trainer_sha256'], 'comparison_code_sha256':file_sha256(Path(__file__)), 'parity':parity,
        'warmup_fingerprint':reference_warm,'passes':{}}
    monitor_tool=Path(__file__).resolve().parents[1]/'tools/training_efficiency_monitor.py'
    for label, template, data in (('reference-1',reference,reference_games),('optimized-1',trainer,games),
                                  ('optimized-2',trainer,games),('reference-2',reference,reference_games)):
        candidate=copy.deepcopy(template)
        root=output/label;root.mkdir()
        (root/'runtime/heartbeats').mkdir(parents=True)
        heartbeat=TrainingHeartbeat(root/'runtime/heartbeats/generation-0256.json',256)
        heartbeat.start()
        (root/'runtime/active-child.json').write_text(json.dumps({'pid':os.getpid(),'started_at':time.time()}))
        parameters=root/'parameters.json'
        parameters.write_text(json.dumps({'run_id':root.name,'training':{'batch_size':64,'updates_per_iteration':updates}}))
        with (root/'monitor.log').open('w') as log:
            monitor=subprocess.Popen([sys.executable,str(monitor_tool),'--run-root',str(root),
                '--parameters',str(parameters),'--generation','256','--output',str(output/(label+'-monitor'))],
                stdout=log,stderr=log)
            try:
                time.sleep(1.2)
                if monitor.poll() is not None:
                    raise RuntimeError('Comparison monitor failed: '+str(root/'monitor.log'))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                walls=[];metrics=[];pulse_start=heartbeat.pulse_writes
                start=time.perf_counter()
                for i in range(updates):
                    tick=time.perf_counter()
                    metrics.append(candidate.step(data))
                    heartbeat.mark('training',i+1,updates)
                    walls.append((time.perf_counter()-tick)*1000)
                wall=time.perf_counter()-start
                torch.cuda.synchronize()
                result={'update_wall_ms':stats(walls),'loop_wall_seconds':wall,'updates':updates,
                    'fingerprint':fingerprint(candidate),'telemetry_sha256':state_hash(metrics),
                    'pulse_writes':heartbeat.pulse_writes-pulse_start,
                    'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved()}
                report['comparison']['passes'][label]=result
                print(label,round(result['update_wall_ms']['mean'],3),'ms/update',flush=True)
            finally:
                heartbeat.close()
                monitor.terminate();monitor.wait(timeout=10)
    passes=report['comparison']['passes']
    expected=passes['reference-1']['fingerprint'];expected_metrics=passes['reference-1']['telemetry_sha256']
    if not all(p['fingerprint']==expected and p['telemetry_sha256']==expected_metrics for p in passes.values()):
        raise RuntimeError('Full measured B64 trajectories or telemetry differ')
    ordinary=statistics.mean(passes[k]['update_wall_ms']['mean'] for k in ('reference-1','reference-2'))
    optimized=statistics.mean(passes[k]['update_wall_ms']['mean'] for k in ('optimized-1','optimized-2'))
    report['comparison']['result']={'reference_ms_per_update':ordinary,'optimized_ms_per_update':optimized,
        'speedup':ordinary/optimized,'wall_reduction_percent':(1-optimized/ordinary)*100,
        'full_measured_trajectory_and_telemetry_bytes_equal':True,
        'end_fingerprint':expected, 'end_telemetry_sha256':expected_metrics}
    report['source_identities_unchanged']=all(file_sha256(Path(p))==sha for p,sha in report['source_sha256'].items())
    report['artifacts_published']=[]
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['source_identities_unchanged']:
        raise RuntimeError('Immutable reference artifacts changed during comparison')
    return {'output':str(output),**report['comparison']['result']}
