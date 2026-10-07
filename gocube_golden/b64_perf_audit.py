"""Bounded measurement of the existing B64 path; invoked only by the V2 entrypoint.

No training artifacts are published. Replay/checkpoints are verified and read in
place. The diagnostic root is deliberately incompatible with a production lineage.
"""
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from .provenance import file_sha256
from .training_profile import Collector, span
from .torus9_five_channel_training import OrdinaryTrainer, TrainingHeartbeat, load_replay


def stats(values):
    values = sorted(values)
    if not values:
        return None
    def quantile(q):
        i = (len(values) - 1) * q
        a = int(i)
        return values[a] + (values[min(a + 1, len(values)-1)] - values[a]) * (i-a)
    return {'mean': statistics.mean(values), 'p50': quantile(.5), 'p95': quantile(.95),
            'min': values[0], 'max': values[-1], 'count': len(values)}


def exact_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact_equal(x, y) for x,y in zip(a,b))
    return a == b


def trace_summary(path, updates, wall_ms):
    events = json.loads(path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    transfers = [e for e in events if 'gpu_memcpy' in e.get('cat', '') and e.get('ph') == 'X']
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in kernels)
    busy = 0.; end = -float('inf'); gaps=[]
    for a,b in intervals:
        if a > end and end != -float('inf'):
            gaps.append((a-end)/1000)
        busy += max(0., b-max(a,end)); end=max(end,b)
    sync = [e for e in events if e.get('cat') == 'cuda_runtime' and
            any(k in e.get('name','') for k in ('Synchronize','Memcpy'))]
    return {'device_activity_available':bool(kernels), 'kernels_per_update': len(kernels)/updates if kernels else None, 'kernel_busy_ms_per_update': busy/1000/updates if kernels else None,
            'kernel_busy_fraction_of_profile_wall': busy/1000/wall_ms if kernels else None,
            'kernel_duration_us': stats([e['dur'] for e in kernels]),
            'gpu_gap_ms': stats(gaps), 'transfers_per_update': len(transfers)/updates,
            'transfer_gpu_ms_per_update': sum(e['dur'] for e in transfers)/1000/updates if transfers else None,
            'runtime_sync_calls_per_update': len(sync)/updates,
            'runtime_sync_wall_ms_per_update': sum(e.get('dur',0) for e in sync)/1000/updates}


def run(payload):
    if payload.get('schema') != 'gocube-b64-perf-audit-v1':
        raise ValueError('Expected gocube-b64-perf-audit-v1')
    output = Path(payload['output']).resolve()
    runs = Path(payload['runs_root']).resolve()
    if output == runs or runs in output.parents:
        raise ValueError('Diagnostic output must be outside runs_root')
    output.mkdir(parents=True, exist_ok=False)
    resolved = Path(payload['resolved_experiment']).resolve()
    experiment = json.loads(resolved.read_text())['workflow']['steps'][0]['config']
    arms = [a for a in experiment['arms'] if a['arm_id'] == 'B64']
    if len(arms) != 1:
        raise ValueError('Expected exactly one B64 arm')
    cfg = arms[0]['config']; training = cfg['training']; execution = cfg['execution']
    if training['batch_size'] != 64 or training['optimizer_steps_per_iteration'] != 2560:
        raise ValueError('Expected production B64/2560 configuration')
    if execution['device'] != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Audit requires the production CUDA device')
    warmup, updates = payload.get('warmup',32), payload.get('updates',128)
    if type(warmup) is not int or not 32 <= warmup <= 64 or type(updates) is not int or not 128 <= updates <= 256:
        raise ValueError('Expected 32–64 warm-up and 128–256 measured updates')
    reference = experiment['parent']
    if reference['checkpoint_id'] != 'M255':
        raise ValueError('Expected immutable M255 parent')
    matches=[runs/'torus9'/state/reference['lineage_id']/reference['path'] for state in ('active','archive')]
    matches=[p for p in matches if p.is_file()]
    if len(matches) != 1 or file_sha256(matches[0]) != reference['sha256']:
        raise ValueError('M255 identity mismatch')
    checkpoint=matches[0]
    rows=cfg['extensions']['offline_ab_replay']
    if [r['generation'] for r in rows] != list(range(256,261)):
        raise ValueError('Expected M256–M260 historical replay metadata')
    identities={str(checkpoint):reference['sha256'], str(resolved):file_sha256(resolved)}
    for row in rows:
        for key in ('fresh_replay','rolling_replay'):
            p=Path(row[key]['path']); identities[str(p)]=row[key]['sha256']
            if file_sha256(p) != row[key]['sha256']:
                raise ValueError('Historical replay manifest identity mismatch')
        for bucket in row['buckets']:
            for shard in bucket['shards']:
                identities[shard['path']]=shard['sha']
    # Audit the first historical generation, exactly as the original B64 arm began.
    row=rows[0]
    for p, digest in identities.items():
        if file_sha256(Path(p)) != digest:
            raise ValueError('Immutable source identity mismatch: '+p)
    print('Loading verified M256 rolling replay',flush=True)
    games=load_replay(row['buckets'],split='train')
    torch.set_num_threads(1)
    trainer=OrdinaryTrainer(checkpoint, learning_rate=training['learning_rate'],
        gradient_clip=training['gradient_clip'], batch_size=64,
        seed=execution['training_master_seed'], device='cuda')
    report={'schema':payload['schema'], 'config':payload, 'parent':reference,
        'source_sha256':identities, 'workload_generation':256,
        'replay_buckets':[b['generation'] for b in row['buckets']], 'replay_games':len(games),
        'replay_positions':sum(len(g['score']) for g in games),
        'measurement_code_sha256':{str(p.relative_to(Path(__file__).resolve().parents[1])):file_sha256(p) for p in [Path(__file__),Path(__file__).with_name('training_profile.py'),Path(__file__).with_name('torus9_five_channel_training.py'),Path(__file__).with_name('torus9_adaptation.py'),Path(__file__).with_name('process_supervision.py')]},
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn':torch.backends.cudnn.allow_tf32},
        'parameter_groups':len(trainer.optimizer.param_groups),
        'parameters':sum(p.numel() for p in trainer.model.parameters()),
        'optimizer_group_flags':[{k:g.get(k) for k in ('foreach','fused','capturable')} for g in trainer.optimizer.param_groups],
        'passes':{}}
    # Observe sampler identity without timing spans on the ordinary parity arm.
    left,right=copy.deepcopy(trainer),copy.deepcopy(trainer)
    plain=Collector(timings=False, capture_positions=True); profiled=Collector(cuda=True, capture_positions=True)
    with plain.activate():
        for _ in range(4): left.step(games)
    with profiled.activate():
        for _ in range(4): right.step(games)
    profiled.finish()
    report['parity']={'updates':4,'sampled_positions_equal':plain.positions==profiled.positions,
        'model_exact':exact_equal(left.model.state_dict(),right.model.state_dict()),
        'adam_exact':exact_equal(left.optimizer.state_dict(),right.optimizer.state_dict()),
        'update_equal':left.update==right.update}
    if not all(v for k,v in report['parity'].items() if k != 'updates'):
        raise RuntimeError('Instrumented path failed exact scientific parity')
    del left,right
    for _ in range(warmup): trainer.step(games)
    torch.cuda.synchronize()
    base=trainer
    monitor_tool=Path(__file__).resolve().parents[1]/'tools/training_efficiency_monitor.py'

    def bench(name, *, monitoring=True, instrumentation=False, profiler=False, count=updates, nsys=False):
        trainer=copy.deepcopy(base)
        root=output/name; root.mkdir()
        (root/'runtime/heartbeats').mkdir(parents=True)
        pulse_collector=Collector()
        heartbeat=TrainingHeartbeat(root/'runtime/heartbeats/generation-0256.json',256,collector=pulse_collector)
        heartbeat.start()
        (root/'runtime/active-child.json').write_text(json.dumps({'pid':os.getpid(),'started_at':time.time()}))
        params=root/'parameters.json'
        params.write_text(json.dumps({'run_id':root.name,'training':{'batch_size':64,'updates_per_iteration':count}}))
        monitor=None
        if monitoring:
            log=(root/'monitor.log').open('w')
            monitor=subprocess.Popen([sys.executable,str(monitor_tool),'--run-root',str(root),
                '--parameters',str(params),'--generation','256','--output',str(output/(name+'-monitor'))],
                stdout=log,stderr=log)
            time.sleep(1.2)
            if monitor.poll() is not None:
                raise RuntimeError('External monitor failed; inspect '+str(root/'monitor.log'))
        collector=Collector(cuda=instrumentation,records=profiler,nvtx=nsys)
        metrics=[]; walls=[]; slices=[]
        context=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if profiler else nullcontext()
        torch.cuda.synchronize()
        if nsys: torch.cuda.cudart().cudaProfilerStart()
        pulse_start=heartbeat.pulse_writes
        begin=time.perf_counter()
        try:
            with context as trace, collector.activate() if instrumentation or profiler or nsys else nullcontext():
                for i in range(count):
                    start=len(collector.rows); tick=time.perf_counter()
                    value=trainer.step(games)
                    with span('metrics.append'): metrics.append(value)
                    heartbeat.mark('training',i+1,count)
                    walls.append((time.perf_counter()-tick)*1000)
                    slices.append((start,len(collector.rows)))
            wall=(time.perf_counter()-begin)*1000
            torch.cuda.synchronize()
            if nsys: torch.cuda.cudart().cudaProfilerStop()
            aggregate=collector.finish() if instrumentation or profiler or nsys else {}
            stages={}
            per_updates=[]
            for a,b in slices:
                per={}
                for label,cpu,start,end in collector.rows[a:b]:
                    values=per.setdefault(label,[0.,0.,0])
                    values[0]+=cpu; values[2]+=1
                    if start is not None: values[1]+=start.elapsed_time(end)
                per_updates.append(per)
                for label,(cpu,gpu,calls) in per.items():
                    r=stages.setdefault(label,{'wall_ms':[],'stream_ms':[],'calls':[]})
                    r['wall_ms'].append(cpu); r['stream_ms'].append(gpu); r['calls'].append(calls)
            result={'update_wall_ms':stats(walls),'loop_wall_ms':wall,'updates':count,
                'stages':{k:{m:stats(v) for m,v in r.items()} for k,r in stages.items()},
                'pulse_writes':heartbeat.pulse_writes-pulse_start, 'pulse_stages':pulse_collector.finish(),
                'loop_durable_writes':count+heartbeat.pulse_writes-pulse_start,
                'loop_fsync_calls':2*(count+heartbeat.pulse_writes-pulse_start),'aggregate':aggregate}
            groups = {
                'pre validate_clocks':['validate.pre'], 'post validate_clocks':['validate.post'],
                'batch sampling':['batch.sampling'], 'torch.stack':['stack.'+k for k in ('observation','pi','z','ownership','score')],
                'H2D':['h2d.'+k for k in ('observation','pi','z','ownership','score')],
                'forward':['forward'], 'losses':['loss.'+k for k in ('policy','wdl','ownership','score')],
                'unused metrics':['metric.'+k for k in ('brier','score_mae_points','policy_entropy')],
                'loss finite check':['loss.finite'], 'backward':['backward'], 'grad clip':['grad_clip'],
                'Adam':['adam'], 'weight finite check':['weights.finite'], 'scalar/telemetry sync':['scalar.telemetry'],
                'metrics.append':['metrics.append'], 'heartbeat/fsync':['mark'],
                'other Python/control':['execution_permit','model.train','zero_grad','loss.sum']}
            result['categories']={label:{'wall_ms':stats([sum(per.get(k,[0,0,0])[0] for k in keys) for per in per_updates]),
                'stream_ms':stats([sum(per.get(k,[0,0,0])[1] for k in keys) for per in per_updates])}
                for label,keys in groups.items()} if instrumentation or profiler or nsys else {}
            if result['categories']:
                covered=set(k for keys in groups.values() for k in keys)
                result['categories']['unattributed / residual']={'wall_ms':stats([
                    w-sum(per.get(k,[0,0,0])[0] for k in covered) for w,per in zip(walls,per_updates)]),'stream_ms':None}
            if profiler:
                path=root/'trace.json' ; trace.export_chrome_trace(str(path))
                result['trace']=trace_summary(path,count,wall)
                result['profile_ops']=[{'name':e.key,'calls_per_update':e.count/count,
                    'self_cpu_ms_per_update':e.self_cpu_time_total/1000/count,
                    'self_cuda_ms_per_update':e.self_device_time_total/1000/count}
                    for e in trace.key_averages()]
            report['passes'][name]=result
            print(name,round(result['update_wall_ms']['mean'],3),'ms/update',flush=True)
        finally:
            heartbeat.close()
            if monitor is not None:
                # Only our isolated external monitor; never signal a production process.
                monitor.terminate(); monitor.wait(timeout=10); log.close()
        return trainer

    if payload.get('nsys_trace_only',False):
        bench('nsys',monitoring=True,nsys=True,count=16)
    else:
        for name, monitoring in [('normal-monitor-1',True),('normal-no-monitor-1',False),
                              ('normal-no-monitor-2',False),('normal-monitor-2',True)]:
            bench(name,monitoring=monitoring)
        instrumented=bench('instrumented',instrumentation=True)
        traced=bench('profiler',profiler=True,count=8)
        del instrumented,traced
    batch=base.batch(games,base.update+1)
    report['h2d_bytes']={k:t.numel()*t.element_size() for k,t in batch.items()}
    report['source_identities_unchanged']=all(file_sha256(Path(p))==h for p,h in identities.items())
    report['artifacts_published']=[]
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return {'output':str(output),'parity':report['parity'],'source_identities_unchanged':report['source_identities_unchanged']}


if __name__ == '__main__':
    from .orchestrator_v2.execution_permit import require_child_execution_permit
    require_child_execution_permit(__name__, action_type='training', topology='torus9')
    run(json.loads(Path(sys.argv[1]).read_text()))
