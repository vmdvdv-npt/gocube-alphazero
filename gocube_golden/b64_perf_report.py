"""Read-only analysis of audit measurements and optional Nsight SQLite export."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics

from .b64_perf_audit import stats


def nsight_summary(database, measured):
    connection = sqlite3.connect('file:'+str(database.resolve())+'?mode=ro', uri=True)
    connection.row_factory=sqlite3.Row
    strings=dict(connection.execute('select id,value from StringIds'))
    kernels=list(connection.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'))
    runtime=list(connection.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start'))
    ranges=list(connection.execute('select * from NVTX_EVENTS where end is not null order by start'))
    transfers=list(connection.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'))
    n=measured['updates']; span_map={}; index=0; active=[]; annotations={}
    for r in runtime:
        while index<len(ranges) and ranges[index]['start']<=r['start']:
            active.append(ranges[index]); index+=1
        active=[a for a in active if a['end']>=r['start']]
        labels=[]
        for a in active:
            label=a['text'] or strings.get(a['textId'],'unknown')
            if a['globalTid']!=r['globalTid']:
                # Autograd dispatch runs on a worker while the main backward call is active.
                same_process=a['globalTid']//(1<<24)==r['globalTid']//(1<<24)
                if label!='backward' or not same_process:continue
            labels.append(label)
        span_map[r['correlationId']]=labels
        name=strings[r['nameId']]
        if 'Synchronize' in name or name == 'cudaMemcpy':
            for label in labels:
                row=annotations.setdefault(label,{'kernels':0,'gpu_ns':0,'sync_calls':0,'sync_cpu_ns':0,'transfer_ns':0})
                row['sync_calls']+=1;row['sync_cpu_ns']+=r['end']-r['start']
    names={}
    span_names={}
    for k in kernels:
        name=strings[k['shortName']];names[name]=names.get(name,0)+1
        for label in span_map.get(k['correlationId'],[]):
            counts=span_names.setdefault(label,{})
            counts[name]=counts.get(name,0)+1
            row=annotations.setdefault(label,{'kernels':0,'gpu_ns':0,'sync_calls':0,'sync_cpu_ns':0,'transfer_ns':0})
            row['kernels']+=1;row['gpu_ns']+=k['end']-k['start']
    for k in transfers:
        for label in span_map.get(k['correlationId'],[]):
            row=annotations.setdefault(label,{'kernels':0,'gpu_ns':0,'sync_calls':0,'sync_cpu_ns':0,'transfer_ns':0})
            row['transfer_ns']+=k['end']-k['start']
    intervals=sorted((k['start'],k['end']) for k in kernels)
    end=0;busy=0;gaps=[]
    for a,b in intervals:
        if end and a>end:gaps.append((a-end)/1e6)
        busy+=max(0,b-max(a,end));end=max(end,b)
    result={'updates':n,'kernel_count':len(kernels),'kernels_per_update':len(kernels)/n,
        'kernel_busy_ms_per_update':busy/1e6/n,
        'kernel_busy_fraction_of_profile_wall':busy/1e6/measured['loop_wall_ms'],
        'kernel_duration_us':stats([(k['end']-k['start'])/1000 for k in kernels]),
        'gpu_idle_gap_ms':stats(gaps), 'transfer_count':len(transfers),
        'transfer_gpu_ms_per_update':sum(k['end']-k['start'] for k in transfers)/1e6/n,
        'kernel_names_counts':names, 'span_kernel_names_counts':span_names, 'spans':{label:{'kernels_per_update':r['kernels']/n,
        'kernel_ms_per_update':r['gpu_ns']/1e6/n,'sync_calls_per_update':r['sync_calls']/n,
        'sync_cpu_ms_per_update':r['sync_cpu_ns']/1e6/n,'transfer_ms_per_update':r['transfer_ns']/1e6/n}
        for label,r in annotations.items()}}
    connection.close()
    return result


def gpu_samples(root):
    samples=[]
    for name in ('normal-monitor-1-monitor','normal-monitor-2-monitor'):
        for line in (root/name/'samples.jsonl').read_text().splitlines():
            s=json.loads(line)
            if (s.get('heartbeat') or {}).get('phase')=='training':
                samples.extend((s.get('gpu') or {}).get('gpus',[]))
    return {field:stats([s[field] for s in samples if isinstance(s.get(field),(int,float))])
        for field in ('utilization.gpu','power.draw','memory.used','temperature.gpu','clocks.sm')}


def write(root, nsys_root=None, database=None):
    report=json.loads((root/'report.json').read_text())
    runs=Path(report['config']['runs_root']).resolve()
    if root.resolve()==runs or runs in root.resolve().parents:
        raise ValueError('Report output must be outside production runs_root')
    passes=report['passes']
    trace=passes['profiler']['trace']
    for key in ('runtime_sync_calls_per_update','runtime_sync_wall_ms_per_update'):
        if key in trace:
            trace[key.replace('runtime_sync','runtime_transfer_or_sync')]=trace.pop(key)
    if not passes['profiler']['trace']['device_activity_available']:
        for operation in passes['profiler']['profile_ops']:
            operation['self_cuda_ms_per_update']=None
    baseline=statistics.mean(passes[k]['update_wall_ms']['mean'] for k in ('normal-monitor-1','normal-monitor-2'))
    no_monitor=statistics.mean(passes[k]['update_wall_ms']['mean'] for k in ('normal-no-monitor-1','normal-no-monitor-2'))
    detailed=passes['instrumented']['update_wall_ms']['mean']
    traced=passes['profiler']['update_wall_ms']['mean']
    gpu=gpu_samples(root)
    nsys=None
    if nsys_root and database:
        nr=json.loads((nsys_root/'report.json').read_text())
        if nr['source_sha256']!=report['source_sha256'] or nr['measurement_code_sha256']!=report['measurement_code_sha256']:
            raise ValueError('Nsight and baseline source identities differ')
        nsys=nsight_summary(database,nr['passes']['nsys'])
        nsys['update_wall_ms']=nr['passes']['nsys']['update_wall_ms']
        nsys['source_parity']=nr['parity']
        nsys['source_identities_unchanged']=nr['source_identities_unchanged']
        report['nsight']=nsys
    categories=passes['instrumented']['categories']
    lookup={
      'pre validate_clocks':['validate.pre'],'post validate_clocks':['validate.post'],
      'batch sampling':['batch.sampling'],'torch.stack':['stack.'+k for k in report['h2d_bytes']],
      'H2D':['h2d.'+k for k in report['h2d_bytes']], 'forward':['forward'],
      'losses':['loss.'+k for k in ('policy','wdl','ownership','score')],
      'unused metrics':['metric.'+k for k in ('brier','score_mae_points','policy_entropy')],
      'loss finite check':['loss.finite'],'backward':['backward'],'grad clip':['grad_clip'],
      'Adam':['adam'],'weight finite check':['weights.finite'],'scalar/telemetry sync':['scalar.telemetry'],
      'metrics.append':['metrics.append'],'heartbeat/fsync':['mark'],
      'other Python/control':['execution_permit','model.train','zero_grad','loss.sum'],
      'unattributed / residual':[]}
    table={}
    for label, row in categories.items():
        spans=[(nsys or {}).get('spans',{}).get(k,{}) for k in lookup[label]]
        table[label]={'wall_ms':row['wall_ms'],'stream_ms':row['stream_ms'],
            'fraction_of_instrumented_wall':row['wall_ms']['mean']/detailed,
            'kernel_ms_per_update':sum(s.get('kernel_ms_per_update',0) for s in spans) if nsys else None,
            'transfer_ms_per_update':sum(s.get('transfer_ms_per_update',0) for s in spans) if nsys else None,
            'kernels_per_update':sum(s.get('kernels_per_update',0) for s in spans) if nsys else None,
            'sync_calls_per_update':sum(s.get('sync_calls_per_update',0) for s in spans) if nsys else None,
            'sync_cpu_ms_per_update':sum(s.get('sync_cpu_ms_per_update',0) for s in spans) if nsys else None}
    validation_ms=sum(table[k]['wall_ms']['mean'] for k in ('pre validate_clocks','post validate_clocks','weight finite check'))
    batch_ms=sum(table[k]['wall_ms']['mean'] for k in ('batch sampling','torch.stack'))
    heartbeat_ms=table['heartbeat/fsync']['wall_ms']['mean']
    # Scenario removal fractions are assumptions, not benchmarked optimizations.
    opportunities={
        'science_neutral_low_risk':{'assumed_validation_wall_removed':.5,'assumed_batch_wall_removed':.5,
            'speedup':detailed/(detailed-.5*validation_ms-.5*batch_ms)},
        'more_invasive_execution':{'assumed_validation_wall_removed':.8,'assumed_batch_wall_removed':.8,
            'assumed_heartbeat_wall_removed':.8,
            'speedup':detailed/(detailed-.8*validation_ms-.8*batch_ms-.8*heartbeat_ms)},
        'zero_kernel_dispatch_and_io_upper_bound':{'speedup':baseline/nsys['kernel_busy_ms_per_update'] if nsys else None,
            'note':'Loose observed-kernel lower bound under Nsight conditions; assumes all idle gaps, transfer and host/I/O cost vanish. Profile overhead and clock drift limit this estimate; not a feasible forecast.'}}
    report['analysis']={'normal_ms_per_update':baseline,'instrumented_ms_per_update':detailed,
        'instrumentation_overhead_percent':(detailed/baseline-1)*100,
        'torch_profiler_overhead_percent':(traced/baseline-1)*100,
        'without_monitor_ms_per_update':no_monitor,'monitor_abba_difference_percent':(baseline/no_monitor-1)*100,
        'gpu':gpu,'table':table,'opportunity_scenarios':opportunities,
        'io':{'mark_writes_per_update':1,'mark_fsyncs_per_update':2,
            'b64_training_mark_writes':2560,'b64_training_mark_fsyncs':5120,
            'expected_training_pulse_writes':baseline*2560/1000/10,
            'note':'Pulse is phase dependent. Projection is not a measured full-generation pulse count. Initial and other lifecycle marks are additional.'}}
    md=['# B64 training update performance audit — 2026-10-07','',
        f'Normal loop: **{baseline:.2f} ms/update** (two 128-update passes with the existing external monitor).',
        f'Instrumented loop: **{detailed:.2f} ms/update**; overhead **{(detailed/baseline-1)*100:+.1f}%**.',
        f'PyTorch CPU+CUDA-runtime trace: **{traced:.2f} ms/update**, overhead **{(traced/baseline-1)*100:+.1f}%** (8 updates).','',
        'M255 parent, first historical M256 rolling window from the immutable M256–M260 offline catalog; '
        f'{report["replay_games"]} train games, {report["replay_positions"]} positions, 32 warm-up updates. '
        'All comparison passes start from the same in-memory warmed model/Adam state. B64, LR 2.5e-5, '
        'clip 8, one real Adam update per iteration, FP32, unchanged sampler and operation order. '
        f'{report["parameter_groups"]} one-parameter Adam groups, {report["parameters"]} parameters. '
        'No torch.compile, fused optimizer, accumulation, async H2D or optimization was introduced.','',
        '## Breakdown','',
        'CPU wall includes dispatch and waits. CUDA below is actual kernel/transfer duration from the independent '
        '16-update Nsight pass, not CPU enqueue time or CUDA Event stream elapsed time. Percentages use the '
        'instrumented denominator so disjoint rows close correctly; profiler inflation must not be interpreted as removable overhead.','',
        '| Stage | CPU wall ms/update | CUDA kernel ms/update | p95 wall ms | % end-to-end | sync calls/update |',
        '|---|---:|---:|---:|---:|---:|']
    for label,t in table.items():
        gpu_ms=t['transfer_ms_per_update'] if label=='H2D' else t['kernel_ms_per_update']
        gpu_text='unavailable' if gpu_ms is None else f'{gpu_ms:.3f}'
        sync='unavailable' if t['sync_calls_per_update'] is None else f'{t["sync_calls_per_update"]:.0f}'
        if label=='unattributed / residual':gpu_text='—';sync='—'
        md.append(f'| {label} | {t["wall_ms"]["mean"]:.3f} | {gpu_text} | {t["wall_ms"]["p95"]:.3f} | {t["fraction_of_instrumented_wall"]*100:.1f}% | {sync} |')
    md+=['','## Bottlenecks','']
    for i,(label,t) in enumerate(sorted(table.items(),key=lambda kv:-kv[1]['wall_ms']['mean'])[:4],1):
        md.append(f'{i}. {label}: {t["wall_ms"]["mean"]:.2f} ms/update.')
    md+=['',f'The gradient-clip wall includes waiting for queued backward work; its {table["grad clip"]["wall_ms"]["mean"]:.2f} ms is not a standalone removable clipping overhead. Heartbeat ({heartbeat_ms:.2f} ms) and Adam ({table["Adam"]["wall_ms"]["mean"]:.2f} ms) are the next largest measured stages.','', 'Substage wall timings (inclusive; do not add these again to the table):','',
        '| Span | mean ms | p50 ms | p95 ms |','|---|---:|---:|---:|']
    for label,row in passes['instrumented']['stages'].items():
        if label.startswith(('validate.pre.','validate.post.','loss.','metric.','h2d.','io.')) or label in ('execution_permit','model.train','zero_grad','scalar.telemetry'):
            s=row['wall_ms'];md.append(f'| {label} | {s["mean"]:.3f} | {s["p50"]:.3f} | {s["p95"]:.3f} |')
    if nsys:
        adam=nsys['spans']['adam'];clip=nsys['spans']['grad_clip']
        md+=['', f'Adam launches {adam["kernels_per_update"]:.0f} kernels/update across 84 singleton groups '
            f'({adam["kernel_ms_per_update"]:.3f} ms GPU). Gradient clipping launches '
            f'{clip["kernels_per_update"]:.0f} kernels/update ({clip["kernel_ms_per_update"]:.3f} ms GPU), '
            'including norm reduction/nonfinite check/scaling, with one CPU wait. The clip trace contains one reduce_kernel, one lpnorm_cleanup, two multi_tensor_apply kernels, one batched-copy kernel and nine vectorized elementwise kernels per update. '
            'The recorded foreach/fused options remain their inherited defaults (None); '
            'PyTorch 2.4.1 already auto-selects foreach internals for the CUDA singleton Adam groups. '
            'The audit has not enabled a new optimizer implementation. '
            'A future optimizer experiment must compare explicit execution/grouping choices to this actual default, '
            'rather than assume the baseline is a pure scalar Adam loop.', '']
    md+=['','Validation Python loop/control is the difference between each inclusive validate span and its four '
        'subspans (step, exp_avg, exp_avg_sq, negative variance); it also includes profiling span setup/teardown. '
        'Adam step tensors are CPU scalars in this checkpoint; their int conversion does not synchronize CUDA.','',
        '## GPU and synchronization','']
    if nsys:
        md+=[f'Nsight loop: {nsys["update_wall_ms"]["mean"]:.2f} ms/update '
             f'({(nsys["update_wall_ms"]["mean"]/baseline-1)*100:+.1f}% vs baseline). '
             f'{nsys["kernels_per_update"]:.0f} kernels/update, {nsys["kernel_busy_ms_per_update"]:.2f} ms kernel busy/update, '
             f'{nsys["kernel_busy_fraction_of_profile_wall"]*100:.1f}% of traced loop wall. '
             f'Median kernel duration {nsys["kernel_duration_us"]["p50"]:.2f} µs; '
             f'p95 GPU idle gap {nsys["gpu_idle_gap_ms"]["p95"]:.3f} ms. '
             'This is timeline busy fraction, distinct from sampled SM utilization.','']
    md+=['The installed PyTorch 2.4.1/CUPTI trace produces CUDA runtime calls but no device activity '
         '(also reproduced with a single multiply). Missing activity is **unavailable**, never zero GPU work. '
         'Nsight Systems 2026.1 records the device activity successfully without changing training/runtime dependencies. '
         'NVIDIA documents WSL2 tracing support for this architecture in '
         '[CUPTI special configurations](https://docs.nvidia.com/cupti/13.1.1/special-configurations/special-configurations.html).','',
         'Actual waits are attributed to CUDA runtime synchronize/copy calls inside spans: Python boolean conversion '
         'of finite/all/any checks in both validations and weight checks; loss boolean check; nonfinite gradient '
         'check in clip_grad_norm_; four float(loss.detach()) and float(grad); and blocking .to(cuda) transfers. '
         'The isfinite/all/any kernels themselves enqueue asynchronously; scalar consumption forces the wait. '
         'clip_grad_norm_ reduction and scaling kernels remain separate from its nonfinite-check wait, which can also absorb queued backward work. '
         'The loss finite check can absorb completion of preceding forward/loss/metric work, so its wall time '
         'cannot all be removed by changing that check.','']
    md+=['Sampled normal-loop device statistics:','']
    for field,s in gpu.items():
        md.append(f'- {field}: mean {s["mean"]:.2f}, p95 {s["p95"]:.2f}, {s["count"]} samples.' if s else f'- {field}: unavailable.')
    md+=['','## Batch, H2D and I/O','',
         f'H2D payload: {sum(report["h2d_bytes"].values())} bytes/update: '+', '.join(f'{k}={v}' for k,v in report['h2d_bytes'].items())+'. '
         'All five .to(cuda) calls retain their original blocking behavior. Stack and sampling are measured independently; '
         'CUDA transfer duration and CPU blocking/enqueue wall are different quantities.','',
         'One training mark performs one durable write and two fsync calls: file plus directory. Thus the 2560 B64 '
         'training marks guarantee **2560 writes and 5120 fsyncs**, plus the concurrent 10-second pulse and lifecycle '
         f'marks. At baseline duration, approximately {baseline*2560/1000/10:.1f} pulse writes are expected during '
         'training; this is a projection, not an observed whole-generation count. A full generation was not run. '
         'Measured bounded-pass counts (exclude initial heartbeat):','']
    for name,p in passes.items():
        md.append(f'- {name}: {p["loop_durable_writes"]} writes, {p["loop_fsync_calls"]} fsyncs, {p["pulse_writes"]} pulse writes.')
    md+=['',f'Mean durable main-thread heartbeat: {passes["instrumented"]["stages"]["heartbeat"]["wall_ms"]["mean"]:.3f} ms. '
         'JSON serialization, exclusive file creation, write/flush, file fsync, replace and directory fsync are '
         'shown in substage timings. Both diagnostic and production heartbeat files use ext4; /tmp is tmpfs '
         'and was excluded from the final I/O comparison.','',
         '## External monitoring and limitations','',
         f'ABBA means: with monitor {baseline:.2f} ms/update; without monitor {no_monitor:.2f} ms/update '
         f'({(baseline/no_monitor-1)*100:+.1f}% difference). This is a bounded comparison, not a causal confidence interval; '
         'thermal/power drift and desktop GPU workloads remain possible. The existing monitor runs at its default '
         'one-second interval, reads diagnostic heartbeats/process telemetry and invokes nvidia-smi. '
         'Only audit-owned monitor PIDs are terminated. No production telemetry is stopped or reconfigured.','',
         'The initial exploratory pass overlapped local CPU tests and is excluded from the final report. '
         'Final measured passes run after those tests completed. No production trainer/monitor process was found at preflight; '
         'desktop device memory/utilization is still device-wide. Other active Python processes belong to codex_bot and codex_telegram_events.py; these host/application processes were retained and their marginal influence was not isolated. Warm-up changes the model by 32 genuine updates; '
         'all benchmark comparisons reuse that same state, and no state is saved. '
         'CUDA Events are read only after each entire measured pass, with no per-stage synchronization; '
         'their stream elapsed values include idle/dispatch gaps and are retained in JSON as stream_ms. '
         'Nsight labels resolve actual device work through CUPTI correlation IDs. Autograd worker-thread launches inherit the enclosing main-thread backward range in the same process; no stage is inferred from GPU execution timestamp alone.','',
         '## Opportunity and next PR','',
         f'Amdahl scenarios (assumptions, **not measured speedups**): science-neutral lower-risk '
         f'~{opportunities["science_neutral_low_risk"]["speedup"]:.2f}× if half the validation/weight-check '
         'and batch wall is saved; more invasive execution '
         f'~{opportunities["more_invasive_execution"]["speedup"]:.2f}× if 80% of these plus heartbeat wall is saved. '
         f'Loose theoretical zero-host/zero-I/O bound: '
         f'{opportunities["zero_kernel_dispatch_and_io_upper_bound"]["speedup"]:.1f}×.' if nsys else 'Theoretical kernel bound unavailable.', '',
         '| Next change | Expected gain hypothesis | Scientific/numerical risk | Bitwise trajectory | Complexity |',
         '|---|---|---|---|---|',
         '| Consolidate existing validation booleans into fewer CPU transfers while retaining every check | Save 30–60 ms/update in the instrumented regime; benchmark separately | Low numerical risk; failure reporting/order needs review | Plausible; require exact parity | Medium |',
         '| Consolidate weight finite and telemetry scalar transfers | Hypothesis: save 5–10 ms/update | Low; preserve fail-closed behavior and returned values | Plausible; require exact parity | Medium |',
         '| Cache cumulative replay offsets for an immutable window; retain RNG and bisect order | Hypothesis: save 4–6 ms/update | Low if replay identity invalidates cache | Expected exact | Low–medium |',
         '| Review durable heartbeat cadence/batching in a separate explicitly authorized policy PR | Hypothesis: save 5–10 ms/update, ceiling ~14 ms | Training math unaffected; supervision/durability risk | Training state can remain exact | Medium |',
         '| Evaluate Adam group coalescing or explicit optimizer execution only after host checks are addressed | Hypothesis: save 3–8 ms/update; current ~15 ms wall is the ceiling, automatic foreach already present | Reduction/order and optimizer rounding may change | Not guaranteed | Medium–high |','',
         'Discarded brier/score_mae_points/policy_entropy remain computed. Removing them would require a separate '
         'change; their measured dispatch/kernel cost is visible above and cannot explain the large fixed update cost. '
         'The Amdahl upper bound also ignores irreducible memory transfers and execution dependencies and should '
         'not be interpreted as an attainable forecast.','',
         '## Preservation and reproduction','',
         'Four-update CUDA parity: sampled positions, model tensors, complete Adam state and update counter '
         'all exactly equal. CPU parity is covered by isolated synthetic-replay tests. All registered source '
         'checkpoint/replay/manifest SHA-256 identities match before and after the audit. No checkpoint/replay '
         'was copied, changed, moved, archived or published; no self-play or Arena was created. '
         'Production notifications, services, live working tree and code pins were untouched.','',
         'Run `.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint b64-perf-audit '
         'configs/diagnostics/b64-perf-audit-20261007.json` from this branch, selecting a new exclusive '
         'output directory on the production filesystem. Source reference paths, SHA-256 identities, '
         'measurement code hashes, environment flags, all pass statistics and Nsight summary are in `report.json`. '
         'Raw profiler/monitor artifacts remain in the diagnostic output directories outside the lineage.','']
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(md))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--nsys-root',type=Path)
    parser.add_argument('--nsys-sqlite',type=Path)
    args=parser.parse_args()
    write(args.root,args.nsys_root,args.nsys_sqlite)
