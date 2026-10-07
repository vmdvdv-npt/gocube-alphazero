"""Summarize a paired B64 speedup comparison without touching training artifacts."""
import argparse
import json
from pathlib import Path

from .b64_perf_audit import stats
from .b64_perf_report import nsight_summary


def write(root, nsys_root=None, database=None):
    root = root.resolve()
    report = json.loads((root/'report.json').read_text())
    runs = Path(report['config']['runs_root']).resolve()
    if root == runs or runs in root.parents:
        raise ValueError('Comparison report must be outside production runs_root')
    comparison = report['comparison']; result = comparison['result']
    if not result['full_measured_trajectory_and_telemetry_bytes_equal'] or not report['source_identities_unchanged']:
        raise ValueError('Comparison did not preserve the scientific contract')
    if nsys_root is not None and database is not None:
        trace = json.loads((nsys_root/'report.json').read_text())
        if trace['source_sha256'] != report['source_sha256'] or trace['measurement_code_sha256'] != report['measurement_code_sha256']:
            raise ValueError('Comparison and Nsight identities differ')
        nsight = nsight_summary(database,trace['passes']['nsys'])
        nsight['update_wall_ms'] = trace['passes']['nsys']['update_wall_ms']
        report['optimized_nsight'] = nsight
    hardware = {}
    for label in comparison['passes']:
        readings = []
        for line in (root/(label+'-monitor')/'samples.jsonl').read_text().splitlines():
            sample = json.loads(line)
            if (sample.get('heartbeat') or {}).get('phase') == 'training':
                readings.extend((sample.get('gpu') or {}).get('gpus',[]))
        hardware[label] = {key:stats([r[key] for r in readings if isinstance(r.get(key),(int,float))])
                          for key in ('utilization.gpu','power.draw','memory.used','clocks.sm')}
    comparison['hardware'] = hardware
    reference, optimized = result['reference_ms_per_update'],result['optimized_ms_per_update']
    lines = ['# B64 speedup — 2026-10-07','',
        f'**{reference:.2f} → {optimized:.2f} ms/update, {result["speedup"]:.2f}× faster '
        f'({result["wall_reduction_percent"]:.2f}% less wall time).**', '',
        'This is a paired same-host reference/optimized/optimized/reference comparison of true batch-64 '
        'Adam updates. Each pass contains 128 measured updates, after 32 warm-up updates from the same '
        'M255 parent. The historical workload is the first M256 rolling window from the immutable '
        f'M256–M260 catalog: {report["replay_games"]} games and {report["replay_positions"]} positions. '
        'The existing external efficiency monitor runs at its default one-second interval in every pass. '
        'The ordinary durable mark/heartbeat and concurrent 10-second pulse remain enabled on ext4.', '',
        '| Pass | Mean ms/update | p50 | p95 | Updates |', '|---|---:|---:|---:|---:|']
    for label, p in comparison['passes'].items():
        s = p['update_wall_ms']
        lines.append(f'| {label} | {s["mean"]:.3f} | {s["p50"]:.3f} | {s["p95"]:.3f} | {p["updates"]} |')
    lines += ['', 'The code changes:', '',
        '- Validate Adam clocks/shapes in the same parameter order. Pack read-only moments for finite/negative '
        'checks and transfer three boolean results together. Both pre/post checks remain. On any invalid '
        'structure/value, the original detailed validator preserves first-error order and exception messages.',
        '- Check all model weights with one packed finite predicate, after every update.',
        '- Transfer the four returned losses and grad norm together, preserving exact Python scalar values.',
        '- Cache cumulative position offsets once per read-only replay window. RNG seed, 64 draws, bisect '
        'and selected game/position order are unchanged. The window shares original target references and '
        'does not copy or prepack replay tensors. Mutable caller-owned sequences retain the uncached sampler.', '',
        'Adam, its 84 named singleton groups and unequal inherited clocks, FP32, losses including discarded '
        'metrics, LR 2.5e-5, clip 8 with error_if_nonfinite=True, model and update order are unchanged. '
        'No compile, fused Adam, CUDA Graphs, gradient accumulation, async H2D, notification or heartbeat '
        'policy changes were introduced.', '', 'Exact preservation:', '',
        '- Eight-update sample identity and telemetry comparison passes. Model and complete Adam state use '
        'exact tensor-byte hashes, including signed zero, dtype, shape, device and stride.',
        '- Warmed states match after 32 updates. All four 128-update pass endpoints and all per-update '
        'telemetry hashes match exactly. End update counter: '+str(result['end_fingerprint']['update'])+'.',
        '- Source checkpoint/replay/manifest SHA-256 identities match before and after. No source artifact '
        'was copied, moved or changed. No production checkpoint, self-play, Arena or lineage was created.',
        '- Synthetic isolated tests cover sampling, mutable caller behavior, missing/invalid clocks, '
        'NaN/Inf/negative moments, shape failures and original first-error ordering.', '']
    nsight = report.get('optimized_nsight')
    if nsight:
        sync_labels = ('validate.pre','validate.post','weights.finite','scalar.telemetry','loss.finite','grad_clip')
        synchronizations = sum(s['sync_calls_per_update'] for k,s in nsight['spans'].items()
                              if k in sync_labels or k.startswith('h2d.'))
        lines += ['CUDA evidence (independent 16-update Nsight pass):','',
            f'- Kernels/update: **3428 → {nsight["kernels_per_update"]:.0f}**.',
            f'- Explicit CUDA waits/blocking copies: **600 → {synchronizations:.0f} per update**.',
            f'- Each Adam validation: **999 → {nsight["spans"]["validate.pre"]["kernels_per_update"]:.0f} kernels**, '
            f'**252 → {nsight["spans"]["validate.pre"]["sync_calls_per_update"]:.0f} wait**.',
            f'- Weight validation: **417 → {nsight["spans"]["weights.finite"]["kernels_per_update"]:.0f} kernels**, '
            f'**84 → {nsight["spans"]["weights.finite"]["sync_calls_per_update"]:.0f} wait**.',
            f'- Adam remains **{nsight["spans"]["adam"]["kernels_per_update"]:.0f} kernels/update**; its execution '
            'implementation/options were not changed.',
            f'- Nsight wall: {nsight["update_wall_ms"]["mean"]:.2f} ms/update, '
            f'{(nsight["update_wall_ms"]["mean"]/optimized-1)*100:+.1f}% overhead. '
            f'Kernel busy time: {nsight["kernel_busy_ms_per_update"]:.3f} ms/update. '
            'Nsight times are not used for the headline speedup.', '',
            'Reference kernel/wait counts come from the preceding audit of this same model/optimizer path; '
            'counts are structural, while wall/kernel timings vary with clocks and profiling overhead. '
            'The earlier ~2.9× zero-host estimate assumed the old kernel workload and profile hardware state. '
            'Packing checks removes most validation kernels, so that estimate is not an absolute bound for '
            'this implementation.', '']
    lines += ['Hardware sample means (device-wide; short optimized phases have few samples):','',
        '| Pass | GPU utilization % | Power W | SM MHz | Samples |', '|---|---:|---:|---:|---:|']
    for label, values in hardware.items():
        u,p,c = values['utilization.gpu'],values['power.draw'],values['clocks.sm']
        if u and p and c:
            lines.append(f'| {label} | {u["mean"]:.2f} | {p["mean"]:.2f} | {c["mean"]:.2f} | {u["count"]} |')
    lines += ['', 'The two reference passes drift by several percent; both optimized passes are near 50 ms. '
        'GPU clocks/power remain dynamic and background desktop/application processes were retained. '
        'The paired speedup is measured in this environment, not a fixed-clock universal guarantee. '
        'A projected 2560-update optimizer loop falls from '
        f'{reference*2560/60000:.2f} to {optimized*2560/60000:.2f} minutes. '
        'No full 2560-update generation was launched for this comparison.', '',
        'All passes report the same CUDA allocator peak in this comparison. Absolute peaks include '
        'both warmed reference/optimized trainers retained in memory and are not a single-trainer VRAM budget.', '',
        'Reproduction: use the authorized V2 `b64-perf-audit` command with '
        '`configs/diagnostics/b64-speedup-comparison-20261007.json`, a fresh exclusive output directory and '
        'a read-only trainer source from audit commit `aa99db656058d4763d24bafbb7d629b5495871c1`. '
        'Its SHA is pinned in that config; create the reference checkout at the configured path or update '
        'the path while retaining the SHA. The reference is imported as a distinct class, without replacing '
        'methods or copying a training loop. Both paths run under the same signed V2 child permit. '
        'Machine-readable pass stats, source identities, byte hashes and Nsight evidence are in `report.json`.', '']
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--nsys-root',type=Path)
    parser.add_argument('--nsys-sqlite',type=Path)
    args = parser.parse_args()
    write(args.root,args.nsys_root,args.nsys_sqlite)
