"""Summarize nsys SQLite exports from central_inference_batch_output.py.

Profile each side separately with cuda,nvtx, sample=none, cpuctxsw=none.
Metrics use the CENTRAL_SELFPLAY NVTX range, excluding startup/10s warmup
and the last 5s. Headline throughput comes from separate unprofiled runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def active_time(intervals, left, right):
    merged = []
    for start, end in sorted(intervals):
        start, end = max(start, left), min(end, right)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path('/tmp')):
        raise ValueError('Benchmark output must be under /tmp')
    conn = sqlite3.connect(f'file:{args.database}?mode=ro', uri=True)
    left, right = conn.execute("""select n.start,n.end from NVTX_EVENTS n
        left join StringIds s on n.textId=s.id
        where coalesce(n.text,s.value)='CENTRAL_SELFPLAY' and n.end is not null""").fetchone()
    first, last = conn.execute('''select min(start),max(end) from CUPTI_ACTIVITY_KIND_MEMCPY
        where copyKind=1 and bytes%1620=0 and bytes>=1620 and start>=? and end<=?''', (left, right)).fetchone()
    left, right = first + int(10e9), last - int(5e9)
    transfers = list(conn.execute('''select copyKind,bytes,count(*) from CUPTI_ACTIVITY_KIND_MEMCPY
        where start>=? and end<=? group by copyKind,bytes order by copyKind,bytes''', (left, right)))
    inputs = [x for x in transfers if x[0] == 1 and x[1] >= 1620 and x[1] % 1620 == 0]
    rows = sum(size * count for _, size, count in inputs) // 1620
    forwards = sum(count for _, _, count in inputs)
    d2h = sum(count for kind, _, count in transfers if kind == 2)
    apis = list(conn.execute('''select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME a
        join StringIds s on s.id=a.nameId where a.start>=? and a.end<=? group by a.nameId''', (left, right)))
    syncs = sum(count for name, count in apis if name.startswith('cudaStreamSynchronize'))
    kernels = list(conn.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?', (left, right)))
    copies = list(conn.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?', (left, right)))
    report = {'window_s': (right - left) / 1e9, 'rows': rows, 'forwards': forwards,
        'mean_batch': rows / forwards, 'profiled_rows_s': rows / ((right - left) / 1e9),
        'd2h_calls': d2h, 'd2h_calls_row': d2h / rows,
        'd2h_calls_forward': d2h / forwards, 'cuda_stream_synchronizations': syncs,
        'cuda_stream_synchronizations_row': syncs / rows,
        'kernels_and_copies_active_pct': active_time(kernels + copies, left, right) / (right - left) * 100,
        'transfers_by_kind_bytes_count': transfers, 'cuda_api_counts': apis}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('transfers_by_kind_bytes_count', 'cuda_api_counts')}, indent=2))


if __name__ == '__main__':
    main()
