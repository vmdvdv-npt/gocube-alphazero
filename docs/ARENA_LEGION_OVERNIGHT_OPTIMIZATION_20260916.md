# Final Arena benchmark sweep on Legion

Date: 2026-09-16  
Scope: execution-only Torus9 Arena tuning after the lane/concurrency repair.

> Any Stage 7 baseline mentioned below is historical / retired validation
> evidence. Do not invoke the removed Stage 7 harness; use the standalone
> `tools/arena.py` implementation and the current Golden presets.

This performance benchmark does not define checkpoint strength methodology:
strength comparisons use the current standard-64 Arena contract, while the
high-volume preset below is execution-only.

## Decision

The single production Arena preset is now:

```text
default workload                 = 192 games
workers                         = 16
games_per_worker                = 12
configured context capacity     = 192
observed steady-state contexts  = 191..192 (max 192)
inference batch cap             = 64 rows
central inference wait          = 4 ms
worker-local wait               = 0 ms
central model owner             = parent
device                          = CUDA
shared memory                   = enabled
strict production gates         = enabled
effective CPU target             = diagnostic only (8 cores)
```

Standard-64 remains an explicit workload override: 64 games with 16 workers
and 4 contexts per worker, cap64, and wait1 ms.

This is the `12/4/cap64` configuration in
`configs/gocube/arena_torus9_legion_v1.json`. It is supported by a second
192-game repeat and a 256-game confirmation, both passing the corrected
throughput/occupancy/technical gates.

The first 192-game `12/4/cap64` run was retained rather than discarded: it
had the highest observed throughput, while effective worker CPU was 7.92
cores. CPU is retained as a regression diagnostic, not a fail-closed
condition; the repeat and long confirmation show the production-safe result
and its run-to-run range.

## Preconditions and fixture

All comparable runs used:

- candidate M10 and reference M5 from the frozen Torus9 Golden checkpoint
  directory;
- candidate model hash
  `sha256:f71ee1d742c8c8eeb719276c703c690f73eb103941291214943e359d5ef4e8d8`;
- reference model hash
  `sha256:9d0c591464c729eb8fff880bd1b719aa4ed78d76b6bb85d62f7adf8106e04bd1`;
- candidate/reference artifact SHA256 values recorded in every manifest;
- master seed `202609131004`;
- komi `0.5`, Torus 9×9 topology, 64 simulations, cpuct 1.25, FPU 0,
  noise OFF, temperature 0, resign OFF, deterministic tie break, watchdog
  1000, paired starts with color swap;
- the same parent-owned model-aware central broker and shared-memory transport.

The scientific contract fingerprint was identical across the sweep:
`sha256:8fe09d8f51e7e1b44eea1b327a774221d94f6c6920f94426d611d0759d6fdb8c`.
Only execution parameters and workload length varied. The read-only Stage 7
baseline was never modified.

## Comparative benchmark table

`Arena wall` is the measured gameplay interval. `E2E wall` is the full
process interval including startup and cleanup. `n/a` means the historical
artifact did not expose that metric. GPU utilization is `n/a` for new runs:
this Legion host exposed no usable `nvidia-smi`; no GPU value was fabricated.

| ID | workers | contexts/worker | actual peak | wait ms | cap | games | Arena wall s | E2E wall s | games/h | moves/s | rows/s | mean batch | p95 batch | forwards | eff CPU | infer wait | GPU avg | startup s | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| canonical historical reference | n/a | n/a | n/a | n/a | n/a | 64 | 1138.094 | n/a | 202.44 | n/a | n/a | 17.907 | n/a | 19334 | n/a | n/a | 27.68% | n/a | historical |
| fixed current baseline | 16 | 4 logical lanes | n/a | n/a | 64 | 64 | 554.381 | n/a | 415.60 | 10.256 | n/a | 5.491 | 11 | 63051 | 1.639 | 85.76% | n/a | n/a | historical |
| fixed-seed repaired baseline, 4/1 | 16 | 4 | 64 | 1 | 64 | 64 | 272.982 | 279.870 | 844.01 | 20.818 | 1267.50 | 13.25 | 34 | 26114 | 5.327 | 91.43% | n/a | 4.132 | degraded |
| Stage A early, 1/6 | 16 | 1 | 16 | 6 | 64 | 64 | 5.003* | n/a | n/a | n/a | n/a | 7.47 | n/a | 638 | n/a | n/a | n/a | n/a | early degraded |
| Stage A early, 2/6 | 16 | 2 | 32 | 6 | 64 | 64 | 5.007* | n/a | n/a | n/a | n/a | 15.85 | n/a | 457 | n/a | n/a | n/a | n/a | early degraded |
| Stage A, 4/6 | 16 | 4 | 64 | 6 | 64 | 64 | 313.226 | 320.196 | 735.57 | 18.153 | 1105.32 | 14.13 | 35 | 24499 | 4.679 | 91.68% | n/a | 4.157 | degraded |
| Stage A, 6/6 | 16 | 6 | 96 | 6 | 64 | 96 | 395.141 | 404.304 | 874.63 | 21.147 | 1289.33 | 20.36 | 55 | 25017 | 5.914 | 93.78% | n/a | 5.051 | degraded: CPU |
| Stage A, 8/6 | 16 | 8 | 128 | 6 | 64 | 128 | 479.377 | 489.112 | 961.25 | 23.222 | 1415.83 | 26.55 | 64 | 25563 | 6.831 | 95.00% | n/a | 4.712 | degraded: CPU |
| Stage A, 12/6 | 16 | 12 | 192 | 6 | 64 | 192 | 648.994 | 661.239 | 1065.03 | 25.595 | 1559.85 | 34.18 | 64 | 29615 | 7.789 | 96.52% | n/a | 4.647 | degraded: CPU |
| Stage B, 12/4 | 16 | 12 | 192 | 4 | 64 | 192 | 632.321 | 645.032 | **1093.12** | **26.294** | **1602.63** | 33.78 | 64 | 29998 | 7.921 | 96.48% | n/a | 5.087 | degraded: CPU |
| Stage B, 12/8 | 16 | 12 | 192 | 8 | 64 | 192 | 666.423 | 678.768 | 1037.18 | 24.948 | 1520.62 | 34.39 | 64 | 29470 | 7.638 | 96.53% | n/a | 4.740 | degraded: CPU |
| Stage B, 12/12 | 16 | 12 | 192 | 12 | 64 | 192 | 695.546 | 707.781 | 993.75 | 23.904 | 1456.95 | 34.25 | 64 | 29584 | 7.368 | 96.56% | n/a | 4.709 | degraded: CPU |
| Stage B neighbor, 8/4 | 16 | 8 | 128 | 4 | 64 | 128 | 483.345 | 494.128 | 953.36 | 23.031 | 1404.21 | 26.10 | 64 | 26002 | 6.980 | 94.97% | n/a | 5.253 | degraded: CPU |
| Stage B neighbor, 8/8 | 16 | 8 | 128 | 8 | 64 | 128 | 511.963 | 522.385 | 900.06 | 21.744 | 1325.72 | 26.88 | 64 | 25252 | 6.692 | 95.03% | n/a | 5.004 | degraded: CPU |
| Stage B neighbor, 8/12 | 16 | 8 | 128 | 12 | 64 | 128 | 541.464 | 551.548 | 851.03 | 20.559 | 1253.49 | 27.47 | 64 | 24704 | 6.348 | 95.06% | n/a | 4.826 | degraded: CPU |
| Stage C, 12/4/cap32 | 16 | 12 | 192 | 4 | 32 | 192 | 727.706 | 740.068 | 949.83 | 22.835 | 1391.70 | 23.27 | 32 | 43524 | 7.611 | 97.02% | n/a | 4.535 | rejected |
| Stage C early, 12/4/cap16 | 16 | 12 | 192 | 4 | 16 | 192 | 5.013* | n/a | n/a | n/a | n/a | 15.82 | n/a | 390 | n/a | n/a | n/a | n/a | early degraded |
| Stage D, 8/8/64 | 8 | 8 | 64 | 4 | 64 | 64 | 310.589 | 316.654 | 741.82 | 18.307 | 1114.70 | 11.99 | 33 | 28866 | 3.950 | 92.60% | n/a | 3.114 | debug degraded |
| Stage D, 12/5/60 | 12 | 5 | 60 | 4 | 64 | 64 | 296.805 | 303.442 | 776.27 | 19.157 | 1166.47 | 13.37 | 34 | 25888 | 4.946 | 91.12% | n/a | 3.730 | debug degraded |
| finalist repeat, 12/4/64 | 16 | 12 | 192 | 4 | 64 | 192 | 684.795 | 698.359 | 1009.35 | 24.274 | 1479.52 | 33.89 | 64 | 29896 | 8.054 | 96.48% | n/a | 5.712 | **PASS** |
| final confirmation, 12/4/64 | 16 | 12 | 192 | 4 | 64 | 256 | 918.461 | 934.989 | 1003.42 | 24.120 | 1469.49 | 36.97 | 64 | 36510 | 8.119 | 96.27% | n/a | 5.009 | **PASS** |

The table is intentionally sorted by experiment stage. By throughput, the
top confirmed observations are the first `12/4` run (1093.12 games/h), the
`12/6` run (1065.03 games/h), the repeat (1009.35 games/h), and the 256-game
confirmation (1003.42 games/h). The final preset is selected from the
confirmed `12/4` region, not from one un-gated maximum.

`*` Early-gate rows stopped after the stated forward sample because mean batch
was already below 16; no long duplicate run was started.

Rows labeled `degraded: CPU` preserve the raw result status from the original
sweep, when the 8-core target was still hard-gated. Under the corrected
contract that target is diagnostic only; the final repeat and confirmation
remain the production evidence.

## Staged sweep findings

### Contexts per worker

The curve is real and monotonic through the tested region:

```text
16 contexts  -> mean batch  7.47 in early gate -> rejected
32 contexts  -> mean batch 15.85 in early gate -> rejected
64 contexts  -> mean batch 14.13 at wait6       -> degraded
96 contexts  -> mean batch 20.36                -> first batch-pass region
128 contexts -> mean batch 26.55                -> higher throughput
192 contexts -> mean batch 34.18                -> highest useful region
```

The 192-context point was fully occupied (`peak=192`, steady-state
`min=191`, `p50=192`, `p95=192`, `max=192`) and every one of the 16 workers
used all lanes `0..11`. The increase in contexts supplies enough pending rows
to make the central model batches materially fuller. The cost is visible in
queue p95 and inference wait, so larger capacity was not accepted merely for
its batch mean.

### Collection wait

At the selected 192-context capacity, wait4 is the best measured point:

| wait | games/h | moves/s | mean batch | queue p95 ms | effective CPU | interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 4 ms | 1093.12 first / 1009.35 repeat | 26.29 / 24.27 | 33.78 / 33.89 | 110.23 / 121.54 | 7.92 / 8.05 | best throughput region; repeat and confirmation pass |
| 6 ms | 1065.03 | 25.59 | 34.18 | 112.67 | 7.79 | slightly fuller but slower |
| 8 ms | 1037.18 | 24.95 | 34.39 | 114.66 | 7.64 | no useful batch gain; more wait |
| 12 ms | 993.75 | 23.90 | 34.25 | 113.79 | 7.37 | throughput loss without fullness gain |

The same wait curve at the neighboring 128-context point is consistent:

| wait | games/h | moves/s | mean batch | queue p95 ms | effective CPU |
|---:|---:|---:|---:|---:|---:|
| 4 ms | 953.36 | 23.03 | 26.10 | 56.90 | 6.98 |
| 6 ms | 961.25 | 23.22 | 26.55 | 57.51 | 6.83 |
| 8 ms | 900.06 | 21.74 | 26.88 | 57.89 | 6.69 |
| 12 ms | 851.03 | 20.56 | 27.47 | 58.40 | 6.35 |

At both context levels, increasing wait makes batches marginally fuller but
reduces throughput and CPU headroom. This independently rejects a global
`8–12 ms` default.

The wait window changes latency and scheduling, not the limiting producer
rate: mean batch remains approximately 34 rows from 4–12 ms while games/h
falls. This is why the winner is not the point with the largest batch mean.

### Batch cap

Cap16 was early-aborted after 390 forwards at mean batch 15.82 despite
192/192 active contexts. Cap32 completed but produced 949.83 games/h and a
163.21 ms queue p95, versus 1093.12 games/h and 110.23 ms at cap64. Cap64
filled to its ceiling in 38.05% of final confirmation forwards and reached
max batch 64. Cap128 was not run: cap64 already filled, while the staged
screening rule says not to add a larger ceiling after a lower cap is already
the throughput boundary and a larger cap has no producer-side evidence.

### Worker count

The production profile rejects non-16 worker counts fail-closed. Explicit
debug diagnostics nevertheless measured 8 workers × 8 contexts (`741.82
games/h`, mean batch 11.99) and 12 workers × 5 contexts (`776.27 games/h`,
mean batch 13.37). Both were below the 16-worker 64-context control (`844.01
games/h`, mean batch 13.25) and neither can be a production preset. This
supports retaining 16 OS workers without starting a long 32-worker sweep.

## Final confirmation details

The 256-game confirmation used `12/worker`, cap64, wait4, 16 workers and the
same seed/model pair. It produced:

- W/L/D `213/43/0`, 256 valid games, 128 valid pairs, 0 technical games;
- Arena wall `918.461 s`, full process wall `934.989 s`, startup `5.009 s`;
- `1003.418 games/h`, `24.120 moves/s`, `1469.487 inference rows/s`;
- `36.967` mean batch, p50 `41`, p95/p99/max `64/64/64`;
- `36510` central forwards and `1,013,378` inference rows;
- effective worker CPU `8.119` cores; process-tree CPU mean/p50/p95/peak
  `9.154/10.119/10.916/11.346` effective cores;
- worker inference wait `96.27%`;
- all 16 workers observed, all lanes `0..11` used, and CUDA initialized in
  none of the search workers;
- dispatch reasons: `13,891` cap-reached and `22,619` deadline-reached;
  queue-drained and shutdown-tail dispatches were zero.

For the final run, aggregate inference timing was H2D mean `0.212 ms`, model
forward mean `9.653 ms`, D2H mean `1.955 ms`, broker queue wait mean/p50/p95/
max `55.318/43.182/122.174/281.734 ms`, and end-to-end inference mean/p50/
p95/max `78.649/66.300/150.309/367.513 ms`. Candidate/reference model
batching remained isolated and balanced: candidate mean batch `36.669` over
18,270 forwards and reference mean batch `37.266` over 18,240 forwards.

The two 192-game `12/4` observations have throughput `1093.12` and `1009.35`
games/h (median `1051.23`, range `1009.35–1093.12`) and effective CPU `7.921`
and `8.054` cores. The 256-game confirmation is `1003.42 games/h` and `8.119`
cores, consistent with the repeat at the longer workload size. All three pass
the primary throughput/occupancy/technical contract, have mean batch
approximately `34–37`, and have zero technical games.

## Startup audit

The instrumented 256-game run did not reproduce the historical ~26-minute
pre-barrier delay. Current startup to the all-workers-ready barrier was
`5.009 s`:

| phase | duration |
|---|---:|
| module import to process start | 0.061 s |
| task corpus construction | 0.851 s |
| CUDA context initialization | 0.035 s |
| candidate model construction/load | 0.384 s |
| reference model construction/load | 0.096 s |
| candidate/reference warmups | 0.159 / 0.004 s |
| shared memory and queue allocation | 0.139 s |
| process spawn | 0.567 s |
| all-workers-ready barrier | 2.676 s |
| first worker request / first CUDA forward | ~5.058 / ~5.063 s after process start |
| first completed move | 11.866 s after process start |
| first completed game | 445.415 s after process start |

Startup and gameplay are separate conclusions. The current measured startup
is normal for this process topology; the historical unexplained 26-minute
interval is not present in the repaired instrumented path and should not be
silently folded into throughput.

## Correctness and integrity

The fixed 64-game repaired `16×4` run exactly reproduced the read-only Stage 7
baseline W/L/D `52/12/0` with zero technical games. Its game IDs matched the
baseline; 62/64 action traces were byte-for-byte identical and the two
remaining traces differed only after batch-shape-dependent float32 near-tie
arithmetic. The deterministic unit comparison of canonical
`SequentialPUCT` against `SequentialPUCTSession` is exact, and all repaired
execution variants produced identical traces to one another. This numerical
reproducibility caveat is retained explicitly.

The final 256-game run had zero technical outcomes, 128/128 valid paired
blocks, no worker death, all expected lane IDs, and `false` for every worker's
post-run CUDA initialization flag. No rules, topology, komi, checkpoint,
search, temperature, noise, resign, pass, seed mapping, or startset semantics
were changed.

## Remaining bottleneck

The next bottleneck is the parent-owned broker/IPC service path under high
request concurrency, not H2D transfer or lack of active contexts. At the
selected point, workers spent `96.27%` of wall time waiting for inference;
broker queue p95 was `122.17 ms`, worker-to-broker transport p95 `41.01 ms`,
and model forward p95 `19.68 ms`. H2D was only `0.212 ms` mean. Increasing
contexts from 64 to 192 raises throughput but also raises queue depth and
latency tails; increasing wait from 4 to 12 ms does not improve fullness and
reduces throughput. The next optimization target, if needed, is broker
service/IPC scheduling and model-forward overlap, with the same model-aware
candidate/reference isolation.

## Performance contract and regression protection

The canonical preset and CLI defaults now encode a default 192-game workload,
`16 workers × 12 contexts`, cap64, wait4, master seed `202609131004`, parent
CUDA ownership, and shared memory. Strict production continues to fail closed
unless:

- at least 64 games complete;
- all 16 worker processes are observed;
- all configured lanes are used and actual occupancy is recorded separately
  from capacity;
- no worker initializes CUDA;
- technical games are zero;
- mean batch is at least 16 rows;
- effective worker CPU is recorded as a diagnostic target; it does not by
  itself fail the run.

`tests/contracts/test_torus9_arena_lockdown.py` now checks both the CLI defaults and the
JSON preset as one explicit performance contract. The behavioral tests also
cover multi-lane interleaving, replenishment, model-aware isolation, session
parity, and fail-closed response identity validation.

## Raw artifacts

The raw JSON/JSONL outputs for every full sweep point, repeat, final
confirmation, and early-degraded diagnostic are committed under
`docs/arena-legion-sweep-20260916/`. Each run contains its `summary.json`,
`manifest.json`, `games.jsonl`, and `hardware-telemetry.jsonl`; early-aborted
runs contain `performance-degraded.json` plus hardware telemetry. The compact
artifact index is in that directory's `README.md`.

No training, self-play, replay, canonical model, or Stage 7 baseline artifact
was modified.

## Final verification

The final local verification passed with `256 passed, 1 skipped`, compileall,
JSON validation, raw-artifact integrity checks, and `git diff --check`. After
the final push, PR #109 completed both required CI jobs successfully:
`Golden production tests` and `Mandatory pinned KataGo rule differential`.
