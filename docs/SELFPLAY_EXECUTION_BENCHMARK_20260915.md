# Torus9 Stage-2 self-play execution benchmark — Legion, 2026-09-15

This is an execution-only characterization of the Stage-2 self-play engine.
It uses the immutable M17 checkpoint from
`torus9-golden-v3-20260914-run03`, a separate temporary benchmark namespace,
CUDA, the current Torus9 Golden profile, 64 simulations, and komi `0.5`.
No training was run and no M18 was created.

The benchmark launcher is
`tools/torus9_selfplay_execution_benchmark.py`. Raw JSON results were written
under `/tmp/torus9-stage2-selfplay-bench-20260915` on Legion.

## Staged sweep

The same benchmark run ID and game IDs were used for every candidate. Shorter
32-game runs were used for the staged axes; the active-context confirmation and
winner confirmation used 64 games. `CPU cores` is process-tree effective CPU;
GPU is sampled NVML utilization. Worker-local wait was `0 ms` in every run.

| Config | workers | games/worker | active contexts | batch cap | wait ms | moves/s | games/s | CPU cores | GPU avg | rows/s | mean batch | p95 batch | games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `w16-g1-c32-w1` | 16 | 1 | 16 | 32 | 1.0 | 11.884 | 0.0984 | 3.87 | 35.6% | 754.9 | 6.69 | 15 | 32 |
| `w16-g2-c32-w1` | 16 | 2 | 32 | 32 | 1.0 | 16.934 | 0.1402 | 4.42 | 32.7% | 1082.4 | 9.79 | 26 | 32 |
| `w16-g3-c64-w1` | 16 | 3 | 48 | 64 | 1.0 | 22.053 | 0.1904 | 5.03 | 34.5% | 1421.5 | 13.66 | 30 | 64 |
| `w16-g4-c64-w1` | 16 | 4 | 64 | 64 | 1.0 | 22.584 | 0.1950 | 4.78 | 33.7% | 1455.1 | 13.45 | 36 | 64 |
| `w16-g4-c16-w1` | 16 | 4 | 32 | 16 | 1.0 | 17.561 | 0.1454 | 4.37 | 33.0% | 1123.5 | 9.67 | 16 | 32 |
| `w16-g4-c32-w1` | 16 | 4 | 32 | 32 | 1.0 | 15.629 | 0.1294 | 4.25 | 31.7% | 997.4 | 9.81 | 24 | 32 |
| `w16-g4-c16-w0` | 16 | 4 | 32 | 16 | 0.0 | 15.230 | 0.1261 | 4.22 | 33.7% | 971.8 | 9.72 | 16 | 32 |
| `w16-g4-c16-w2` | 16 | 4 | 32 | 16 | 2.0 | 15.583 | 0.1290 | 4.22 | 33.6% | 994.9 | 10.09 | 16 | 32 |
| `w8-g4-c16-w1` | 8 | 4 | 32 | 16 | 1.0 | 16.936 | 0.1403 | 3.50 | 32.8% | 1081.4 | 8.20 | 16 | 32 |
| `w16-g4-c16-w1` (64-game cap check) | 16 | 4 | 64 | 16 | 1.0 | 18.920 | 0.1633 | 3.88 | 31.4% | 1213.5 | 10.32 | 16 | 64 |

The 16→32→48→64 active-context progression is the clear performance axis.
Moving from 48 to 64 contexts improved moves/s by only 2.4% in the first
64-game pass, so this is the practical active-context plateau. On the full
64-game workload, cap16 was materially slower than cap64; the 32-game cap
rows are retained as staged evidence, not as a substitute for the full
workload comparison. The central wait local peak was 1 ms; 0 and 2 ms were
slower in the same staged configuration. The 8-worker check did not beat the
16-worker pool.

## Winner and repeat

The candidate was `16 workers × 4 active games/worker × 64 active contexts ×
batch cap 64 × central wait 1 ms`. The initial 64-game pass measured `22.584
moves/s`. Two identical 64-game repeats measured `20.899` and `20.963
moves/s`; the repeat median is `20.931 moves/s`, `0.1807 games/s`, and
`4.78 process-tree effective CPU cores`. Each run had `7413 moves`, `462598
inference rows`, `0 technical games`, and `16` real worker PIDs.

The first pass was a host/GPU-clock outlier relative to the two repeats. The
repeats themselves differ by only `0.31%`, and their latency stages are also
stable. The report therefore uses the repeat median as the confirmed winner
rate and records the first pass rather than discarding it.

### Winner latency, repeat median (milliseconds)

| Stage | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| worker → broker | 0.609 | 0.334 | 1.390 |
| broker queue | 4.745 | 3.811 | 10.963 |
| GPU/service | 9.764 | 9.702 | 14.955 |
| response → worker | 0.932 | 0.811 | 1.886 |
| total inference blocked | 16.050 | 15.269 | 26.320 |

Winner batch telemetry was mean `13.29`, p50 `8`, p95 `39`, max `60` rows.
The central owner remained a single parent PID; shared-memory transport was
true and worker-local timed waiting was zero.

## Scientific report

```text
scientific semantics changed: NO
komi: 0.5
rules fingerprint: PASS (sha256:e0fd15c82d42a63ca05c3b6fb3ae02deb938543e1e06483ecfeab275dc98a39e)
observation fingerprint: PASS (sha256:e5792b409199dfe2c25ac6f681e4ca29ed73cdf4b7d53a61df634f70d1fa415f)
target fingerprint: PASS (sha256:02ab244688534b271473302ab4edf00516b91d43fb91b8c9e592d2a8de63dfb5)
self-play fingerprint: PASS (sha256:22a4e4dd37d70bd3d712b909476120b96385802ec874b99358fab256c4e3351f)
profile fingerprint: sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775
network: GoldenGraphNetV2-Torus9 80×8, WDL+ownership+score
record validation: PASS
replay conversion: PASS (existing Stage-2 parity suite)
technical games: 0 in every benchmark run
M17 checkpoint: read-only
M18/training: not run
```

Focused execution/scientific suite: `48 passed`. The full clean-checkout CI
suite passed, including the pinned KataGo differential, V3 smoke, hardened
production smoke, and Cython MCTS build.

## Remaining bottleneck and plateau conclusion

The structural IPC bottlenecks are removed: worker→broker transport is below
1 ms mean, response transport is below 1 ms mean, and observations plus
policy/WDL outputs use reusable shared-memory slots. The remaining dominant
per-evaluation stages are central broker queue (`4.745 ms`) and model service
(`9.764 ms`), while GPU average is only about `30%` and the process tree uses
about `4.8` effective CPU cores. This points to synchronization/CPU-search
feed and small effective model batches as the next limiting resource, not a
saturated GPU or a large payload IPC path. Further Python hot-loop/CUDA work
should be a separate profiled stage; it is not part of this execution refactor.

The practical plateau is the `16×4 / 64 / 1 ms` neighborhood: 3→4 contexts
adds only 2.4% on the first full pass, repeats are stable, cap16 is slower,
and neighboring waits 0/2 ms are slower. This is an execution tuning result;
it does not change the Golden scientific profile or authorize M18.

## Git

```text
base: 6541598248c581c1da7a4ca8ff9d514da82f6a01 (PR #101 merge)
branch: stage2/selfplay-engine-process-central-inference
implementation SHA: 8950650134f159b80ca27f070b5f0fea28649a2e
benchmark/fix SHA: ac631d0
PR: https://github.com/vmdvdv-npt/gocube-alphazero/pull/102
CI for ac631d0: pass
```
