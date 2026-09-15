# Torus9 Legion self-play performance reference

The execution recommendation in `gocube_golden.execution_reference` is
separate from the Torus9 scientific profile. It must not be added to a
scientific fingerprint or treated as a scientific validation gate.

Validated Legion execution path:

```text
16 workers × 4 active games/worker
at least 64 games / 64 effective active contexts
batch cap 64, central wait 1 ms
CUDA + shared memory + one parent inference owner
```

Provenance: PR #102 performance validation and the 2026-09-15 current-main
reproduction at `85e37cd28e203345984c58ea2f56db818cab0c2d`.

The current-main reproduction measured `21.09873 moves/s`, mean batch
`13.255`, p95 batch `39`, and `4.796` process-tree effective CPU cores.
The historical PR #102 repeat median was `20.931 moves/s`.

`total_active_contexts` is only a ceiling. A run with 32 games can create no
more than 32 game contexts even when the requested ceiling is 64; this is an
underfilled workload, not a SelfPlayEngine regression. The advisor reports
that condition before self-play and records it in execution telemetry.

The post-run comparison is diagnostic only. Small throughput differences do
not fail a scientific run; a comparable full workload that falls by at least
15% is marked `PERFORMANCE_DEGRADED` in telemetry.
