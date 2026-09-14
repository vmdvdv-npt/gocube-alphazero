# Torus 9x9 Arena lockdown — 2026-09-14

## Decision

There is one Torus 9x9 production Arena policy from this point forward.

The execution foundation is the real OS-process Arena used by the 2026-09-13
Torus9 proof line (source commit
`9723bb5ac8eb28d55d21d607e3673da5bd894315`) and the independently proven
Golden process executor from PR #83. PR #83 demonstrated 480/480 bit-exact
games, zero divergence, effective parallelism 13.492/16, and 4.372x speedup.

Those historical numbers are evidence for the process architecture, not a
performance target for the current 80x8 Torus9 network.

## What is frozen

The current `run_torus9_batched_arena` / `Torus9BatchedPUCT` production path is
frozen. Its `workers=16` are logical game lanes inside one Python process, not
16 OS MCTS workers. Its inference wait is applied after a chunk is already
formed, so it cannot aggregate requests arriving during the wait.

The historical implementations are preserved byte-for-byte under explicit
`tools/_frozen_*` module names. Their public front doors are fail-closed:

- `tools/continue_torus9_golden_m1_m100.py`
- `tools/torus9_ownership_ab.py`
- `tools/torus9_alpha_score_ab.py`
- `tools/torus9_komi_calibration.py`

All historical Arena execution requires the operator to invoke the public
script with `--allow-frozen-arena`. Programmatic execution entry points such as
`run`, `_backfill_missing_m5_arena`, `run_experiment`, and calibration `main`
are blocked. The komi calibration module keeps only its read-only/offline
analysis helpers available through the normal import path. `--request-stop`
for an already-running continuation remains available without an override.

Artifacts produced through a frozen override must not be promoted as current
Golden production evidence.

Sequential Arenas remain correctness/reference oracles only. Cube, Torus5,
and legacy AlphaZero Arena implementations are not deleted in this lockdown
because they serve other topology/reference scopes, but no public Torus9
operational runner may select them as production engines.

## Canonical facade

`gocube_golden.torus9_arena_runtime` is the only supported Torus9 production
policy facade.

`run_torus9_process_foundation()` exposes the historical real-process
implementation only when `acknowledge_foundation_only=True`. It is restricted
to:

- 16 OS workers;
- CPU execution;
- komi 0.5.

CUDA is intentionally rejected there because that historical implementation
loads both models in every worker process.

`run_current_torus9_production_arena()` remains fail-closed until the final
architecture is implemented and proven.

## Production unlock gate

The next production implementation must preserve the current scientific Arena
contract while combining the good execution properties:

1. 16 persistent OS processes perform CPU MCTS selection/expansion/backup.
2. Exactly one parent/central CUDA inference broker owns each model.
3. Requests from different worker PIDs are coalesced into central GPU batches.
4. No CUDA model is loaded in an MCTS worker.
5. Paired starts/color swaps, 64 games, 64 sims, cpuct 1.25, FPU 0, no noise,
   no temperature, no fast search, no resign, watchdog 1000, deterministic
   tie-break, fail-closed technical outcomes, and komi 0.5 remain unchanged.
6. Bit-exact/semantic parity against the reference Arena is demonstrated.
7. Legion telemetry proves 16 real worker PIDs, effective CPU parallelism,
   broker PID, queue wait/depth, cap-hit rate, cross-worker batch membership,
   GPU utilization, and end-to-end wall time.
8. `mean_inference_batch_rows >= 16` is a hard acceptance floor, not an
   optimization target.
9. The implementation must be benchmarked on the current 80x8 network before
   any cap/wait setting is promoted to Golden.

Until all gates pass, M18/current Golden production Arena stays locked.
