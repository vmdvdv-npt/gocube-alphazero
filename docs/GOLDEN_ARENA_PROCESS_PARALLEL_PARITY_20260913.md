# Golden Arena process-parallel parity

Run: `torus-golden-arena-process-parity-20260913-final`
Execution date: 2026-09-13 (Asia/Tbilisi)
Base branch: `codex/torus-rebuild-v1` at merge commit `6cf0fc7` (PR #82)
Feature branch: `codex/golden-arena-process-parallel`
Source commit: `92f1ab17974642b2b3a60621965f9ac2f598c31c`

## Verdict

**PARALLEL GOLDEN ARENA: PASS**

The change accelerates only Golden Arena execution. It creates one task per
independent game, runs tasks in persistent OS processes, loads each checkpoint
once per worker, and reconstructs results in canonical pair/input order. The
worker calls the existing `SequentialGoldenArena.play_game`; search, inference,
batching, self-play, replay, and training code are unchanged.

## Parity gates

| Gate | Result |
|---|---:|
| Sequential vs process-parallel games | 480 / 480 identical |
| Full action traces | bit-exact; first divergence: none |
| Color-swapped pair order | PASS |
| Frozen expected W/L/D | PASS for all six comparisons |
| Technical games | 0 |
| Worker counts 1/2/4/8/16 | 4 / 4 identical for every count |

The process pool uses at most 16 workers. A single-game request remains on the
existing sequential path. Worker CPU threads are pinned to the same one-thread
contract used by the reference Arena caller, preventing nested oversubscription
without changing search behavior.

## Frozen Arena results

The immutable Stage4 v8 checkpoints and frozen starts were used. Each primary
comparison has 64 start pairs and 128 games; adjacent progression checks have
16 start pairs and 32 games.

| Comparison | Games | W/L/D | Technical | Trace parity |
|---|---:|---:|---:|---|
| M1 vs M0 | 128 | 91/37/0 | 0 | identical |
| M4 vs M0 | 128 | 124/4/0 | 0 | identical |
| M4 vs M1 | 128 | 113/15/0 | 0 | identical |
| M2 vs M1 | 32 | 25/7/0 | 0 | identical |
| M3 vs M2 | 32 | 19/13/0 | 0 | identical |
| M4 vs M3 | 32 | 18/14/0 | 0 | identical |

## Performance

| Measurement | Value |
|---|---:|
| Sequential wall time, 480 games | 795.107 s |
| Parallel wall time, 16 workers | 181.853 s |
| Aggregate speedup | 4.372x |
| Parallel child CPU time | 2453.624 s |
| Effective parallelism | 13.492 / 16 |
| Worker utilization | 84.33% |
| Peak parent RSS | 614.4 MB |
| Peak child RSS | 413.7 MB |

RSS is measured with POSIX `getrusage`; the child value is the peak reported
for child processes. Process startup and checkpoint loading are included in
parallel wall time.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python tools/torus_golden_process_arena.py \
  --workers 16 \
  --output-dir runs/golden-arena-process-parallel/torus-golden-arena-process-parity-20260913-final
```

The complete machine-readable result is tracked at
`docs/GOLDEN_ARENA_PROCESS_PARALLEL_BENCHMARK_20260913.json`; detailed per-game
records and parity files are written below the ignored `runs/` artifact tree.

## Verification

Targeted Arena and metadata tests:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_golden_arena_process_parallel.py \
  tests/test_torus_golden_stage1_meta.py \
  tests/test_torus_golden_stage2_arena.py \
  tests/test_golden_cube_arena.py
```

Full local pytest on the final implementation commit completed with `1207
passed, 1 failed, 57 warnings` in 385.27 s. The sole failure is the existing
data-dependent `tests/test_gocube_s1_replay_audit.py` assertion: the ignored
local `data/` directory contains 8208 fork records while the test expects 0.
No tracked data or test was changed; the targeted Golden Arena suite passed
with 38 tests.
