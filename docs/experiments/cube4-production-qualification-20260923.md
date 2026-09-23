# Cube4 production qualification — 2026-09-23

## Result

**PASS.** The bounded qualification used the proposed Cube4 production
configuration on the Legion RTX 3060 Laptop GPU. It did not create a real
production lineage: M0, replay, self-play records, and Arena output were
written below a temporary directory and removed after the run. No new
`runs/cube4/active` or `runs/cube4/archive` production lineage was created;
pre-existing ignored Stage-8 evaluation fixtures under
`runs/cube4/evaluations` were not used or modified.

Source provenance at qualification start:

- commit: `5cc09ab69eaf586e828c3411dd654ddf58461170`
- tree: `9af13be91f406a82a7e99ffa81948f03178869c6`
- working tree: clean
- host: `Legion`, WSL2, 16 logical CPUs
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB
- CUDA: 12.4

## Selected configuration

The effective preset is
[`configs/gocube/cube4_production_stage9_v1.json`](../../configs/gocube/cube4_production_stage9_v1.json).

Self-play used 64 games, 64 simulations, `cpuct=1.25`, `fpu=0`, root noise
`epsilon=0.25`, `alpha=0.11`, temperature on plies 1–8 and then 0, no resign,
komi `0.5`, and technical move cap 600. Execution was 16 workers × 4 active
games = 64 contexts, inference cap 64, wait 1 ms, CUDA, `spawn`.

The production Arena target remains 192 games, 64 simulations, `cpuct=1.25`,
`fpu=0`, watchdog 1200 s, 16 workers × 12 games, batch cap 64, wait 4 ms,
CUDA, strict production. Qualification intentionally used the monitoring
acceptance path with 64 games so that the bounded run could validate wiring and
correctness without claiming a 192-game production Arena.

## Measured self-play

| Measure | Value |
| --- | ---: |
| Games / formal / technical | 64 / 64 / 0 |
| Technical reasons | none; 600 cap was sufficient |
| Moves | 5,720 |
| Moves per game | 89.375 |
| Wall time | 553.286 s |
| Games per hour | 416.421 |
| Moves per hour | 37,217.610 |
| Replay positions | 5,720 |
| Inference calls / rows | 26,974 / 367,057 |
| Mean / p50 / p95 / max batch rows | 13.608 / 11 / 39 / 60 |
| Peak concurrent contexts | 64 |
| Peak VRAM (hardware) | 1,077 MiB |
| Mean / peak GPU utilization | 33.7% / 70.0% |
| Effective CPU / process-tree CPU | 5.88 / 8.89 cores |

There were no worker errors, central-inference failures, invalid records, or
non-`MOVE_LIMIT` technical terminations. The 1000/1600 fallback caps were not
needed.

## Measured Arena qualification

| Measure | Value |
| --- | ---: |
| Requested / valid / technical / invalid games | 64 / 64 / 0 / 0 |
| Candidate wins / reference wins / draws | 32 / 32 / 0 |
| Paired colors | candidate black 32, candidate white 32 |
| Wall time | 604.630 s |
| Games per hour | 381.060 |
| Inference calls | 13,437 |
| Mean / p50 / p95 / max batch rows | 31.679 / 32 / 41 / 60 |
| Effective CPU | 8.18 cores |
| Peak VRAM | 1,077 MiB |
| Performance status | HEALTHY |
| Performance warnings | none |

GPU utilization was not emitted by the Arena summary; the self-play hardware
telemetry above is the available direct utilization measurement. The Arena
run used the same M0 as candidate and reference. This is a correctness and
execution qualification only; no model-strength gate was applied.

## Acceptance

- self-play completed all 64 games formally;
- all records validated deeply and produced replay positions;
- Arena completed all requested games with zero technical or invalid games;
- colors were paired 32/32;
- the strict Arena execution path reported `HEALTHY` with no warnings;
- no new production M0, checkpoint, replay, or Arena result was left in the
  repository's active/archive lineage directories.

The real canonical M0 and the subsequent M1/Arena/resume/M2/continuous chain
remain post-merge operations, guarded by the clean-main preflight in
[`tools/publish_cube_m0.py`](../../tools/publish_cube_m0.py).
