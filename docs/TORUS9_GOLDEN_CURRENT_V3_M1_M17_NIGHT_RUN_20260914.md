# Torus 9×9 Golden current-v3 night run: M1→M17

Run: `torus9-golden-v3-20260914-run03`

This report records the stopped night run and the repository work that made
the continuation resumable. It does not authorize or describe M18, a new
training run, a new cap/wait sweep, or a parallel-MCTS refactor.

## Final run state

- M1 was reused from the existing checkpoint; it was not replayed.
- M2…M17 completed with 64 self-play games per iteration and `technical=0`.
- M17 is the last clean resumable boundary.
- The run is `STOPPED_USER_REQUEST`; `scientific_validity=PASS`.
- M18 did not start.
- The persisted run namespace and all M1…M17 checkpoints, replay, state, and
  telemetry artifacts remain unchanged except for their normal continuation
  outputs.

## Self-play measurements

The table reports wall time, moves/sec, inference rows/sec, mean/p50/p95
inference batch rows, and sampled GPU utilization. M1 is the existing baseline.
Every row M2…M17 is scientifically valid under the unchanged Golden semantic
contract; execution throughput is reported separately from scientific model
selection.

| iteration | self-play wall (s) | moves/s | rows/s | batch mean / p50 / p95 | GPU avg |
|---:|---:|---:|---:|---:|---:|
| M1 | 2260.359 | 2.209 | 139.569 | 1.000 / 1 / 1 | 34.87% |
| M2 | 1796.726 | 4.059 | 248.919 | 3.883 / 4 / 4 | 32.35% |
| M3 | 1193.232 | 4.686 | 294.499 | 7.251 / 8 / 8 | 30.18% |
| M4 | 1298.727 | 5.194 | 325.385 | 10.315 / 11 / 12 | 30.81% |
| M5 | 1215.654 | 4.989 | 312.854 | 10.245 / 11 / 16 | 32.70% |
| M6 | 1138.473 | 5.291 | 330.223 | 9.711 / 10 / 12 | 30.00% |
| M7 | 1182.553 | 4.982 | 311.320 | 9.152 / 10 / 12 | 30.16% |
| M8 | 1327.749 | 4.921 | 307.438 | 9.257 / 10 / 12 | 29.77% |
| M9 | 1292.143 | 4.800 | 297.599 | 9.348 / 10 / 12 | 28.32% |
| M10 | 1227.785 | 4.860 | 299.723 | 9.248 / 10 / 12 | 28.49% |
| M11 | 1312.672 | 4.851 | 301.048 | 9.313 / 10 / 12 | 28.88% |
| M12 | 1556.115 | 4.896 | 306.964 | 9.838 / 10 / 12 | 27.96% |
| M13 | 1413.409 | 4.861 | 300.904 | 9.526 / 10 / 12 | 28.05% |
| M14 | 1735.006 | 4.758 | 295.021 | 8.960 / 10 / 12 | 27.31% |
| M15 | 1763.555 | 4.690 | 293.921 | 9.078 / 10 / 12 | 27.36% |
| M16 | 1510.760 | 4.792 | 297.372 | 9.280 / 10 / 12 | 26.41% |
| M17 | 1348.265 | 4.765 | 292.513 | 9.002 / 10 / 12 | 26.65% |

Observed process CPU was approximately 1.11 core-equivalent with
`workers=16`; sampled GPU average across M2…M17 was approximately 26.4–32.7%.
This is a performance finding, not a reason to invalidate the scientific
transitions. Restoring genuine process-level parallel MCTS is explicitly a
separate follow-up task.

## Execution-only tuning

The measured execution sweep selected `inference_batch_cap=12` and
`inference_batch_wait_ms=4`. This `12/4` choice is retained as historical
telemetry only and is **invalid for Golden selection**: it must not be treated
as a new semantic Golden standard or used to authorize another long run.

The existing profile fingerprint and scientific contract were not changed.
The previously appended cap/wait rows were removed from the connected
`Golden Standart` → `TORUS 9×9` sheet and verified absent; no Arena winner row
was added.

## Arena

The valid Arena result completed during the run was:

- M10 vs M5: 64/64 valid games, `52/12/0` candidate W/L/D, technical=0.
- Arena wait: 4 ms; Arena batch size: 8; paired starts/color swap remained
  enabled under the fixed Arena contract.

The scheduled M5 vs M1 and M5 vs M0 controls did not occur. The cause was a
recoverable continuation-run exit between M5 cap selection and the scheduled
Arena dispatch. The existing continuation code includes the guarded
`--backfill-m5-arena` recovery path and the cap-summary compatibility fix;
those controls were intentionally **not** backfilled in this task. The gap is
documented and operationally guarded, and it does not make M1…M17
scientifically invalid because no missing Arena result was counted as a
training or scientific W/L result.

## Repository changes included

- continuation/resume orchestration from the existing M1 checkpoint;
- atomic-boundary state and clean-stop/resume behavior;
- execution-only Arena `inference_batch_wait_ms` application;
- NVML GPU telemetry fallback when `nvidia-smi` is unavailable;
- continuation telemetry and execution reporting;
- the `tools/` launcher `sys.path` fix;
- current Torus 9×9 targeted tests;
- technical-outcome exclusion from Arena wait selection;
- cap-selection compatibility for persisted `batch_cap` summaries;
- guarded post-run M5 Arena backfill support.

Unrelated `docs/torus9-komi-reeval-20260913/` is intentionally excluded.

## Source artifacts

The complete run artifacts are preserved at:

`runs/torus9/active/torus9-golden-v3-20260914-run03`

The final state is recorded in `continuation-state.json`, the M17 metrics in
`iter-17-summary.json`, and the lineage in `manifest.json`.
