# Torus9 9×9 controlled `inference_batch_wait_ms` benchmark

Status: **STOPPED — correctness/parity gate failed; no performance winner**

The benchmark used the production `run_torus9_selfplay_games` front door and
stopped after self-play. It did not run training, replay construction,
optimizer steps, or checkpoint creation. The original Golden Google Sheet was
read-only; no `.xlsx` was created or used.

## Partial results before STOP

| wait | wall | moves | moves/s | games/h | mean batch | p95 batch | CPU | GPU | technical |
| ---: | ---: | ----: | ------: | ------: | ---------: | --------: | --: | --: | --------: |
| 2.0 ms | 284.720 s | 6,998 | 24.579 | 809.2 | 18.399 | 40 | 5.846 | N/A | 0 (64/64) |
| 0.5 ms | 284.983 s | 6,998 | 24.556 | 808.5 | 18.172 | 40 | 5.904 | N/A | 0 (64/64) |

Additional telemetry:

- `wait=2.0`: p50/max batch 19/60; 23,674 inference calls; 1,529.85 rows/s; broker queue mean/p95 5.352/12.217 ms; worker blocked-wait mean/p95 18.302/29.275 ms.
- `wait=0.5`: p50/max batch 16/60; 23,970 inference calls; 1,528.44 rows/s; broker queue mean/p95 5.450/12.453 ms; worker blocked-wait mean/p95 18.598/30.450 ms.
- GPU utilization and memory were unavailable because `nvidia-smi` returned no GPU samples in the benchmark environment.

## Correctness result

The actual order was `wait-2.0ms → wait-0.5ms`. The normalized per-game
records differed for exactly one game: `torus9-wait-benchmark-game-0004`.
The benchmark intentionally stopped before running `wait=1.0 ms`; therefore
the primary three-way comparison is invalid and no Golden change is
recommended. `wait=0 ms` was not run because the correctness gate stopped the
experiment first.

The direct CUDA probe used the same model and identical input rows, comparing
one batched forward with concatenated single-row forwards. It observed a
maximum absolute difference of approximately `2.62e-6` in policy logits and
`4.77e-7` in WDL logits. This establishes a plausible mechanism: changing
batch composition changes float32 neural outputs, which can alter a marginal
MCTS decision under the same seed. Until that is eliminated or the production
parity contract is explicitly relaxed, wait values are not a valid isolated
execution-only comparison.

## Frozen provenance

- Benchmark commit: `4591889e22ee91e390ec167b7d87b55ef5400b8d`.
- Checkpoint catalog reference: `torus9-golden-v3-20260914-run03@17`.
- Checkpoint path: `runs/torus9/active/torus9-golden-v3-20260914-run03/checkpoints/M17.pt` (reference only; not copied).
- Model SHA-256: `sha256:2b0d04c735874f4667712bc859db54560feefb3ad6cb5d2e3c769dd80f0c0ff5`.
- Artifact SHA-256: `sha256:86722afe70fefd1d4a2a408e47c3492c7b6da43e86f283888d8815da5b037e53`.
- Master seed: `202609131002`; per-game seed: `derive_seed(master_seed, run_id, game_id, "game")`.
- Fixed game IDs: `torus9-wait-benchmark-game-0000` through `-0063`; the same IDs and seeds were used for both completed variants.
- Fixed execution: 16 workers; 4 active games/worker; 64 active contexts; inference cap 64; coalescing ON; parent-owned central CUDA inference; shared memory ON; device `cuda`.
- Frozen scientific contract: Torus 9×9; komi 0.5; 64 simulations; cpuct 1.25; FPU 0; root noise ON; Dirichlet ε=0.25/α=0.11; temperature 1.0 on plies 1–8 then 0; fast search OFF; resign OFF; watchdog 500; GoldenGraphNetV2-Torus9 80×8 M17.

Evaluation artifacts are under
`runs/torus9/evaluations/torus9-wait-benchmark-20260916-run01/`, with status
`ARCHIVED` / `FAILED`; no checkpoint or replay artifacts were created.
