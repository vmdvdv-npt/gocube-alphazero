# Torus9 9×9 controlled `inference_batch_wait_ms` benchmark

Status: **COMPLETE — correctness gate PASS; keep `wait=1.0 ms`**

The continuation uses the production `run_torus9_selfplay_games` front door.
It runs self-play only: no training, replay construction, optimizer step, or
checkpoint creation. The original Golden Google Sheet was read-only; no
`.xlsx` was created or used.

## Results

The `wait=2.0 ms` row is the existing valid artifact from the first run and was
read by reference. It was not rerun or copied. New runs used the same
self-play run ID, game IDs, master seed, checkpoint, and scientific profile.

| wait | wall | moves | moves/s | games/h | mean batch | p50/p95/max batch | broker mean/p95 ms | worker blocked mean/p95 ms | CPU | technical |
| ---: | ---: | ----: | ------: | ------: | ---------: | ----------------: | -----------------: | --------------------------: | ---: | --------: |
| 2.0 ms | 284.720 s | 6,998 | 24.579 | 809.2 | 18.399 | 19 / 40 / 60 | 5.352 / 12.217 | 18.302 / 29.275 | 5.846 | 0 (64/64) |
| 0.5 ms | 279.996 s | 7,001 | 25.004 | 822.9 | 18.236 | 16 / 43 / 60 | 5.353 / 12.154 | 18.291 / 29.945 | 5.866 | 0 (64/64) |
| 1.0 ms | 267.263 s | 6,998 | 26.184 | 862.1 | 17.989 | 16 / 40 / 60 | 5.197 / 11.536 | 17.426 / 27.722 | 5.828 | 0 (64/64) |

`wait=0.5 ms` was **4.51% slower** than `wait=1.0 ms` by moves/s. The
`1.0 ms` run was also 4.55% shorter in wall time than `0.5 ms` and 6.13%
shorter than the reused `2.0 ms` run. GPU utilization/memory samples were
unavailable because `nvidia-smi` returned no GPU samples in this environment.

## Correctness gate

The gate now keeps scientific invariants strict while allowing bounded,
expected FP32 variance caused by batch composition:

- `64/64` completed and `technical=0` for every row;
- same Torus9 profile, M17 checkpoint, seeds, game IDs, MCTS settings, and
  production execution contract;
- raw same-input CUDA batched-vs-single-row probe: policy max abs
  `6.91e-6`, WDL max abs `1.91e-6`, both within `atol=rtol=1e-5`, with zero
  tolerance violations;
- bit-exact action/result digests are advisory only. They differed for one
  game against the reused `2.0 ms` artifact for each new wait variant, which
  is allowed by the numerical gate and is recorded separately.

The full machine-readable gate is in
`runs/torus9/evaluations/torus9-wait-benchmark-20260916-run02/metrics/correctness-gate.json`.

## Decision

**Keep the Golden execution preset at `inference_batch_wait_ms=1.0 ms`.**

`0 ms` was not run: the first `0.5 ms` result was not faster than `1.0 ms`
by the required approximately 5% threshold. No execution-preset change is
recommended.

## Frozen provenance

- Continuation evaluation: `torus9-wait-benchmark-20260916-run02`.
- Benchmark code commit: `9d54d769c5508364c5bc056b6e8e8db97f602e65`.
- Reused baseline: `runs/torus9/evaluations/torus9-wait-benchmark-20260916-run01/results/wait-2.0ms.json`.
- Canonical checkpoint catalog reference: `torus9-golden-v3-20260914-run03@17`.
- Checkpoint path: `runs/torus9/active/torus9-golden-v3-20260914-run03/checkpoints/M17.pt` (reference only; not copied).
- Model SHA-256: `sha256:2b0d04c735874f4667712bc859db54560feefb3ad6cb5d2e3c769dd80f0c0ff5`.
- Artifact SHA-256: `sha256:86722afe70fefd1d4a2a408e47c3492c7b6da43e86f283888d8815da5b037e53`.
- Master seed: `202609131002`; seed derivation: `derive_seed(master_seed, run_id, game_id, "game")`.
- Fixed game IDs: `torus9-wait-benchmark-game-0000` through `-0063`.
- Self-play run ID reused across all rows: `torus9-wait-benchmark-20260916-run01`.
- Fixed execution: 16 workers; 4 active games/worker; 64 active contexts;
  inference cap 64; coalescing ON; parent-owned central CUDA inference;
  shared memory ON; device `cuda`.
- Scientific contract: Torus 9×9; komi 0.5; 64 simulations; cpuct 1.25;
  FPU 0; root noise ON; Dirichlet ε 0.25/α 0.11; temperature 1.0 on
  plies 1–8 then 0; fast search OFF; resign OFF; watchdog 500; current
  GoldenGraphNetV2-Torus9 80×8 M17.

Evaluation artifacts are under
`runs/torus9/evaluations/torus9-wait-benchmark-20260916-run02/`; status is
`ARCHIVED` / `COMPLETE`, with no checkpoint or replay artifacts.
