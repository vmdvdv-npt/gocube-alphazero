# Golden Arena Legion repair and overnight optimization

Date: 2026-09-16  
Scope: execution-only repair of the Torus9 Arena path on the existing Golden
scientific contract.

## Result

The repaired path restores real independent lane concurrency: 16 OS worker
processes, four active game contexts per worker, and one central parent-owned
CUDA inference broker. The selected execution preset is
`configs/gocube/arena_torus9_legion_v1.json`:

```text
workers=16
games_per_worker=4
configured_context_capacity=64
inference_batch_rows=64
inference_batch_wait_ms=1.0
worker_local_inference_batch_wait_ms=0.0
central_model_owner=parent
device=cuda
master_seed=202609131004
```

The exact fixed-seed run reproduced the read-only Stage 7 baseline result
`W/L/D=52/12/0` with `technical_games=0`. The repaired run observed all 16
worker PIDs and lane IDs `[0,1,2,3]` in every worker; peak and steady-state
active contexts were 64.

## Root cause found

After the legacy runtime cleanup, the Arena profile still carried lane-shaped
transport, shared-memory, and response-queue structures, but the execution
loop hard-coded `lane_id=0`, consumed `response_queues[0]`, and ran one
synchronous `SequentialPUCT.search()` to completion before starting another
game. The configured `games_per_worker=4` therefore described storage, not
observed concurrency.

## Repair boundary

- Each lane now owns an independent cooperative
  `SequentialPUCTSession` and candidate/reference response transport.
- A lane suspends only at `SearchEvaluationRequest`; the worker scheduler polls
  all lanes and resumes whichever response is ready.
- A completed lane is immediately replenished from the worker's initial queue
  or the shared global task queue.
- The parent remains the sole model/CUDA owner; worker processes never load or
  initialize CUDA.
- Candidate/reference queues remain model-aware and are never mixed in one
  forward. Shared-memory rows, worker, lane, ticket, generation, role, and
  model-hash identities are validated on both sides.
- No virtual loss, same-tree parallelism, alternate MCTS implementation, or
  scientific search-setting change was introduced.

## Telemetry added

Every completed run records configured capacity separately from observed
occupancy, including worker PIDs, observed lanes per worker, current/mean/p50/
p95/max active contexts, steady-state occupancy, per-worker occupancy
distributions, replenishments, runnable/inference/queue-blocked time, batch
p50/p95/p99/max, rows pending at dispatch, dispatch reasons, workers per batch,
candidate/reference split, H2D/forward/D2H timing, startup phases, process-tree
CPU samples, and best-effort GPU/RAM telemetry. `startup-failure.json` is
written atomically when an instrumented startup phase fails.

`hardware-telemetry.jsonl` is diagnostic only. This host exposed no usable
`nvidia-smi`, so GPU utilization, temperature, power, and VRAM remain absent
instead of being fabricated.

## Benchmark sweep

All repaired runs used the same M10/M5 checkpoint artifacts and the same
16×4 shape. The first three sweep points used the default exploratory seed;
the final winner used the fixed Stage 7 seed for comparability.

| variant | games | wall s | moves/s | games/h | mean batch | p50 / p95 / p99 / max | technical | observed capacity |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| read-only fixed baseline | 64 | 554.381 | 10.256 | 415.599 | 5.491 | 6 / 11 / n/a / 14 | 0 | 16 logical lanes; old run did not expose 64 active contexts |
| repaired cap64/wait1, exploratory seed | 64 | 291.850 | 20.990 | 789.447 | 13.649 | 6 / 36 / 44 / 49 | 0 | 64 peak; 64 steady |
| repaired cap64/wait2, exploratory seed | 64 | 297.509 | 20.591 | 774.432 | 13.766 | 7 / 35 / 41 / 48 | 0 | 64 peak; 64 steady |
| repaired cap32/wait1, exploratory seed | 64 | 292.614 | 20.935 | 787.385 | 13.527 | 6 / 32 / 32 / 32 | 0 | 64 peak; 64 steady |
| repaired cap64/wait1, fixed seed | 64 | 272.982 | 20.818 | 844.012 | 13.250 | 4 / 34 / 42 / 47 | 0 | 64 peak; 64 steady |

The fixed-seed winner is approximately 2.03× the baseline moves/s and 2.03×
the baseline games/h. Its worker-only effective CPU metric was 5.327 cores;
the interval process-tree sampler measured mean 6.199, p50 8.218, p95 8.836,
peak 9.212 effective cores. The host hard gate is intentionally still
`PERFORMANCE_DEGRADED` because mean batch 13.25 is below 16 and the aggregate
worker-only metric is below 8. Strict production mode therefore fails closed;
the debug benchmark does not promote a degraded host to production evidence.

The fixed-seed result has identical W/L/D to the read-only baseline. 62/64
game action traces are byte-for-byte identical; two traces differ after
batch-size-dependent float32 GPU arithmetic changes a near-tie action. A
deterministic unit evaluator gives exact `SequentialPUCT` versus
`SequentialPUCTSession` parity, and all execution sweep variants produced
identical traces to one another. This is reported explicitly as a numerical
reproducibility caveat rather than hidden as a semantic claim.

## Startup and safety evidence

The fixed-seed repaired run recorded approximately 4.13 s startup to the
all-workers-ready barrier, with model construction/load, warmup, shared-memory
allocation, spawn, and first-request/first-forward/first-move timestamps
separated in `summary.json`. It completed 346,005 inference rows, 26,114
central forwards, and zero technical outcomes.

The response protocol fails closed on malformed, stale, cross-lane,
cross-worker, wrong-generation, wrong-role, or wrong-model-hash responses.
Atomic publication avoids leaving a valid-looking partial `summary.json` or
`games.jsonl` after a write interruption.

## Verification

Focused and behavioral tests cover model-aware isolation, broker ingress during
slow forward, real multi-lane interleaving with a blocked lane, replenishment,
session parity, and identity fail-closed behavior. The production profile
continues to use only the canonical Golden PUCT core through
`SequentialPUCTSession`.

The fixed baseline directory
`runs/torus9-stage7/torus9-golden-stage7-20260915-run01/arena/fixed-M10-vs-M5-64`
was read-only throughout this task. No training, self-play, canonical model,
replay, or scientific artifact was modified.
