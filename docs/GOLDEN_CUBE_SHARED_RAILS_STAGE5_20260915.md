# Golden Cube Shared Rails — Stage 5

## Base and scope

Stage 5 is based on the freshly updated `main` at `1b775d15659286b8d6de6506a190dc23a1818b28`, the merge commit for PR #105. The implementation branch is `codex/stage5-cube-shared-rails`.

This stage changes execution plumbing only. Existing Golden Cube checkpoints and proof artifacts are read-only.

## Before

```text
CubeSelfPlayRunner
→ Cube ProcessPool
→ model per worker
→ inference batch 1

Cube custom replay/training/checkpoint orchestration
```

## After

```text
CubeSelfPlayAdapter
→ SelfPlayEngine
→ shared-memory observation/policy/WDL slots
→ one central model owner
→ batching between independent games

CubeTrainingAdapter
→ TrainingEngine
→ cumulative replay
→ transactional checkpoint publication
```

The public `run_cube_selfplay_games(...)` front door is retained and now routes to the shared rails. `CubeSelfPlayRunner` remains only as the serial parity oracle; it is not a production path.

## Arena boundary

```text
current Cube Arena path: SequentialGoldenCubeArena in the transfer-proof comparison tool
shared Arena migration required: NO
reason: it is a proof/reference evaluation path, not the Cube self-play or training production front door
```

Stage 5 does not create a fourth Arena engine or alter the Stage-4 GoCube Protocol V1 path.

## Scientific boundary

```text
scientific semantics changed: NO
Cube topology changed: NO
Cube rules changed: NO
network changed: NO
observation changed: NO
target changed: NO
self-play search changed: NO
training math changed: NO
replay semantics changed: NO
komi: 0.5
```

The frozen Cube profile fingerprint is unchanged. Execution settings are separate from profile identity:

```text
workers = 16
active_games_per_worker = 4
total_active_contexts = 64
inference_batch_cap = 64
central wait = 1 ms
```

These settings are represented as execution-only configuration and are explicitly marked non-semantic in training checkpoint metadata.

## Correctness evidence

Focused local checks completed:

```text
Cube observation writer byte parity: PASS
self-play shared-memory smoke: PASS
self-play scheduling/replenishment smoke: PASS
self-play old/new CPU parity on fixed fixture: PASS
Cube cumulative replay no-eviction guard: PASS
Cube TrainingEngine transaction/checkpoint smoke: PASS
Cube checkpoint reload/resume smoke: PASS
Stage-4 Protocol regression: PASS
```

The old/new self-play gate compares formal/technical result, action trace, state identity, side-to-move, legal mask, root visits, policy target and selected action for the fixed fixture.

Training continues to call the existing `train_cube_batch_schedule(...)` primitive through `CubeTrainingAdapter`; deterministic sampling remains `random.Random(seed).sample(...)`, with one cumulative replay exposure per new position and 64-row batching. Adam state and the model hash survive checkpoint reload/resume.

## Local test policy

Only focused Cube, shared-engine, training-engine and Stage-4 tests are run locally. Full local pytest is **NOT RUN**. Local KataGo differential is **NOT RUN**, because Stage 5 does not change rules semantics.

## Execution report

The generic engine reports wall time, games/sec, moves/sec, NN rows/sec, forward/sec, worker PIDs/CPU, active contexts, batch percentiles, broker wait, H2D, forward and response latency, tail duration and technical games. The short fixture observed real cross-game batches (`mean_inference_batch_rows = 3.2`, `max = 4`) with a single parent-side inference owner.

## Git handoff

```text
base SHA: 1b775d15659286b8d6de6506a190dc23a1818b28
branch: codex/stage5-cube-shared-rails
implementation commit: 1cd7397
final branch SHA: determined after publication
PR: pending push
CI run: pending final push
```
