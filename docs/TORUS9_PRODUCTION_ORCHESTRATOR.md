# Torus9 Production Orchestrator

This is the current production integration for the game-independent training orchestrator. Torus 9×9 is wired now. Cube is intentionally not wired yet: when a production Cube training path exists, it should be added as a separate profile driver without changing the orchestrator core.

## Start a Torus9 lineage

```bash
.venv/bin/python tools/training_orchestrator.py start \
  --spec configs/gocube/torus9_training_orchestrator_v1.json \
  --lineage <new-lineage-id> \
  --max-generations <N>
```

The concrete spec is `configs/gocube/torus9_training_orchestrator_v1.json`. It pins the current canonical Torus9 profile and the validated Legion execution path rather than duplicating scientific settings inside `gocube_golden.orchestrator`.

## Current production binding

The Torus-specific adapter is `tools/torus9_orchestrator_driver.py`. For each generation it calls the existing production paths:

- 64 self-play games per generation;
- 64 MCTS simulations;
- 16 OS workers;
- 4 active games per worker / 64 total active contexts;
- parent-owned CUDA inference with shared-memory transport;
- inference batch cap 64 and 1 ms wait;
- the existing `TrainingEngine` through `Torus9TrainingAdapter`;
- Adam, 80 optimizer steps × batch 64;
- rolling replay of the last 3 generations, capped at 20,000 positions.

The execution-only self-play values are checked against `LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE` before work starts. The driver does not silently continue with an unvalidated production execution shape.

## Periodic Arena

The production spec schedules Arena every five committed generations:

```text
M5  vs M0
M10 vs M5
M15 vs M10
...
```

The periodic comparison uses the existing universal Arena engine and Torus9 profile:

- 64 deterministic paired games with color swap;
- 64 simulations;
- cpuct 1.25, FPU 0;
- no root noise, temperature, fast search or resign;
- 16 workers;
- 12 game contexts per worker;
- central inference batch cap 64, 4 ms wait;
- CUDA, strict production mode;
- zero technical games required.

Full Arena artifacts are stored in the canonical cross-check location under `runs/torus9/evaluations/`. The lineage keeps the orchestrator Arena result and metrics; Arena never mutates or gates the training checkpoint implicitly.

The orchestrator pins fingerprints for both the periodic Arena preset and its deterministic start-set contract. A retry may reuse an already-complete matching evaluation, but a mismatched result fails closed.

## Crash and resume semantics

`generation-XX.complete.json`, written by `TrainingEngine`, is the authoritative training transaction marker.

If a process dies **before** that marker exists, the generation is uncommitted. Explicit orchestrator resume removes only known artifacts belonging to that current uncommitted generation and deterministically regenerates it from the previous committed checkpoint/replay state.

If a process dies **after** the marker exists but before the driver result is published, resume does not train the generation twice. The driver reloads and validates the committed checkpoint and replay, validates the persisted self-play artifact referenced by the committed summary, recreates the durable resume proof, and publishes the missing driver result.

Every completed driver result includes SHA-256 validation for the checkpoint, checkpoint metadata, fresh and rolling replay, training metrics, iteration summary, completion marker, self-play record and resume-state document.

The resume-state proof covers:

- model;
- optimizer;
- replay;
- generation number;
- deterministic RNG/seed schedule.

Technical self-play or Arena outcomes fail closed.

## Long-run protection and reporting

The Torus driver maintains a durable heartbeat while self-play, training, reload verification or Arena is active. The generic supervisor additionally monitors process liveness, heartbeat staleness, disk/RAM pressure, artifact integrity, performance and training-clock progress.

The production spec compares self-play speed against the validated Legion reference of 21.09873 moves/s. Below 85% is a warning; below 70% is fail-closed. Inference batch quality is also tracked against the validated mean batch reference.

Learning reports include policy/value/ownership/score/total losses, parameter delta, gradient norm, optimizer update clock, sample-consumption clock and self-play speed. A stalled optimizer/sample clock is fail-closed. Model-strength evidence remains the periodic Arena rather than loss alone.

## Current scope boundary

The current Torus integration creates a fresh M0 lineage. A generic `--parent-*` reference is rejected by the Torus driver at M1 because silently discarding or partially reconstructing an external optimizer/replay state would violate the training contract. Cross-lineage continuation needs its own explicit migration contract before it can be enabled.

Cube remains future work. The intended extension is another driver implementing the same generation/Arena result contracts after the Cube production engine is ready; it must not require Torus-specific branches in `gocube_golden.orchestrator`.
