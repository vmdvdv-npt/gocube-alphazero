# Torus 9×9 Arena — single production path

Date: 2026-09-14

## One production entry point

The only normal Torus9 Arena command is:

```bash
.venv/bin/python tools/torus9_arena.py --candidate PATH_TO_CHECKPOINT.pt
```

`--reference` defaults to the candidate checkpoint, so the shortest command is
already a self A/B test. For two different models:

```bash
.venv/bin/python tools/torus9_arena.py \
  --candidate runs/.../M14.pt \
  --reference runs/.../M8.pt
```

Production defaults are fixed to the Golden scientific contract and a safe
Legion execution preset:

- 64 games / 32 paired starts with color swap;
- 64 simulations, cpuct 1.25, FPU 0;
- noise OFF, temperature 0, fast OFF, resign OFF;
- watchdog 1000;
- komi 0.5;
- 16 real OS CPU search workers;
- 4 concurrent game lanes per worker;
- one parent CUDA inference owner;
- global inference batch cap 64 rows;
- 4 ms real coalescing window, applied before the model forward.

The execution preset is immediately usable but is not declared the final
performance optimum until the current 80×8 A/B benchmark confirms it.

## Identity lock

For frozen-model experiments, use the expected identity flags. Example:

```bash
.venv/bin/python tools/torus9_arena.py \
  --candidate runs/.../M14.pt \
  --expected-candidate-model-hash sha256:15fcb2e70ebbf1ea250d91d578f18f8799cf529ad399d5d341f248c44e60540f \
  --expected-candidate-artifact-sha256 05cc0182670e2bda3f599c6138c7dc14507bdb3d149699d85501106be739bf93
```

When `--reference` is omitted the same checkpoint is used on both sides.

## Hard validity gates

A production run fails closed if any of these conditions is violated:

- fewer than 64 games;
- workers != 16;
- central device is not CUDA;
- fewer than 16 distinct worker PIDs are observed;
- CUDA initializes inside a search worker;
- technical outcomes occur;
- mean inference batch rows < 16;
- aggregate worker CPU time corresponds to < 8 effective CPU cores.

The summary always records worker PIDs, parent/model-owner PID, aggregate CPU,
effective CPU cores, batch distribution, cross-worker batch rate, queue wait,
cap-hit rate, model hashes, artifact hashes, and end-to-end throughput.

For unit/debug work only, `--debug-non-production` permits reduced CPU runs.
Those results are explicitly non-production and cannot satisfy the Golden gate.

## Frozen historical Arenas

The old Torus9 executors are removed from the public `gocube_golden` API and
direct calls are guarded. Historical operational wrappers require the explicit
`--allow-frozen-arena` flag. They must never be used as current Golden evidence.

The internal `Torus9BatchedPUCT` class remains only as a deterministic local
search primitive used inside each OS worker. It is not a production Arena
executor and is not exported by `gocube_golden`.

## Architecture

```text
16 OS worker processes
  └─ CPU selection / expansion / backup
     └─ up to 4 active games per worker
        └─ shared-memory inference requests
                    ↓
one parent inference broker / model owner
  └─ collect requests from multiple worker PIDs for up to 4 ms
  └─ one CUDA forward per coalesced model batch
  └─ shared-memory policy/WDL responses
                    ↓
workers continue search
```

This restores the proven process-parallel design principle from PR #83 and the
historical central-batching Arena, while avoiding CUDA model copies in worker
processes.
