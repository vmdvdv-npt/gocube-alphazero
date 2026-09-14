# One universal Arena engine

Date: 2026-09-14

## Rule

The repository has one production Arena execution engine for every supported
board/game:

```bash
.venv/bin/python tools/arena.py --candidate PATH_TO_CHECKPOINT.pt
```

There is no separate Torus9/Cube/Torus5 multiprocessing Arena. Game-specific
code is an Arena profile only.

The default `--profile auto` reads checkpoint metadata and resolves the profile.
For an explicit profile:

```bash
.venv/bin/python tools/arena.py \
  --profile torus9 \
  --candidate runs/.../M14.pt \
  --reference runs/.../M8.pt
```

When `--reference` is omitted, the candidate plays itself.

## Architecture boundary

`tools/arena_engine.py` owns only execution:

- OS worker process lifecycle;
- central request queue and shared-memory transport;
- cross-worker inference coalescing;
- one parent model owner / inference broker;
- batching cap and pre-forward wait;
- worker PID / CPU / queue / batch telemetry;
- fail-fast performance and validity gates;
- artifact publication.

It must not import a board, topology, scoring rule, checkpoint family, or
game-specific search implementation.

`tools/arena_profiles/<profile>.py` owns only game semantics:

- checkpoint/profile recognition;
- observation shape and policy/WDL sizes;
- model construction/loading;
- state/startset serialization;
- legal move/application semantics;
- search primitive and scientific search settings;
- formal termination/scoring;
- paired-start construction;
- result summarization;
- scientific contract.

Adding another game means adding a profile to `tools/arena_profiles/`. It must
not copy or fork the Arena engine.

## Torus 9×9 profile

Current production profile: `torus9`.

Its scientific contract remains unchanged:

- Torus 9×9;
- komi 0.5;
- 64 games / 32 paired starts with color swap;
- 64 simulations;
- cpuct 1.25;
- FPU 0;
- noise OFF;
- temperature 0;
- fast search OFF;
- resign OFF;
- watchdog 1000;
- deterministic tie-break;
- technical outcomes fail closed.

Current Legion execution defaults:

- 16 real OS CPU search workers;
- up to 4 active games per worker;
- one parent CUDA inference owner;
- global inference batch cap 64 rows;
- 4 ms real pre-forward coalescing window.

The cap/wait values remain benchmark-tunable execution parameters and are not
scientific semantics.

## Frozen model identity

Example frozen self A/B:

```bash
.venv/bin/python tools/arena.py \
  --candidate runs/.../M14.pt \
  --expected-candidate-model-hash sha256:15fcb2e70ebbf1ea250d91d578f18f8799cf529ad399d5d341f248c44e60540f \
  --expected-candidate-artifact-sha256 05cc0182670e2bda3f599c6138c7dc14507bdb3d149699d85501106be739bf93
```

## Hard Torus9 production gates

A production run fails closed if:

- games < 64;
- workers != 16;
- central inference is not CUDA;
- fewer than 16 distinct worker PIDs are observed;
- CUDA initializes inside a search worker;
- technical outcomes occur;
- mean inference batch rows < 16;
- aggregate worker CPU time corresponds to < 8 effective CPU cores.

For small unit/debug runs, `--debug-non-production` permits reduced CPU
execution. Such runs are explicitly not production evidence.

## Historical Arenas

Old Torus9 Arena executors remain frozen solely for reproduction. They are not
public production APIs and require `--allow-frozen-arena` through their
historical wrappers.

`Torus9BatchedPUCT` remains an internal Torus9 search primitive used by the
Torus9 profile. It is not an Arena executor.

## Non-regression rule

CI must enforce all of the following:

1. exactly one `production_allowed=True` Arena symbol;
2. that symbol is `tools.arena.run_arena`;
3. `tools/arena_engine.py` contains no Torus9/board-specific imports or symbols;
4. `tools/torus9_arena.py` does not exist;
5. historical executors remain fail-closed;
6. profile-specific modules do not spawn their own worker processes.

This makes another accidental "Arena v2 for one board" an explicit CI failure
rather than a silent architectural fork.
