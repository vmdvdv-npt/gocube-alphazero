# GoCube Golden production path

This is the current execution map after Stage 6. There is one supported
production path:

```text
Golden rules/state/search
        ↓
SelfPlayEngine
        ↓
TrainingEngine
        ↓
Golden .pt + metadata sidecar
        ↓
standalone Arena / GoCube Protocol V1
```

## Current entrypoints

Torus9 learning uses:

```bash
.venv/bin/python tools/torus9_golden_learning.py
```

The current Cube profile is exposed through `gocube_golden.cube_training` and
`gocube_golden.cube_training_adapter`; its self-play adapter and training
adapter both terminate at the shared engines.

The Protocol V1 service is:

```bash
.venv/bin/python -m alphazero.envs.gocube.integration.server \
  --checkpoint-dir runs --host 127.0.0.1 --port 8765
```

The stable endpoints are `GET /v1/health`, `GET /v1/checkpoints`, and
`POST /v1/games`. Only Golden `.pt` checkpoints with a matching metadata
sidecar are listed or loaded. Unsupported artifacts fail closed.

## Scientific boundary

Current profiles keep the frozen rules, topology, observations, targets,
replay policy, optimizer, network architecture, deterministic seeds and
`komi=0.5`. Search is `SequentialPUCT` / `SequentialPUCTSession`.

The standalone Arena CLI is `tools/arena.py`; its profile adapters do not
provide another search or checkpoint execution implementation.

## Historical material

Earlier GoCube path maps, experiment reports and migration evidence remain in
Git history and dated `docs/` reports. They are archival references only and
must not be used as current launch commands. The pinned KataGo rule oracle is
also retained as a separate CI-only differential reference, not as a Golden
production backend.
