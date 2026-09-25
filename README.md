# GoCube Golden runtime

This repository contains the supported GoCube Golden production path:

```text
Golden rules/state/search
        ↓
SelfPlayEngine
        ↓
TrainingEngine
        ↓
Golden .pt checkpoint + metadata sidecar
        ↓
standalone Arena / GoCube Protocol V1
```

Torus9 and Cube use profile-specific scientific adapters over the shared
`SelfPlayEngine` and `TrainingEngine`. Search is `SequentialPUCT` (or its
incremental `SequentialPUCTSession`). The standalone Arena CLI is
`tools/arena.py`.

## Torus9 `new_komi` transition

The current Torus9 transition is documented in
`docs/TORUS9_GOLDEN_BEST.md`.

`new_komi` is a fresh-history lineage bootstrapped from canonical M137 after
the verified 6CH -> 5CH model conversion. Adam state is preserved everywhere
with an unambiguous mapping; only the folded input bias moments are reset.
Parent replay/history is not inherited.

The bootstrap does not choose a new rules komi. Training remains blocked until
a fresh 5-channel komi calibration selects it.

## GoCube Protocol V1

Run the service with the Golden catalog:

```bash
.venv/bin/python -m alphazero.envs.gocube.integration.server \
  --checkpoint-dir runs --host 127.0.0.1 --port 8765
```

The stable endpoints are `GET /v1/health`, `GET /v1/checkpoints`, and
`POST /v1/games`. The catalog accepts only a Golden `.pt` artifact with a
matching `.metadata.json` sidecar. Unsupported artifacts, including legacy
pickle files, are ignored by the catalog and are never loaded as a fallback.

The immutable Torus9 `torus9-golden-v3-20260914-run03/M17` artifact and the
current Cube Golden checkpoint remain compatible with this loader and
Protocol V1.

## Verification

Focused local checks use the repository virtual environment:

```bash
.venv/bin/python -m pytest -q \
  tests/test_stage6_cleanup.py \
  tests/test_gocube_golden_protocol_stage4.py \
  tests/test_current_golden_boundary.py \
  tests/test_selfplay_engine.py \
  tests/test_selfplay_engine_shared_memory.py \
  tests/test_training_engine_stage3.py
```

KataGo differential tests are authoritative CI-only checks and are intentionally
not run as part of the local smoke policy.
