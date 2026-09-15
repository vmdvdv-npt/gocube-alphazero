# Stage 6 — физическое удаление legacy и Golden-only production path

Дата: 2026-09-15
Base: `0d341d1a24aa4668bf5abd3dca96dd947f3c7f33`
Branch: `codex/stage6-remove-legacy`

## Результат

После Stage 6 production topology одна:

```text
Golden rules/state/search
        ↓
Torus9SelfPlayAdapter / CubeSelfPlayAdapter
        ↓
SelfPlayEngine
        ↓
Torus9TrainingAdapter / CubeTrainingAdapter
        ↓
TrainingEngine
        ↓
Golden .pt + metadata sidecar
        ↓
standalone Arena / GoCube Protocol V1
```

В репозитории больше нет поддерживаемого второго AlphaZero execution path.
Оставленные исторические документы и pinned KataGo oracle не являются
production backends.

## Before / after

До Stage 6 одновременно присутствовали Golden adapters и старые generic
Coach/self-play/MCTS/checkpoint/Arena ветки, включая временную загрузку
legacy `.pkl`.  После cleanup runtime и integration boundary принимают только
текущую Golden identity. Неподдерживаемый формат завершается обычной ошибкой,
без fallback и угадывания формата.

## Explicit audit table

| Component | Before | After |
|---|---|---|
| Self-play engine | Golden engine плюс старые worker/game loops | `SelfPlayEngine` only |
| Training orchestration | Golden engine плюс старый orchestration | `TrainingEngine` only |
| Search | Golden PUCT плюс legacy MCTS/batched variants | `SequentialPUCT` / `SequentialPUCTSession` |
| Checkpoint serving | `.pt` и временный `.pkl` путь | Golden `.pt` + metadata sidecar only |
| GoCube generation | conditional legacy/Golden backends | `GoldenCheckpointLoader` → `GoldenGameGenerator` |
| Arena | несколько исторических execution paths | один standalone `tools/arena.py` + profile adapters |
| Generic wrapper/player runtime | generic wrapper/player modules | physically removed |
| Coach runtime | old training/gating entrypoint | physically removed |
| Self-play agent runtime | old Cython worker entrypoint | physically removed |
| Legacy configs/CLI | old presets and obsolete commands | current Torus9/Cube Golden commands only |

## Удалено

Удалены старые runtime families и их adapters:

- generic core: `Coach`, `Arena.pyx`, `Evaluator`, `Game`, generic players,
  legacy `MCTS`, `NNetWrapper`, `SelfPlayAgent`, Cython game/runtime files,
  old inference batching, old search contract, pit/round-robin helpers and
  bookkeeping/worker shims;
- GoCube legacy integration: legacy contract/manifest/model/generation,
  registration and dev-launcher modules;
- old Golden orchestration: old training, Arena/process-Arena, player,
  replay/serialization, experiment-profile, Stage 3/4 and standard/demo
  modules;
- old Torus/Cube experiment entrypoints, transfer/serial Arena CLIs, obsolete
  Torus5 and historical Stage 1–4 configs;
- legacy-only tests for the removed paths and unrelated non-GoCube generic
  environment/GUI entrypoints that could expose a second runtime.

The standalone Arena was retained and its Torus9 profile now calls the common
`SequentialPUCT` path. Cube self-play retains only the shared cooperative
engine; the former serial self-play parity runner was removed.

## Checkpoint and Protocol V1 boundary

`CheckpointCatalog` discovers only `M<integer>.pt` files with a valid Golden
metadata sidecar. `.pkl` files are neither listed nor loaded. Loader metadata
is checked against the exact current profile, architecture, topology, rules,
observation, target, model hash and artifact identity. There is no
`try-golden-then-legacy` fallback.

The existing Protocol V1 endpoints and request/response shapes remain:

```text
GET  /v1/health
GET  /v1/checkpoints
POST /v1/games
```

Only the catalog capability changed: legacy checkpoints no longer appear.
The immutable Torus9 `torus9-golden-v3-20260914-run03 / M17` compatibility
test remains read-only when the artifact is present. The current Cube Golden
checkpoint fixture remains covered by the Golden loader path.

## Intentionally retained

- `gocube_golden` rules, topology, state, scoring, observation, target, PUCT,
  root-noise, replay and network primitives; Stage 6 does not change their
  scientific semantics;
- Torus9 and Cube adapters, `SelfPlayEngine`, `TrainingEngine`, the universal
  Arena engine and Protocol V1 server;
- `alphazero/envs/gocube/katago_v3.py`, its pinned reference tests and the
  separate `katago_reference` CI job. This is a rule oracle, not a playable
  production backend and is intentionally not run locally;
- immutable checkpoint/replay evidence and historical reports under `docs/`.
  Historical references to removed names are archive evidence, not imports;
- small numerical/reference utilities used for scientific verification, where
  they do not provide a production execution route.

## Current execution paths

```text
Torus9 self-play:  Torus9SelfPlayAdapter → SelfPlayEngine
Cube self-play:    CubeSelfPlayAdapter → SelfPlayEngine
Torus9 training:   Torus9TrainingAdapter → TrainingEngine
Cube training:     CubeTrainingAdapter → TrainingEngine
Search:            SequentialPUCT / SequentialPUCTSession
Arena:             standalone universal Arena
Serving:           Golden .pt → Golden loader → Golden rules/search → V1
```

## Guards and verification

Focused guards cover:

- physically absent generic runtime entrypoints and deleted Golden legacy
  orchestration files;
- source-level forbidden imports and obsolete checkpoint/config switches;
- `.pkl` hidden/unsupported behavior;
- current Torus9 profile, architecture, fingerprints and `0.5` komi;
- current M17 loader/Protocol V1 generation when the immutable artifact is
  available;
- Cube loader-facing metadata, shared self-play rails and TrainingEngine
  transaction;
- one universal Arena and canonical Sequential PUCT;
- self-play determinism, replay, rules, scoring, pass and optimizer behavior.
- tracked-tree hygiene: no `.pyc`, `.pyo`, `__pycache__` or retired project
  directories;

Current/runtime/config/test source contains no literal `7.5`; the invalid-komi
guard uses an equivalent numeric expression so the stale value cannot be
silently accepted. Canonical current komi remains `0.5`.

## Size / removal report

Counts are measured against the final Stage 6 plus cleanup-fixup worktree diff:

```text
files deleted:                         362
files substantially simplified:        24
lines removed:                       81,487
legacy production entrypoints removed: primary generic/legacy families listed above
legacy tests removed:                  109
```

The large deletion count includes obsolete generic environments/GUI modules,
old configs, tools and tests; it is not a goal by itself. The goal is that
there is no selectable old production pipeline.

## Testing policy

Local verification is focused only. Full local `pytest` is not run, pinned
KataGo differential tests are not run locally, and no long training or long
Arena run is started. CI has two explicit jobs:

1. `Golden production tests` runs `pytest -m "not katago_reference"` and
   current compile/entrypoint checks.
2. `Mandatory pinned KataGo rule differential` builds the pinned rule-only
   oracle and runs only the four explicit `katago_reference` test modules:
   random, rectangular fixtures, topology bridge and topology symmetry.

The Torus9 M17 artifact, current Cube Golden artifacts and historical evidence
were not mutated; no M18 was created.
