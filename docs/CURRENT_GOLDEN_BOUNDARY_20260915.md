# Current Golden boundary — Torus 9×9 M0→M17 (2026-09-15)

This document freezes the dependency and scientific boundary proven by the successful Torus 9×9 line `M0→M17`. It is an audit/ownership document, not a redesign of self-play, training, MCTS, the network, or Arena.

## 1. Immutable reference

| Item | Immutable value |
|---|---|
| Reference run | `torus9-golden-v3-20260914-run03` |
| PR #93 merge commit | `152be924db12530ff04b37266925838da191e948` |
| PR #93 head | `2e70a3a3e6a64e82fa784d6760df9c51283b2e65` |
| Common Git tree | `d4b438ff6b8442aae03ef16cafacbad4ad7ece14` |
| M1 baseline run Git SHA | `d54de530fc4026d22409ca017f329f88a920dec2` |
| Original base commit | `53946d0c84fca5a6f81a387bfd399ea62e34b088` |
| Semantic/lineage profile fingerprint | `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775` |
| Sanitized current JSON content fingerprint | `sha256:7e97c50e1697641fb8f5b9a3566144f0a58c105e3b688940f42e7b6154fb0831` |
| Last clean resumable boundary | M17 |
| M18 | never started |

The run report is `docs/TORUS9_GOLDEN_CURRENT_V3_M1_M17_NIGHT_RUN_20260914.md`. The external run namespace is `runs/torus9-golden-v3-active/torus9-golden-v3-20260914-run03` on Legion. M2…M17 completed with 64 games/iteration and `technical=0`.

The two fingerprints have deliberately different roles. `36911…` remains the immutable **semantic/lineage identity** stored in the M0→M17 manifest/checkpoint metadata and expected by the continuation runtime. It must remain stable because no scientific contract changed. `7e97…` is a separate **content-integrity fingerprint** for the sanitized current JSON, calculated with `profile_fingerprint` and `content_fingerprint` excluded. This lets the repository remove forbidden historical sentinel metadata without invalidating M17 resume identity.

No topology, rules, observation, target, network, search, replay, training, seed, checkpoint, or Arena scientific semantics changed.

## 2. Production and preservation entrypoints

| Purpose | Entrypoint / boundary | Status |
|---|---|---|
| Torus9 continuation front door | `tools/continue_torus9_golden_m1_m100.py` | KEEP CURRENT as fail-closed compatibility/front-door boundary |
| Preserved M1→M100 runtime | `tools/_frozen_continue_torus9_golden_m1_m100.py` | KEEP CURRENT for M0→M17 lineage/resume compatibility; frozen from redesign |
| Standalone Arena CLI | `tools/arena.py` | KEEP CURRENT |
| Standalone Arena engine | `tools/arena_engine.py` | KEEP CURRENT |
| Torus9 Arena adapter/profile | `tools/arena_profiles/torus9.py` | KEEP CURRENT; duplicated scientific literals guarded against canonical profile |
| Torus9 checkpoint/replay/training library boundary | `gocube_golden/torus9.py` | KEEP CURRENT |
| GoCube product Protocol V1 service | `python -m alphazero.envs.gocube.integration.server` | KEEP CURRENT |

The continuation front door resolves the preserved runtime through the module identity `tools._frozen_continue_torus9_golden_m1_m100` and remains fail-closed around the historical Arena path. The preservation requirement in this PR is that the M17 lineage can still be validated/resumed under its original semantic fingerprint; it does not promote old Arena execution to current production Arena.

## 3. Proven M0→M17 dependency graph

```text
tools/continue_torus9_golden_m1_m100.py
  -> tools._frozen_continue_torus9_golden_m1_m100
  -> configs/gocube/torus9_golden_current_v3.json
  -> gocube_golden.torus9_contract.load_torus9_current_profile()
  -> gocube_golden.torus9
       -> gocube_golden.topology
       -> gocube_golden.state
       -> gocube_golden.rules
       -> gocube_golden.search
       -> gocube_golden.neural
       -> self-play/inference coordination
       -> replay/training
       -> checkpoint + optimizer state
       -> resume loaders
```

The preserved runtime does **not** route M0→M17 through `alphazero.Coach`, `alphazero.SelfPlayAgent`, or `alphazero.NNetWrapper`.

## 4. Scientific contract frozen to M0→M17

**Rules and topology**

- Torus 9×9, 81 points, row-major IDs, wrap X/Y;
- exact graph-area scoring;
- positional superko;
- suicide forbidden;
- two-pass termination;
- Benson automatic ending OFF;
- komi **0.5 only**.

**Observation and targets**

- observation `6×81`;
- action count 82; pass index 81;
- observation fingerprint `sha256:e5792b409199dfe2c25ac6f681e4ca29ed73cdf4b7d53a61df634f70d1fa415f`;
- W/D/L side-to-move target;
- policy target from root visits before the chosen move;
- ownership and score auxiliary targets ON;
- technical outcomes excluded;
- target fingerprint `sha256:02ab244688534b271473302ab4edf00516b91d43fb91b8c9e592d2a8de63dfb5`.

**Network**

- `GoldenGraphNetV2-Torus9`;
- hidden 80, blocks 8, input channels 6;
- heads policy `[82]`, value `[3]`, ownership `[81,3]`, score `[1]`;
- explicit symmetry augmentation OFF.

**Self-play/search**

- 64 games/iteration;
- 64 simulations, cpuct 1.25, FPU 0;
- root noise ON, epsilon 0.25, alpha 0.11;
- temperature 1.0 on plies 1–8, then 0;
- fast search OFF, resign OFF, watchdog 500;
- self-play fingerprint `sha256:22a4e4dd37d70bd3d712b909476120b96385802ec874b99358fab256c4e3351f`.

**Replay/training**

- rolling last 3 generations, cap 20,000, deterministic sampling;
- Adam, LR 0.001, weight decay 0;
- batch 64, 80 optimizer steps/iteration, 5,120 samples/iteration;
- LR scheduler none, model gating false;
- ownership loss ON, score loss ON.

## 5. Checkpoint/resume identity

M17 is the last clean resumable boundary. The physical M17 checkpoint, replay state, optimizer state and continuation state live on Legion rather than in Git, so GitHub CI cannot honestly reload those bytes.

The repository can and does protect the compatibility contract:

- `profile_fingerprint` remains `36911…`, matching M1→M17 checkpoint/manifest metadata and the preserved runtime expectation;
- the sanitized profile file is independently protected by `content_fingerprint=7e97…`;
- the loader validates both identities fail-closed;
- a focused regression test calls the preserved runtime's profile resolution and proves it still resolves the M17 lineage fingerprint;
- scientific sub-contract fingerprints remain unchanged.

This is intentionally **not** an M18 fingerprint migration. No new checkpoint identity is introduced merely because forbidden non-semantic metadata was removed.

## 6. Execution mechanics are not scientific semantics

The M0→M17 run was scientifically valid but performance-degraded. Historical self-play telemetry showed roughly one CPU core-equivalent despite `workers=16`, with mean inference batches commonly around 9–10 rows. Those measurements are execution debt, not a reason to alter the scientific contract in this PR.

The current standalone Arena is a separate later boundary. Its execution mechanics are authoritative in `tools/arena_engine.py` / `tools/arena.py`, not in the historical Arena execution fields retained in the M0→M17 profile. Current production Arena defaults/gates include:

- at least 64 games;
- 16 OS search workers;
- 4 active games per worker;
- one parent CUDA inference owner;
- central batch cap 64;
- central wait 1 ms;
- worker-local wait 0;
- mean inference batch rows >=16;
- effective worker CPU cores >=8;
- technical outcomes fail closed.

No process/thread architecture or throughput optimization is changed here.

## 7. Canonical source of truth

The only current Torus9 scientific profile is:

`configs/gocube/torus9_golden_current_v3.json`

The profile owns topology/rules, observation/targets, 80×8 network shape, self-play semantics, replay/training budgets and seeds. `gocube_golden/torus9_contract.py` validates it fail-closed and separates semantic lineage identity from file-content integrity.

Arena scientific semantics are canonical in that same profile. `tools/arena_profiles/torus9.py` still repeats several fixed scientific literals; this is a temporary duplicate protected by parity tests until a narrow direct-profile-resolution migration removes it. Arena **execution** defaults remain owned by the universal engine/CLI.

## 8. Legacy inventory

### KEEP CURRENT / SHARED

- current Torus9 profile and contract;
- `gocube_golden/torus9.py`;
- shared Golden topology/state/rules/search/neural/provenance helpers;
- continuation front door and preserved runtime needed for M17 lineage compatibility;
- universal standalone Arena and Torus9 adapter;
- Protocol V1 integration surface.

### KEEP TEMPORARILY FOR MIGRATION

- Torus9 Arena duplicate scientific literals, guarded by parity tests;
- `alphazero/NNetWrapper.py`, because Protocol V1 model loading still depends on it;
- `alphazero/GenericPlayers.py` / legacy Cython MCTS, because Protocol V1 game generation still uses `MCTSPlayer`;
- loader/migration compatibility needed for historical GoCube descriptors.

### LEGACY — REMOVE IN LATER STAGE

- `alphazero/Coach.py` for the current Golden line;
- current-line dependence on `SelfPlayAgent` once broad legacy CI/build obligations are removed;
- retired Torus9 learning/ownership experiment profiles;
- superseded Torus training/demo/calibration entrypoints;
- superseded checkpoint-Arena/gating runners;
- historical one-off diagnostics/docs that still describe retired paths as current.

Unknown historical fixtures referenced by tests must stay UNKNOWN until their remaining consumer is proven absent.

## 9. GoCube product boundary

Protocol V1 remains a stable frontend/API boundary. Its backend still has explicit migration blockers:

1. `integration/models.py` imports `alphazero.NNetWrapper` and loads historical `.pkl` checkpoints;
2. legacy model/game metadata is still resolved for old descriptors;
3. `integration/generation.py` creates `GenericPlayers.MCTSPlayer`.

The next GoCube migration should replace those backend adapters while preserving Protocol V1 endpoints/schema.

## 10. Regression guards

Focused tests lock:

- exactly one Torus9 profile with `status=current`;
- semantic/lineage fingerprint `36911…` and sanitized content fingerprint `7e97…` as distinct identities;
- M17 continuation runtime resolves the original lineage identity;
- content tampering fails closed;
- komi 0.5 and rejection of noncanonical explicit komi;
- 80×8 network and exact heads;
- observation, target and self-play fingerprints;
- self-play/replay/training budgets;
- retired profiles absent from continuation selection;
- continuation path does not depend on `Coach`, `SelfPlayAgent` or `NNetWrapper`;
- Arena adapter scientific semantics remain equal to the canonical profile;
- the forbidden historical komi literal is absent from the current path.

Repository CI continues to run full pytest, smoke suites and the pinned KataGo differential. No long training run is added.

## 11. Boundary conclusion

`M0→M17` remains reproducibly identifiable by immutable Git/run provenance and its original semantic lineage fingerprint. The current profile is sanitized without changing that checkpoint/resume identity, and its sanitized bytes are protected by a separate content fingerprint. Training/resume lineage, standalone Arena and GoCube product boundaries are explicitly separated.

**Scientific semantics changed: NO.**

**Execution architecture changed: NO.**
