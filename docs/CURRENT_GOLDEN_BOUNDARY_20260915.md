# Historical Golden boundary — Torus 9×9 M0→M17 + Self-play Stage 2 (2026-09-15)

> Superseded by `GOLDEN_LEGACY_REMOVAL_STAGE6_20260915.md`. This dated
> document is retained as scientific lineage evidence; its pre-Stage-6
> compatibility names and paths are not current launch instructions.

This document preserves the immutable scientific/dependency boundary proven by the successful Torus 9×9 line `M0→M17` and records the later Stage-2 extraction of **current self-play execution**. The extraction changes execution architecture only; it does not redefine rules, search semantics, targets, replay, training, the network, checkpoint identity, Arena semantics or the M0→M17 lineage.

## 1. Immutable reference

| Item | Immutable value |
|---|---|
| Reference run | `torus9-golden-v3-20260914-run03` |
| PR #93 merge commit | `152be924db12530ff04b37266925838da191e948` |
| PR #93 head | `2e70a3a3e6a64e82fa784d6760df9c51283b2e65` |
| Common Git tree | `d4b438ff6b8442aae03ef16cafacbad4ad7ece14` |
| M1 baseline run Git SHA | `d54de530fc4026d22409ca017f329f88a920dec2` |
| Original base commit | `53946d0c84fca5a6f81a387bfd399ea62e34b088` |
| PR #100 / Stage-2 base | `1f80db26cfb8f46a7661cc13383cead65918a5f7` |
| Semantic/lineage profile fingerprint | `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775` |
| Sanitized current JSON content fingerprint | `sha256:7e97c50e1697641fb8f5b9a3566144f0a58c105e3b688940f42e7b6154fb0831` |
| Last clean resumable boundary | M17 |
| M18 | never started |

The immutable run report remains `docs/TORUS9_GOLDEN_CURRENT_V3_M1_M17_NIGHT_RUN_20260914.md`. The physical namespace is `runs/torus9-golden-v3-active/torus9-golden-v3-20260914-run03` on Legion and remains read-only for parity/benchmark work. No M18 is created by Stage 2.

The two fingerprints retain their original distinct roles. `36911…` is the semantic/lineage identity stored in M0→M17 metadata and required by the continuation runtime. `7e97…` is the sanitized profile content-integrity fingerprint. Stage 2 changes neither.

## 2. Current production ownership

| Purpose | Entrypoint / boundary | Ownership |
|---|---|---|
| Torus9 continuation front door | `tools/continue_torus9_golden_m1_m100.py` | compatibility/front door |
| Preserved M1→M100 runtime | `tools/_frozen_continue_torus9_golden_m1_m100.py` | frozen orchestration/training compatibility |
| Current Torus9 public library facade | `gocube_golden/torus9.py` | stable imports + current self-play front door |
| Universal self-play engine | `gocube_golden/selfplay_engine.py` | **EXECUTION** |
| Torus9 self-play adapter | `gocube_golden/torus9_selfplay.py` | **SCIENTIFIC/profile adapter** |
| Preserved Torus9 scientific/replay/training implementation | `gocube_golden/torus9_monolith.py` | scientific + unchanged downstream training/replay/checkpoint code |
| Standalone Arena CLI | `tools/arena.py` | unchanged |
| Standalone Arena engine | `tools/arena_engine.py` | unchanged |
| Torus9 Arena adapter/profile | `tools/arena_profiles/torus9.py` | unchanged |
| GoCube product Protocol V1 | `python -m alphazero.envs.gocube.integration.server` | unchanged |

`gocube_golden/torus9.py` intentionally re-exports the established Torus9 library surface from the preserved implementation and overrides only `run_torus9_selfplay_games(...)` with the Stage-2 engine adapter. There is therefore one current public self-play route.

## 3. Dependency graph

Before Stage 2, current-profile self-play was mixed inside `gocube_golden/torus9.py`:

```text
run_torus9_selfplay_games(...)
  -> ThreadPoolExecutor lanes                    EXECUTION
  -> Torus9InferenceCoordinator/thread/Queue     EXECUTION
  -> Torus9SelfPlayRunner                        SCIENTIFIC
  -> Torus9 search/rules/observation/records     SCIENTIFIC
  -> model evaluator                             MIXED
```

After Stage 2:

```text
tools/continue_torus9_golden_m1_m100.py
  -> tools._frozen_continue_torus9_golden_m1_m100
  -> gocube_golden.torus9                         public facade
       -> gocube_golden.torus9_selfplay           Torus9 scientific adapter
            -> gocube_golden.selfplay_engine      process/IPC/batching execution
            -> gocube_golden.torus9_monolith      unchanged game/search/model/record semantics
       -> gocube_golden.torus9_monolith            unchanged replay/training/checkpoint downstream
```

The new engine does not import `alphazero.Coach`, `alphazero.SelfPlayAgent`, `alphazero.NNetWrapper`, Torus9 rules, topology, board dimensions, action count, Torus9 model classes or MCTS.

The pre-extraction `run_torus9_selfplay_games` implementation remains physically present only in the preserved monolith for source/reference compatibility. It is not the public current self-play front door and is not used by the current continuation import path.

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

**Replay/training — unchanged downstream**

- rolling last 3 generations, cap 20,000, deterministic sampling;
- Adam, LR 0.001, weight decay 0;
- batch 64, 80 optimizer steps/iteration, 5,120 samples/iteration;
- LR scheduler none, model gating false;
- ownership loss ON, score loss ON.

## 5. Self-play execution boundary

The universal `SelfPlayEngine` owns only:

- worker process creation/termination;
- task scheduling and canonical game ordering;
- IPC request/response transport;
- one central inference service;
- batching/coalescing, batch cap and batch wait;
- timeouts and fail-closed error propagation;
- clean shutdown;
- execution telemetry.

`Torus9SelfPlayAdapter` owns:

- the Torus9 scientific profile identity;
- canonical `6×81` observation building;
- current 80×8 model/head interpretation;
- the unchanged Torus9 game/search runner;
- search contract, root noise, temperature and watchdog semantics;
- Torus9 record/provenance validation;
- canonical game seed derivation.

CUDA production shape:

```text
16 OS search workers
        -> observation IPC
        -> one parent CUDA inference owner
        -> batched forward
        -> per-worker response IPC
```

Workers do not own CUDA model copies. CUDA uses `spawn`; CPU can use `fork` where available.

## 6. Reproducibility and result ordering

Scientific RNG is scheduling-independent. The canonical game seed remains:

```text
derive_seed(master_seed, run_id, game_id, "game")
```

PID, worker number, queue timing, completion order and inference batch composition do not enter the seed. Search seeds remain derived by the unchanged Torus9 runner from game seed + ply.

The engine sorts game IDs before scheduling and returns records in that same canonical order regardless of completion order.

A focused CPU/single-lane parity test compares the complete game record against the pre-extraction `Torus9SelfPlayRunner` under the same fixed model, seeds and scientific contract.

## 7. Fail-closed execution

The complete self-play batch fails on:

- worker exception;
- abnormal worker process death;
- central inference exception;
- malformed request/response;
- batch response count mismatch;
- inference timeout;
- incomplete result set;
- duplicate completion;
- unclean shutdown.

Inference transport failures are infrastructure failures, not scientific game outcomes; they intentionally escape the game runner's `except Exception` conversion and invalidate the batch. No partial iteration is returned as valid.

## 8. Execution telemetry and Legion gates

Standalone self-play telemetry includes:

- configured workers, concrete worker PIDs and PID count;
- peak concurrent search workers;
- child process CPU seconds and effective CPU cores;
- worker failures/restarts;
- games requested/completed/failed/technical;
- moves, wall time, games/sec and moves/sec;
- inference requests/forwards/rows;
- batch rows mean/p50/p95/max;
- rows/sec and forwards/sec;
- batch cap, wait, device and central inference owner PID.

The Torus9 compatibility layer also preserves the telemetry keys consumed by the frozen continuation runtime.

Legion production validation remains:

- workers = 16;
- CUDA inference;
- central batching ON;
- mean inference batch rows >=16 when workload can form such batches;
- effective CPU cores >=8 on a sufficiently long smoke;
- no technical/infrastructure failures.

If mean inference batch rows is below 16, the run is performance-degraded; do not repeat a long benchmark with identical settings.

Reference-vs-new benchmarking must use M17 read-only, a separate temporary namespace and identical checkpoint, game IDs, seeds, simulations, device, rules and komi 0.5. The reference checkout is `152be924db12530ff04b37266925838da191e948` or equivalent PR #93 tree.

## 9. Checkpoint/resume identity

M17 remains the last clean resumable boundary. Its checkpoint, replay, optimizer state and continuation state live on Legion and are not GitHub repository contents.

Stage 2 does not create a new checkpoint identity:

- `profile_fingerprint` remains `36911…`;
- sanitized profile content fingerprint remains `7e97…`;
- scientific sub-contract fingerprints remain unchanged;
- the continuation path keeps its established call shape and telemetry compatibility;
- replay/training/checkpoint downstream code remains unchanged.

This is **not** an M18 migration.

## 10. Arena and GoCube boundaries

The standalone Arena remains a separate subsystem in `tools/arena_engine.py` / `tools/arena.py`; Stage 2 does not redesign Arena.

Protocol V1 remains stable. Existing migration blockers around `NNetWrapper`, historical descriptors and `GenericPlayers.MCTSPlayer` are unchanged and remain deferred.

## 11. Legacy/deferred inventory

### KEEP CURRENT

- current Torus9 scientific profile/contract;
- `gocube_golden/selfplay_engine.py` execution boundary;
- `gocube_golden/torus9_selfplay.py` Torus9 adapter;
- `gocube_golden/torus9.py` public facade;
- shared Golden topology/state/rules/search/neural/provenance;
- continuation front door/frozen runtime;
- standalone Arena + Torus9 Arena adapter;
- Protocol V1 integration surface.

### KEEP TEMPORARILY

- `gocube_golden/torus9_monolith.py`, because training/replay/checkpoint extraction is explicitly deferred;
- Torus9 Arena duplicate scientific literals guarded by parity tests;
- `alphazero/NNetWrapper.py`, `GenericPlayers.py` and legacy Cython MCTS for Protocol V1 migration compatibility.

### DEFERRED

- training extraction;
- Cube self-play adapter and GoCube integration migration;
- legacy deletion;
- Arena redesign;
- MCTS algorithm rewrite;
- network/optimizer/LR/replay-policy changes;
- long training;
- M17→M18 continuation.

## 12. Boundary conclusion

`M0→M17` remains reproducibly identifiable by its immutable Git/run provenance and original scientific lineage fingerprint. Stage 2 changes only current self-play execution ownership: OS search workers now communicate with one central inference owner through a generic engine, while Torus9 scientific semantics and existing training remain downstream and unchanged.

**Scientific semantics changed: NO.**

**Training semantics changed: NO.**

**Self-play execution architecture changed: YES.**
