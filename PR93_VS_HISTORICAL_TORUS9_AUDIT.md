# PR93 vs Historical Torus9 Audit

Date: 2026-09-14

Audited post-merge `main`: `152be924db12530ff04b37266925838da191e948` (PR #93 merged 2026-09-14 05:53:44Z)

Historical comparison anchor: pre-PR93 base `53946d0c84fca5a6f81a387bfd399ea62e34b088`, plus the older proven Legion multiprocessing/central-inference path and the Golden Torus9 PR sequence.

## Executive conclusion

PR93 preserves the current Torus9 scientific rules, target semantics, 80x8 network contract, fixed training clock, deterministic replay, optimizer continuation, most resume invariants, and deterministic Arena search semantics. It does **not** preserve the previously proven performance architecture.

The current coalesced self-play path uses `ThreadPoolExecutor` lanes around Python MCTS. The only Torus9 `ProcessPoolExecutor` path is the non-coalesced legacy path, where every worker loads its own CUDA model and therefore does not provide the required central cross-process GPU broker. Because the coalesced path runs MCTS in Python threads, `workers=16` is not evidence of 16-core CPU execution and is exposed to the GIL.

The current batched Arena similarly uses logical in-process lanes in `Torus9BatchedPUCT`; it does not preserve PR83's proven OS-process Arena execution. PR83 had bit-exact 480/480 parity, effective parallelism 13.492/16 and 4.372x speedup at 16 workers.

M18 must remain blocked until process-level MCTS plus central GPU batching is restored and passes the real-Legion regression gate.

## Evidence sources

Primary implementation sources:

- `gocube_golden/torus9.py`
- `gocube_golden/torus9_contract.py`
- `configs/gocube/torus9_golden_current_v3.json`
- `tools/continue_torus9_golden_m1_m100.py`
- `tests/contracts/test_torus9_current_golden.py`
- pre-PR93 `alphazero/envs/gocube/training_common.py`
- pre-PR93 `alphazero/inference_batching.py`
- pre-PR93 `alphazero/SelfPlayAgent.pyx`

Historical implementation/proof chain includes PRs #12, #38, #45-#48, #64-#70, #72-#80, #82-#90, #92 and #93.

## 1. Self-play architecture

### Current PR93 coalesced path

`run_torus9_selfplay_games(..., coalescing=True)` creates a single `Torus9InferenceCoordinator` and runs games using `ThreadPoolExecutor(max_workers=workers)`. Every game lane executes `Torus9SelfPlayRunner` / Python MCTS and calls the coordinator for neural evaluation.

Inference is central in the process and serialized through the coordinator's model-forward lock. This is scientifically safe but does not provide OS-process MCTS parallelism. The Python selection/expansion/backup loops are not a GIL-free C/Cython search engine, so 16 threads cannot be treated as 16 effective CPU workers.

The coordinator lock protects the model forward, not the complete MCTS path. Its queue is a `queue.Queue`; each lane submits one prepared state request and blocks on an event for the result. The server thread coalesces requests until the configured deadline or cap.

### Current non-coalesced path

The fallback path can use `ProcessPoolExecutor`, but each process initializes and loads a Torus9 model/checkpoint onto CUDA. That restores process-level CPU execution but loses the required single central cross-worker inference broker and can duplicate GPU model state/work. It is not the target architecture.

### Historical proven path

The older GoCube self-play architecture used OS `SelfPlayAgent` worker processes, shared-memory inference input/output tensors and a parent central inference broker. The parent aggregated ready-worker requests across processes before one GPU forward, with routing/slot bookkeeping around results. This architecture was actually observed on Legion as many Python worker PIDs at `workers=16`.

A historical Legion B-run artifact also showed central inference batches around 60 rows and a 256-game self-play phase around 7m24s. These numbers are architecture/performance evidence, not a current scientific Golden configuration.

### Verdict

- Real process-level parallelism in current coalesced self-play: **NO**.
- Central batching in current coalesced self-play: **YES, but only across threads/lanes**.
- Old central cross-process batching preserved: **NO**.
- GIL regression risk: **YES**.

## 2. Inference batching

Current `Torus9InferenceCoordinator` collects one row per pending lane request. Flush occurs when the explicit `batch_cap` is reached or the `wait_ms` deadline expires after the first request. Telemetry exposes forward calls, total rows, raw batch row counts, mean/max rows and maximum concurrent model forwards.

This is not equivalent to the older cross-process broker. The old broker aggregated worker requests backed by shared-memory slots; historical runs reached batches around 55-60 rows. In the current design there are at most approximately one immediately pending row per game lane, and PR93's execution sweep used explicit low caps. A cap of 4/8/12/16 would make historical ~60-row behavior impossible even if enough work existed.

The self-play execution cap/wait is therefore **not Golden science**. It remains execution tuning and must not be promoted until the restored architecture passes regression testing.

## 3. Historical performance regression baseline

| Baseline | workers | CPU/process evidence | GPU/inference batching | rows/s | moves/s | wall | network | sims | wait/cap | games | Use |
|---|---:|---|---|---:|---:|---:|---|---:|---|---:|---|
| Historical Legion central-broker B-run | 16 | many worker PIDs observed | mean batch approx. 60 rows | not present in located excerpt | not present in located excerpt | approx. 7:24 self-play | exact model not safely recoverable from located excerpt | not safely recoverable | not safely recoverable | 256 | architecture/performance baseline only |
| PR83 Golden Arena benchmark | 16 | effective parallelism 13.492/16; 4.372x speedup | not a self-play GPU-batch baseline | n/a | n/a | benchmark-specific | Golden Arena | benchmark | n/a | 480/480 parity corpus | OS-process Arena baseline |
| PR93 M1 degraded current path | 16 logical lanes | no process proof | mean/p50/p95 batch=1; GPU avg 34.87%, peak 79%, VRAM peak ~1098 MiB | 139.57 | 2.2089 | 2260.36 s self-play | current 80x8 | 64 | batch effectively 1 in artifact | 64 | degraded baseline |
| Old Torus9 validation8 line | 16 historical workers | historical engine path | historical batching path | not mixed into baseline without raw artifact | not mixed | run-specific | GraphNet 64x6 | 100 regular; fast search existed | historical | 256/iteration | learning-history only; **scientifically invalid for current Golden because legacy komi/rules differ** |

Unknown cells above are intentionally not invented. Before closing P0, the implementation should archive one machine-readable old-performance baseline artifact with the complete metric schema from `TELEMETRY_SCHEMA_OLD_VS_NEW.md`.

## 4. Scientific contract

The post-PR93 current profile and implementation satisfy the requested contract:

- Torus 9x9 wrap X/Y, exact graph-area scoring.
- positional superko; suicide forbidden.
- two-pass termination; Benson auto-ending OFF.
- komi = **0.5**.
- 6 observation planes.
- 82 policy actions, PASS=81.
- WDL is side-to-move perspective.
- policy target is root visit distribution before the chosen move.
- ownership and score auxiliary targets are enabled.
- technical-invalid outcomes are fail-closed/excluded from scientific W/L.
- current self-play: root noise ON, epsilon .25, Dirichlet alpha .11, temperature 1.0 for plies 1-8 then 0, fast OFF, resign OFF.
- deterministic seed derivation is validated by tests.
- Arena deterministic tie-break is explicit; batched search ignores Arena seeds for choice ordering.
- explicit symmetry augmentation is OFF after the prior equivariance audit.

`7.5` exists only as a legacy sentinel in the current profile/test and is explicitly reject/fail-closed. No active current-Golden 7.5 configuration was found. Any future executable path accepting 7.5 must fail the contract.

**Scientific verdict: PASS.**

## 5. Network contract

Actual current class: `Torus9CurrentGraphNet`, derived from the ownership+score model and hard-fixed to the current architecture.

- architecture/version: `GoldenGraphNetV2-Torus9`
- hidden channels: 80
- residual blocks: 8
- input: 6
- policy: 82
- WDL: 3
- ownership: 81x3
- score: 1 scalar
- ownership aux: ON
- score aux: ON
- explicit symmetry augmentation: OFF

Checkpoint load verifies exact architecture configuration, head shapes, topology and komi 0.5, loads optimizer state when supplied, and verifies the model hash.

## 6. Training contract and backend use

The values are not only manifest metadata. `Torus9OwnershipScoreTrainer` executes them:

- Adam
- learning rate 0.001
- weight decay 0
- training batch 64
- exactly 80 optimizer updates/iteration
- exactly 5120 sample exposures/iteration
- no scheduler
- no warmup
- rolling last 3 generations
- replay cap 20,000
- deterministic seeded replay sampling
- gating OFF/decoupled
- ownership loss ON
- score loss ON

The continuation runner hard-fails if the optimizer-step count, sample count, batch-size sequence or auxiliary-loss enablement drifts.

**Training clock verdict: PASS.**

## 7. Self-play fixed scientific parameters

Current profile matches the requested science: 64 games/iteration, 64 sims, cpuct 1.25, FPU 0, root noise ON, epsilon .25, alpha .11, temperature plies 1-8 then 0, fast OFF, resign OFF, watchdog 500, workers target 16, komi .5.

`workers=16` is presently a scientific/execution target value, not proof of process-level execution. Execution cap/wait remains non-Golden until the performance gate passes.

## 8. Arena

Current Arena scientific switches match the required configuration: 64 games, 64 sims, cpuct 1.25, FPU 0, noise OFF, temperature 0, fast OFF, resign OFF, watchdog 1000, workers target 16, batched ON, arena batch size 8, komi .5, paired starts/color swaps, deterministic tie-break and technical fail-closed/excluded.

`inference_batch_wait_ms` is genuinely applied in `Torus9BatchedPUCT._evaluate_entries`: the scheduler sleeps before a model batch. It is not merely parsed by CLI.

However, current `run_torus9_batched_arena` advances many active games inside one process. `workers` controls logical lanes/batch geometry, not 16 OS processes. This regresses PR83's proven process-parallel Arena architecture.

**Arena science: PASS. Arena performance architecture: FAIL.**

## 9. Resume / atomicity

Strengths in `tools/continue_torus9_golden_m1_m100.py`:

- contiguous completion requires checkpoint+metadata, self-play, fresh replay, rolling replay, training metrics and iteration summary.
- replay is rebuilt and compared with persisted state.
- self-play game IDs and derived seeds are validated.
- optimizer state is restored and continued.
- incomplete iteration artifacts are not treated as completed generations.
- state/manifest/json/jsonl writes use temp-and-replace helpers.
- checkpoint publication uses temporary checkpoint/metadata paths followed by `os.replace`.
- stop requests and SIGINT/SIGTERM are handled at clean atomic boundaries.

Caveats:

- checkpoint `.pt` and metadata are two separate replaces, not one filesystem transaction. Completion logic prevents a half-pair from being counted complete, but publication is not literally atomic as a pair.
- `_complete_iterations` is primarily existence-based; stronger content/hash validation should gate recovery.
- a resumed run already beyond M5 can leave the mandatory M5 Arena missing until the late backfill path after M100. This is the documented M5 scheduling gap and is a P0 blocker.

**Resume core: mostly preserved. Mandatory Arena scheduling after resume: FAIL.**

## 10. Historical “69-8” / fixes-accounting matrix

No unique repository artifact literally defining the label “matrix 69-8” was found. To avoid silently omitting work, this audit accounts for the complete Golden sequence beginning at PR69 plus the earlier performance/resilience chain that the later work depended on.

| Historical change | Achievement that must not be lost | PR93/current status | Action |
|---|---|---|---|
| PR12 | parent central inference batching across OS self-play workers | **missing in current Torus9 coalesced path** | restore as P0 with current four-head model |
| PR38 | hardware telemetry, Arena batching/performance visibility | partially present; schema weakened/changed | restore semantic telemetry as P0 |
| PR45-47 | crash-resumable sweeps, atomic history/storage, preflight/provenance/supervisor | resume/provenance partly preserved; supervisor/recovery not complete | P0 resume invariants; P1 supervisor |
| PR48 | MCTS inference routing correctness | equivalent current lane routing exists; historical cross-process routing absent | preserve routing keys in restored broker |
| PR64 | deterministic Arena model/color bookkeeping | preserved | keep tests |
| PR65 | multi-game identity, slot generation, out-of-order/stale protection, four-head coalescing | scientific identity mostly preserved; cross-process bookkeeping not carried into current broker | restore IPC bookkeeping P0 |
| PR66 | checkpoint-contract restoration and batched sanity | preserved scientifically | keep |
| PR67-68 | retired legacy training path; learning-perspective/root-value diagnostics | current side-to-move contract preserved | keep |
| PR69 | independent fail-closed Golden Arena reference | concept preserved | retain as reference/oracle |
| PR70 | komi .5 baseline; 7.5 hard-fail | preserved as current sentinel reject | keep; any active 7.5 is error |
| PR72 | Golden passport: graph-area, superko, WDL STM, root visits, no fast | preserved | keep |
| PR73 | independent pure-Python Golden rules core | preserved | keep |
| PR74 | sequential Golden Arena/search qualification, deterministic tie-break | preserved | keep reference equivalence tests |
| PR75 | post-terminal fail-closed evidence | preserved | keep |
| PR76 | provenance, paired/color-swapped starts, exact hashes/seeds | mostly preserved | strengthen resume hash checks |
| PR77 | solver/PUCT hardening, policy validation, typed evidence | preserved | keep |
| PR78 | neural Golden self-play equivalence gate | semantic grouping tests preserved; performance gate absent | add process-broker equivalence/perf gate P0 |
| PR79 | deterministic diagnostics/replay | preserved | keep |
| PR80 | Golden learning proof | learning contract lineage preserved | keep |
| PR82 | optimized Golden rules/search, bit-exact hot-path speedups | current science uses Golden core; ensure no regression during multiprocessing work | benchmark/reference gate |
| PR83 | persistent OS-process Golden Arena, 480/480 parity, 13.492/16 effective, 4.372x speedup | **lost in current batched Arena** | restore P0 |
| PR85 | standalone Torus9 proof, 16-worker preflight | science carried forward; process preflight no longer proves current path | restore real PID/core gate |
| PR86 | fixed 80x64 training clock, Adam .001/wd0, last3/cap20k, optimizer continuity | preserved | keep |
| PR87 | technical outcomes excluded | preserved | keep |
| PR88 | watchdog split 500 self-play / 1000 Arena, komi .5 | preserved | keep |
| PR89 | ownership auxiliary + batched Arena metrics | ownership preserved; Arena batching architecture changed | keep head; restore process architecture |
| PR90 | symmetry audit; explicit augmentation OFF | preserved | keep |
| PR92 | versioned current profiles/source-of-truth | preserved | keep |
| PR93 | 80x8 current model, ownership+score, alpha .11, resumable M1-M100 | preserved scientifically; execution architecture regressed | fix P0 before M18 |

All discovered historical correctness/performance achievements are therefore explicitly classified; the ones marked missing are not accepted as intentional deletions.

## 11. Golden Standart status

The `Golden Standart` sheet, tab `TORUS 9x9`, was updated during this audit to make the model/training/Arena contract self-contained. Added/clarified rows include score head, warmup, ownership loss, score loss, Arena batching, deterministic tie-break, canonical startset semantics, the Arena worker-process caveat, and the fact that current self-play batch-size/cap/wait observations are not promoted as a new Golden performance winner.

Self-play inference cap/wait was deliberately **not** promoted to Golden winner status.

## 12. P0/P1/P2 classification

Detailed implementation plan is in `TORUS9_POST_PR93_FIX_PLAN.md`.

P0 before M18: restore real process-level self-play and Arena parallelism with one central GPU broker; restore routing/bookkeeping; version and repair telemetry semantics; add historical performance gate; close M5 Arena resume scheduling gap; harden generation publication/recovery; prove no duplicate training/self-play across crash boundaries.

P1 before long M18->M100: supervisor/recovery/heartbeat/stall detection; hardware monitoring; disk/artifact checks; stronger persisted benchmark/provenance ledger.

P2: cleanup/refactor only after scientific/performance validity is restored.

## 13. M18 performance acceptance gate

M18 is blocked until a non-training real-Torus9 smoke on Legion demonstrates all of the following:

1. 16 configured workers yield 16 distinct self-play/Arena worker processes (allowing normal short startup/teardown variance).
2. aggregate worker CPU is substantially above the 1-2-core failure mode; report `effective_cpu_cores = aggregate_worker_cpu_seconds / wall_seconds`.
3. one central GPU inference owner aggregates requests across worker processes.
4. `mean_inference_batch_rows >= 16`.
5. batch p50/p95/max and cap-hit metrics show no artificial low-cap bottleneck.
6. rows/s and moves/s are materially above the degraded PR93 M1 baseline (139.57 rows/s, 2.2089 moves/s), and are compared with the archived historical broker baseline.
7. GPU is not idle for most of the self-play phase; persist avg/peak, VRAM and idle-duty fraction.
8. fixed-seed scientific outputs are unchanged versus the qualified reference path: legal actions, selected actions/root visits where deterministic, WDL/ownership/score routing, targets, terminal outcomes and replay-row identity.

Do not rerun a long Arena with the same settings if `mean_inference_batch_rows < 16`.

## Final verdict

PR93 SCIENTIFIC CONTRACT: PASS

OLD PERFORMANCE ARCHITECTURE PRESERVED: NO

TELEMETRY SEMANTICS EQUIVALENT: NO

REAL 16-WORKER PARALLELISM: NO

HISTORICAL 69–8 FIXES ACCOUNTED FOR: YES

GOLDEN STANDARD COMPLETE: YES

SAFE TO START M18: NO
