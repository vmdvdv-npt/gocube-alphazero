# Torus9 Post-PR93 Fix Plan

Date: 2026-09-14

Scope: plan only. This audit branch must not implement the fixes below. M18 stays blocked until all P0 gates pass.

> Historical / retired plan. Do not use this document as the current Torus 9×9
> Arena or training entrypoint. Its provisional Arena settings are preserved
> only as historical planning evidence; use `tools/arena.py` and the current
> Golden standard-64/high-volume presets instead.

## P0 — mandatory before M18

### P0.1 Restore process-level self-play MCTS with one central GPU inference owner

Target architecture:

- 16 persistent OS worker processes on Legion run rules + MCTS selection/expansion/backup.
- Workers do **not** own CUDA models.
- One parent/broker owns the current `GoldenGraphNetV2-Torus9` CUDA model.
- Workers submit prepared inference rows through bounded IPC/shared-memory slots or an equivalently audited mechanism.
- Parent aggregates requests across processes, performs one batched forward and routes WDL/policy/ownership/score results back to the exact request.
- `workers=16` must be verifiable from PIDs/effective CPU cores, not a config field.

Reuse the proven concepts from the older `SelfPlayAgent`/`processSelfPlayBatches` path and `alphazero/inference_batching.py`, but adapt them to the current pure-Golden rules/search and four-head model. Do not reintroduce retired scientific semantics or the legacy 7.5-komi path.

Required routing identity per request should cover at least worker/process identity, game identity, game/slot generation and monotonically unique request sequence. Stale, duplicate, out-of-order or wrong-generation responses must fail closed.

### P0.2 Keep the current thread coordinator only as a reference/debug path

The post-PR93 thread-lane implementation is useful as a semantic oracle and debugging fallback. It must not remain the production Golden performance path on Legion.

Add equivalence tests between the restored process broker and the qualified reference path using fixed seeds and a small deterministic corpus. Compare legal actions, selected/root-visit behavior where deterministic, WDL, ownership, score, terminal outcomes and replay-row identities.

### P0.3 Restore process-level Golden Arena

Use the PR83 architecture/proof as the minimum standard:

- persistent OS process workers, max 16 on Legion;
- current deterministic Golden search semantics;
- paired starts/color swaps and frozen startset identity;
- central neural inference batching where neural evaluation is used;
- fail-closed technical outcomes;
- no root noise, temperature, fast search or resign;
- games>=64, sims=64, cpuct=1.25, FPU=0, watchdog=1000, komi=.5;
- batched ON, `arena_batch_size>=4` (default 8), `inference_batch_wait_ms>=4` (default 6).

Preserve the PR83 semantic-parity requirement while restoring its OS-process speedup model.

### P0.4 Replace low-cap-by-design batching with an architecture that can accumulate real rows

Do not choose the new self-play cap/wait winner before process-level aggregation exists. The historical broker reached batches around 55-60 rows, so any explicit cap below that range must be treated as a possible bottleneck until measured otherwise.

After restoring the broker, run a short execution-only sweep over sensible caps/waits. Selection criteria must include moves/sec, rows/sec, batch distribution, cap-hit ratio, actual queue wait, CPU effective cores and GPU duty. Cap/wait remains execution tuning, not scientific Golden state.

### P0.5 Version and repair telemetry semantics

Implement the schema in `TELEMETRY_SCHEMA_OLD_VS_NEW.md`.

Mandatory additions before M18:

- worker PID count/set;
- aggregate worker CPU seconds and effective CPU cores;
- process-tree CPU avg/p95/peak;
- batching aggregation scope;
- actual queue-wait distribution;
- queue depth distribution;
- cap-hit count/ratio;
- GPU idle-duty fraction;
- total iteration wall;
- unambiguous generated/replay/unique/reused/sample-exposure counters;
- versioned technical-outcome categories.

Do not keep the current `cpu_utilization_pct` as if it represented total worker utilization: it is based on parent `RUSAGE_SELF` normalized by host CPU count.

### P0.6 Add a historical performance regression gate

Create a machine-readable benchmark artifact and command that can be rerun on Legion without advancing training.

The gate must compare against:

- historical central-broker evidence (16 OS workers; mean batch ~60; 256-game self-play ~7m24s where directly comparable),
- PR83 process-parallel Arena evidence (13.492/16 effective, 4.372x speedup, 480/480 parity),
- degraded PR93 M1 floor (139.57 rows/s, 2.2089 moves/s, batch mean 1, GPU avg 34.87%).

Unknown historical values must be recorded as unknown, not borrowed from a different run.

### P0.7 Close the resume/Arena scheduling gap

A resumed run must reconcile a durable Arena obligation ledger **before** advancing the next training transition.

At startup:

1. determine completed scientific checkpoints;
2. determine every mandatory Arena comparison due for those checkpoints;
3. validate artifact identity/startset/checkpoint hashes for existing Arena results;
4. execute any missing due Arena before further self-play/training, unless the user explicitly stops;
5. persist completion atomically.

The M5 Arena may not be deferred silently until after M100.

### P0.8 Harden atomic publication and crash recovery

Keep temp+replace, but make generation completeness content-based rather than only existence-based.

Required:

- iteration manifest/commit marker published last;
- hashes/identities for checkpoint, metadata, self-play, fresh replay, rolling replay and training metrics;
- on resume, reject/quarantine incomplete or mismatched artifacts;
- no double-training after checkpoint publication boundaries;
- no repeated canonical self-play after a fully committed generation;
- incomplete generations are safely rerun from previous committed checkpoint;
- checkpoint `.pt` and metadata pair must be validated as one logical object even if filesystem publication uses two replaces.

Add crash-injection tests at phase boundaries and between individual publication operations.

### P0.9 Preserve current scientific/training contract exactly while changing execution

The multiprocessing/performance fix must not alter:

- Torus9 graph-area/PSK/suicide/termination semantics;
- komi .5;
- 6 observation planes;
- 82 actions / PASS=81;
- WDL side-to-move;
- root-visits policy target;
- ownership and score targets/losses;
- current 80x8 four-head model;
- 64 self-play games, 64 sims, cpuct1.25/FPU0;
- noise epsilon .25, alpha .11;
- temperature plies1-8 then0;
- fast/resign OFF;
- fixed Adam .001, wd0, 80x64=5120 training clock;
- deterministic rolling replay last3/cap20k;
- gating OFF/decoupled;
- symmetry augmentation OFF.

Any active komi 7.5 is a hard failure.

## P0 M18 acceptance procedure

Run this only after the final P0 implementation is batched into one completed change set.

### Stage A — semantic parity smoke

Use a small fixed-seed corpus with the current 80x8 checkpoint. Compare restored process-broker outputs against the qualified reference path. Require exact game/action/root-visit/replay identity wherever the contract is deterministic and numerically tight neural-head parity under the existing tolerance where floating-point batching order can vary.

### Stage B — real Legion performance smoke, no training advancement

Use 16 worker processes and the real current network/search workload. Persist telemetry.

Hard gates:

- real worker process count proves 16 process-level workers during the steady-state window;
- aggregate CPU is substantially above the one/two-core failure mode; report effective cores explicitly;
- one central inference owner aggregates requests from multiple worker PIDs;
- `mean_inference_batch_rows >= 16`;
- no unexplained routing/technical failures;
- scientific parity remains PASS.

Performance comparison gates:

- rows/sec and moves/sec materially exceed PR93 M1's degraded 139.57 rows/s and 2.2089 moves/s;
- compare p50/p95/max batch rows against the old large-batch path;
- GPU must not spend most of the phase idle; report avg/p95/peak and idle-duty fraction;
- cap-hit ratio must show whether the chosen cap is constraining throughput.

If `mean_inference_batch_rows < 16`, classify the path performance-degraded and do not launch/repeat a long Arena with unchanged settings.

### Stage C — Arena smoke

At least 64 games, workers16, batched, arena batch size8, wait6, sims64, watchdog1000, komi.5. Prove OS process PIDs/effective CPU and mean inference batch >=16. Validate deterministic paired/color-swapped scientific accounting.

### Stage D — resume fault tests

Exercise clean stop and injected crashes before/after self-play publication, replay publication, training, checkpoint publication, summary publication and due-Arena scheduling. Require no duplicate committed generation and no skipped mandatory Arena.

Only after A-D pass may `SAFE TO START M18` change to YES.

## P1 — required before a long M18->M100 run

### P1.1 Supervisor and recovery

- systemd/user-service or equivalent production supervisor on Legion;
- heartbeat and phase-progress timestamp;
- bounded automatic restart only from the last committed atomic boundary;
- explicit user-stop state that is never mistaken for a crash;
- startup preflight for CUDA/device, disk, checkpoint/replay hashes and configuration fingerprint.

### P1.2 Stall and utilization monitoring

Alert/fail a run when:

- worker PID count collapses unexpectedly;
- effective CPU cores remain in the one/two-core regime;
- mean inference batch falls below 16 for a sustained window;
- GPU idle duty becomes dominant;
- queue age/depth grows without forward progress;
- no canonical games complete within a board-scaled watchdog interval.

### P1.3 Artifact and storage robustness

- disk-space preflight and reserve threshold;
- atomic retention policy for old checkpoints/replay/history;
- preserve benchmark and telemetry artifacts needed for later scientific audit;
- do not delete the last known-good resume boundary during cleanup.

### P1.4 Durable run passport

Every long run should snapshot:

- commit/tree and dirty-state evidence;
- current Golden profile fingerprint;
- model/checkpoint hashes;
- all scientific parameters;
- execution architecture ID and execution tuning separately;
- telemetry schema version;
- hardware/device identity;
- Arena startset identity and due/completed ledger.

## P2 — cleanup/refactor after validity is restored

- unify duplicate telemetry field names around one versioned schema;
- separate scientific config objects from execution tuning types;
- mark/remove dead legacy entrypoints only after replay/reproduction needs are documented;
- consolidate broker abstractions shared by self-play and Arena where semantics genuinely match;
- simplify historical scripts after their artifacts have been indexed;
- improve docs/naming so “worker”, “lane”, “request”, “row”, “batch”, “game” and “process” cannot be conflated.

P2 must not delay the P0 performance/scientific gate and must not be mixed into P0 unless necessary for correctness.

## Implementation order

1. Freeze current reference tests/artifacts.
2. Implement central cross-process self-play broker + routing safety.
3. Add telemetry schema and process/hardware accounting.
4. Restore process-level Arena.
5. Fix due-Arena reconciliation and publication/recovery invariants.
6. Run targeted semantic/fault tests locally/CI.
7. Final push of the complete change set.
8. Run CI only after that final push.
9. Run Legion Stage A-D acceptance smoke.
10. Only if all gates pass, authorize M18.

No merge is part of this plan without an explicit separate instruction.
