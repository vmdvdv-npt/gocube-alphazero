# Production Training Orchestrator V3

## Scope

V3 is the production supervisor for long AlphaZero training lineages. It is deliberately game-independent: Cube/Torus rules, topology semantics, search, model, replay and optimizer behaviour stay in scientific profiles/adapters and the existing engines.

The boundary is:

`research / Golden Sheet / task specification -> immutable run-spec -> generic supervisor -> profile adapter -> SelfPlayEngine / TrainingEngine / Arena`

The supervisor chooses no scientific or execution tuning values on its own.

## Immutable run-spec

A new run uses schema `gocube-production-run-spec-v3`. The one-shot JSON explicitly pins:

- topology and board size;
- adapter identity and process transport;
- scientific profile path + fingerprint;
- self-play workload and execution parameters;
- device and reproducibility seeds;
- Arena enablement, cadence, mode, workload, seed, fixed startset identity and execution gates;
- liveness/resource thresholds;
- progress-stall thresholds and bounded restart policy;
- soft-stop window;
- performance baselines/thresholds;
- learning metrics/stall rules;
- required generation/Arena metrics.

Missing policy is an error. There is no production fallback to Golden Sheet, Legion values, old configs or historical benchmark constants.

At creation the complete run-spec is copied once to:

`runs/<topology>/active/<lineage-id>/run-spec.json`

Its canonical SHA-256 is stored in `manifest.json`. Resume/status/stop read only this lineage-owned copy and fail closed on drift.

## Storage ownership

One lineage uses one stable directory:

`runs/<topology>/active/<lineage-id>/`

All lineage-owned checkpoints, self-play/replay, logs, metrics, reports and periodic same-lineage Arena evidence remain under that directory. Periodic Arena therefore writes to:

`runs/<topology>/active/<lineage-id>/arena/generation-NNNN/`

Only comparisons between independent lineages belong under:

`runs/<topology>/evaluations/<evaluation-id>/`

Checkpoints are referenced, never copied between lineages.

## Terminal commands

Start detached and survive terminal disconnect:

```bash
.venv/bin/python tools/training_orchestrator.py start \
  --spec /path/to/run-spec.json \
  --lineage <lineage-id> \
  --max-generations <N>
```

`start` does not return success merely because `Popen` succeeded. It waits for the detached supervisor to publish a `RUNNING` heartbeat inside the explicit startup timeout. Supervisor stdout/stderr goes to:

`runs/<topology>/active/<lineage-id>/logs/orchestrator-supervisor.log`

Live status:

```bash
.venv/bin/python tools/training_orchestrator.py status --lineage <lineage-id>
.venv/bin/python tools/training_orchestrator.py status --lineage <lineage-id> --watch
```

Status includes lifecycle state, health classification, generation/phase, semantic progress token, progress age, latest speed metrics, last Arena, warning, soft-stop target and report path.

Logs:

```bash
.venv/bin/python tools/training_orchestrator.py logs --lineage <lineage-id> --follow
```

Soft stop:

```bash
.venv/bin/python tools/training_orchestrator.py stop --lineage <lineage-id>
.venv/bin/python tools/training_orchestrator.py stop --lineage <lineage-id> --minutes 60
```

The active safe unit is allowed to finish. A stop request does not start a new generation or a new Arena. If a generation committed exactly on an Arena cadence, that Arena remains pending and runs first after explicit resume.

Resume:

```bash
.venv/bin/python tools/training_orchestrator.py resume --lineage <lineage-id>
```

No `--spec` is accepted for resume/status/stop; the lineage-owned immutable run-spec is authoritative.

## Liveness versus progress

Driver heartbeat schema V2 separates:

- `liveness_at`: process is responsive;
- `progress_at` + `progress_token`: meaningful adapter progress advanced;
- optional structured `progress` (`completed`, `total`, `unit`).

A process that keeps heartbeating while semantic progress is frozen is therefore detectable. Warning and critical progress ages come from the run-spec. Critical health is fail-closed: the supervisor terminates only the active child process group it owns, marks the lineage `RECOVERY_REQUIRED`, persists the reason and never uses broad `killall`/`pkill` cleanup.

Progress thresholds must be chosen for the slowest legitimate safe unit in the run. They are stall guards, not assumptions about expected iteration duration.

## Fault recovery

Each generation is transactional. A checkpoint/replay/resume-state is validated and hashed before the generation transaction is marked committed. Partial current-generation files may be cleaned by the adapter during resume; previously committed generations are never deleted by recovery.

A non-zero generation child exit can receive a bounded number of automatic resume attempts, explicitly configured by `supervision.max_generation_restarts`. Hangs, stale progress, critical resource conditions, worker/inference fatal reports and artifact validation failures fail closed to `RECOVERY_REQUIRED` rather than looping indefinitely.

## Performance and learning supervision

Generation metrics may include self-play throughput, MCTS/move throughput, inference batch telemetry, training duration/updates per second, replay volume, losses, gradient norm and parameter delta. `performance.checks` compare selected metrics with an explicit baseline and warning/fail ratio.

`learning.metrics` and `learning.stall_checks` track whether learning is moving across generations. Arena results are independent evidence and are never treated as training data or gating state.

Reports are stored in the lineage:

- `reports/training-report.md` — rolling operator report;
- `reports/final-report.json` — machine-readable final evidence;
- `reports/final-report.md` — human-readable final summary;
- `metrics/history.jsonl` — generation/Arena metric history;
- `logs/orchestrator-events.jsonl` — warnings, restarts and lifecycle events.

## Archive and discard

Archive moves the whole stopped lineage without copying:

```bash
.venv/bin/python tools/training_orchestrator.py archive --lineage <lineage-id>
```

Discard is intentionally harder and requires an exact lineage confirmation, explicit reason and useful-result statement. It refuses deletion if retained JSON artifacts reference the lineage/checkpoint hashes, writes the required short record under `docs/experiments/discarded/`, then removes the heavy lineage directory:

```bash
.venv/bin/python tools/training_orchestrator.py discard \
  --lineage <lineage-id> \
  --confirm-lineage <lineage-id> \
  --reason "..." \
  --useful-result "none"
```

## Universal adapter boundary

`gocube_golden.orchestration_adapter.TrainingAdapter` defines the game-independent lifecycle surface (`prepare`, self-play, replay, train, checkpoint save/validation, Arena, metrics, report, resume, cleanup). The current production transport is process-based so long-running scientific work remains isolated from the supervisor.

A fake Cube adapter is exercised in regression tests through the same generic supervisor. The supervisor itself imports no Cube/Torus engine or rules module.
