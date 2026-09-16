# Production Training Orchestrator V1

`gocube_golden.orchestrator` is the game-independent supervisor for long AlphaZero runs. It does not own Cube/Torus rules, network architecture, loss weights, replay policy, search settings, komi, seeds, or other scientific hyperparameters. Those remain in the canonical profile and in the profile-specific generation/Arena drivers.

The orchestrator owns only the production lifecycle:

```text
create/resume lineage
    -> generation driver (self-play -> replay -> train -> checkpoint -> reload verification)
    -> transactional generation commit
    -> scheduled observational Arena
    -> health/performance/learning reporting
    -> next generation
    -> soft stop / completion
```

## Storage contract

This implementation follows `docs/RUN_STORAGE_AND_ARCHIVING_POLICY.md`:

- one lineage lives only in `runs/<topology>/active/<lineage-id>/`;
- `manifest.json` is the lineage source of truth;
- checkpoints are never copied between lineages;
- a parent model is a reference (`lineage_id`, `path`, `sha256`), not a copied file;
- all generation data, replay, logs, metrics, reports and within-lineage Arena results remain inside that lineage;
- cross-lineage evaluations remain under the existing `runs/<topology>/evaluations/` path and are not created by this supervisor;
- checkpoint hashes are refreshed in `manifest.json` after every committed generation.

The orchestrator rejects `.xlsx` as a profile source. Production uses the canonical JSON preset derived from the project Golden Standard workflow; it does not read or export spreadsheet copies.

## Driver boundary

V1 uses a subprocess driver contract so the supervisor remains valid for Cube, Torus, other board sizes and future topologies. A driver is invoked once per generation. It receives:

- `AZ_LINEAGE_ID`, `AZ_TOPOLOGY`, `AZ_RUN_ROOT`;
- `AZ_PROFILE_PATH`, `AZ_PROFILE_FINGERPRINT`;
- `AZ_GENERATION`, `AZ_GENERATION04`;
- `AZ_RESUME=0|1`;
- `AZ_SOFT_STOP_REQUEST_PATH`;
- `AZ_DRIVER_HEARTBEAT_PATH`;
- `AZ_GENERATION_RESULT_PATH`.

A generation driver owns the scientific pipeline for that profile and must atomically publish the result JSON last. The result schema is `gocube-generation-driver-result-v1` and must prove:

- requested generation completed;
- canonical profile fingerprint matches;
- checkpoint was successfully reloaded and validated (`checkpoint_reload_verified=true`);
- checkpoint, replay and durable resume state are present and SHA-256 validated;
- resume state covers **model, optimizer, replay, generation and RNG**;
- technical games = 0 and invalid games = 0;
- metrics may include loss, games/hour, moves/sec, inference batch statistics, optimizer updates/sec and hardware observations.

An interrupted generation is never blindly repeated. If its transaction is `RUNNING`, the orchestrator requires `execution.generation_resume_command`; otherwise it fails closed instead of risking duplicated self-play/replay/training data.

## Arena boundary

Arena is observational and decoupled from training. V1 requires a cadence from **5 through 10 generations**. For required Arena, the spec must pin both:

- `preset_fingerprint`;
- `startset_fingerprint`.

The Arena result schema `gocube-arena-driver-result-v1` must match those fingerprints, report `training_mutated=false`, and have zero technical/invalid games. Arena output for the lineage is stored under `arena/generation-NNNN/`; it never replaces a training checkpoint or gates the training lineage implicitly.

## Terminal commands

Foreground:

```bash
.venv/bin/python tools/training_orchestrator.py run \
  --spec <orchestrator-spec.json> \
  --lineage <lineage-id>
```

Detached overnight run:

```bash
.venv/bin/python tools/training_orchestrator.py start \
  --spec <orchestrator-spec.json> \
  --lineage <lineage-id>
```

One-shot status:

```bash
.venv/bin/python tools/training_orchestrator.py status \
  --spec <orchestrator-spec.json> \
  --lineage <lineage-id>
```

Live terminal status:

```bash
.venv/bin/python tools/training_orchestrator.py status \
  --spec <orchestrator-spec.json> \
  --lineage <lineage-id> --watch
```

Explicit resume of the same lineage after a soft stop or recoverable fail-closed condition:

```bash
.venv/bin/python tools/training_orchestrator.py resume \
  --spec <orchestrator-spec.json> \
  --lineage <lineage-id>
```

`resume` is detached by default; add `--foreground` for an attached session. The command clears only the durable soft-stop control request. It never creates a new lineage and never rewrites scientific identity.

Soft stop with a target window from 30 to 100 minutes (60 by default):

```bash
.venv/bin/python tools/training_orchestrator.py stop \
  --spec <orchestrator-spec.json> \
  --lineage <lineage-id> --minutes 60
```

Soft stop is deliberately not `SIGKILL`. The request is durable and visible to both supervisor and driver. No new generation is started after a stop request. The active driver is allowed to reach its next safe boundary. If the target time is exceeded, the supervisor emits a warning instead of silently killing training and creating ambiguous scientific artifacts.

`SIGINT`/`SIGTERM` delivered to the supervisor are converted into the same durable soft-stop request. Generation drivers run in their own process session, so a terminal interrupt cannot accidentally kill the active training process first.

## Health and warnings

While a generation/Arena driver is active, the supervisor records its own heartbeat and checks:

- child process liveness and return code;
- driver heartbeat age;
- free disk;
- available RAM (Linux `/proc/meminfo`);
- published artifact integrity;
- technical/invalid games;
- profile/config fingerprint drift.

Critical disk/RAM pressure automatically creates a soft-stop request. Stale heartbeat is warning/critical telemetry but does not produce an unsafe hard kill.

Warnings and critical events are printed in foreground mode and persisted in `logs/orchestrator-events.jsonl`. A detached run writes normal terminal output to `logs/orchestrator-supervisor.log`; `status --watch` exposes current state and the latest warning.

## Performance and learning velocity

The orchestration spec may declare performance checks against a confirmed operational baseline. Each metric has a warning ratio and an optional fail-closed ratio. A generation is committed first after all scientific artifacts validate; a performance fail-closed condition then stops further progress without making the completed generation disappear or be repeated.

Raw generation and Arena metrics are appended to `metrics/history.jsonl`. `reports/training-report.md` is refreshed after each generation. The report exposes current generation, Arena cadence/results, warnings and configured learning-metric deltas. On `COMPLETED`, `SOFT_STOPPED` or `RECOVERY_REQUIRED`, `reports/final-report.json` and `reports/final-report.md` summarize the lineage. The JSON report contains full generation/Arena metric history, warning/critical events and interrupted-generation resume events.

The orchestrator does not pretend that decreasing loss proves stronger play. Profile drivers should publish policy/value/aux losses plus prediction metrics, while Arena supplies strength evidence. The `learning.metrics` list selects which numeric series are summarized as deltas; richer derived learning-velocity rules can be added without changing game science.

## Resume and transaction rules

A lineage is pinned to:

- git SHA recorded at creation;
- full orchestrator config fingerprint;
- canonical profile fingerprint.

Resume rejects config/profile drift. `runtime/generations/generation-NNNN.json` is the transaction record. A generation advances `last_committed_generation` only after the driver result, required artifacts, hashes, checkpoint reload proof, resume-state proof and technical-game gates all pass.

This means a crash may leave a `RUNNING` generation, but it cannot advance the lineage. The profile driver is responsible for resumable in-generation semantics; the orchestrator explicitly selects the resume command rather than guessing that rerunning is safe. A committed generation whose required Arena failed is remembered: after explicit resume, that Arena is completed before any new generation starts.

## Required profile-driver work

The supervisor is intentionally not a second Cube/Torus training implementation. A production profile integration must provide two small executable adapters:

1. **generation driver** — call the existing `SelfPlayEngine` + `TrainingEngine`, publish durable checkpoint/replay/resume state and metrics using the result contract above;
2. **Arena driver** — call the existing standalone Arena with its frozen evaluation preset/startset and publish the Arena result contract.

Cube/Torus/board-size differences belong only in those adapters and canonical presets. Adding a new board size must not require changes to `gocube_golden.orchestrator`.

## Test contract

`tests/test_training_orchestrator.py` uses a deterministic fake driver to exercise the supervisor without performing scientific training. It covers:

- five committed generations plus scheduled Arena;
- durable 30–100 minute soft-stop validation;
- interrupted-generation resume path;
- technical-game fail-closed behavior;
- corrupted artifact hash;
- non-zero/killed driver behavior;
- critical resource soft-stop request;
- Arena cadence validation;
- config drift rejection.

A multi-hour/multi-day soak test belongs in the hardware acceptance stage after the real Cube/Torus generation drivers are wired. Soak testing must not change canonical scientific parameters.
