# M137-5CH komi=1.5: human-gated adaptation

Entry point: `python -m gocube_golden.orchestrator_v2.adaptation`.
This workflow uses the existing cooperative SelfPlayEngine and delegates arenas
to production_entrypoint.run_arena_from_config / ArenaRunnerV2. It must never
call the Arena engine directly or mint its own execution permits.
It does not modify the canonical bootstrap or start the ordinary continuous loop.

## Production run

The standalone adaptation launcher is not authorized to compute. The existing
Orchestrator V2 workflow now owns a bounded `adaptation_phase` action:

```json
{
  "mode": "workflow",
  "workflow_id": "torus9-adaptation-partial-reviewed",
  "topology": "torus9",
  "steps": [{
    "step_id": "partial",
    "action": "adaptation_phase",
    "config": {
      "root": "/absolute/runs/torus9/active/ADAPTATION_RUN",
      "target_stage": 2,
      "review": {
        "report_sha": "sha256:REVIEWED_REPORT_SHA",
        "allow_inconclusive": true,
        "reason": "Explicit operator decision to continue despite inconclusive evaluation"
      }
    }
  }]
}
```

Run this plan with `python -m gocube_golden.orchestrator_v2.production_entrypoint
run PLAN.json` under the existing user service. The action restores the existing
checkpoint/replay, executes only the selected phase, evaluates through ArenaRunnerV2,
and returns at NEEDS_REVIEW. `allow_inconclusive` requires a reason and the exact
reviewed report hash; it never changes the recorded gate and cannot admit FAIL.
An ordinary PASS needs no exception.

A reviewed code rollover uses `runtime/execution-pin.json`, bound to the original
scientific config hash and source commit. Do not rewrite config.json or checkpoint
metadata to change the execution revision: this preserves Adam/replay restore
identities. `adaptation_artifacts.publish_checkpoints` registers real ancestry
and adaptation replay ledgers in the existing graph before evaluation.

Launch the production entrypoint under a user systemd service with `KillMode=control-group` so workers
cannot survive a supervisor stop. A single run lock prevents concurrent owners.
Restarting a RUNNING process restores the last committed checkpoint and shards;
an incomplete self-play chunk of up to 128 games is regenerated with the same
seeds. A FAILED run requires investigation and an explicit retry, not an automatic
training restart. A NEEDS_REVIEW run remains paused after restart.

The bounded workflow action returns after its review report. Delivery state is durable;
uncertain delivery is not automatically repeated.
The machine/Windows host must remain awake. The process does not depend on an
active Codex chat. `heartbeat.json` is updated every 15 seconds, `state.json` is
the durable source of truth, and `reports/stage-*/report.md` is the review entry.
The progress ETA is for the current operation, not a promise for the whole pilot.

## Review

```bash
.venv/bin/python -m gocube_golden.orchestrator_v2.adaptation status --root RUN_ROOT
```

Standalone `decide --action continue/expand` is blocked too: it must not wake
an old runner still loaded in memory. Such decisions require V2 authority after
the registered adaptation_phase action. `stop` and read-only status/report inspection remain usable.

`continue` requires PASS and a remaining adaptation phase. `expand` runs another,
larger arena with new seeds, up to two expansions; all attempts remain available.
It does not automatically pool repeated tests or claim improved confidence from
multiple looks. `stop` ends the workflow. Final pilot completion cannot be used
to enter ordinary continuous training without a separate reviewed setup.

For a reviewed rollback and lower LR, create a new run:

```bash
.venv/bin/python -m gocube_golden.orchestrator_v2.adaptation retry --root RUN_ROOT --retry-root NEW_RUN_ROOT --lr-scale 0.5
```

This restores the complete phase-entry checkpoint, RNG, optimizer, replay
references and local clocks, scales LR, and preserves the rejected run as evidence.
It only prepares the new run; launch it under its own service after review.

## Scientific and operational choices

- Self-play: 200 simulations, 16 workers, 4 active games/worker, batch cap 64,
  wait 1 ms, komi 1.5. Production arenas: 256 simulations, color-swapped paired
  starts, komi 1.5, noise/resign off, 16 workers, 12 games/worker, batch cap 192.
- Initial corpus: at least 512 games AND 50,000 training positions; extend by 128
  games until both are met (maximum 2048 before human investigation). About 10%
  of initial games form fixed validation. Every game is split as a whole.
- Later data: 384 fresh games per 160 updates, rolling six generation buckets;
  validation stays fixed and never enters the training buffer. Only the accepted
  champion generates fresh games.
- Checkpoints every 40 updates. Phase endpoints: 160, 480, 1120, 2400. Every phase
  including initial collection pauses with a Telegram report and copied Codex prompt.
- Reset only the folded bias Adam moments and local step. Frozen parameters have
  grad=None and unchanged state. All other inherited states remain intact.
- For this first pilot stabilization conservatively HOLDS shared LR at 5e-6 and
  heads at 1e-5, rather than automatically reaching the optional maxima in the
  research specification. Any increase is a separately reviewed configuration.
- L2-SP is calibrated at update 200 to 5% of the shared supervised gradient; its
  coefficient halves in stabilization. No old-policy/WDL distillation is applied.
- NaN, invalid targets, clock mismatch, technical arena games, large update
  spikes and repeated validation regression fail closed. No automatic model promotion.
- Telegram reserves each send durably before HTTP. Slow sends retain their
  reservation after flush times out and persist a late receipt. Ambiguous sends
  are DELIVERY_UNCERTAIN and are not automatically repeated, including after a
  restart. Only an explicit HTTP 429 rejection permits a delayed retry.
  This favors avoiding duplicates over guaranteed delivery after a lost response.

## Acceptance evidence

`tests/test_torus9_adaptation.py` checks Adam migration/freeze/phase transitions,
reproducible reload, komi targets, replay contract validation, review token checks,
notification retry/deduplication and phase rollback. The historical CUDA smoke
tool also obeys the engine boundary and cannot execute standalone. Unit tests
explicitly use test-only authority; production commands have no bypass flag.
The bounded smoke workload uses four games and eight simulations.
Production data never imports smoke games or weights. Training tests which require
the canonical local bootstrap skip when it is unavailable in CI.

## Mandatory Arena boundary

`tools.arena_engine.run_arena` requires a signed, parent-PID-bound V2 child
permit for action `arena` and the matching evaluation run ID before any model,
output directory, GPU or worker is initialized. This also applies to smoke runs.
A V2 environment marker or in-process authority does not authorize the engine.
Only the existing production entrypoint / ArenaRunnerV2 / arena_child route
owns arena lifecycle notifications, supervision and canonical evaluation outputs.
The adaptation smoke command now verifies self-play, training and reload only.

Adaptation requires `metadata/arena-checkpoint-refs.json` with a `reference`
CheckpointRef and `candidates` mapping checkpoint SHA to CheckpointRef. These
must resolve through the existing V2 artifact graph, including owner manifest,
checkpoint node, effective config and provenance; this file alone does not
register a checkpoint. Missing registration fails closed. Do not create fake
genesis nodes for trained checkpoints or a second runner to bypass registration.

The already running legacy arena may finish on its loaded code. At NEEDS_REVIEW,
before approving another phase or expanded arena, stop its paused service,
register its real checkpoint ancestry/replay in V2, and restart on the reviewed
code revision with an explicit code-pin migration. Never change the live code
pin or restart an active arena just to install this boundary. No migration or
subsequent training is implied by installing the source change.

## All computation belongs to V2

The common SelfPlayEngine.run and TrainingEngine.run_iteration, the Torus
trainer methods, Cube training adapter, and adaptation trainer enforce the same
V2 capability boundary. Generation children may perform self-play and training;
an arena permit cannot authorize either. Topology-specific calls also check the
topology. CLI wrappers, direct Python calls, CPU and smoke workloads obey the
same checks. Pure target builders, checkpoint inspection and inference for the
interactive game are not training or self-play launches.

Only existing production entrypoint/runtime modules may mint execution authority;
a repository test rejects new issuers in engines, tools or additional launchers.
These are application execution guards, not a security sandbox against someone
who can edit Python source or run a historical checkout.
