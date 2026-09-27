# M137-5CH komi=1.5: human-gated adaptation

Entry point: `python -m gocube_golden.orchestrator_v2.adaptation`.
This is a separate workflow in orchestrator v2. It uses the existing cooperative
SelfPlayEngine, Arena engine, atomic persistence and Telegram notification store.
It does not modify the canonical bootstrap or start the ordinary continuous loop.

## Production run

Use the repository venv and the exact code commit recorded in the run config.

```bash
.venv/bin/python -m gocube_golden.orchestrator_v2.adaptation init --root runs/torus9/active/torus9-m137-5ch-komi15-adaptation-20260927-v1
.venv/bin/python -m gocube_golden.orchestrator_v2.adaptation run --root runs/torus9/active/torus9-m137-5ch-komi15-adaptation-20260927-v1
```

Launch `run` under a user systemd service with `KillMode=control-group` so workers
cannot survive a supervisor stop. A single run lock prevents concurrent owners.
Restarting a RUNNING process restores the last committed checkpoint and shards;
an incomplete self-play chunk of up to 128 games is regenerated with the same
seeds. A FAILED run requires investigation and an explicit retry, not an automatic
training restart. A NEEDS_REVIEW run remains paused after restart.

The process stays alive while awaiting review so Telegram delivery can retry.
The machine/Windows host must remain awake. The process does not depend on an
active Codex chat. `heartbeat.json` is updated every 15 seconds, `state.json` is
the durable source of truth, and `reports/stage-*/report.md` is the review entry.
The progress ETA is for the current operation, not a promise for the whole pilot.

## Review

```bash
.venv/bin/python -m gocube_golden.orchestrator_v2.adaptation status --root RUN_ROOT
.venv/bin/python -m gocube_golden.orchestrator_v2.adaptation decide --root RUN_ROOT --action continue --expected-report-sha sha256:EXACT_HASH_FROM_STATE
```

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
- Telegram transport is existing durable infrastructure. At-least-once delivery
  can duplicate a message after ambiguous network failures; event IDs are stable.
  Configuration errors/exhausted retries remain inspectable in notification state.

## Acceptance evidence

`tests/test_torus9_adaptation.py` checks Adam migration/freeze/phase transitions,
reproducible reload, komi targets, replay contract validation, review token checks,
notification retry/deduplication and phase rollback. The real CUDA smoke tool is:

```bash
.venv/bin/python -m tools.torus9_adaptation_smoke --output artifacts/UNIQUE_SMOKE_DIR
```

The smoke uses four games and eight simulations to verify execution, not strength.
Production data never imports smoke games or weights. Training tests which require
the canonical local bootstrap skip when it is unavailable in CI.
