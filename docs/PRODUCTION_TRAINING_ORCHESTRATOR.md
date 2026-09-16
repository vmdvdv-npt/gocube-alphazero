# Production Training Orchestrator V2

## Purpose

The orchestrator is a lifecycle/safety mechanism. It does **not** decide how a model should be trained or evaluated.

The production boundary is:

`research / Golden Sheet / task specification -> one-shot run-spec -> orchestrator -> profile driver -> engines`

There are no production tuning defaults in the active CLI path. A run must provide every policy-bearing value that the supervisor or driver needs. Missing values fail before lineage creation.

## Immutable one-shot run-spec

A new run starts with a JSON document using schema `gocube-production-run-spec-v2`.

The task that launches the run owns the values in that document, including execution choices, Arena schedule/workload, health thresholds, soft-stop window, performance references, learning checks, seeds and required metrics.

The repository does not carry a canonical production orchestrator preset with those choices.

## Create freezes policy

`create`, `run`, and `start` require `--spec`.

At lineage creation the complete semantic run-spec is written to:

`runs/<topology>/active/<lineage-id>/run-spec.json`

Its canonical SHA-256 fingerprint is recorded in `manifest.json` as both the lineage `config_fingerprint` and `run_spec.fingerprint`.

After creation, the source JSON passed on the command line is no longer authoritative. It may be moved or deleted.

`resume`, `status`, `stop`, and the detached supervisor discover the lineage and read only the saved `run-spec.json`. If the saved file or its manifest fingerprint changes, execution fails closed.

This prevents a repository preset, later edit, or different command line from silently changing an existing training lineage.

## CLI

New lineage:

```bash
.venv/bin/python tools/training_orchestrator.py create \
  --spec /path/to/one-shot-run.json \
  --lineage <lineage-id>

.venv/bin/python tools/training_orchestrator.py start \
  --spec /path/to/one-shot-run.json \
  --lineage <lineage-id> \
  --max-generations <N>
```

Existing lineage:

```bash
.venv/bin/python tools/training_orchestrator.py status --lineage <lineage-id>
.venv/bin/python tools/training_orchestrator.py status --lineage <lineage-id> --watch
.venv/bin/python tools/training_orchestrator.py stop --lineage <lineage-id>
.venv/bin/python tools/training_orchestrator.py resume --lineage <lineage-id>
```

There is deliberately no `--spec` on resume/status/stop.

## What remains inside the orchestrator

The supervisor owns only mechanism:

- lineage creation and Run Storage paths;
- exclusive run lock;
- durable state machine;
- transactional generation bookkeeping;
- subprocess supervision;
- atomic state writes;
- artifact/hash verification;
- fail-closed technical-result validation;
- heartbeat/resource observation using thresholds supplied by the run-spec;
- performance and learning checks using rules supplied by the run-spec;
- Arena scheduling using the cadence supplied by the run-spec;
- safe soft-stop at generation boundaries;
- crash/recovery bookkeeping;
- terminal warnings/status;
- machine- and human-readable reports.

The active production path does not choose games, workers, contexts, batch size, wait, Arena cadence, performance baseline, alert threshold, stop window, or seed.

## Scientific profiles versus run policy

A scientific profile remains a separately fingerprinted input. For Torus9 it owns game/search/training semantics such as rules, network, MCTS scientific contract, optimizer and replay semantics.

The run-spec explicitly points to the profile and pins its fingerprint. If a task needs a scientifically different profile, it must supply a different explicitly fingerprinted profile; the orchestrator must not synthesize such a change.

- **profile** = what the experiment/model semantics are;
- **run-spec** = how this particular run is scheduled, executed and monitored;
- **orchestrator** = safe execution only.

## Arena

`arena.every_generations` is any positive integer chosen by the run-spec. The old hard limit of 5..10 is not part of V2 production policy.

If Arena is disabled, the run-spec must say so explicitly. If enabled, command, cadence, driver config and startset identity are explicit and fingerprinted.

## No policy fallback

V2 treats absent policy as an error. Examples:

- missing `health.poll_seconds` -> error;
- missing soft-stop default/min/max -> error;
- missing performance `fail_ratio` -> error;
- missing Arena cadence -> error;
- missing Torus worker/batch/wait setting -> driver error.

An empty list is valid only when the task explicitly chooses no checks for that category.

## Storage policy

All run-owned artifacts remain under the stable lineage directory. Cross-lineage evaluations remain under `runs/<topology>/evaluations/`. Checkpoints are referenced rather than duplicated, and all existing Run Storage and Archiving Policy rules continue to apply.
