# Stage 1: structured events and durable notification delivery

This document records the implementation boundary for the first notification
refactor.  The computation result remains authoritative; Telegram is an
at-least-once delivery channel.

## Producer inventory

| Previous producer / call site | New owner | Event type | Compatibility |
| --- | --- | --- | --- |
| `orchestrator_v2/arena_runner.py` start | `ArenaRunner` | `ARENA_STARTED` | old `arena-start:*` adapter |
| `orchestrator_v2/arena_runner.py` returned verified result, including reuse | `ArenaRunner` / `reconcile_completed` | `ARENA_COMPLETED` | old `arena-complete:*` adapter |
| `orchestrator_v2/arena_runner.py` execution exception | `ArenaRunner` | `ARENA_FAILED` | old `arena-failed:*` adapter |
| `_continuous_training_core.py` training/run boundaries | continuous lineage coordinator | `TRAINING_STARTED`, `RUN_COMPLETED`, `RUN_FAILED`, `STOP_REQUESTED`, `RUN_STOPPED` | old `continuous:*` adapter |
| `continuous_training.py` generation boundary | confirmed checkpoint boundary | `GENERATION_STARTED`, `GENERATION_COMMITTED` | old adapter only for legacy injected notifiers |
| `experiment_runner.py` experiment/stage decisions | experiment coordinator | `EXPERIMENT_STARTED`, `EXPERIMENT_STAGE_DECIDED`, `EXPERIMENT_COMPLETED` | old `experiment:*` adapter |
| `komi_calibration.py` calibration state transitions | calibration coordinator | `CALIBRATION_STARTED`, `CALIBRATION_DECIDED`, `CALIBRATION_HANDOFF_COMPLETED`, `RUN_FAILED` | old calibration key adapter |
| `production_entrypoint.py` dependency composition | no fact owner | creates one dispatcher and closes it | `TelegramNotifier` name is a factory compatibility alias |
| `telegram_notifier.py`, `torus9_run_owned.py` | legacy compatibility surface | old outbox/receipt and presentation policy | retained until migration is complete |

`ARENA_COMPLETED` is emitted only after `ArenaRunResult` has been returned from
the existing Arena engine and contains an explicit `validity` plus W/L/D.  A
Telegram outage therefore cannot create `ARENA_FAILED` or cause another Arena
execution.

## Event storage

For owner root `R`, the new service stores:

```
R/notifications/events/<sha256(event_id)>.json
R/notifications/delivery/<sha256(event_id)>.json
R/notifications/dispatcher.lock
R/notifications/diagnostics.jsonl
```

Event and delivery records are written with the existing unique-temp-file
atomic writer.  `dispatcher.lock` is held only while a dispatcher drains a
root.  It is never held while an Arena, training child, or other engine runs.
There is at most one active sender per root; a second dispatcher can still
publish durable events and will leave delivery for the owner holding the lock.

The event ID is derived from the event type and full logical identity.  It does
not use PID, launch time, or formatted text.  A repeat publication returns the
first immutable event.  If scientific payload differs for the same ID, the
first event is retained and both evidence references are recorded in local
diagnostics.

## Delivery states and recovery

`PENDING`, `RETRY_WAIT`, `DELIVERED`, and `BLOCKED_CONFIGURATION` are stored
separately from event files.  A receipt is written only after a positive
transport response.  A crash after Telegram accepts a message but before the
receipt is durable can duplicate a message; this is the documented
at-least-once boundary, not exactly-once delivery.

`NotificationDispatcher.close()` and `flush()` use one total time budget.
Undelivered events remain on disk.  A standalone process does not deliver while
it is stopped; a future drain command can construct the same dispatcher without
starting training or Arena.  No cron job or system service is installed by
this change.

The compatibility reader recognizes the PR215 locations
`telegram-outbox/*.json` and `telegram-notifications.jsonl`.  It ignores
malformed lines, keeps unknown keys diagnosable, never deletes source files,
and only maps an old key to a new scientific event when the caller supplies explicit
evidence.  An old successful receipt is imported as `DELIVERED`; an old pending
envelope is represented as a `legacy-envelope` compatibility event and delivered
as its saved text without pretending that W/L/D can be recovered from that text.

## Telegram transport contract

The transport sends one JSON `sendMessage` request with a finite timeout.  HTTP
429 and `parameters.retry_after` use bounded retry scheduling; network and
5xx errors retry with capped exponential backoff.  credential, destination,
serialization, and message-size errors become `BLOCKED_CONFIGURATION` with a
safe error code.  Tokens and full request URLs are never put in diagnostics.

## Rollout and rollback

Before a production rollout, copy the owner root and record the new event and
delivery files plus the legacy pending/receipt list.  Start one dispatcher
owner at a safe process boundary.  To stop the new sender, stop delivery while
retaining `R/notifications`; continue computation only if its result storage is
healthy.  Do not point the old sender at new event files without first draining
or explicitly exporting pending events on a copy.  Legacy source files are not
removed automatically.
