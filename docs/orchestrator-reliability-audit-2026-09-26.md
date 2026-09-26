# Orchestrator reliability audit after PR #214

Base: `246f454779769e94c897026eb4e423ed26ec759a` (merged PR #214).

## Confirmed defects and fixes

- **Standalone Arena completion was silent.** The public production Arena runner
  sent a start notification but returned without sending results. Only selected
  coordinators sent their own completion messages. The shared runner now sends
  `ARENA COMPLETED` with candidate/reference, validity, W/L/D, and execution
  revision for both fresh and reused results. Failures send `ARENA FAILED` and
  preserve the original exception. Continuous training avoids a second generic
  completion; experiment winner decisions remain separate messages.
- **Transport failures discarded notifications.** After the immediate HTTP
  attempts failed, the old notifier only wrote an error log. Notifications now
  enter an atomic disk outbox before delivery. A background worker retries and
  a new notifier resumes its outbox. Directly constructed V2 notifiers participate
  in shutdown flushing. Receipt errors stay fail-open; malformed receipt lines
  no longer hide later valid receipts.
- **Concurrent heartbeat writes shared a temporary filename.** The Arena child
  publishes from both its heartbeat thread and progress callback. The shared
  writer used only the PID in its temporary name, allowing one writer's rename
  to remove the other writer's source. Each write now uses an exclusive unique
  temporary file, retaining atomic replacement and fsync. A barrier-based test
  reproduces the old `FileNotFoundError` deterministically.
- **Result reuse misreported execution provenance.** A cached Arena result used
  the newly requested commit in its returned result. It now reads the original
  execution revision from validated saved provenance, or leaves it unknown for
  older artifacts that did not record a revision.

- **Incomplete-output cleanup erased notification state.** Standalone notifiers
  can create the evaluation directory before its identity is published. Recovery
  now retains the outbox, delivery receipts, and notifier error log while removing
  incomplete Arena products. Identity publication explicitly permits that cleared
  directory, but still refuses to replace an existing identity marker.
- **An orphaned malformed/foreign execution intent could be erased.** When no
  active-child file remained, cleanup skipped supervisor validation. It now
  consults the supervisor plan before reclaiming an intent-only directory and
  preserves rejected ownership evidence.

## Verification

The original focused suite passed 189 tests despite these defects. New regression
checks cover the production completion path, notification recovery and dedupe,
background retry without a new training event, receipt disk errors, concurrent
heartbeat publication, and stored execution provenance.

Four new regression checks fail against a separate archive of the base revision:
concurrent heartbeat publication, undelivered completion recovery, production
completion notification, and reused execution provenance. They pass with the fix.

PR #214's dead-child recovery regression remains covered. Additional checks verify
that live children, explicit stops, malformed active-child records, foreign
active-child identities, and foreign execution intents remain untouched.

Commands:

```sh
.venv/bin/python -m compileall -q gocube_golden
.venv/bin/python -m pytest -q -m 'not katago_reference'
```

Final verification: **847 passed, 1 skipped, 102 deselected** in the full command
above; compilation and `git diff --check` passed. The 102 deselected checks are
the explicitly excluded KataGo-reference tests.

Telegram tests replace the HTTP transport; they do not contact a real channel.

## Operational limits

Outbox delivery is at-least-once: a process crash after Telegram accepts a message
but before its receipt is saved can cause a duplicate. Pending messages retry
while the notifier process runs, and resume when a notifier for the same runtime
root starts again. An exited standalone command cannot retry during downtime.

Existing results and failed-send logs can diagnose a particular missed message,
but this audit did not inspect a live training host or verify its Telegram setup.
Changes to pinned child code require a new immutable execution revision; already
running children do not load these edits. No live run was restarted by this audit.
