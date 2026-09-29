# Operator-controlled training

The user owns changes to the training system. A request to launch, resume,
monitor, stop, or tune a run authorizes editing its declarative configuration and
using the existing documented Orchestrator V2 commands only. For new Torus9 5CH
jobs, use the simple `gocube-operator-job-v1` JSON and the `production_entrypoint
job` command documented in `docs/orchestrator-v2/operator-job.md`. Do not invent
a low-level plan, launcher, or service as a substitute for a missing parameter.

Do not change orchestrator, engine, model, replay, artifact, supervision,
notification, launcher or service code/settings without the user's explicit
permission for that specific code change. This includes
`gocube_golden/orchestrator_v2/`, `gocube_golden/notifications/`, training and
self-play adapters, `training_engine.py`, `selfplay_engine.py`, `tools/`, systemd
units, environment overrides and scripts outside the repository. Moving code
elsewhere, monkey-patching, or writing a one-off launcher is not an exception.
A previous permission applies only to its stated task, not future launches.

If a requested parameter is unsupported, report the exact limitation and ask
for a separate code-change instruction. Never silently patch the implementation
in order to make a training request work.

Notifications are part of the standard orchestrator lifecycle. Preserve their
configured defaults; do not suppress credentials, disable delivery, manually
send lifecycle messages or add a parallel notification loop. If notifications
are unavailable, diagnose and report the problem without changing policy.

Do not restart, stop, migrate, change a code pin, modify an immutable artifact,
or change the working tree/HEAD used by a live run without explicit permission.
Develop authorized fixes in a separate worktree. Tests must use isolated data
and mocked transports, never production games or unsolicited Telegram messages.

Changes to these rules also require the user's explicit permission.
AGENTS.md is an instruction, not a security boundary.
