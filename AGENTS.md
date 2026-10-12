# Operator-controlled training

Before launching, resuming, or reconfiguring Orchestrator V2, read `docs/orchestrator-v2/ORCHESTRATOR_V2_LAUNCH_GUIDE.md` from the executing revision. Follow documented interfaces and run `production_entrypoint job <parameters.json> --check` before launch.

For new Torus9 5CH jobs, prefer `gocube-operator-job-v1` as documented in `docs/orchestrator-v2/operator-job.md`.

## Code changes

Requests to launch, resume, monitor, stop, or tune training authorize necessary configuration changes **and modifications to `gocube_golden/orchestrator_v2/`**, including fixing bugs or implementing missing orchestrator functionality.

Other subsystems (engine, model, replay, adapters, notifications, tools, services, environment) require explicit permission before modification. Do not bypass supported interfaces with temporary launchers or monkey-patches.

When changing public interfaces, update the launch guide and its reviewed interface fingerprint.

For repository changes: commit, push the branch, and create or update a PR. Batch related changes and run CI only after the final push.

## Runtime safety

Never modify immutable artifacts, checkpoints, replay, or evaluation results. Do not restart, migrate, change the code pin, or modify the working tree of a live run unless explicitly requested.

Develop fixes in a separate worktree. Use isolated test data and mocked transports; never use production games for tests.

Preserve standard orchestrator notifications. Do not disable delivery, suppress credentials, or send unsolicited lifecycle messages.

For passive throughput/GPU monitoring, use `tools/training_efficiency_monitor.py` and write outputs outside the live run.

Changes to these rules require explicit user permission.
