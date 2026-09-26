# Scenario operations quick reference

Each mode remains one command. The coordinator owns the plan and resumes the
same durable state; it is not necessary to launch individual Arena batches or
copy checkpoints.

```bash
# ordinary continuous training
.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint \
  continuous config.json --runs-root runs

# explicit performance tuning
.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint \
  performance-tuning tuning.json --runs-root runs

# A/B experiment with optional Stage 2
.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint \
  experiment experiment.json --runs-root runs
```

For a declarative run-spec, use `run run-spec.json`; its `mode` selects the
same handlers. A workflow continues to use the existing `WorkflowRunner` and
its dependency/output/select/stop contract.

## Recovery quick check

- A live child has matching owner/execution identity and a live process group;
  reattach through `SupervisorV2`.
- A ready result has matching request/evaluation identity, committed refs and
  validity evidence; reuse it without a new engine launch.
- An undelivered message has a saved event plus a non-`DELIVERED` delivery
  record; drain notifications without rerunning science.
- A durable stop, foreign owner, malformed state, or integrity mismatch is a
  prohibition to continue. Preserve evidence and stop at the boundary.

The coordinator order is workflow lock -> scenario owner state -> execution
owner. Notification delivery is outside that lock order and never decides a
retry or scientific branch.
