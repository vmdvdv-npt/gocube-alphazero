# Orchestrator V2 workflows

Orchestrator V2 accepts one JSON scenario file and persists its state under
`runs/<topology>/orchestration/workflows/<workflow_id>/`.  Run or resume a
scenario with:

```bash
.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint \
  run configs/gocube/orchestrator_v2_examples/komi_select_training.json \
  --runs-root runs
```

Rerunning the same command resumes `RUNNING` actions, reattaches through the
existing action supervisor, and never repeats a `COMPLETED` action.  Inspect
`state.json` for `steps`, `resolved_config`, `attempts`, `events`, `outputs`,
and `stop_reason`.  A workflow `stop` action durably records `STOPPED`; later
steps remain pending and another invocation cannot bypass that stop.

The examples use placeholder checkpoint references and are templates, not
production commands. Replace every `PLACEHOLDER` value with an artifact from
the intended run storage before execution.

The four supported compositions are:

- `arena_only.json`: an evaluation with search and execution settings owned by
  this Arena run.
- `continuous_with_arena.json`: resumable training with periodic Arena.
- `komi_select_training.json`: calibration results → valid-result selection →
  the selected komi applied to the next effective training config.
- `ab_select_continue.json`: two training arms → Arena comparison → selection
  → continuation from the selected checkpoint and effective config.

References use `${step.outputs.field}`.  List results can be addressed with
`${step.outputs.1.metrics.bias}`.  Selection refuses an explicitly invalid
result and requires the selected metric to be present and finite. Arena
results expose a JSON-only checkpoint/effective-config/metrics contract;
model and replay contents are never copied into workflow state. Arena
parameters that the current engine cannot vary (noise, temperature, fast
search, resign and non-deterministic ties) must be supplied with their fixed
supported values or are rejected before work starts.

A technical child failure is recovered by the action's Supervisor V2 budget.
Workflow retries are a separate, opt-in budget for a new action invocation;
they do not acknowledge `supervisor-stop.json` automatically.  Configuration,
compatibility, data-integrity and ownership errors stop the workflow and need
operator review.  `KeyboardInterrupt` and `SystemExit` are propagated as
operator actions, not converted into retries.

The ordinary Arena path records low CPU/GPU utilization, batching, lane and
active-context observations as diagnostics.  Use the explicit
`strict_performance: true` Arena setting only when those performance targets
are intended to be hard gates; scientific validity, technical outcomes,
checkpoint compatibility and process ownership remain enforced normally.
