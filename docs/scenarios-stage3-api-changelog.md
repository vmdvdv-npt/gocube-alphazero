# Scenario API changelog

Added:

- `gocube_golden.scenarios.contracts.ActionRequest` and `ActionOutcome` for
  stable intent/result envelopes. Completed actions require result references,
  commit evidence, and an explicit scientific validity.
- `gocube_golden.scenarios.experiment.policy` as the sole owner of the existing
  strict A/B winner rule and durable `WinnerDecision` representation.
- `gocube_golden.scenarios.komi.policy` as the sole owner of Wilson statistics,
  batch-ledger aggregation, strict extension threshold, and candidate tie-break.
- `gocube_golden.scenarios.calibration.CalibrationRunner` for the independent
  ordered-list calibration mode.

Moved behind compatibility facades:

- `orchestrator_v2.experiment_runner` -> `scenarios.experiment.runner`.
- `orchestrator_v2.komi_calibration` -> `scenarios.komi.runner`.

The old imports, result dataclasses, CLI aliases, state names, evaluation ids,
checkpoint refs, and config fingerprints remain unchanged. The production komi
subclass keeps its old module as a compatibility adapter and delegates its
mathematics to `scenarios.komi.policy`.
