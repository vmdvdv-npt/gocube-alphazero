# Torus9 Production Orchestrator Adapter

## Scope

`tools/torus9_run_driver.py` is the active Torus9 adapter for Production Training Orchestrator V2.

It deliberately contains no preferred Legion worker count, context count, batch cap, wait, Arena workload, reference gap, seed, heartbeat interval, or performance baseline.

Those values are supplied once by the task's immutable run-spec.

## Generation driver config

`generation.driver_config` must explicitly provide:

- `games`;
- `device`;
- `workers`;
- `active_games_per_worker`;
- `total_active_contexts`;
- `inference_batch_cap`;
- `inference_batch_wait_ms`;
- `coalescing`;
- `heartbeat_interval_seconds`;
- `model_init_seed`;
- `selfplay_master_seed`;
- `training_master_seed`.

The adapter checks only structural/runtime consistency. It does not compare values with a preferred Legion preset.

`games` is checked against the referenced scientific profile. To change scientific workload, the task must explicitly provide a profile whose fingerprint contains that change rather than silently overriding the profile inside the orchestrator.

## Arena driver config

When enabled, `arena.driver_config` explicitly provides:

- `reference_gap`;
- `games`;
- `master_seed`;
- `heartbeat_interval_seconds`;
- `execution.workers`;
- `execution.games_per_worker`;
- `execution.inference_batch_rows`;
- `execution.inference_batch_wait_ms`;
- `execution.device`;
- `execution.strict_production`.

`arena.startset` pins `master_seed` and `pairs`; the adapter verifies `pairs == games / 2`.

`strict_production` is itself a run-spec decision. When true, the underlying Torus9 Arena profile may apply its own production compatibility guards. When false, the orchestrator does not force the historical Legion preset.

Arena scientific/game semantics remain in the Torus9 Arena profile, code-pinned by the lineage Git commit. The orchestration layer does not duplicate MCTS/scoring/rules logic.

## Performance gates

All performance baselines and thresholds live in `performance.checks` in the run-spec. The active adapter does not import or compare against a preferred performance reference.

## Resume

The adapter receives the persisted lineage run-spec through `AZ_RUN_SPEC_PATH` and its manifest-pinned fingerprint through `AZ_RUN_SPEC_FINGERPRINT`.

A resumed generation therefore uses the same execution policy and seeds as the original attempt. The external source JSON used at creation is irrelevant after lineage creation.

## Legacy implementation module

`tools/torus9_orchestrator_driver.py` remains an internal compatibility implementation for the transaction/checkpoint mechanics introduced in PR #120. V2 never invokes its CLI and overrides its old execution-policy hooks from the immutable run-spec before generation work. Periodic Arena execution is implemented directly in the V2 adapter using external run policy.

Do not invoke the legacy driver directly for new production runs.
