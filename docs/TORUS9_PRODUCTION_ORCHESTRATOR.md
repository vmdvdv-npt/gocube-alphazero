# Torus9 Production Adapter V3

`tools/torus9_run_driver.py` is the single active Torus9 process adapter for Production Training Orchestrator V3.

It does not import or monkey-patch historical orchestrator drivers. Production execution policy is read exclusively from the immutable lineage run-spec.

## Scientific boundary

The referenced fingerprinted Torus9 scientific profile remains authoritative for rules, network, MCTS/search semantics, optimizer, replay, targets and the intended self-play workload. The adapter validates that explicit run-spec scientific bindings such as game count and reproducibility seeds agree with that profile rather than silently overriding it.

Changing scientific semantics therefore requires a different explicitly fingerprinted profile. The orchestrator never synthesizes a scientific change.

## Generation execution policy

`generation.driver_config` must explicitly contain:

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

There are no active production defaults for these values in the driver CLI.

## Arena policy

When Arena is enabled, `arena.driver_config` explicitly contains:

- `comparison_mode` (`candidate-vs-prior-generation` for periodic Torus9 Arena);
- `reference_gap`;
- `games`;
- `master_seed`;
- `heartbeat_interval_seconds`;
- `execution.workers`;
- `execution.games_per_worker`;
- `execution.inference_batch_rows`;
- `execution.inference_batch_wait_ms`;
- `execution.device`;
- `execution.strict_production`;
- `execution.min_mean_inference_batch_rows`;
- `execution.min_effective_cpu_cores`;
- `execution.early_gate_enabled`;
- `execution.early_gate_min_forwards`;
- `execution.early_gate_min_wall_sec`.

The fixed startset block is separately fingerprinted and must explicitly contain a generator/schema, master seed and pair count matching the Arena workload.

Periodic same-lineage Arena output is lineage-owned:

`runs/torus9/active/<lineage>/arena/generation-NNNN/`

It is not written to the global cross-lineage `evaluations/` namespace.

## Progress heartbeat

The adapter publishes heartbeat V2 with separate liveness and semantic progress timestamps. Phase transitions such as load-state, self-play completion, training, reload verification and Arena verification advance the semantic progress token; the supervisor independently updates process liveness monitoring and fails closed on stale progress according to the run-spec.

## Resume and transaction safety

A failed current generation may remove only its uncommitted temporary/current-generation artifacts before re-running. Previous committed checkpoints, replay, metrics and Arena evidence remain untouched.

Generation completion is published only after checkpoint reload verification, replay/resume-state persistence and artifact SHA-256 validation. Arena snapshots training-owned artifacts before and after evaluation and must explicitly prove `training_mutated=false`.

## Cross-lineage continuation

A new Torus9 lineage may continue from a committed parent generation without
copying the parent artifacts. The creation command requires the parent
checkpoint, its SHA-256, the parent generation and the corresponding rolling
replay artifact. The CLI also records the checkpoint metadata and replay
identities in `manifest.json` after validating them locally.

On the first child generation, the adapter loads that external checkpoint and
rolling replay, verifies their recorded identities, and preserves the parent
training clock and replay eviction count. Later generations use the child
lineage's own committed artifacts. A checkpoint-only reference is rejected:
true continuation requires the optimizer-bearing checkpoint and its matching
rolling replay.

## Canonical implementation

The former `tools/torus9_orchestrator_driver.py` duplicate has been removed. New production work has one Torus runtime entrypoint: `tools/torus9_run_driver.py` through the generic orchestrator.
