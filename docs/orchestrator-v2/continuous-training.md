# Orchestrator V2 continuous training

Orchestrator V2 exposes two coordinator-level use cases:

- `ExperimentRunnerV2` runs the A/B experiment flow.
- `ContinuousTrainingRunnerV2` advances one stable lineage.

The continuous runner receives a parent `CheckpointRef`, a lineage id, an
`EffectiveConfig`, a finite generation budget or `None`, a positive Arena
cadence, and an `ArenaExecutionConfig`. It prepares
`runs/torus9/active/<lineage-id>/` through `Torus9ProductionLineage`, then
repeatedly calls `ProductionTrainOne`. The runner never builds replay,
creates checkpoints, or invokes the production driver directly.

`runtime/state.json` records the original parent, current committed child,
effective-config identity, budget, cadence, and completed Arena identities.
`control/soft-stop.json` is a durable operator request. A request lets the
active `train_one` finish, then leaves the lineage `SOFT_STOPPED`; a later
launch (or `resume()`) consumes the request and starts at the first
unfinished generation. Existing committed children are reused by
`ProductionTrainOne`.

Arena cadence is relative to the supplied parent generation. Same-lineage
Arena output is owned by the lineage under `arena/generation-NNNN/`; a
cross-lineage comparison uses the canonical `evaluations/` namespace. No
checkpoint is copied between lineages.
