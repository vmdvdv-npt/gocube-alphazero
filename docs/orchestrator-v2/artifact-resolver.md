# Orchestrator V2 Stage 1 — ArtifactResolver

`gocube_golden.orchestrator_v2.artifact_resolver.ArtifactResolver` is the
single runtime owner of V2 checkpoint-node opening, immediate-parent traversal,
ancestor lookup, replay-window construction, cycle detection, generation
checks, and bounded immutable artifact verification.

The canonical V2 node locator is:

```text
<lineage-root>/metadata/checkpoints/<checkpoint_id>.json
```

This is a storage detail and is not a new persisted contract. It is separate
from legacy `checkpoints/M*.metadata.json` files. Nodes contain only one
immediate parent and one generation-specific `fresh_replay`; full ancestry and
replay windows are never persisted.

`checkpoint()` delegates physical checkpoint opening to
`gocube_golden.run_storage.resolve_checkpoint()`. Effective-config and
provenance references are opened relative to the checkpoint's owning lineage,
and fresh replay artifacts are opened only when `replay_window()` requests
them. No legacy inference or production artifact migration is performed.
