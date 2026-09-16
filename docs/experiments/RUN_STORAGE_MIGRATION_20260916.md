# Run storage migration — 2026-09-16

The local run tree was migrated to the run-storage policy without deleting
non-empty training, replay, log, report, or checkpoint artifacts.

The first move-only pass performed 81 whole-directory moves and 48 artifact
merges. It produced 68 lineage directories and 32 evaluation directories.
The follow-up split moved 39 items from multi-lineage legacy bundles, producing
16 additional direct lineage/evaluation locations. Bundle-level metadata was
retained under `docs/experiments/legacy-bundles/`.

Current lineage states are represented by manifests:

- `runs/torus9/active/torus9-golden-v3-20260914-run03/` — the active Torus9
  continuation point;
- `runs/<topology>/archive/<lineage-id>/` — retained completed lineages;
- `runs/<topology>/evaluations/<evaluation-id>/` — independent Arena and
  evaluation results without checkpoint copies.

The 42 duplicate physical copies of six byte-identical legacy Cube-4 model
checkpoints were replaced with relative references to the canonical
`c4-t001-c4-c001` checkpoint files. Each affected manifest records the parent
lineage, checkpoint, canonical path, and SHA-256. The deduplication record is
`checkpoint-deduplication-20260916.json`.

`data/.gocube-game-ids` remains in place as global service state. The old
root-level artifact directories are empty compatibility remnants only; new
runtime code writes through `gocube_golden.run_storage`.

Audit records:

- `run-storage-migration-20260916.json` — initial move/merge inventory;
- `run-storage-split-legacy-bundles-20260916.json` — legacy bundle split;
- `checkpoint-deduplication-20260916.json` — checkpoint identity cleanup.
