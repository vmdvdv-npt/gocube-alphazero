# Run Storage and Archiving Policy

This repository keeps runtime artifacts on the local run filesystem. Git
tracks source code and concise scientific documentation; the `runs/` tree is
ignored because it may contain large checkpoints and replay data.

## 1. One lineage = one stable directory

Every independent training lineage lives in:

```text
runs/<topology>/active/<lineage-id>/
```

A continuation or resume of the same training reuses that directory. It does
not create a new lineage. Checkpoints, self-play/replay data, logs, metrics,
reports, lineage-specific Arena results, and `manifest.json` belong inside it.

Tracked summaries in `docs/` are repository-level scientific records. They
must reference the canonical run path but are not a second physical copy of
the run artifact tree.

## 2. Checkpoints are referenced, not copied

When a new lineage starts from an existing checkpoint, its manifest records:

- parent lineage;
- checkpoint identifier and canonical path;
- SHA-256 hash.

The referenced checkpoint remains retained while any retained lineage or
evaluation depends on it.

## 3. Evaluations live separately

Arena comparisons between independent lineages live in:

```text
runs/<topology>/evaluations/<evaluation-id>/
```

An evaluation stores results, configuration, and checkpoint references/hashes;
it never stores model checkpoint copies.

## 4. Manifest

Each lineage has a machine-readable `manifest.json` containing at least:

```text
lineage_id
topology
status
parent_checkpoint
git_commit
config_fingerprint
created_at
checkpoint_hashes
```

Allowed lineage statuses are `ACTIVE`, `ARCHIVED`, and `DISCARDED`.
`run.md` is optional human-readable context. For evaluations, `manifest.json`
also records both compared checkpoints and their hashes.

## 5. State transitions

An active lineage is moved, never copied, to:

```text
runs/<topology>/archive/<lineage-id>/
```

when it is no longer active but remains scientifically or operationally
useful. A discarded lineage has its heavy artifacts removed only after a
short record is written to `docs/experiments/discarded/`.

## 6. No duplicate models

Checkpoint identity is deduplicated by SHA-256. One canonical physical copy
is retained; other lineages and evaluations reference it. A model that is
semantically equivalent but has a different artifact hash is not silently
deduplicated.

## 7. Path ownership

New training, self-play, Arena, and evaluation code uses
`gocube_golden.run_storage` (or a caller-provided path produced by it). New
code must not write lineage artifacts to unrelated root-level directories such
as `checkpoint/`, `data/`, `arena-results/`, or `training_reports/`.

Global service state, caches, and repository-level metadata are exceptions.
The persistent game-ID registry at `data/.gocube-game-ids` is one such global
service file. Tracked scientific summaries under `docs/` are repository-level
metadata and point to, rather than duplicate, ignored run artifacts.
