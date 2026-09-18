# Orchestrator V2 Stage 0 — contracts and legacy artifact audit

Status: **Stage 0 complete: contracts frozen, live legacy audit executed, strict parsing enforced**
Date: 2026-09-18  
Baseline: `main` at `997e981bff21fbad6507b1e3b7f9ab1be36174c4` (`Honor resolved cross-lineage replay sources (#139)`)

This document fixes the persistence contracts that Stage 1 may treat as architecture, not as open design questions. It also records the completed read-only legacy audit of the live run-storage tree.

## Scope and evidence boundary

Stage 0 did **not** run self-play, training, Arena, migration, or orchestration. The live audit was read-only: it did not move, rewrite, archive, hash-scan, or delete any production artifact.

The final audit scanned the mounted live tree on 2026-09-18. It covered 104 run/evaluation directories, 3,035 files, 139 manifests, 118 `generation-N.complete.json` records, 267 `M*.metadata.json` checkpoint metadata records, and 10 artifact catalogs. The detailed classification below focuses on Torus9, including the useful M0-M17, M18-M47, M48-M54, M55-M80, M81+, post-AB, and failed/incomplete continuation records.

The earlier repository-only first pass is retained later in this document for provenance, but its availability/status statements are superseded by section 15. The audit deliberately separates:

1. facts proven by committed durable records, chiefly `docs/experiments/run-storage-migration-20260916.json`, `run-storage-split-legacy-bundles-20260916.json`, `checkpoint-deduplication-20260916.json`, and the migration report; and
2. facts proven by the final live read-only scan; and
3. facts that remain unproven and therefore fail closed pending explicit bootstrap or stronger durable evidence.

No parent edge is inferred from an `M<number>` name, adjacent generation numbers, a lineage name, directory proximity, or historical expectation. A legacy edge that is not proven by durable metadata fails closed and is classified as requiring explicit bootstrap/mapping until stronger evidence is read.

The current Torus9 Golden values were reviewed as context only. They are configuration data, not schema constraints; non-Golden supported values are valid. Historical `komi = 7.5` material is legacy/error context and is not a Torus9 default.

---

# 1. Fixed architecture invariants

1. The checkpoint graph is the only ancestry model. A node stores one immediate parent or explicitly declares itself genesis.
2. Full ancestry is computed by parent traversal. It is never persisted as `ancestors`, promotion tables, fork tables, bootstrap ancestry tables, or `replay_references`.
3. Lineage is a storage container, not ancestry. Crossing a lineage boundary is a normal parent edge.
4. Checkpoint and replay artifacts are not copied across lineages. Cross-lineage use is by immutable reference.
5. SHA-256 is the immutable physical-artifact identity.
6. `gocube_golden.run_storage.resolve_checkpoint()` remains the low-level checkpoint opener/ownership verifier. Stage 1 `ArtifactResolver` must build on it.
7. Resolve once: the future resolver owns ancestry/replay resolution; runners receive resolved inputs and do not rediscover them.
8. Run state chooses the next business step but is not checkpoint history and is not commit truth.
9. A generation is committed only when the existing atomic commit contract and required committed artifacts say it is committed.
10. Arena identity and Arena result validity are separate facts. Decision provenance is separate from both.

---

# 2. Canonical serialization and hashing

All V2 immutable content fingerprints use the existing repository primitives in `gocube_golden.provenance`:

```python
canonical_json(value)
sha256_fingerprint(value)
file_sha256(path)
```

Canonical JSON therefore uses sorted keys and compact separators. Contract fingerprints are `sha256:<64 lowercase hex>` of that canonical JSON. Artifact SHA fields use the same canonical string form.

No second canonical-JSON implementation is introduced.

Every persisted V2 top-level structure has both an explicit schema name and `version = 2`.

---

# 3. Contract A — CheckpointNode V2

Schema: `gocube-checkpoint-node-v2`, version `2`.

Canonical shape:

```json
{
  "schema": "gocube-checkpoint-node-v2",
  "version": 2,
  "checkpoint": {
    "topology": "torus9",
    "lineage_id": "lineage-b",
    "checkpoint_id": "M83",
    "generation": 83,
    "path": "checkpoints/M83.pt",
    "sha256": "sha256:..."
  },
  "genesis": false,
  "parent": {
    "topology": "torus9",
    "lineage_id": "lineage-a",
    "checkpoint_id": "M82",
    "generation": 82,
    "path": "checkpoints/M82.pt",
    "sha256": "sha256:..."
  },
  "fresh_replay": {
    "path": "...lineage-relative fresh replay artifact...",
    "sha256": "sha256:..."
  },
  "effective_config": {
    "artifact": {"path": "...lineage-relative config snapshot...", "sha256": "sha256:..."},
    "fingerprint": "sha256:..."
  },
  "provenance": {
    "path": "...lineage-relative provenance record...",
    "sha256": "sha256:..."
  }
}
```

## 3.1 Checkpoint reference

The canonical checkpoint reference contains exactly the information needed to make a cross-lineage reference unambiguous and tamper-detectable:

- topology;
- owning `lineage_id`;
- logical `checkpoint_id`;
- generation;
- lineage-relative artifact path;
- SHA-256.

The low-level `run_storage` reference semantics already use owning lineage, relative path, and SHA for cross-lineage resolution. V2 adds logical checkpoint id/generation/topology to the contract because they are part of the graph node's logical identity; Stage 1 still delegates opening/ownership/SHA verification to `resolve_checkpoint()` rather than implementing a competing opener.

A short operator name such as `M122` is **not** a persisted checkpoint reference. Stage 1 may accept it only as a lookup query: one usable match is allowed; zero or multiple matches require explicit disambiguation. It must never guess.

## 3.2 Parent semantics

`genesis = true` requires `parent = null`. `genesis = false` requires exactly one full parent reference. Generation number alone does not define genesis and does not prove a parent relation.

Same-lineage and cross-lineage parents use the same structure. There is no lineage-boundary special case.

## 3.3 Replay semantics

`fresh_replay` is only the replay artifact created by this node's generation. It is not a replay window and not an ancestry list. The future `replay_window(parent, count)` is computed by traversing parent nodes and selecting each node's fresh replay.

`ancestors` and `replay_references` are explicitly rejected by the machine-readable schema implementation.

---

# 4. Contract B — EffectiveConfig V2

Schema: `gocube-effective-config-v2`, version `2`.

Stage 0 chooses **one** model: an immutable effective-config snapshot is persisted as a standalone artifact; each CheckpointNode stores one `EffectiveConfigRef` containing the snapshot artifact `{path, sha256}` plus its canonical content fingerprint.

There is no competing embedded-snapshot model.

This choice matches existing artifact-catalog/provenance practice: immutable records remain independently hashable and catalogable, while a checkpoint node stays small. A continuation can resolve the node, resolve exactly one config artifact, verify both file SHA and content fingerprint, and reconstruct the actual parameters without Golden-name inference.

Canonical top-level domains:

```json
{
  "schema": "gocube-effective-config-v2",
  "version": 2,
  "topology": "torus9",
  "compatibility": {},
  "self_play": {},
  "training": {},
  "replay": {},
  "execution": {},
  "arena": {},
  "supervision": {},
  "extensions": {}
}
```

Required identity fields are `topology` plus a non-empty `compatibility` object. The compatibility object is the place for board/rules/model/checkpoint/replay-format identities required by the selected topology/runner. The other domains are extensible JSON objects and are not universally required, because not every topology/runner supports every setting.

The schema does not whitelist Golden values. Examples such as `mcts_simulations = 256`, `learning_rate = 0.0002`, or `games_per_iteration = 90` are structurally valid if the runner supports them.

Scientific and execution settings are distinct domains/objects even when persisted in the same immutable snapshot.

---

# 5. Declarative parameter changeability

The machine-readable module owns versioned policy metadata `PARAMETER_CHANGEABILITY_V2`; the future orchestrator must not contain a parameter-by-parameter `if/elif` switch.

The policy has three classes:

| Class | Meaning | Examples |
|---|---|---|
| `next_generation` | first not-yet-started generation | self-play sims/games/search knobs, training LR/steps/batch, replay window/cap, Arena cadence/games/reference gap |
| `next_execution_unit` | next child/process start | workers, active contexts, inference batching/wait, watchdog, supervision thresholds |
| `incompatible_new_run` | runtime amendment rejected; new run/compatibility boundary required | topology and compatibility identity |

The metadata may evolve as a versioned policy table, independently of `orchestrator_v2.py` control flow.

---

# 6. Contract C — Persistent RunState V2

Schema: `gocube-orchestrator-run-state-v2`, version `2`.

Business states are deliberately small:

```text
READY
RUNNING_GENERATION
GENERATION_COMMITTED
RUNNING_ARENA
STOPPED
TERMINAL_FAILURE
```

The persisted record contains:

- run id and `continuous | experiment` mode;
- lineage id where applicable;
- current business state;
- last committed full checkpoint reference **together with** an authoritative generation commit artifact reference;
- active generation or Arena execution unit and attempt;
- bounded-retry attempt state;
- soft-stop request;
- pending required step;
- optional already-durable queued transition;
- immutable base config reference;
- ordered identities of applied amendment artifacts;
- creation/update timestamps.

`GENERATION_COMMITTED` is invalid without an external commit-artifact reference. RunState therefore cannot manufacture commit truth. Recovery must validate the referenced generation commit and required committed artifacts using the commit layer.

Internal supervisor states may exist transiently but do not expand this business state machine.

---

# 7. Contract D — RuntimeAmendment V2

Schema: `gocube-runtime-amendment-v2`, version `2`.

The base run config is immutable. Accepted changes are append-only amendment artifacts. Canonical content includes:

- amendment id;
- run id;
- `accepted_at`;
- `requested_changes` list;
- for each change: parameter path, old value, new value, change class, fully resolved boundary;
- base effective-config artifact identity/fingerprint;
- resulting effective-config artifact identity/fingerprint.

A persisted accepted amendment never says merely “next generation”. The resolved boundary is concrete, for example:

```json
{
  "path": "self_play.mcts_simulations",
  "old_value": 128,
  "new_value": 256,
  "change_class": "next_generation",
  "resolved_boundary": {"kind": "generation", "generation": 122}
}
```

Class-3 changes cannot be represented as accepted runtime amendments. They require a new run/compatibility decision instead.

---

# 8. Contract E — Arena EvaluationIdentity V2

Schema: `gocube-arena-evaluation-identity-v2`, version `2`.

The generic identity is a generalization of the full identity/reuse checks already present in the staged harness merged through PR #136. It contains:

- candidate full checkpoint reference including SHA;
- reference full checkpoint reference including SHA;
- games/workload;
- master seed;
- startset id, artifact SHA, and startset fingerprint;
- scientific Arena contract;
- material execution/reproducibility contract.

The entire structure is canonically serialized/fingerprinted. Changing candidate SHA, reference SHA, games, seed, startset, scientific search, or material execution settings produces a different evaluation fingerprint.

## 8.1 Result validity is not identity

Arena result validity is a separate value:

```text
VALID
INVALID
TECHNICAL
CRITICAL
```

Only `VALID` is reusable as a scientific result. Existing output files are insufficient for reuse if the recorded result is invalid, technical, critical, tampered, or identity-mismatched.

## 8.2 Decision provenance is separate

Rules such as `W > L`, tie handling, staged promotion, or another deterministic winner rule do not change Arena execution identity. They belong to experiment decision provenance, not the EvaluationIdentity.

---

# 9. Ownership table

| Fact | Single owner |
|---|---|
| immediate parent / parent-chain traversal | Stage 1 `ArtifactResolver` over CheckpointNode records |
| checkpoint physical path/owner/SHA verification | artifact store / `run_storage.resolve_checkpoint()` |
| effective config snapshot + fingerprint | effective-config resolver/store |
| generation commit truth | generation transaction/commit layer |
| fresh replay identity of a checkpoint | CheckpointNode, verified through artifact store/catalog |
| replay-window composition | `ArtifactResolver` traversal (future Stage 1) |
| Arena execution identity | Arena identity layer / future ArenaRunner |
| Arena result validity | Arena result validation layer |
| process liveness/heartbeat/retry mechanics | Supervisor |
| business next step | Orchestrator V2 |
| accepted runtime configuration changes | append-only RuntimeAmendment records |

No immutable fact has two authoritative owners.

---

# 10. Existing V1 mapping

| V2 need | Existing source | Reuse | Missing / do not carry forward |
|---|---|---|---|
| checkpoint opening/owner/SHA | `gocube_golden/run_storage.py` | `resolve_checkpoint()`, active/archive lookup, cross-lineage refs, ownership/SHA checks | no graph traversal belongs here |
| canonical JSON/hash | `gocube_golden/provenance.py` | `canonical_json`, `sha256_fingerprint`, `file_sha256` | no duplicate implementation |
| immutable artifact index | `gocube_golden/artifact_catalog.py` | path/SHA/size and generation artifact indexing | catalog is not ancestry |
| durable process mechanics | V1 orchestrator/run lifecycle | atomic writes, fsync/replace, transactions, active-child identity, process groups, heartbeat, bounded retry, soft stop, recovery | V2 must not subclass the large V1 orchestrator or inherit its topology-specific control flow |
| run configuration | `run_spec.py` and current configs | source material and validation patterns | names/Golden profiles cannot substitute for per-checkpoint effective config |
| staged experiment | PR #136 harness | config-driven arms, immutable spec, staged order, full Arena identity/reuse validation, tamper rejection | Torus9 hardcoding, equal-budget requirement, Golden permission gate, generic topology checks |
| cross-lineage replay resolution | main through PR #139 | resolved external lineage/path sources | replay-source lists are execution inputs, not a second permanent ancestry model |
| lineage manifests | run-storage policy | storage/index identity, checkpoint hashes, run description | manifest must not become a full ancestry database |

---

# 11. Read-only legacy artifact audit — historical first pass

The subsection below preserves the repository-only evidence and limitation from the initial Stage 0 pass. It is retained for provenance; the final live results in section 15 supersede its availability/status statements.

## 11.1 What the committed records prove

The 2026-09-16 run-storage migration report records a move-only migration into `runs/<topology>/{active,archive,evaluations}`. The committed inventory contains checkpoint file names and SHA-256 values for the then-active Torus9 lineage `torus9-golden-v3-20260914-run03`; the inventory shows `M0` through at least `M17` with recorded hashes. The split report records movement of earlier Torus9 lineage directories out of legacy bundles, including `torus9-golden-v3-20260913-run01`, `torus9-golden-v3-20260914-run02`, several learning-proof/stable-learning lineages, and independent evaluations.

The same audit records also document Cube-4 checkpoint deduplication by immutable SHA/reference. That dedup relationship is a **physical storage identity relation**, not proof of a training parent edge, and V2 must not reinterpret it as ancestry.

## 11.2 What the committed records do not prove

The committed migration inventory does not provide a per-checkpoint immediate-parent edge for the useful Torus9 checkpoint sequence. Searching the durable migration snapshot for parent metadata does not yield node-by-node `child -> parent` evidence. The split record proves directory relocation, not model ancestry. The committed snapshot also does not, by itself, prove one immutable effective-config snapshot and one fresh-replay artifact for every checkpoint.

Consequently Stage 0 cannot honestly classify those nodes as V2-ready merely because `M16.pt` and `M17.pt` coexist and have hashes.

The live post-2026-09-16 production lineages (including later M47/M80+ continuation/experiment history) were not present in the Git tree available to the initial pass, so their generation transactions/provenance/results could not be inspected in that pass. The final live read-only scan is recorded in section 15.

## 11.3 Legacy matrix

`A/B/C/D` below describes what can be proven from the evidence available to this Stage 0 execution. A later read-only scan may upgrade a `C` row to `B` if it finds explicit durable parent/config/replay records; it may not upgrade it by inference.

| Lineage/range | Parent coverage | SHA coverage | Fresh replay coverage | Effective-config coverage | Cross-lineage boundary | Class | Notes |
|---|---|---|---|---|---|---|---|
| `torus9-golden-v3-20260914-run03`, committed migration snapshot, M0..M17 | not proven node-by-node | recorded in migration inventory | not proven per node by committed inventory | not proven per node by committed inventory | no boundary edge proven by snapshot | **C** | physical checkpoint identity is strong; ancestry/config/replay still need live durable evidence or explicit bootstrap |
| split archived Torus9 lineages from `torus9-legacy-20260913` / `torus9-invalid-20260914` bundles | split record does not prove edges | physical artifacts were moved under canonical lineage locations; exact per-node coverage must be read live | not proven in split record | not proven in split record | lineage boundaries exist physically but child/parent checkpoint edges are not established by split record | **C** | scan generation provenance/transactions/results/manifests before any mapping |
| legacy Torus9 bundle wrapper metadata retained under `docs/experiments/legacy-bundles/` | not a canonical node graph | bundle metadata only | not authoritative | not authoritative | not authoritative | **D as ancestry source** | retained for audit/history; must not become a second ancestry database |
| Cube-4 SHA dedup/reference records | dedup parent/source is not training ancestry | strong SHA/canonical physical reference evidence | outside dedup contract | outside dedup contract | dedup may cross storage locations, not training graph | **C for V2 ancestry** | never reinterpret dedup links as parent edges |
| post-2026-09-16 live Torus9 production/experiment lineages | **not inspectable in this execution environment** | not inspectable here | not inspectable here | not inspectable here | exact later boundaries not inspectable here | **C fail-closed pending live audit** | may become B if live durable metadata proves every field; otherwise requires explicit bootstrap |

### V2-ready (A)

None could be proven from the repository-only migration snapshot. This did **not** assert that no live checkpoint was V2-ready; it said the live metadata needed to prove it was unavailable in that pass.

### Deterministically migratable (B)

None could be promoted to B from the committed migration inventory alone because parent + fresh replay + effective config had not all been proven for a specific node. The final live read-only scan identifies B nodes where generation provenance/result/transaction records supply those fields unambiguously.

### Explicit bootstrap/mapping required (C)

All useful checkpoint ranges for which one or more of parent, fresh replay, or effective config cannot be proven after the live scan. A bootstrap row must identify a full child checkpoint reference and a full parent checkpoint reference including SHA; never a bare `M82 -> M81` assertion.

### Unusable/inconsistent (D)

Legacy bundle wrappers are unusable as canonical ancestry sources. Any individual live node with conflicting SHAs, conflicting parent records, missing physical artifact, or incompatible identity must also be classified D rather than auto-repaired.

---

# 12. Cross-lineage findings

1. The existing storage layer already has the correct low-level shape for cross-lineage checkpoint references: owning lineage + path + SHA, with active/archive resolution and ownership checks.
2. Main also contains the PR #139 fix that honors resolved cross-lineage replay sources. This is useful execution plumbing but is not a second ancestry model.
3. No committed record available to this Stage 0 execution proves a specific useful later Torus9 `child checkpoint -> parent checkpoint` edge across a lineage boundary. Therefore no such edge is written into this report from names or generation adjacency.
4. A future migration can cross any number of lineage boundaries simply by writing normal CheckpointNodes whose `parent` points to the owning lineage of the immediate parent. No fork table or lineage-boundary rule is required.

---

# 13. Migration requirements (not implementation)

A separate migration step is required before the useful legacy history can be traversed safely. It must:

1. scan the actual `runs/*/active/*`, `runs/*/archive/*`, and relevant `runs/*/evaluations/*` metadata read-only;
2. inspect only lightweight manifest, generation transaction/result, provenance, artifact catalog, run-spec/effective-config, evaluation identity, and filesystem metadata unless a conflict requires stronger verification;
3. for each checkpoint, collect physical canonical artifact + recorded SHA;
4. prove exactly one immediate parent from durable records, or require an explicit bootstrap mapping keyed by **full child reference -> full parent reference**, both with SHA;
5. identify exactly one fresh-replay artifact for that generation from durable generation output/catalog records;
6. materialize/identify one immutable EffectiveConfig V2 snapshot representing the actual settings and record its artifact SHA + content fingerprint;
7. produce ordinary CheckpointNode records as the migration output;
8. fail on conflicting parent/SHA/config/replay evidence and classify the node D instead of guessing;
9. treat migration/bootstrap input as one-time evidence only. After canonical nodes exist, that input is not consulted by ArtifactResolver and never becomes a second ancestry database.

No historical artifact should be mutated as a side effect of the audit itself. The actual migration requires its own artifact-policy-controlled task/PR.

---

# 14. Stage 1 ArtifactResolver may now assume

Stage 1 does **not** need to decide any ancestry format. It may treat these facts as fixed:

- a checkpoint's full reference is `{topology, lineage_id, checkpoint_id, generation, path, sha256}`;
- `CheckpointNode.genesis` explicitly distinguishes genesis from non-genesis;
- a non-genesis node has exactly one full immediate-parent reference;
- parent traversal is the only ancestry mechanism;
- a node's `fresh_replay` is the only replay datum persisted on the node;
- replay windows are derived by parent traversal, never from a persisted replay ancestry list;
- a checkpoint points to exactly one immutable effective-config artifact + fingerprint;
- the low-level checkpoint opener remains `run_storage.resolve_checkpoint()`;
- short names are lookup queries and are accepted only when unique;
- canonical JSON and fingerprints use `gocube_golden.provenance` primitives;
- Arena reuse is keyed by the full generic EvaluationIdentity and separately gated by `VALID` result status.

The Stage 1 API can therefore be implemented as:

```python
checkpoint(ref)
parent(checkpoint)
ancestor(checkpoint, n)
ancestors(checkpoint, count)
replay_window(parent, count)
open_artifact(ref)
```

without another ancestry-schema decision. Legacy data may still need migration/bootstrap **data**, but not a new ancestry **format**.


# 15. Final live audit result — 2026-09-18

## 15.1 Scope and method

The final audit was read-only. It inspected lightweight metadata under `runs/torus9/{active,archive,evaluations}`: manifests, checkpoint metadata, generation completion/transaction/result records, provenance/config references, artifact catalogs, evaluation metadata, and filesystem existence. It did not load model weights and did not rehash the full checkpoint corpus. Recorded SHA-256 values were compared with the corresponding artifact-catalog/path records where those records existed.

The live scan covered 104 run/evaluation directories, 3,035 files, 139 manifests, 118 `generation-N.complete.json` records, 267 `M*.metadata.json` checkpoint metadata records, and 10 artifact catalogs. For a migratable node, the evidence chain had to provide the checkpoint identity, one immediate parent identity, a generation-specific fresh replay artifact, and a deterministic effective configuration assembled from durable actual configuration records. A Golden profile name alone was not accepted as effective config.

The audit found no persisted `gocube-checkpoint-node-v2` records and no standalone V2 effective-config artifacts in the live tree. Therefore A is empty. The B classification means that ordinary V2 records can be materialized deterministically from the existing durable metadata; it does not claim that migration has already been performed.

## 15.2 Final A/B/C/D matrix

| Live lineage/range | Durable evidence | Class | Migration disposition |
|---|---|---:|---|
| `torus9-golden-v3-20260914-run03`, M0-M17 | Checkpoint metadata and physical SHA exist, but there is no per-node immediate-parent record, no per-node fresh-replay identity, and no per-node effective-config snapshot. | **C** | Requires an explicit bootstrap/mapping for the root and each historical edge/fact that cannot be proven. Never infer the M sequence from names or adjacency. |
| `torus9-golden-v3-production-20260917-m17`, M18-M47 | Complete records, parent refs, fresh replay/catalog entries, checkpoint metadata, run-spec/profile/config fingerprints, and physical artifacts are present. | **B** | Deterministically migratable; M18 explicitly crosses to M17 in the preceding lineage. Run-spec fingerprint: `sha256:cc427fb3c19afa8c195fa8ae3581979ec1af4b8e3d5eef99b081a0577870ac85`; profile fingerprint: `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775`. |
| `torus9-golden-v3-plateau-exit-m47-lr3e4-r6-40k-s128-20260917-v2`, M48-M54 | Complete records and exact parent/fresh replay/config/catalog evidence are present. | **B** | Deterministically migratable; M48 explicitly crosses from production M47. Run-spec fingerprint: `sha256:56fdcdc4ecdb89036cf845654d62a0739b80c8498474757ba2cc6bcf1fb383c4`; profile fingerprint: `sha256:e4b8dde145555a3ba4633faaf273f1211a6e8bb717d04e64fcc7dff2688a2036`. |
| `torus9-golden-v3-plateau-exit-m54-lr3e4-r6-40k-s128-20260917-v3`, M55-M80 | Complete records and exact parent/fresh replay/config/catalog evidence are present. | **B** | Deterministically migratable; M55 explicitly crosses from plateau-exit v2 M54. It preserves the v2 run-spec/profile fingerprints above. |
| `torus9-staged-cadence-m80-20260918-v1-g64`, M81-M86 | Complete records, exact parent/fresh replay/catalog evidence, and staged-cadence actual configuration records are present. | **B** | Deterministically migratable; M81 explicitly crosses from plateau-exit v3 M80. Profile fingerprint: `sha256:9d8d00619d7891de860edbd724bcb3c8aea72d475334aacc9b516d5a4c06f2be`. |
| `torus9-staged-cadence-m80-20260918-v1-g128`, M81-M83 | Complete records, exact parent/fresh replay/catalog evidence, and staged-cadence actual configuration records are present. | **B** | Deterministically migratable; M81 explicitly crosses from plateau-exit v3 M80. |
| `torus9-staged-cadence-m80-20260918-v1-g192`, M81-M82 | Complete records, exact parent/fresh replay/catalog evidence, and staged-cadence actual configuration records are present. | **B** | Deterministically migratable; M81 explicitly crosses from plateau-exit v3 M80. |
| `torus9-post-ab-m80-g128-20260918-v1`, M84-M93 | Complete records, exact parent/fresh replay/catalog evidence, and post-AB actual configuration records are present. | **B** | Deterministically migratable; M84 explicitly crosses from staged-cadence g128 M83. Run-spec fingerprint: `sha256:f57ed0cd3aae133f276961a60e5d5fb33c4d29741af919de2785af5bc9d8ddac`. |
| `torus9-replay80k-m80-g128-20260918-v1`, attempted M81 continuation | Run-level provenance and actual config are present, but no child checkpoint, generation-complete record, fresh replay, or catalog entry exists. The report is `RECOVERY_REQUIRED` while the stale runtime state still says `RUNNING`; the catalog is empty. | **D** | No checkpoint node may be synthesized. Preserve as a failed/incomplete run requiring explicit recovery/resume handling outside this Stage 0 migration. |
| Archived Stage7 side branches `torus9-stage7-post-pr118-20260916-run01`, `run02`, `run04`, M1 | Each side node has explicit parent metadata, generation completion/fresh replay/config metadata, and physical artifact evidence. | **B**, outside the primary chain | Migratable as independent side branches; each M1 explicitly points to the old M0, but these branches are not silently merged into the primary lineage. |
| Legacy wrapper/bundle records and any record with conflicting identity evidence | Wrapper metadata is not a canonical node graph. Conflicting SHA, parent, config, replay, or missing-artifact evidence is not repairable by inference. | **D** | Preserve as audit evidence only; do not make wrappers or guessed repairs an ancestry source. |

Summary: **A = 0**; **B = 84 useful primary-chain/arm nodes** (30 + 7 + 26 + 6 + 3 + 2 + 10), plus the three independently migratable Stage7 side branches; **C = M0-M17 and any bootstrap-dependent legacy facts**; **D = wrappers, the incomplete replay80k continuation, and inconsistent records**.

## 15.3 Actual cross-lineage edges

These are the cross-lineage edges proven by durable child metadata. Each entry is a full checkpoint reference on both sides: topology is `torus9`, the lineage is encoded by the directory/lineage id, the generation/checkpoint id is explicit, and the path and SHA-256 are recorded.

| Child full reference | Parent full reference | Evidence |
|---|---|---|
| `runs/torus9/active/torus9-golden-v3-production-20260917-m17/checkpoints/M18.pt`, M18, `sha256:571e71def674e8f0acacb47135c85a4e2e5d4a4797df59dd00021453d0b4e14a` | `runs/torus9/active/torus9-golden-v3-20260914-run03/checkpoints/M17.pt`, M17, `sha256:86722afe70fefd1d4a2a408e47c3492c7b6da43e86f283888d8815da5b037e53` | production M18 metadata/complete record |
| `runs/torus9/active/torus9-golden-v3-plateau-exit-m47-lr3e4-r6-40k-s128-20260917-v2/checkpoints/M48.pt`, M48, `sha256:dcf7c2ec494437b65eaa67bad1bf6eb4d95c0366bee106e4511d1952dbc3c2e9` | `runs/torus9/active/torus9-golden-v3-production-20260917-m17/checkpoints/M47.pt`, M47, `sha256:49671a5c8df991aafd5cbb84ef74230043ac530fcba33615a79603d96ee18b1a` | plateau v2 M48 metadata/complete record |
| `runs/torus9/active/torus9-golden-v3-plateau-exit-m54-lr3e4-r6-40k-s128-20260917-v3/checkpoints/M55.pt`, M55, `sha256:9fb3eaa41d5f90cdac750396fe7808c6edf53047b15b00f113abe2fb3a428765` | `runs/torus9/active/torus9-golden-v3-plateau-exit-m47-lr3e4-r6-40k-s128-20260917-v2/checkpoints/M54.pt`, M54, `sha256:d72b3d7dc203a0e7ad318bedfa871884aba08386fd3d36bc08469e1615917c63` | plateau v3 M55 metadata/complete record |
| `runs/torus9/active/torus9-staged-cadence-m80-20260918-v1-g64/checkpoints/M81.pt`, M81, `sha256:ef6a75f458b83dc59577073d60270c7b1c1eadaa56149b2c225593765633445d` | `runs/torus9/active/torus9-golden-v3-plateau-exit-m54-lr3e4-r6-40k-s128-20260917-v3/checkpoints/M80.pt`, M80, `sha256:a3efc33fbbaaa0640deb24bd4d8ba1eeb4563823ee8819487c957b8dacbbe1f` | staged g64 M81 metadata/complete record |
| `runs/torus9/active/torus9-staged-cadence-m80-20260918-v1-g128/checkpoints/M81.pt`, M81, `sha256:ec14ee1550e6be84abcf6676109cb1346d3f34066411daddc2206b8e5f06d847` | `runs/torus9/active/torus9-golden-v3-plateau-exit-m54-lr3e4-r6-40k-s128-20260917-v3/checkpoints/M80.pt`, M80, `sha256:a3efc33fbbaaa0640deb24bd4d8ba1eeb4563823ee8819487c957b8dacbbe1f` | staged g128 M81 metadata/complete record |
| `runs/torus9/active/torus9-staged-cadence-m80-20260918-v1-g192/checkpoints/M81.pt`, M81, `sha256:fb0e8b623c527aaa8d0fe9e83ece1527199b8b459d98b2a232687e7f2ba6ef4f` | `runs/torus9/active/torus9-golden-v3-plateau-exit-m54-lr3e4-r6-40k-s128-20260917-v3/checkpoints/M80.pt`, M80, `sha256:a3efc33fbbaaa0640deb24bd4d8ba1eeb4563823ee8819487c957b8dacbbe1f` | staged g192 M81 metadata/complete record |
| `runs/torus9/active/torus9-post-ab-m80-g128-20260918-v1/checkpoints/M84.pt`, M84, `sha256:8df29e44d8674ab835b0c4e40139e2a2c86e66c6bfb9f47d1b31d3d9a320e18d` | `runs/torus9/active/torus9-staged-cadence-m80-20260918-v1-g128/checkpoints/M83.pt`, M83, `sha256:8cd4429702b236151be2627cff4a11924497cd551490cef1f6c3ecf15c860ee5` | post-AB M84 metadata/complete record |
| `runs/torus9/archive/torus9-stage7-post-pr118-20260916-run01/checkpoints/M1.pt`, M1, `sha256:a7b9cc5646b3836603d13ce87b8ca6b9aa6306a35e098c324a83eef52325a5e9` | `runs/torus9/active/torus9-golden-v3-20260914-run03/checkpoints/M0.pt`, M0, `sha256:2da9b40576f67eba2fbf4afa1fa7287a71151f1f379723d358cfa2a8bb91c800` | Stage7 side-run01 M1 metadata |
| `runs/torus9/archive/torus9-stage7-post-pr118-20260916-run02/checkpoints/M1.pt`, M1, `sha256:41db685492590ec6f8166b825f84f4aeb688f7f323d555f46fe6016095967dda` | `runs/torus9/active/torus9-golden-v3-20260914-run03/checkpoints/M0.pt`, M0, `sha256:2da9b40576f67eba2fbf4afa1fa7287a71151f1f379723d358cfa2a8bb91c800` | Stage7 side-run02 M1 metadata |
| `runs/torus9/archive/torus9-stage7-post-pr118-20260916-run04/checkpoints/M1.pt`, M1, `sha256:503c398e7f4cf96de115ae2c1e2b2b8c04f2944392f20140a956a9df0fcd456f` | `runs/torus9/active/torus9-golden-v3-20260914-run03/checkpoints/M0.pt`, M0, `sha256:2da9b40576f67eba2fbf4afa1fa7287a71151f1f379723d358cfa2a8bb91c800` | Stage7 side-run04 M1 metadata |

No parent edge was inferred from a checkpoint filename, generation adjacency, or directory proximity. The table is migration input evidence; it is not itself a second ancestry database.

## 15.4 Migration and Stage 1 prerequisites

The B rows are ready for a separate, controlled migration that materializes ordinary `CheckpointNode` records and immutable EffectiveConfig artifacts. That migration must preserve each child checkpoint's full identity, the exact parent identity, the generation-specific `iter-N-fresh` artifact (not a rolling replay-window artifact), and the actual effective configuration assembled from durable run-spec/profile/operator/checkpoint records. It must fail closed on any conflict and leave the live run tree unchanged.

The C M0-M17 range still needs an explicit bootstrap decision/mapping. The bootstrap must state the full child reference and full parent reference, including SHA, or explicitly declare the canonical genesis fact where appropriate. M0-M17 must not be upgraded to B merely because the physical files and numeric names are present.

Stage 1 may now consume the frozen contracts and treat the 84 B nodes plus any explicit bootstrap output as migration inputs. Stage 0 does not add an ArtifactResolver, ancestry API implementation, runner, orchestration logic, training/Arena behavior, or production mutation.


# 16. Strict durable parsing audit

The contract parsers now reject malformed durable values instead of coercing or silently dropping them:

- required strings, path components, relative paths, SHA-256 values, integers, and booleans are type-checked before validation;
- optional structured fields (`parent`, `fresh_replay`, `active_execution`, commit/checkpoint refs, and `queued_transition`) reject present values of the wrong type;
- every element of `requested_changes` and `applied_amendments` is parsed and validated; malformed elements are not skipped;
- `genesis` and `soft_stop_requested` require JSON booleans, so strings and numeric truthy values cannot change persisted state;
- legacy `artifact_sha256` is accepted only as the existing checkpoint-reference compatibility alias and is still required to be a string with canonical SHA syntax.

Targeted negative tests cover these cases and the existing contract suite passes: `29 passed` in `tests/test_orchestrator_v2_contracts.py`.
