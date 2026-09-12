# Torus Golden v2 machine passport

`configs/gocube/torus_golden_v2.json` is the machine-readable Stage-2 passport for the standalone Golden Torus 5×5 reference line. It extends the historical v1 experiment contract without rewriting it.

## Scope

The v2 passport pins the standalone 5×5 research topology, graph-area rules, komi `0.5`, observation and WDL target contracts, the sequential Golden Arena/search contract, checkpoint semantic identity, and evidence reproducibility requirements. The production Torus factory is not extended or reused for this line.

The passport is fingerprinted as canonical JSON. Its `experiment_fingerprint` covers every semantic section of v2; the Golden Arena search contract has its own independent fingerprint so Arena settings cannot silently drift with checkpoint/train metadata.

## P1: exact player identity

A free-form `player_id` is display metadata only. Canonical Golden evidence requires a structured `PlayerIdentity` for both A and B. For checkpoint-backed players this identity pins:

- SHA256 of the exact model artifact;
- fingerprint of the complete checkpoint metadata;
- rules, topology and komi compatibility;
- PointId order and board size;
- observation schema/version/fingerprint;
- target contract/version/fingerprint;
- value-head semantics and network head shapes;
- parent/source run identity.

Missing or mismatching metadata fails closed. Checkpoint metadata cannot override Arena-owned search settings.

## P1: exact mirrored-pair schedule

Counts are not sufficient evidence of a paired Arena. `RunManifest.pair_schedule` persists each entry as:

```text
(pair_id, A-black game_id, B-black game_id)
```

Validation reconstructs this schedule from raw `GameRecord` evidence and requires an exact match. Every pair must contain exactly two distinct games, one A-black/B-white and one B-black/A-white, with identical run identity, player identities, start state/history, rules/topology, and search contract.

Incomplete pairs, third games, duplicate game IDs, same-color pairs, pair-id drift, or schedule substitution are rejected.

## Seed and run provenance

The seed contract remains `golden-arena-seed-v1`:

```text
seed_game = H64(master_seed : pair_id : game_id : "game")
seed_A    = H64(seed_game : "A")
seed_B    = H64(seed_game : "B")
```

The validator never trusts persisted derived seeds. It recomputes all three from the persisted master seed and game identity.

Each `GameRecord` also carries an immutable run-identity fingerprint over the experiment profile, git commit/tree identity, Arena/search contract, master seed, and exact A/B player identity fingerprints. The validator recomputes this fingerprint independently.

## Code identity and canonical evidence

A run records full git commit SHA, tree SHA, and working-tree cleanliness. Canonical evidence requires a clean tree and exact checkpoint identities for both players. Research/scripted evidence can still be replayed and validated, but cannot be promoted to canonical checkpoint evidence.

## Formal terminal boundary

The Stage-2 evidence validator retains the strict terminal boundary: after a replayed formal Golden terminal (`state.is_terminal == True`) no further `ActionEvidence` may exist. A post-terminal action cannot transform a valid rule result into a technical failure.

## Persisted artifacts

`write_run_evidence()` writes:

```text
<evidence-dir>/manifest.json
<evidence-dir>/records.jsonl
```

Before writing, it validates the complete manifest, exact pair schedule, all raw records, player identities, seeds, code identity and canonical-evidence requirements.
