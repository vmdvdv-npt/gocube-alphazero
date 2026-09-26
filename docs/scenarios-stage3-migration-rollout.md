# Scenario migration and rollout notes

## Canonical records

| Decision | Canonical record | Rebuildable projections |
| --- | --- | --- |
| experiment plan | experiment `state.json` identity/fingerprint plus existing `plan.json` | reports and notifications |
| pinned parent | `parent_checkpoint` ref/hash in calibration state | readable parent fields |
| accepted Arena | stage `arena` result ref/evaluation identity | `results.json`, report |
| winner | stage `winner` record bound to evaluation id/fingerprint/W/L/D | `winner.json`, Telegram text |
| calibration batches | `candidate_batches` ledger | aggregate candidate result and report |
| child handoff | child lineage identity and handoff state | human-readable handoff event |
| terminal outcome | scenario `state` plus terminal evidence | summaries and notifications |

Atomic JSON writes protect each record; they do not make several files one
transaction. Resume reads the canonical record first and recreates a missing
projection only after its referenced evidence validates.

## Compatibility matrix

| Input/state | Reader behavior | Rollback note |
| --- | --- | --- |
| experiment v2 legacy state | existing exact migration to v3, requiring the legacy fingerprint | keep the original file until migrated evidence validates |
| experiment v3 state | read in place; old public import remains available | previous coordinator can read unchanged experiment fields |
| komi v1 state | read in place, including all historical state names | do not rename states without a dedicated migration |
| old CLI aliases/configs | entrypoint adapters preserve them | rollback uses the same request/evaluation identity |
| new action envelopes | optional boundary metadata; existing Arena identity is unchanged | a coordinator that cannot read the envelope stops before launch |

Migration is deterministic and does not create a new action id. No checkpoint,
replay, lineage id, evaluation id, or scientific fingerprint is rewritten by
the package extraction.

## Rollout / rollback

1. Run a new scenario in an isolated `runs_root`.
2. Resume a copied completed state and verify zero Arena/generation launches.
3. Resume a copied interrupted state with a short injected executor.
4. Run one controlled production scenario within its explicit budget.
5. Verify event recovery, result refs, child uniqueness, and terminal evidence.

Rollback is allowed only after checking that the old reader understands the
new state and does not point away from committed results. If it cannot, stop at
the durable boundary and use an adapter; do not rewrite a winner, generation,
or batch number by hand.
