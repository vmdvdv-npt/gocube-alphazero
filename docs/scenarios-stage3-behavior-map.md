# Scenario extraction: behavior map

This map records the behavior that the extraction must preserve. Existing
`state.json` and `plan.json` locations remain canonical in the first rollout;
the new `gocube_golden.scenarios` modules are adapters/policy owners.

## Experiment

| Old state | Next action | Required evidence | Technical retry owner |
| --- | --- | --- | --- |
| `RUNNING` | Resume the first missing arm, in A then B order | matching config fingerprint, target generation, common parent | Supervisor for a generation attempt |
| arm record has `final_checkpoint` | Validate and reuse the checkpoint | committed checkpoint, parent ancestry, lineage/config identity | none |
| `STAGE1_ARENA` | Reconcile/reuse the Arena, then make one winner decision | candidate/reference refs, valid Arena, matching evaluation identity and W/L/D | Arena/Supervisor |
| valid Stage 1 winner, Stage 2 enabled | Start control and C from that winner | durable winner decision and two fresh lineage identities | Supervisor per action |
| `ARENA_INVALID` | Stop and fail closed | invalid Arena evidence | none; no scientific retry |
| `STOPPED` | Reconstruct result only | all final refs, Arena evidence, winner decision | none |

Stage 1 always compares `B-final` to `A-final`. Stage 2 compares `C-final` to
`control-final`; both use the strict candidate-if-`wins > losses` rule.

## Komi calibration

| Old state | Next action | Required evidence | Technical retry owner |
| --- | --- | --- | --- |
| `WAITING_FOR_M137` | Observe the configured parent; do not guess a checkpoint | committed M137 completion evidence | parent runner/supervisor |
| `STOPPING_PARENT` | Reconcile the idempotent stop request | acknowledged safe boundary, not only a control file | parent runner/supervisor |
| `M137_PINNED` | Run the frozen candidate startset | pinned checkpoint ref/hash and frozen startset | Arena/Supervisor |
| `CALIBRATION_*` | Reuse each committed batch or run the same batch action id | valid result, batch number, evaluation identity, startset and counters | Arena/Supervisor |
| `CALIBRATION_EXTENSION` | Add batch 2 only when initial bias difference is `< 0.01` | complete batch 1 ledger and new valid batch 2 | scientific extension is a new planned action |
| `KOMI_SELECTED` | Create/reuse exactly one child lineage | selected komi and parent replay refs | lineage service |
| `CHILD_LINEAGE_CREATED` / `TRAINING_RESUMED` | Reattach the matching child | handoff intent and child commit evidence | child runner/supervisor |
| `COMPLETE_HANDOFF` | Return the recorded child result | committed child generation | none |
| `CALIBRATION_FAILED` | Remain stopped | failure reason and original evidence | none |

Production aggregation sums black/white/draw counters across the ordered batch
ledger and recomputes Wilson statistics. A stray extra batch is not used when
extension was not required.

## Ownership and retry table

| Concern | Single owner | Durable evidence |
| --- | --- | --- |
| process/PID/PGID/heartbeat/technical retry | `SupervisorV2` | active-child, supervisor result, stop marker |
| committed/reusable generation or Arena | generation/Arena boundary | checkpoint/evaluation refs and commit markers |
| stage, winner, extension, selected komi, handoff | scenario | scenario `state.json` and ledger |
| dependency/status/outputs | `WorkflowRunner` | workflow state |
| delivery | notifications dispatcher | event and delivery receipt |
