# GoCube learning failure diagnosis (2026-09-09)

## Conclusion

The demonstrated failure was in the retired GoCube V3 legacy search boundary,
not in the optimizer or checkpoint serializer. V3 targets encode value relative
to the side to move:

```text
[WIN(side-to-move), LOSS(side-to-move), NO_RESULT]
```

The legacy `MCTS.process_results()` read the first two slots as absolute Black
and White values. On a white-to-move position, `[1, 0, 0]` therefore became a
Black win instead of a White win. That reverses the value signal on every such
inference and corrupts search decisions, self-play policy targets, and the next
training iteration. The reproduction is in
`tests/test_gocube_legacy_training_retirement.py`.

The fix is deliberately fail-closed: GoCube V3 accepts only the pinned search
contract, and the retired legacy training entrypoint cannot create new V3
data. The pinned path converts the player-relative value to absolute search
utility once at the search boundary.

No independent downstream defect was found in the tested current path. The
short current Cube4 run also proved that a late checkpoint can beat an earlier
checkpoint after the boundary fix; this is a pipeline sanity signal, not a
production-strength claim.

## Evidence chain

| Link | Reproducible evidence | Verdict |
| --- | --- | --- |
| rules | V3 cleanup, scoring, Benson/topology, terminal and rules-fingerprint suites; the run manifest pins the rules fingerprint and `komi=0.5` | REJECTED: rules drift is not the observed cause |
| self-play | `test_gocube_selfplay_records.py`, pinned MCTS integration, and the 20-game run with 2,855 recorded positions | REJECTED: self-play was not a no-op |
| samples/targets | replay tensor validation, target provenance, no-result/score masks, and exact row/counter accounting in the run | REJECTED: rows and targets reach training with the current schemas |
| trainer | `test_real_trainer_overfits_tiny_dataset_and_optimizer_changes_parameters` drives real GraphNet/NNetWrapper loss down by over 95% with >0.95 target probabilities | REJECTED: trainer/optimizer failure |
| checkpoint | `test_checkpoint_round_trip_preserves_trained_predictions` verifies predictions before save and after load | REJECTED: checkpoint corruption |
| inference | `test_production_contract_has_one_explicit_v3_semantic_path` and the model-contract suites pin shape, schema, heads, and target semantics | REJECTED: model-contract mismatch in current path |
| MCTS | `test_v3_training_target_network_mcts_preserves_absolute_white_utility`, batched-vs-single routing tests, and pinned exploration tests | REJECTED for the pinned path; CONFIRMED for the retired legacy boundary |
| Arena | `test_arena_distinguishes_good_and_bad_deterministic_models` gets 8–0 for a known one-ply good player; fixed checkpoint Arena uses equal settings and balanced colors | REJECTED: Arena cannot distinguish models |
| gating | `--model-gating` fails closed because production gating is intentionally observational/off; Arena telemetry remains available | REJECTED: silent wrong-model acceptance is not active in this path |
| late checkpoint | Current pinned Cube4 run: `iteration-0005` vs `iteration-0002`, 8W/0L/0D/0NR at 50 simulations | PASS: minimal end-to-end improvement signal |

## Hypothesis cards

### H1 — player-relative value was interpreted as absolute

**Verdict: CONFIRMED for the retired path.**

The test runs the actual legacy and pinned MCTS update paths with the same V3
target. Black-to-move agrees by accident; white-to-move is reversed by the
legacy slot lookup and correct in the pinned path. The V3 legacy path is now
rejected before neural inference, so this cannot silently generate new data.

### H2 — trainer does not learn or optimizer does not update

**Verdict: REJECTED.**

The real production network wrapper overfits a 16-row synthetic target set:
the parameter digest changes, 200 optimizer steps are reported, combined
policy/value loss falls below 15% of its initial value, and both target
probabilities exceed 0.95.

### H3 — checkpoint save/load changes predictions

**Verdict: REJECTED.**

The trained network is saved with optimizer state and loaded into a fresh
wrapper. Policy and value predictions match to `1e-6`.

### H4 — game and Arena use different observation semantics

**Verdict: REJECTED for the current contract.**

The baseline and structural adapters produce the expected shapes and preserve
the same semantic state; the contract fingerprints reject incompatible model
variants.

### H5 — pinned MCTS mishandles value perspective, policy, or search targets

**Verdict: REJECTED for the pinned path.**

The white-to-move value matrix is tested for black win, white win, draw, and
no-result. Single and batched searches agree on root values, visits, and policy
targets. Exploration tests verify that the post-LCB target is the one used for
training while played-move selection has the pinned temperature behavior.

### H6 — replay or training budget is silently empty/incorrect

**Verdict: REJECTED.**

The diagnostic run saved 2,855 rows, consumed 2,855 optimizer examples, and
performed 93 updates. Each iteration has a complete replay marker and all
tensor rows agree. Existing budget tests also cover partial batches and resume
accounting.

### H7 — Arena or batched inference routes the wrong model/game row

**Verdict: REJECTED.**

The Arena unit proof separates known GOOD and BAD players. Existing batched
MCTS tests verify per-game leaf routing, stale-slot rejection, model identity,
single-vs-batched equality, and color balancing.

### H8 — gating accepted a regression and hid the failure

**Verdict: REJECTED as a current-path cause.**

Production model gating is explicitly disabled and the CLI rejects the opt-in;
the Arena is observational. The implementation has tests for the disabled
configuration and for the candidate/previous/anchor telemetry, so a future
gate must be enabled only after an observational Arena pass.

### H9 — the entire current pinned loop still fails to improve

**Verdict: REJECTED by the minimal end-to-end run.**

The current run exercised real self-play, replay assembly, training, checkpoint
loading, inference, pinned MCTS, and fixed Arena. The fifth checkpoint beat the
second 8–0 in eight games. This is intentionally a smoke-level signal, not a
claim that five tiny iterations are sufficient for useful GoCube strength.

## Commands

Cheap automated proofs:

```bash
.venv/bin/python -m pytest -q \
  tests/test_gocube_learning_pipeline_diagnostics.py \
  tests/test_gocube_legacy_training_retirement.py \
  tests/test_gocube_value_perspective_integration.py \
  tests/test_gocube_batched_mcts_state_routing.py \
  tests/test_gocube_exploration_integration.py
```

Minimal current-path learning run:

```bash
.venv/bin/python -m alphazero.envs.gocube.katago_train \
  --topology cube --size 4 --workers 1 --sims 1 --arena-sims 1 \
  --games-per-iteration 4 --iterations 5 --train-batch-size 32 \
  --train-samples-per-new-sample 1 --endgame-sample-weight 1 \
  --fast-game-prob 0 --no-arena \
  --run-name learning-proof-cube4-20260909 --allow-dirty-source \
  --seed 20260909
```

Fixed Arena evaluation must use the checked-in evaluator with `--sims 50`,
noise/temperature disabled, an even number of games, and a fresh output file.
The full diagnostic report must record both checkpoint paths, their contracts,
seed, game count, no-result count, color balance, and the W/L/D result.
