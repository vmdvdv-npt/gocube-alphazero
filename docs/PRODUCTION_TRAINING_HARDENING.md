# Production training hardening

This document describes the production GoCube training contract added on top of the pinned KataGo search port.

## Scope

The hardened path intentionally does **not** change the network architecture, loss weights, GoCube rules, or the selected search budgets. Production Cube 4 keeps:

- komi: **0.5**;
- regular self-play search: **50 simulations**;
- fast search: **20 simulations**;
- fast-game probability: **0.25**;
- observational Arena: 50 simulations, deterministic move selection, no root noise, no model gating.

The KataGo reference remains commit `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.

## 1. Pinned chosen-move temperature

The played self-play move follows the pinned self-play configuration exactly:

- `chosenMoveTemperatureEarly = 0.75`;
- `chosenMoveTemperature = 0.15`;
- `chosenMoveTemperatureHalflife = 19`;
- `chosenMoveSubtract = 0`;
- `chosenMovePrune = 1`.

KataGo's `interpolateEarly` scales elapsed halflives by `19 / sqrt(board area)`. GoCube substitutes the logical topology point count for planar board area. This is the same topology substitution already used by root-policy temperature.

The generic framework temperature callback is prevented from applying a second legacy schedule after the pinned self-play temperature is computed. The hardened Arena uses a constant-zero temperature callback, so it remains deterministic on every move.

## 2. Value-weighted tree statistics

Production KataGo-mode MCTS stores, per node:

- utility mean;
- utility squared mean;
- total statistical weight;
- squared-weight sum;
- score mean when available.

Child aggregation applies the pinned `valueWeightExponent = 0.5` bad-child downweighting. The weighting distribution is the same **Student-t distribution with 3 degrees of freedom** used by KataGo, not a Gaussian approximation. The child weights are normalized back to the original total child weight after downweighting.

When root Dirichlet noise is active, pinned `chosenMoveSubtract/chosenMovePrune` is also applied during root value aggregation before normalization, matching `recomputeNodeStats`.

PUCT uses the resulting child statistical weight rather than treating every child subtree as if `weight == visits`.

Legacy/non-KataGo MCTS remains on its previous code path.

## 3. LCB play selection and policy targets

Pinned selection values use:

- `useLcbForSelection = true`;
- `lcbStdevs = 5.0`;
- `minVisitPropForLCB = 0.15`;
- `useNonBuggyLcb = true` semantics.

LCB uses the weighted effective sample size derived from `weightSum^2 / weightSqSum`, plus KataGo's low-playout variance prior.

Before LCB, non-best root children are retrospectively reduced by inverting PUCT against the non-LCB best child. As in pinned `searchresults.cpp`, the reduced value for each non-best child is then rounded with `ceil` before LCB processing.

The self-play action and the policy target intentionally differ, matching KataGo training behavior:

- the **actually played self-play move** uses the pinned chosen-move temperature with LCB disabled;
- the **policy training target** is extracted with LCB enabled;
- deterministic Arena keeps LCB enabled.

Root forced-exploration correction remains observable separately from the post-LCB policy target in per-move search telemetry.

## 4. Atomic checkpoint writes

The hardened network wrapper never writes directly to a visible production checkpoint filename. It writes a same-directory staging file, flushes and `fsync`s it, then publishes it with `os.replace`, followed by a directory `fsync`.

A process crash before the replace leaves the previous visible checkpoint unchanged. A process crash after the replace leaves a complete new checkpoint.

## 5. Atomic replay logical commits

Each iteration's six replay tensors are first written into a staging directory on the same filesystem:

1. observation data;
2. policy targets;
3. value targets;
4. score targets;
5. ownership targets;
6. ownership masks.

Only after all six files exist are they promoted into the run directory. A completion marker `iteration-NNNN-complete.json` is written **last**.

Replay loading is fail-closed: an iteration without a valid marker, with a missing tensor, or with inconsistent row counts is ignored rather than partially entering the replay window.

## 6. Resume contract

Resume no longer derives the next iteration from `len(glob(checkpoints))`.

The hardened coach scans `iteration-0000.pkl`, `iteration-0001.pkl`, ... in order and stops at the first missing or structurally unreadable checkpoint. Any later checkpoint files are treated as an untrusted trailing tail and ignored. The selected checkpoint is then loaded through the existing search/training contract validation, so a structurally readable but semantically incompatible checkpoint still fails.

`--allow-existing-run` is fail-closed: it requires an existing namespace containing at least one checkpoint. It cannot silently turn a partially created directory into a new run.

Old checkpoints are not silently reinterpreted under the new semantics. The exploration contract is versioned as `katago-pinned-exploration-v2`, and hardened checkpoints additionally persist the recovery and move/value/LCB fields.

## Entrypoint

Start a new Cube 4 run:

```bash
tools/run_cube4_katago_from_scratch.sh
```

Intentional resume after interruption:

```bash
RUN_NAME=<same-run-name> RESUME=1 tools/run_cube4_katago_from_scratch.sh
```

`RESUME=1` is rejected for a namespace that does not already exist, while an existing namespace is rejected unless resume was explicitly requested.

The direct Python entrypoint is:

```bash
python -m alphazero.envs.gocube.hardened_train ...
```

Use `--allow-existing-run` only for an intentional resume of an existing hardened namespace.
