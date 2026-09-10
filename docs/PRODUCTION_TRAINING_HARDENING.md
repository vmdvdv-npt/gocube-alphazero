# Production training hardening

> Historical/frozen-path note: this document describes the existing Cube-4
> production contract, not the project-wide komi policy. Its `0.5` value is a
> local reproducibility pin for that path. New Torus/reference development
> follows `docs/KOMI_POLICY.md`, where 0.5 is the default baseline but fair
> Torus komi remains an open research question and legacy 7.5 is forbidden.

This document describes the production GoCube training contract added on top of the pinned KataGo search port.

## Scope

The hardened path intentionally does **not** change the network architecture, loss weights, GoCube rules, or the selected search budgets. Production Cube 4 keeps:

- historical baseline komi: **0.5**;
- regular self-play search: **50 simulations**;
- fast search: **20 simulations**;
- fast-game probability: **0.25**;
- observational Arena: 50 simulations, deterministic move selection, no root noise, no model gating.

The KataGo reference remains commit `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.

## 0. Formal rules versus episode runtime

The production episode budget is `256 + 24 * point_count` (2560 for Cube 4).
It is owned by the self-play runner and counted separately from formal/history
turns, so fork and synthetic cleanup episodes do not inherit a stale budget.
`apply_v3_action` and MCTS clones never turn this budget into a terminal. When
the real runner reaches it, the current position is force-scored and records
carry `termination_reason=episode_move_limit`, `result_provenance=runtime`,
and `runtime_forced=true`. Formal pass, pass-alive, and cycle results remain
separate.

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

Each iteration's seven replay tensors and required target-provenance sidecar
(replay format v4) are first written into a staging directory on the same
filesystem:

1. observation data;
2. policy targets;
3. value targets;
4. score targets;
5. score applicability masks;
6. ownership targets;
7. ownership point masks.

The eighth artifact is `-target-provenance.pkl`, a row-aligned `uint8 [N]`
sidecar using encoding `gocube-target-provenance-encoding-v1` (`1=formal`,
`2=rule_no_result`, `3=runtime`; `0` is reserved for unknown/legacy data and
is rejected for new S3 rows). Only after all seven tensors and the sidecar
exist are they promoted into the run directory. A completion marker
`iteration-NNNN-complete.json` is written **last** and records the provenance
semantics, encoding, suffix, row count, and termination contract.

Replay loading is fail-closed: an iteration without a valid marker, with a missing tensor, or with inconsistent row counts is ignored rather than partially entering the replay window.

## 6. Resume contract

Resume no longer derives the next iteration from `len(glob(checkpoints))`.

The hardened coach scans `iteration-0000.pkl`, `iteration-0001.pkl`, ... in order and stops at the first missing or structurally unreadable checkpoint. Any later checkpoint files are treated as an untrusted trailing tail and ignored. The selected checkpoint is then loaded through the existing search/training contract validation, so a structurally readable but semantically incompatible checkpoint still fails.

`--allow-existing-run` is fail-closed: it requires an existing namespace containing at least one checkpoint. It cannot silently turn a partially created directory into a new run.

Before a resume loads any checkpoint, the immutable `run-manifest.json` and
`effective-config.json` are compared with the current invocation. A changed
effective parameter, topology, rules fingerprint, pinned KataGo commit, komi,
master seed, target semantic, sample-clock contract, or replay version aborts
the run. The rich manifest is never overwritten during resume.

Old checkpoints are not silently reinterpreted under the new semantics. The exploration contract is versioned as `katago-pinned-exploration-v2`, and hardened checkpoints additionally persist the recovery and move/value/LCB fields. S1 requires replay format v4, while retaining training contract v3 and the target semantics `win-loss-noresult-s1-v2`, `normalized-score-with-applicability-mask-s1-v2`, `formal-v3-s1-with-point-mask-v2`, and score initialization `katago-boardhistory-clear-v1`. S3 additionally requires the provenance sidecar and encoding; older artifacts fail closed.

## 7. Initial network reproducibility

The production entrypoint applies `master_seed` before constructing the initial trainable network. Therefore fresh runs from the same source and configuration with the same `master_seed` start from the same initial `train_net` parameters.

This does not claim bitwise reproducibility of the complete multiprocessing training run. Worker and game RNG streams continue to use the existing seed-derivation contract.

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
