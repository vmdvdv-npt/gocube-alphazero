# GoCube production training loop contract

This is the authoritative contract for the current KataGo Rules V3 training
loop. The pinned KataGo reference commit is
`f6bc4b19a1686caa2d088b56251e8c11c8be6d51`, and production komi is exactly
`0.5`.

## Canonical launch

The from-scratch Cube 4 entrypoint is:

```bash
tools/run_cube4_katago_from_scratch.sh
```

It uses the project virtualenv, defaults the master seed to `0`, and passes
that seed through the deterministic `gocube-seed-derivation-v1` coordinate
hash. A worker/game seed is derived from master seed, iteration, worker,
game slot, and game sequence number; no production process relies on an
implicit no-argument RNG seed.

## Versioned sample clock and replay

The optimizer clock is `sample-clock-v2`: scheduler state advances by the
number of examples consumed, not by wall-clock time or nominal iteration
number. Checkpoints and resume state must carry this contract identifier.

Replay format v3 is one atomic logical commit of seven tensors, in this exact
order:

1. observations;
2. policy targets;
3. value targets with three classes `[win, loss, no_result]`;
4. normalized black-minus-white score targets;
5. score applicability masks;
6. formal Rules V3 ownership targets;
7. ownership point masks.

The completion marker is published last. Replay v1 and v2 are rejected: their
score/ownership labels do not carry the corrected S1 semantics.

Score initialization is the explicit contract
`katago-boardhistory-clear-v1`. Setup, ordinary replay, and synthetic cleanup
starts all initialize the equivalent of KataGo
`BoardHistory::clear(..., encorePhase)`, including setup stones and capture
counters. `main_moves` is telemetry and never selects a scoring algorithm.

Replay v3 (the S1 tensor-format identifier) also records the termination/target provenance contract; missing S3 marker fields are rejected. Per-game
records carry the exact `termination_reason`, `result_provenance`, episode
type, and runtime move count; historical records without enough information
are classified as `unknown_legacy_termination` rather than rewritten.

## Target semantics

The value target contract is `win-loss-noresult-s1-v2`. A scored win/loss is a
one-hot win/loss target relative to the player to move. A scored draw is
`[0.5, 0.5, 0]`. A genuine `NO_RESULT` is `[0, 0, 1]`; its score target is
`NaN` with score mask `0`, and its ownership target is zero with ownership
mask `0`. Losses select active rows or points before arithmetic, so masked
`NaN` values can never contaminate gradients.

The score contract is
`normalized-score-with-applicability-mask-s1-v2`; the ownership contract is
`formal-v3-s1-with-point-mask-v2`.

The target provenance contract is `formal-runtime-result-provenance-v1`.
`episode_move_limit` uses scored value/score/ownership targets produced by the
current position, but remains explicitly marked as runtime-forced. A genuine
cycle remains `NO_RESULT` with its existing masked auxiliary targets.

## Immutable run identity

Every hardened run writes `run-manifest.json`, `effective-config.json`, and an
environment artifact before training. The manifest records the effective
configuration, source and KataGo commits, rules fingerprint, topology,
observation/action/value shapes, komi, target contracts, and parameter
origins. Resume is allowed only when those immutable fields match exactly;
configuration or rules drift is a hard failure.

The reference differential suite is mandatory in CI and can be run locally
with:

```bash
.venv/bin/python -m pytest -m katago_reference
```
