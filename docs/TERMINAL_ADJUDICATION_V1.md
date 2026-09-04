# Terminal Adjudication V1 — self-play policy

This document selects the first total terminal adjudicator permitted by `COMPATIBILITY_SPEC_V1.md`.

## Decision

The first enabled self-play ruleset is **Chinese area scoring** using:

```text
terminal_adjudicator_id = gocube-conservative-area-v1
```

The adjudicator is deliberately conservative:

1. two consecutive passes end normal play;
2. Stage A runs the production-parity `AssistedEndgameClassifier` port;
3. every Stage-A `alive`, `dead`, or `seki` result is preserved exactly;
4. every remaining Stage-A `unresolved` group is explicitly retained as `alive` for self-play scoring;
5. the resulting complete classification is scored by the GoCube-compatible Chinese scoring implementation.

This makes terminal evaluation total and deterministic for every two-pass position without inventing additional dead stones.

## Why unresolved becomes alive in this adjudicator

This is **not** a claim that an unresolved group is objectively alive under human Go adjudication. It is a self-play cleanup convention.

The safety property is one-sided: an unproven group remains physically present on the board. The resolver never deletes an unproven stone group and never awards prisoners/territory by guessing that it is dead. Therefore a self-play agent that wants an opponent group removed from an area-scored final position must either:

- reach one of the narrow positions where Stage A proves it dead, or
- physically capture it before both players pass.

The fallback is versioned and recorded in terminal evidence. It is never silent.

## Why V1 does not enable Japanese self-play

Production GoCube Japanese scoring depends on dead-stone/prisoner classification. Simply retaining every unresolved group as alive would make unresolved dead groups block territory and avoid prisoner credit, while simply declaring them dead would fabricate captures and territory.

A sound self-play implementation for Japanese/territory scoring therefore needs a separately specified **cleanup phase** in which disputed stones can be resolved by continued play with score-preserving compensation/phase rules. That is outside this first adjudicator.

Accordingly:

```text
Chinese ruleset  -> enabled with gocube-conservative-area-v1
Japanese ruleset -> semantic/scoring core available, self-play terminal adjudication disabled
```

Trying to use this adjudicator with `ruleset=japanese` is a configuration error, not a draw and not an adjudication fallback.

## Compatibility boundary

The following remain direct production GoCube parity:

- board topology and point identity;
- action indexing;
- captures/liberties/suicide/simple ko/pass semantics;
- Stage-A automatic alive/dead/seki/unresolved proposal;
- Chinese scoring after a complete classification.

Only the final transformation

```text
Stage-A unresolved -> alive
```

is the declared self-play extension identified by `gocube-conservative-area-v1`.

Models, replay data, checkpoints, and experiment metadata MUST retain the adjudicator identifier so later adjudicator revisions are not mixed silently.

## Required conformance

Before MCTS integration, tests must prove that:

- Stage-A proven statuses are unchanged;
- every unresolved group becomes alive and remains on the scoring board;
- the resolver always returns a complete classification for Chinese endgame states;
- repeated adjudication is deterministic;
- Japanese invocation fails closed;
- no MCTS terminal node can become `all-zero win_state + zero valid moves` once the framework adapter is enabled.
