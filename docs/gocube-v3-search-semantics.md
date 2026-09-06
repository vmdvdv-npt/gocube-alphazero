# Search semantics vs rules semantics

GoCube Japanese V3 deliberately separates the game rules from the search heuristics used by MCTS.

## Rules semantics are authoritative

The V3 rules engine owns legality, phase transitions, ko/repetition handling, cleanup, terminal detection, final scoring, ownership training targets, and the final winner. MCTS must not rewrite any of those rules.

In particular:

- PASS legality comes from `v3_valid_moves()` and remains unchanged by search.
- MAIN / CLEANUP1 / CLEANUP2 transitions come from the V3 state machine.
- Terminal positions use the formal V3 adjudication and exact final score.
- `NO_RESULT` remains a rule-side terminal outcome and is not converted into a neural score.
- Search heuristics never remove stones, alter captures, change komi, or redefine territory.

## Search semantics

The GoCube V3 search contract is versioned as `katago-v3-score-aware-v1` and uses four neural outputs:

1. policy,
2. win/draw value,
3. normalized Black-minus-White score,
4. per-point Black / White / neutral ownership.

Legacy games and legacy checkpoints remain on the historical policy/value-only search path.

For GoCube V3, MCTS combines win/loss utility with a KataGo-style smooth score utility. The score component is expressed in score-point space first and then transformed into search utility. Root ending adjustments are also expressed in points and converted through the same score-utility transform; they are never subtracted directly from Q as if points and utility were interchangeable units.

The score transform and dynamic score-center behavior are based on the pinned KataGo reference commit `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.

## Ownership-aware root behavior

Ownership is used only as a search signal. It does not become a second scoring engine.

At the root, ownership can help distinguish:

- neutral / dame-like points that are still useful to play,
- pointless filling of strongly owned territory,
- deep invasions into strongly opponent-owned territory,
- local moves that may still matter because they touch opponent stones or connect own groups.

The implementation uses only the logical topology (`point_count` and neighbor relations). It has no rectangular-board assumptions, so the same logic applies to cube and torus boards.

## Second PASS handling

After exactly one PASS in MAIN, PASS remains legal. Search may suppress PASS from the returned visit distribution only when an actually searched non-PASS move:

- improves the estimated score by at least the configured point threshold, and
- does not reduce estimated win probability beyond the configured tolerance.

This is a root decision rule, not a game rule. If no qualifying non-PASS move exists, PASS remains selectable. A high PASS policy prior alone is therefore not enough to terminate play when search has evidence that a non-PASS move preserves the win while materially improving the score.

## Root ending bonus

The root ending bonus is a small search preference intended to reduce premature PASS and pointless territory filling. PASS receives the territory-style ending adjustment in score-point units. Ownership can also apply an adjustment to clearly pointless filling or deep, unsupported invasions.

These adjustments affect root move evaluation only. They do not modify legal moves or final scoring.

## Self-play sampling

V3 does not use one blanket endgame oversampling multiplier. Sampling weights are independent for:

- `main_after_one_pass`,
- `cleanup1`,
- `cleanup2`.

Ordinary MAIN positions remain weight 1. The exact effective weights are saved in the run metadata so resumed training cannot silently change the search/sampling contract.

## Arena and reproducibility

Arena evaluation uses the same score-aware search heads and search utility contract as self-play while keeping Arena-specific exploration disabled as configured. Checkpoints and run manifests persist the search contract and search settings. Resume validation is fail-closed for incompatible V3 search contracts.

This separation is intentional: the rules engine answers **what is legal and how the game ends**; the search layer answers **which legal move is preferable to explore or select**.
