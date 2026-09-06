# GoCube KataGo-compatible Japanese adjudication V3

## Contract

- Adjudicator: `gocube-katago-japanese-v3`
- Observation schema: `gocube-observation-v3`
- KataGo rules document: Rules Version 3
- KataGo reference commit: `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`
- Reference source anchors: `docs/rules.html`, `cpp/game/board.cpp`, `cpp/game/board.h`, `cpp/game/boardhistory.cpp`, `cpp/game/boardhistory.h`
- Benson reference: J. Benson, *Life in the Game of Go* (1976)

The production rule profile is Japanese-like KataGo: `KoRule=SIMPLE`, `ScoringRule=TERRITORY`, `TaxRule=SEKI`, multi-stone suicide disabled, Button disabled, White handicap bonus 0, and KataGo self-play optimizations enabled.

`gocube-japanese-cleanup-v2` remains available only as a legacy experimental adjudicator for replay, evaluation, diagnostics, and historical reproducibility. It is not authoritative for new production training.

## Topology independence

Rule semantics operate only on occupancy plus `Topology.neighbor_indices(...)`. Cube seams and Torus wraparound are ordinary graph edges. Production scoring, group, liberty, region, Benson/pass-alive, ko, and cleanup code contains no board-edge, row/column, corner, or rectangular flood-fill assumptions. Rectangular topology exists only in tests for conformance-style fixtures.

## State machine

`MAIN -> CLEANUP_1 -> CLEANUP_2 -> SCORED`, with `NO_RESULT` as a separate terminal kind.

MAIN and each cleanup phase can end through formal pass/repetition rules. A cycle that reaches the Rules V3 repeated-state termination condition is `NO_RESULT`. The emergency move cap also produces only `NO_RESULT`; it never forces a score.

With self-play optimization enabled, if every logical point belongs to a pass-alive group or pass-alive territory, MAIN may terminate directly with the score-equivalent result.

## Benson and pass-alive

`benson-pass-alive-v1` semantics are retained as a generic graph fixed-point proof of unconditional life. V3 exposes black/white pass-alive groups and black/white pass-alive territory separately.

`not pass-alive` means only `not proven pass-alive`; it is never interpreted as dead.

Pass-alive territory follows Rules V3 over maximal non-own-color regions, including the possibility of opposing stones inside such territory. It does not use the V2 dead/seki heuristics.

## Cleanup 1

After MAIN ends under territory scoring, the game continues. Players make real moves, captures mutate the board and capture counts, and Rules V3 cleanup ko semantics apply. There is no +1-per-move score compensation in this phase.

## Cleanup 2

At entry, all ko-recapture blocks are cleared and the complete grid coloring is stored as `second_cleanup_start_colors`. Real moves and captures continue. Each real move made by a player in CLEANUP_2 contributes +1 point to that player's final score. This compensates for physically filling territory while removing dead stones.

## Cleanup ko

A cleanup ko-move is a pseudolegal move for which the opponent has a pseudolegal reply restoring the exact previous grid coloring.

Cleanup state includes `ko-recapture-blocked` points and phase-local ko history. A cleanup ko capture is forbidden when it captures a region containing a blocked point, or when the same player already made the same ko capture at the same point from the same exact grid coloring in the phase. A legal ko capture marks its played point blocked; blocks whose points become empty are removed.

KataGo pass-for-ko has two board-point forms, both represented in the existing `point_count + PASS` action space and both leaving the board unchanged while consuming the turn. A player may choose the blocked opposing single stone in atari itself, or choose the empty ko-capture point whose unique capturable one-stone target is that blocked stone. In either case the corresponding ko-recapture block is removed. No extra action type is introduced.

## Final Tax=SEKI scoring

Before scoring, KataGo self-play semantics remove each color's stones located in the opponent's pass-alive territory; these removals count as captures.

An independent-life region for a color is a maximal non-opponent region containing neither a dame region nor a stone region in atari. This is the V3 source of Tax=SEKI territory; no heuristic `seki` classifier is authoritative.

For each color:

1. +1 for every empty point inside that color's independent-life regions.
2. + captures of the opposing color.
3. +1 for every real move made by that color during CLEANUP_2.
4. -1 for every stone of that color outside its independent-life regions that was not that color at the start of CLEANUP_2.
5. White additionally receives komi.

Winner, margin, score-head target, and ownership target derive only from this final V3 calculation. Cleanup captures are therefore part of the formal final result.

The implementation follows the corrected post-issue-#1158 behavior: unassigned single-color empty components are not silently dropped, and scoring is based on the complete final position rather than only stones matching `second_cleanup_start_colors`.

## NO_RESULT and training targets

Framework utility may expose `NO_RESULT` through the draw slot because MCTS requires a terminal utility vector. The training collector separately checks `training_valid`; a `NO_RESULT` game emits no ordinary policy/value/score/ownership terminal samples.

Only `terminal_kind == SCORED` can produce value and score targets. An actual scored draw is valid training data; `NO_RESULT` is not.

Ownership labels are Black, White, or Neutral based on final formal independent-life/area results. Dame and seki are Neutral. The target includes a point mask so auxiliary loss can exclude points that cannot be authoritatively labeled without inventing alive/dead status.

## Observation V3

The observation preserves Black stones, White stones, previous Black, previous White, current player, pass state, Black captures, and White captures, and adds CLEANUP_1 flag, CLEANUP_2 flag, ko-recapture-blocked mask, second-cleanup-start Black/White masks, CLEANUP_2 Black/White move counts, repetition pressure, and current ko-repeat-forbidden mask.

This state exposes cleanup legality and score-relevant state used by the current policy/value inference contract.

## Compatibility and fingerprint

V1 Chinese, Japanese Cleanup V2, and V3 have separate Game classes and manifest versions. New `train.py` defaults only to V3 and the namespace `gocube-{topology}-{size}-japanese75-katago-v3`.

Checkpoint/run metadata records rule set, komi, terminal adjudicator, observation schema, topology, size, KataGo rules version, KataGo reference commit, and deterministic SHA-256 rules fingerprint. The rules implementation version is bumped whenever production move legality or adjudication semantics change; adding the second KataGo pass-for-ko form therefore changes the V3 fingerprint. V3 loading fails closed if required metadata is missing or differs. V2 samples/checkpoints are never silently resumed as V3.