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

MAIN and each cleanup phase can end through formal pass/repetition rules. A cycle that reaches the Rules V3 repeated-state termination condition is `NO_RESULT`. The self-play episode budget is not a rule transition: the runner may force-score at `256 + 24 * point_count`, with `termination_reason=episode_move_limit` and runtime provenance. Search clones never apply that budget.

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

## Score initialization and final Tax=SEKI scoring

Every formal V3 state carries the equivalent of KataGo's
`BoardHistory::whiteBonusScore`. `BoardHistory::clear` initializes it to the
black setup-stone count minus the white setup-stone count, then applies the
capture-counter correction using KataGo's captured-colour counters. GoCube's
capture tuple is stored by capturing player, so the equivalent is
`black_stones - white_stones - black_captures + white_captures`. A real move in
MAIN or CLEANUP_1 adds `+1` for Black or `-1` for White; CLEANUP_2 adds no such
offset because its start-colour accounting is part of the formal area score.
This value is explicit state and is rebuilt by synthetic cleanup rebasing.

`main_moves`, `cleanup1_moves`, and `cleanup2_moves` remain useful replay
telemetry, but none of them selects a scoring implementation.

For a position that enters CLEANUP_2 and ends without a real CLEANUP_2 move,
KataGo's Rules V3 scorer keeps the board and capture counters unchanged for a
pass-only CLEANUP_2 finish. If a real CLEANUP_2 move was played, the port uses
the pinned second-encore start-colour accounting before independent-life
scoring. This avoids inventing captures for an immediate pass-only phase.

An independent-life region for a color is a maximal non-opponent region containing neither a dame region nor a stone region in atari. This is the V3 source of Tax=SEKI territory; no heuristic `seki` classifier is authoritative.

The authoritative score is the formal board-area score plus the explicit
white-oriented bonus above and komi. For the public GoCube breakdown, the
bonus is represented on White so that `white - black` is the canonical final
margin; it is not a territory/prisoner decomposition. The formal area itself
is computed as follows:

1. +1 for every empty point inside that color's independent-life regions.
2. remaining stones count before CLEANUP_2, or only when their color matches
   `second_cleanup_start_colors` in CLEANUP_2;
3. White additionally receives the explicit bonus and komi.

Winner, margin, score-head target, and ownership target derive only from this final V3 calculation. Cleanup captures are therefore part of the formal final result. A formal cleanup score uses `termination_reason=formal_pass`; an early pass-alive score uses `pass_alive`.

The implementation follows the corrected post-issue-#1158 behavior: unassigned
single-color empty components are not silently dropped, and scoring is based
on the complete final position rather than a `main_moves`-selected fallback.

## NO_RESULT and training targets

Framework utility may expose `NO_RESULT` through its dedicated third value slot because MCTS requires a terminal utility vector. The training collector retains the policy and value samples from every visited `NO_RESULT` position, assigns value target `[0, 0, 1]`, and masks score and ownership targets. A `NO_RESULT` score target is stored as `NaN` with a zero score mask; loss code selects active rows before arithmetic.

Only `terminal_kind == SCORED` can produce score and ownership targets. An actual scored draw is valid training data and uses the win/loss mixture `[0.5, 0.5, 0]`; `NO_RESULT` is a separate value class, not a scored draw. A runner force-score also uses the scored targets, but its `result_provenance=runtime` and `termination_reason=episode_move_limit` are retained in the terminal, target bundle, and game record.

The V3 neural value head is player-to-move-relative: its first two classes are `[WIN for side to move, LOSS for side to move]`, followed by `NO_RESULT`. Thus a Black-to-move and White-to-move position encode the same absolute winner in different first/second slots. A scored draw is `[0.5, 0.5, 0]`; `NO_RESULT` is `[0, 0, 1]`.

Ownership labels are Black, White, or Neutral based on final formal independent-life/area results. Dame and seki are Neutral. The target includes a point mask so auxiliary loss can exclude points that cannot be authoritatively labeled without inventing alive/dead status.

## Observation V3

The observation preserves Black stones, White stones, previous Black, previous White, current player, pass state, Black captures, and White captures, and adds CLEANUP_1 flag, CLEANUP_2 flag, ko-recapture-blocked mask, second-cleanup-start Black/White masks, CLEANUP_2 Black/White move counts, repetition pressure, and current ko-repeat-forbidden mask.

This state exposes cleanup legality and score-relevant state used by the current policy/value inference contract.

## Compatibility and fingerprint

V1 Chinese, Japanese Cleanup V2, and V3 have separate Game classes and manifest versions. New production training defaults only to V3 and the namespace `gocube-{topology}-{size}-katago-v3-pilot`.

Checkpoint/run metadata records rule set, komi, terminal adjudicator, observation schema, topology, size, KataGo rules version, KataGo reference commit, score initialization contract, target contracts, deterministic SHA-256 rules fingerprint, search contract, and the termination contract. The local rules implementation version is `5` for S3; the upstream KataGo rules version remains `3` at the pinned commit. The pinned search contract is `katago-pinned-search-v3`; replay format is v4 and training contract is v3. Replay v4 adds the required `uint8 [N]` target-provenance sidecar with encoding `gocube-target-provenance-encoding-v1`; it remains outside the seven training tensors. V3 loading fails closed if required metadata is missing or differs. Replay/checkpoint artifacts from pre-S1/S3 contracts, and old search semantics, are never silently resumed under the new boundary.
