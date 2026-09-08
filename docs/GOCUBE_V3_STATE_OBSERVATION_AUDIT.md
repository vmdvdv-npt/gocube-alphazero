# GoCube V3 state and observation audit (H1) / exact simple ko (M1)

Status: H1 audit completed; M1 production fix completed.

This audit was run against `codex/h1-m1-state-ko-audit` at `main` commit
`84a878e`, after verification-groundwork and S2/model-contract work. The
audited production V3 observation has 17 planes. The pinned self-play wrapper
adds plane 17 (`passWouldEndPhase`) and uses
`gocube-observation-v4-pass-would-end-phase` with 18 planes. G1 structural
features were not included in this audit and must be re-probed after G1 is
merged.

## State inventory

The authoritative exact game state is the frozen `V3State` in
`alphazero/envs/gocube/katago_v3.py`. `GoGame` is the framework facade and
`Pinned*JapaneseGame` adds search/self-play metadata. `V3State` has 25 fields:

| Field | Type | Legality | Phase | Score | Root/search | Technical stop | In observation | Exactly reconstructible | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `board` | `uint8[N]` | yes | yes | yes | yes | no | planes 0/1 | state/replay | A |
| `current_player` | `int` | yes | yes | no | yes | no | plane 4 | state/replay | A |
| `turns` | `int` | no direct | no | no | no | move cap | no | state/replay only | D |
| `consecutive_passes` | `int` | yes | yes | no | yes | yes | plane 5 | immediate state | A |
| `captures` | `(int, int)` | no direct | no | yes | score context | no | planes 6/7, normalized | replay/state exact count | A |
| `white_bonus_score` | `float` | no direct | no | yes | score utility context | no | no | replay/state only | D |
| `previous_board` | `uint8[N] \| None` | simple ko | no | no | yes | no | planes 2/3 | replay/state | A |
| `phase` | `str` | yes | yes | yes | yes | no | planes 8/9 | state/replay | A |
| `ko_recap_blocked` | tuple of points | cleanup ko | yes | no | yes | no | plane 10 | state/replay | A |
| `phase_history` | tuple of state keys | cycle checks | yes | no | yes | cycle | no | replay only | D |
| `history_since_pass` | tuple of state keys | cycle checks | yes | no | yes | cycle | no | replay only | D |
| `black_pass_states` | tuple of state keys | PASS transition | yes | no | yes | no | no | D |
| `white_pass_states` | tuple of state keys | PASS transition | yes | no | yes | no | no | D |
| `ko_capture_history` | tuple of ko records | cleanup repeat | yes | no | yes | no | no | D |
| `second_cleanup_start_colors` | `bytes \| None` | no direct | CLEANUP_2 | yes | yes | no | planes 11/12 | state/replay | A |
| `cleanup2_moves` | `(int, int)` | no direct | yes | no | search context | no | planes 13/14, normalized | replay/state exact count | A |
| `main_moves` | `(int, int)` | no | accounting | no; telemetry | no | no | replay/state | C |
| `cleanup1_moves` | `(int, int)` | no | accounting | no | search context | no | no | replay/state | C |
| `terminal_kind` | `str \| None` | terminal gate | yes | yes | yes | yes | no; terminal API | state/replay | D |
| `no_result_reason` | `str \| None` | no | yes | no | yes | yes | no; terminal API | state/replay | D |
| `pass_alive_early_end` | `bool` | no | terminal gate | no | root stop | yes | no | replay/state | D |
| `entered_cleanup1` | `bool` | no | provenance | no | no | no | no | replay/state | C |
| `entered_cleanup2` | `bool` | no | provenance | no | no | no | no | replay/state | C |
| `cleanup_captures` | `int` | no | diagnostic | no under current scorer | no | no | no | replay/state | C |
| `ko_unblock_actions` | `int` | no | diagnostic | no | no | no | no | replay/state | C |

Classification follows the H1 rule: A means directly encoded, B means exactly
derivable, C means absent without an immediate semantic effect, D means a
potentially relevant hidden field without a reachable semantic collision in
the tested bound, and E means a proven reachable same-observation semantic
collision. No field reached category E in this audit.

The per-field lifecycle/preservation audit is:

| Field | Created/updated by | Clone | Supported record |
| --- | --- | --- | --- |
| `board` | initial/setup; every placement | preserved | final board |
| `current_player` | initial/setup; every accepted action and phase reset | preserved | final position |
| `turns` | initial/setup; every accepted action | preserved | final position |
| `consecutive_passes` | initial/setup; PASS increments, placement and phase reset clear | preserved | final position |
| `captures` | setup; every placement capture | preserved | final position |
| `white_bonus_score` | setup from board/captures; MAIN/CLEANUP_1 placement updates | preserved | final position |
| `previous_board` | setup optional; every accepted action | preserved | final position |
| `phase` | setup; second PASS, cycle, cap, and scoring transitions | preserved | final position |
| `ko_recap_blocked` | setup; cleanup ko placement/unblock; phase reset clears | preserved | final position |
| `phase_history` | setup; every accepted action; phase reset starts a segment | preserved | omitted; replay only |
| `history_since_pass` | setup; placement/unblock/cycle; PASS starts a segment | preserved | omitted; replay only |
| `black_pass_states` | setup; black PASS; phase reset clears | preserved | omitted; replay only |
| `white_pass_states` | setup; white PASS; phase reset clears | preserved | omitted; replay only |
| `ko_capture_history` | setup; cleanup ko capture; phase reset clears | preserved | omitted; replay only |
| `second_cleanup_start_colors` | setup in CLEANUP_2; CLEANUP_1 -> CLEANUP_2 transition | preserved | final position |
| `cleanup2_moves` | setup; CLEANUP_2 placements | preserved | final position/diagnostics |
| `main_moves` | setup; MAIN placements | preserved | final position/diagnostics |
| `cleanup1_moves` | setup; CLEANUP_1 placements | preserved | final position/diagnostics |
| `terminal_kind` | setup terminal fixtures; cycle/cap/phase scoring | preserved | final position and terminal |
| `no_result_reason` | cycle/cap terminal transitions | preserved | final position and terminal |
| `pass_alive_early_end` | initialized false; pinned/game wrapper early-end check | preserved | final position/diagnostics |
| `entered_cleanup1` | setup if cleanup; first cleanup transition | preserved | final position/diagnostics |
| `entered_cleanup2` | setup if CLEANUP_2; second cleanup transition | preserved | final position/diagnostics |
| `cleanup_captures` | initialized; cleanup placements that capture | preserved | final position/diagnostics |
| `ko_unblock_actions` | initialized; cleanup PASS-for-ko point actions | preserved | final position/diagnostics |

`GoGame.clone()` passes the immutable `V3State` through the facade constructor,
and the frozen dataclass contains no mutable rule state outside its arrays.
The tests still compare every field explicitly so an added field cannot be
silently omitted from the audit.

`main_moves` and `cleanup1_moves` are retained telemetry. The current scorer
uses the explicit `white_bonus_score`/capture/setup state and does not select
a score algorithm from those counters. `cleanup_captures` and
`ko_unblock_actions` are diagnostic counters. `turns` remains technically
relevant because `apply_v3_action` has an emergency cap, but its semantics are
owned by S3.

## Wrapper and pinned state

The framework facade stores `_board`, `_player`, `_turns`, `last_action`,
`_state`, and `_terminal`; these mirror or expose V3 state and are not a
second source of rules truth. The pinned wrapper stores:

| Wrapper field | Purpose | Clone behavior | Observation/rule role |
| --- | --- | --- | --- |
| `_pinned_auto_end_pass_alive` | game-level early terminal switch | copied | game-level terminal behavior |
| `_pinned_root_prune_useless_moves` | root-only pruning switch | copied | root search only |
| `_pinned_selfplay_semantics` | enables pinned game path | copied | integration only |
| `_pinned_seki_fork_hack_prob` | sampling probability | copied | generation only |
| `_pinned_is_search_clone` | distinguishes MCTS clone | set on clone | search control |
| `_pinned_at_search_root` | root-only lifetime marker | set on clone | root control |
| `_pinned_started_from_seki_fork` | provenance | copied | generation metadata |
| `_pinned_start_phase` | starting provenance | copied | generation metadata |
| `_pinned_move_history` | ordered `(player, action)` history | copied | fork/root pruning |
| `_pinned_state_history` | saved state prefix for forks | copied | fork reconstruction |
| `_pinned_state_history_offset` | absolute history alignment | copied | fork reconstruction |

The diversified wrapper additionally stores early/ordinary fork probabilities,
fork-pool capacity, fork provenance/suppression, training state history, and
its offset. The self-play agent separately owns MCTS objects, policy
histories, temperatures, and reset markers. These controls do not enter the
NN observation and are not V3 rule state.

Clone tests compare every `V3State` field, pinned wrapper accumulator, and
representative diversified accumulator. The two intentional clone marker
changes (`_pinned_is_search_clone` and `_pinned_at_search_root`) are asserted
separately.

## Observation inventory

`GoGame.observation()` emits spatial constant planes in canonical
`PointId` order:

| Plane | Meaning |
| ---: | --- |
| 0, 1 | current black / white stones |
| 2, 3 | immediately previous accepted board, black / white |
| 4 | side to move (`+1` black, `-1` white) |
| 5 | `consecutive_passes == 1` |
| 6, 7 | captures by capturing player, divided by point count |
| 8, 9 | `CLEANUP_1` / `CLEANUP_2` |
| 10 | cleanup ko-recap block mask |
| 11, 12 | `second_cleanup_start_colors`, black / white |
| 13, 14 | `cleanup2_moves`, divided by point count |
| 15 | repetition-pressure projection |
| 16 | cleanup ko-repeat-forbidden projection |
| 17 (pinned only) | exact `passWouldEndPhase` bit |

The previous-board planes are sufficient for immediate MAIN simple-ko
context. The cleanup projections are intentionally not treated as a lossless
encoding of all hidden history. Pinned search receives exact `V3State` and
history separately; the NN input is therefore an approximation boundary, not
the engine's complete rules state.

## PASS and phase transition audit

Every accepted action increments `turns` and switches the player. A placement
resets `consecutive_passes`; a PASS copies the current board into
`previous_board` and increments the consecutive-pass counter. The second
consecutive PASS transitions MAIN -> CLEANUP_1, CLEANUP_1 -> CLEANUP_2, and
CLEANUP_2 -> SCORED. The transition resets phase-local history and preserves
captures; entering CLEANUP_2 records the current board as
`second_cleanup_start_colors`.

`black_pass_states` and `white_pass_states` record the state before each
player's PASS. A repeated same-player pass state can end a phase even when
the consecutive-pass count alone would not. The pinned plane 17 is computed
from this actual transition (`pass_would_end_phase`), not from plane 5 alone.

Cleanup PASS-for-ko is a point action, not an ordinary capture: it consumes a
turn and clears the relevant cleanup ko block while leaving board occupancy
and capture counts unchanged. Both blocked-stone and empty-ko-point forms are
covered by the existing cleanup tests and the M1 regression suite.

## Score, root, and technical semantics

The exact scorer uses board area, captures, komi `0.5`, explicit
`white_bonus_score`, and `second_cleanup_start_colors`. It does not infer
score initialization from `main_moves` or `cleanup1_moves`. No S1 scorer
change was made here.

The root ending branch is computed once per root and consumes the exact
simple-ko result described below. For a true active simple ko it suppresses
the stone-move ending adjustment; for an ordinary one-stone capture it still
evaluates the adjustment. PASS remains governed by its own area/phase branch.
The audit found no other production consumer of the old helper: root pruning
continues to use its independent pass-alive condition, and cleanup legality
continues to use the V3 rule transition/ko-block state.

Current technical behavior is an emergency cap of
`256 + 24 * point_count` accepted actions in `apply_v3_action`. The raw V3
path emits `NO_RESULT` with reason `move-cap`; the pinned wrapper converts
that specific crossing to an immediate scored terminal to match its current
KataGo training path. This is recorded as current behavior only. The
`turns=cap-1` same-observation probe is kept as a manually injected diagnostic
and is not included in the reachable collision search. Any redesign of
move-cap/episode-limit semantics is deferred to S3.

## M1 exact simple-ko fix

### Root cause and reproduction

The old `_simple_ko_likely_active` implementation treated
`changed_points == 2` as sufficient. That is only a shape prefilter:

```text
two changed points != simple ko
```

The required Cube 4 false-positive fixture is:

```text
black: front:0:1, front:1:0, front:1:2, front:2:0
white: front:1:1
black plays: front:2:1
```

The capture changes exactly two points, so the pre-fix predicate is `True`.
White's apparent recapture at `front:1:1` is suicide. The independent graph
proof and production rule transition both report `simple_ko = False`.

The paired true fixture changes exactly two points, permits the immediate
recapture under local capture/suicide rules, and restores the full prior board:

```text
black: front:1:0, left:0:3, top:3:0
white: front:0:0, front:0:2, front:1:1, top:3:1
black plays: front:0:1
white recaptures: front:0:0
```

The exact rule helper now:

1. uses `previous_board` as the immediately preceding accepted snapshot;
2. applies the two-point shape only as a cheap prefilter;
3. identifies the removed stone as the only candidate recapture point;
4. runs the real graph capture-before-suicide transition without ko blocking;
5. requires one captured stone and byte-for-byte restoration of the previous
   board.

The cleanup ko block is intentionally ignored by this helper. It answers the
unblocked local rule fact; cleanup policy separately enforces the block and
PASS-for-ko release.

### M1 search effect and contract

The root ending calculation now calls the exact rule-derived result. The false
fixture no longer disables its ending-bonus branch; the true fixture still
does. The applicable rectangular graph fixture and the Cube fixture both
verify the same transition without renderer coordinates.

Because root move preferences change, `KATAGO_SEARCH_CONTRACT` was bumped from
`katago-pinned-search-v2` to `katago-pinned-search-v3` through the existing S2
contract machinery. No observation shape, network architecture, replay
schema, target semantics, komi, or search budget changed. Existing checkpoints
with v2 are weights-compatible in principle but search-semantics-incompatible;
the current loader fails closed on the contract mismatch and does not rename
metadata automatically. No retraining was performed.

### M1 performance

Targeted local timing on the Cube 4 false fixture (Python 3.9, 10,000 helper
calls / 500 root computations, warm import):

| Measurement | Result |
| --- | ---: |
| old two-point helper | 1.69 us/call |
| exact helper | 25.76 us/call |
| root ending, old predicate | 1.44 ms/root |
| root ending, exact predicate | 1.87 ms/root |

The exact check is a local graph transition, not a full replay. Root bonuses
are precomputed once per root, so this measured overhead is bounded to root
setup rather than every PUCT edge.

## Reachability methodology

The test-only H1 infrastructure is in `tests/support/h1_probe.py` and
`tests/test_gocube_h1_state_audit.py`.

The ordinary probe uses legal actions from `v3_valid_moves`,
`apply_v3_action` transitions, a complete V3 state key, seed `20260908`,
depth 6, and a 256-state cap. A legal six-PASS path is added explicitly so
MAIN, CLEANUP_1, CLEANUP_2, and SCORED boundaries remain covered despite the
BFS cap.
Fork samples use the official diversified plain-fork and seki-fork pools.
Synthetic cleanup samples use the official `rebase_cleanup_training_state`
helper. No main collision search constructs contradictory `V3State` values.

Observation keys use shape, dtype, raw C-order bytes, and SHA-256; Python's
persistent built-in `hash()` is not used. Each observation group is compared
using this immediate semantic signature:

```text
valid move mask
phase and side to move
PASS result (phase, side, terminal kind, reason)
score-relevant offset/capture/start-color state
exact simple-ko context
whether the next accepted action reaches the technical cap
```

The signature deliberately does not include learned network output.

## Collision search results

The checked run examined 267 reachable or officially rebased samples in 263
observation groups:

| Family | Samples |
| --- | ---: |
| ordinary bounded start | 256 |
| ordinary phase boundary | 6 |
| ordinary fork | 1 |
| early fork | 1 |
| seki fork | 1 |
| synthetic CLEANUP_1 | 1 |
| synthetic CLEANUP_2 | 1 |

There were three observation collision groups and zero semantic collisions.
The three groups were expected aliases: two ordinary PASS-boundary samples
with identical immediate semantics and one shared candidate state materialized
through ordinary/early/seki fork labels. Therefore the result is:

```text
No reachable collision with different immediate semantics was found within
the tested state families and bounds. This is not a proof of full Markov
completeness.
```

The probe has dedicated regression tests for legality comparison, PASS
transition comparison, deterministic generation, clone preservation, and
inclusion of both synthetic cleanup phases and all three fork families.

### Classification of candidate issues

| Candidate | Result | Classification / follow-up |
| --- | --- | --- |
| hidden phase/cycle/pass history | no semantic collision in bound | D; rerun with larger/future state families |
| cleanup ko/repetition context | no semantic collision in bound; exact M1 helper used | D; preserve fixture coverage |
| second-cleanup start colors | direct planes 11/12; no collision | A for the encoded colors |
| captures and score initialization | direct normalized capture planes; exact offset remains engine state | D for `white_bonus_score`; revalidate after any S1 changes |
| `turns` / technical budget | no reachable cap collision searched | `BLOCKED_BY_S3_SEMANTICS`; diagnostic only |
| raw game-record serialization | final-position record omits full history tuples | serialization boundary, not confirmed H1 observation collision |

The artificial `turns=0` versus `turns=cap-1` construction remains only a
technical diagnostic. It is not used as an H1 finding because it is not a
legal replay or official setup. S3 must decide whether this budget remains
part of the environment state.

## Clone and serialization boundary

Clone preserves all rule-relevant V3 fields, including board, phase, side,
previous-board context, captures, cleanup metadata, counters, and history.
Pinned and diversified fork accumulators are copied; clone markers are changed
only to identify a search clone and root.

The supported JSON game record stores board, previous board, counters, phase,
captures, terminal metadata, and selected cleanup diagnostics. It does not
store `phase_history`, `history_since_pass`, `black_pass_states`,
`white_pass_states`, or `ko_capture_history` as a raw snapshot. Those fields
can only be reconstructed if the accepted move list and setup context are
available and replayed under the same contract. A final-position record alone
is therefore not an exact V3 raw-state serialization. Replay format rewrite
is explicitly deferred.

## Dependency boundaries

### S1-dependent cases

Score initialization and `white_bonus_score` are audited against the current
S1 contract (`katago-boardhistory-clear-v1`). Any future S1 scoring change
requires revalidation of the score-offset rows and cleanup fixtures. This PR
does not change scoring.

### S3-dependent cases

Move-cap crossing and technical episode termination are documented as current
behavior, not redesigned. The H1 technical-budget candidate must be rerun
after S3 decides whether the budget is removed, changed, or represented in the
search state.

### G1-dependent follow-up

G1 changes structural NN inputs and architecture. This branch deliberately
does not touch observation shape, channels, masks, network projection, replay,
or architecture IDs. After G1 merges, rerun the H1 collision suite against
the new observation and record whether any candidate collision disappears or
remains.

## Tests and changed files

Targeted M1/H1 tests:

```text
9 passed   tests/test_gocube_search_ko_exact.py
12 passed  tests/test_gocube_h1_state_audit.py
3 passed   tests/test_gocube_h1_observation_probe.py
628 passed full repository suite (`pytest -q`)
```

Production changes:

```text
alphazero/envs/gocube/katago_v3.py
alphazero/envs/gocube/__init__.py
alphazero/search_contract.py
```

Tests and audit infrastructure:

```text
tests/test_gocube_search_ko_exact.py
tests/test_gocube_h1_state_audit.py
tests/support/h1_probe.py
```

Documentation:

```text
docs/GOCUBE_V3_STATE_OBSERVATION_AUDIT.md
docs/GOCUBE_VERIFICATION_GROUNDWORK.md
docs/KATAGO_PINNED_SEARCH.md
```

Scope check: no G1 architecture change, S3 move-limit change, S1 scorer
change, new observation channel, training hyperparameter change, replay
migration, or target-semantics change was made in this change set. The only
production semantic change is M1 exact simple-ko detection and its explicit
search-contract version bump.
