# Golden Stage 2 — Sequential Arena V1 and search qualification

## Scope

Stage 2 extends only the independent `gocube_golden` reference line from Stage 1.
It does not import `alphazero/Arena.pyx`, production Arena bookkeeping, production
terminal/scoring, batching, worker queues, asynchronous inference, training, or
Cube code.

Production Golden identity remains:

- rules: `graph-area-v1`
- topology: canonical Golden Torus 5×5
- komi: `0.5`
- Arena/search contract: `golden-arena-search-v1`
- execution watchdog: 500 applied actions, checked only after formal rule terminal

`komi=0.0` is allowed only for explicit test/research fixtures such as the DRAW
mapping test. `7.5` remains a fail-closed legacy-contamination error.

## Stage-1 hardening

Canonical live states now require both:

1. `superko_history[-1] == board_key`
2. no duplicate stone positions in canonical live superko history

PASS does not append board history and therefore preserves both invariants.
Arbitrary board fixtures are explicitly tagged `synthetic-test-research` through
`research_state_from_stones`; the compatibility `state_from_stones` helper is a
synthetic-fixture alias. Sequential Golden Arena accepts only canonical live
state/history and requires every noninitial start state to replay exactly from a
legal start trace.

## Arena result boundary

Golden referee returns only absolute `BLACK`, `WHITE`, or `DRAW`. Player/model
identity is not visible to scoring. Only after `GoldenResult` exists does Arena
map the absolute color result to `A_WIN`, `B_WIN`, or `DRAW`.

Technical outcomes are separate:

- `TRUNCATED_MOVE_LIMIT`
- `ERROR_ILLEGAL_PLAYER_ACTION`
- `ERROR_PLAYER_EXCEPTION`
- `ERROR_SEARCH`

They never carry a Golden result, WDL mapping, or score.

Every `GameRecord` is frozen and contains start evidence, independent seeds,
color assignment, full action evidence, final board, absolute/mapped result (if
any), score, contract/settings, and typed termination. `validate_game_record`
replays raw evidence independently. `recompute_summary` rejects duplicate
`game_id` and derives all counters/pair summaries from records rather than
mutable in-flight counters.

## RNG discipline

A master seed deterministically derives:

1. a game seed,
2. an A seed,
3. a B seed,
4. per-move child seeds for each slot.

A's stream therefore does not depend on how many random calls B makes.

## Search decision: SEARCH PATH B

The existing pinned Cython MCTS was inspected before writing a replacement. It
does not pass the new Golden boundary without semantic adapters that Stage 2 is
specifically intended to eliminate:

- it imports production GoCube/KataGo exploration and search helpers;
- pinned search requires `score` and `ownership` heads;
- its current value adapter expects player-relative
  `[WIN, LOSS, NO_RESULT]`, then converts to absolute Black/White;
- its utility aggregation is White-perspective and includes score utility and
  root-ending logic tied to production GoCube state.

Adapting Golden `[WIN, DRAW, LOSS]` side-to-move values into that stack would
require hidden perspective/semantic conversions and production dependencies.
Therefore Stage 2 takes **PATH B**.

The replacement is intentionally small: `golden-sequential-puct-v1`.

- one tree
- sequential Python
- no batching
- no tree reuse
- no transposition table
- no virtual loss
- no fast simulations
- no noise in Arena
- exact terminal value only from Golden referee

Model boundary:

`[WIN, DRAW, LOSS]` relative to side-to-move → `u = P(WIN) - P(LOSS)`

Internal convention:

`edge-Q-from-parent-side-to-move`

The one turn-boundary conversion is the named child→parent sign flip. There is
no absolute Black/White neural value representation in Stage-2 search.

## Tiny oracle

`solve_exact` exhausts tiny Golden research graphs. If the node budget is not
enough, it returns `UNKNOWN`; a partial tree is never reported as an oracle.

The qualification suite uses a three-point line graph. Exhaustive solve proves
the unique best opening action is the center point. Sequential PUCT with a
uniform fake evaluator selects the same action at the qualification budget.

## Demos

```bash
.venv/bin/python tools/torus_golden_stage2_arena_demo.py
.venv/bin/python tools/torus_golden_stage2_search_demo.py
```

The first prints the paired color-swap proof with absolute and A/B results. The
second prints side-to-move, legal actions, fake policy/WDL, root visits,
selected action, exact oracle action, and PASS legality.
