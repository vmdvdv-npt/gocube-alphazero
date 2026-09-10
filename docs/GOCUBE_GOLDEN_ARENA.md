# GoCube Golden Arena V1

## Purpose

`tools/gocube_golden_arena.py` is the deliberately slow reference evaluator for
GoCube. It exists to answer two questions without relying on the production
Arena bookkeeping:

1. What absolute terminal result did the game reach: **Black**, **White**,
   **Draw**, or **No Result**?
2. Did the production terminal scorer compute exactly the same numeric score
   and winner as an independent implementation?

Golden Arena is an oracle and diagnostic tool, not the high-throughput Arena
used for routine training gates.

The Rules-V3 score arithmetic was checked against the repository's pinned
KataGo source commit `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`, specifically
`BoardHistory::countTerritoryAreaScoreWhiteMinusBlack`,
`BoardHistory::endAndScoreGameNow`, and the territory-scoring
`whiteBonusScore` update in `BoardHistory::makeBoardMoveAssumeLegal`. The
reference scorer remains a separate Python implementation over generic graph
adjacency; it does not call KataGo or the production GoCube scorer at runtime.

## Trust boundary

Golden Arena deliberately does **not** import or call `alphazero.Arena`.

It runs:

- one process;
- one game at a time;
- CPU inference;
- one Torch thread;
- no inference batching;
- no worker queues or routing;
- no fast search;
- no root noise;
- no root temperature;
- no move temperature;
- fixed 50-simulation MCTS;
- paired color swaps.

The two games in a pair share one RNG seed. Model A is Black in the first game
and White in the second game. This is intentionally simple and repeatable.

Golden Arena does reuse the repository's current neural-network loader,
observation adapters, and MCTS. Therefore it is a reference for **Arena
orchestration, color attribution, and terminal adjudication**, not yet an
independent proof that the current MCTS search algorithm itself is correct.

## Independent winner determination

The Golden scorer lives in `alphazero/envs/gocube/golden_scoring.py`. It does
not import production `final_v3_score`, `independent_life_analysis`,
`terminal_from_state`, or `win_state`.

It consumes only:

- the final occupancy vector (`empty`, `black`, `white`);
- graph adjacency for every logical point;
- terminal kind (`scored` or `no_result`);
- the Rules-V3 white score offset;
- the CLEANUP_2 start-color snapshot when present;
- capture counters for evidence;
- komi, hard-pinned to `0.5`.

It independently computes connected stone groups, liberties, Benson-style
pass-alive groups, pass-alive territory, independent-life areas, seki
exclusion, the remaining-stones clause, the Rules-V3 score offset, and the
final absolute outcome.

`NO_RESULT` is a distinct result type. It is never represented as `DRAW` in
the Golden API and it never receives a numeric score.

For a scored game, the winner rule is intentionally trivial at the final
boundary:

- `black_score > white_score` -> `BLACK`;
- `white_score > black_score` -> `WHITE`;
- equality -> `DRAW`.

No model index participates in this step. Only after the absolute color winner
has been established is it mapped to `win` or `loss` for model A.

## Independent reconstruction of score inputs

For a normal Golden Arena game starting from the empty board, the runner does
not simply trust the score-relevant counters stored in the semantic state.
While observing each legal state transition it reconstructs:

- captures by detecting removed opponent stones;
- `whiteBonusScore`: +1 for each real Black placement and -1 for each real
  White placement during MAIN/CLEANUP_1;
- the board snapshot at the transition into CLEANUP_2;
- the formal start-color snapshot used by pass-alive early termination.

Before scoring, every reconstructed value must equal the production state's
corresponding value. A mismatch aborts the run.

## Dual adjudication: disagreement is an error

Every completed Golden game is scored twice:

1. by `golden_scoring.py` from the raw final state plus independently
   reconstructed inputs;
2. by the production terminal scorer already attached to the game state.

Golden Arena compares terminal kind, Black score, White score, komi, margin,
and winner. Any discrepancy raises `GoldenScoreMismatch`; the game is not
counted and no aggregate result is emitted.

This fail-closed behavior is deliberate. A reference evaluator must prefer
"unknown because the implementations disagree" over silently choosing one
answer.

## Cube and Torus

The reference scorer is graph-based. It does not assume rows, columns, borders,
or a rectangular plane. Therefore Cube seams and Torus wrap-around are handled
through the same `neighbors_by_index` graph used to identify logical points.

The graph itself is validated before scoring:

- one adjacency list per point;
- no duplicates;
- no self-edges;
- all neighbor indices in range;
- every edge reciprocal.

Regression tests include both Cube and Torus territory across seams/wraps.

## Evidence

Golden output is write-once JSON. Existing evidence files are never silently
overwritten.

Each game record contains the absolute color result, model-A result, signed
margin, full action sequence plus digest, final occupancy vector, independently
reconstructed score inputs, terminal kind, and score breakdown. This is
deliberately redundant evidence: a later audit can re-adjudicate a saved result
without trusting the Arena's aggregate counters.

The run also records SHA-256 for both checkpoint files and the complete Golden
execution contract.

## Komi

Golden Arena accepts exactly `0.5`. The constant is intentionally duplicated
inside the independent scorer rather than imported from the production
contract, and a regression test asserts that both remain equal. Any other komi
fails closed.

## Running

Example:

```bash
PYTHONPATH="$PWD" .venv/bin/python tools/gocube_golden_arena.py \
  --run-a RUN_A --iteration-a 5 \
  --run-b RUN_B --iteration-b 2 \
  --games 8 --seed 20260910
```

Use small paired runs for diagnostics. The purpose is reference correctness,
not throughput. Production Arena performance requirements do not apply to this
single-threaded oracle.

## Interpretation

If Golden Arena and production Arena disagree on **who won the same finished
state**, treat that as a correctness failure in result accounting or terminal
scoring.

If both agree on terminal results but choose different moves from the same
positions, the remaining fault domain is search/inference/execution rather
than winner accounting. A future independent-MCTS oracle can narrow that layer
further without changing this terminal reference.
