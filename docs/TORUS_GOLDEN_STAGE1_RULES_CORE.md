# Torus Golden Stage 1 — Rules Core

Stage 1 adds a deliberately small, pure-Python referee for the Stage-0
`gocube-torus-golden-graph-area-v1` passport.

## Base and identity

- Stage-1 base branch: `codex/torus-rebuild-v1`
- Stage-1 base HEAD: `dc220ba770f720f6a13feadc5a4f6d760b891e25`
- Golden profile: `gocube-torus-golden-graph-area-v1`
- Rules profile: `graph-area-v1`
- Golden topology: `torus-5x5-row-major-v1`
- Topology fingerprint: `sha256:b4097c32d4ab5034b84300fa41f951b353a5fcf0d8226e83922889b5552289ef`
- Rules fingerprint at the proof komi: `sha256:8eac3337443a70893fa5ad359580f7ba92b18958e06f0d775c29f08791796842`
- Proof komi: `0.5`

The Stage-0 passport itself records the earlier pre-Stage-0 source SHA as part
of its immutable experiment identity. The SHA above is the actual branch HEAD
from which Stage 1 was created, after Stage 0 was merged.

## Architectural boundary

The oracle is the top-level package `gocube_golden/`, not a child of
`alphazero.envs.gocube`.

This placement is intentional. Importing the existing `alphazero` package
loads training/NN machinery, and importing `alphazero.envs.gocube` loads the
legacy/production rule stack. A reference referee that lived underneath those
packages would therefore acquire unwanted dependencies merely through Python
package initialization.

`gocube_golden` imports only Python standard-library modules and its own
modules. It does not import or call Japanese-V3, production `valid_moves` or
`play_action`, production terminal/scoring code, Benson, Coach, MCTS, PyTorch,
multiprocessing, Cython, or a production topology factory.

**The production Torus factory is not extended for 5×5.** The canonical Golden
Torus 5×5 lives entirely inside the reference package. Tiny/custom graph
construction is exposed only as explicit Golden test/research support.

## Implemented rule semantics

The Stage-1 state is immutable and records stones, absolute side to move, exact
positional-superko history, consecutive passes, topology/rules identities, and
komi. Board identity and full-state identity are deliberately different; full
state identity includes the entire superko history.

Stone moves use the Stage-0 order:

1. require an empty PointId;
2. place the moving side's absolute-colour stone;
3. find all adjacent opponent groups with zero liberties in the common
   post-placement snapshot;
4. remove all such groups simultaneously;
5. reject the move if the new own group still has zero liberties;
6. reject an exact board arrangement already present in positional-superko
   history;
7. append the new arrangement, switch side, and reset the pass count.

`PASS` is exempt from positional-superko repetition. It leaves both the board
and board-history unchanged, switches side, and increments the pass counter.
Only two consecutive passes create a rule-level terminal state. A stone move
resets the counter. Every action after terminal is rejected fail-closed.

## Scoring and result

Scoring is available only after formal double-pass terminal and uses literal
`graph-area-v1` semantics. Stones are counted as area. Each empty connected
component is Black territory only when its boundary contains Black and no
White, White territory only when its boundary contains White and no Black, and
neutral otherwise (including an empty boundary). No dead-stone cleanup occurs.

Komi is applied only to final `margin_black = black_area - white_area - komi`.
The first proof profile uses `0.5`, while the scorer remains capable of an
explicit finite research komi. The known legacy sentinel `7.5` always raises a
fail-closed error instructing escalation to the owner; it is never coerced.

A formal terminal produces an immutable `GoldenResult` with winner, areas,
komi, margin, `DOUBLE_PASS`, and the rules/topology fingerprints. Technical
truncation/error statuses are not Golden rule results.

## Replay and diagnostics

`replay(initial_state, actions)` applies a trace in order, stops at the first
illegal action, preserves its exact reason, and returns the state reached so
far. Replaying the same initial state and trace is deterministic.

The human-readable proof script is:

```bash
python3 tools/torus_golden_stage1_demo.py
```

Its pinned trace is:

```text
[0, 1, 2, 10, 6, 11, 21, 3, PASS, PASS]
```

Black's move at PointId 21 captures the White stone at PointId 1 through the
Torus north/south wrap relation. The trace then continues with White 3 and two
passes. Expected terminal breakdown:

```text
black area: 5
white area: 3
neutral points: 17
komi: 0.5
margin_black: 1.5
winner: BLACK
terminal reason: DOUBLE_PASS
```

## Colour-swap transformation

The colour-swap metamorphic test is not the false "swap stones only" symmetry.
It transforms all semantic colour-bearing state:

- Black ↔ White stones on the current board;
- Black ↔ White in every historical board arrangement;
- side-to-move Black ↔ White;
- komi `k -> -k` for the mathematical colour-swapped research state.

With this transformation, Black and White raw areas exchange and
`margin_black` negates exactly.

## Tests

Targeted Stage-1 tests cover canonical topology, groups/liberties, single and
multi-stone capture, simultaneous capture, Torus-wrap capture, pure suicide,
capture-before-suicide, positional superko (immediate and longer-history), pass
semantics, post-terminal rejection, graph-area/ownership/neutral scoring,
wrap-connected territory, exact komi, legacy-7.5 rejection, board-vs-full-state
identity, illegal-move immutability, deterministic replay, vertex permutation,
correct colour swap, tiny explicit graphs, dependency isolation, and the demo.

The repository already contains a pinned rectangular KataGo harness, but its fixture
contract is Japanese-V3-oriented (cleanup phases, seki tax, prisoner contribution,
and no-result cycle handling). Those semantics are not the Stage-1 graph-area-v1
contract, so no misleading rectangular differential is claimed here. The existing
harness remains available for its own Japanese-V3 verification line.

Stage 1 intentionally contains no Arena, MCTS, NN adapter, self-play, target
builder, trainer, checkpoint, batched inference, worker, Cube, product API,
fair-komi research, Benson, or Japanese-V3 compatibility implementation.
