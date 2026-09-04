# GoCube ↔ AlphaZero Compatibility Specification V1

**Status:** normative implementation contract for the first GoCube training environment.

This document fixes the semantic boundary between the production GoCube engine and the `gocube-alphazero` training engine. It does **not** introduce a second generic game API: the training implementation remains an implementation of the existing `alphazero.Game.GameState` contract.

Normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are used in the RFC 2119 sense.

## 1. Source anchors

V1 is defined against these exact repository states:

- GoCube: `vmdvdv-npt/GoCube@764ace0302426cde1ee6c811a54c808d28c36b22`
- AlphaZero fork: `vmdvdv-npt/gocube-alphazero@7135b4d9cd7f0e5482d0c855c7744b2047767ed6`

The GoCube source of truth for this contract is:

- `src/core/topology/Topology.ts`
- `src/core/topology/TorusTopology.ts`
- `src/core/topology/CubeTopology.ts`
- `src/core/game/types.ts`
- `src/core/game/GameEngine.ts`
- `src/core/history/LinearHistory.ts`
- `src/core/rules/SimpleKoPolicy.ts`
- `src/core/endgame/EndgameClassifier.ts`
- `src/core/endgame/AssistedEndgameClassifier.ts`
- `src/core/endgame/ManualEndgameClassifier.ts`
- `src/core/game/GameSession.ts`
- `src/core/scoring/Scoring.ts`
- `src/core/scoring/ChineseScoring.ts`
- `src/core/scoring/JapaneseScoring.ts`
- `src/app/TorusGameController.ts`
- `src/app/Cube2DGameController.ts`

The AlphaZero framework source of truth is `alphazero/Game.py` and the existing MCTS/self-play code that consumes that class.

Any future GoCube change that alters a rule, topology, point order, scoring rule, or terminal-adjudication meaning MUST trigger a compatibility review. A deliberately incompatible change requires a new specification version rather than silently changing V1.

## 2. Scope and non-goals

V1 fixes:

1. semantic game state;
2. point identity and canonical ordering;
3. neural action indexing;
4. topology/neighbour semantics;
5. move legality, capture, suicide, simple ko, pass, and player switching;
6. two-pass transition semantics;
7. Chinese and Japanese scoring semantics;
8. the automatic terminal-adjudication contract for self-play;
9. the mapping to the existing AlphaZero `GameState` lifecycle.

V1 intentionally does **not** fix the neural observation-plane layout, network architecture, symmetry augmentation, resign action, superko, handicap, or time controls. Those may be specified separately without changing the game semantics fixed here.

## 3. Environment identity

A training environment is bound to one immutable configuration:

```text
topology_kind ∈ {torus, cube}
size          = N
ruleset       ∈ {chinese, japanese}
komi          = finite number
```

This binding is important because `alphazero.Game.GameState.action_size()` and `observation_size()` are static methods. A single runtime `Game` class MUST NOT mix board sizes or topology kinds whose action spaces differ. Separate bound classes/configurations or separate training runs MAY be used.

Current production size constraints at the source anchor are:

- Torus: `N ∈ {9, 13, 19}`.
- Cube topology: any safe integer `N >= 2`; the product UI currently exposes `N ∈ {2, 3, 4, 5, 6, 7}`.

## 4. Semantic state contract

The minimum semantic state required by the training engine is:

```text
topology_kind
size
board
current_player
previous_board (or an exact-equivalent previous-position key)
consecutive_passes
captures
ruleset
komi
```

### 4.1 Field semantics

`topology_kind`
: `torus` or `cube`. It selects the exact graph defined in section 7.

`size`
: face/torus side length `N`.

`board`
: one occupancy value for every canonical point: `empty`, `black`, or `white`. A dense internal representation is encouraged, but its element at point index `i` MUST mean the same point as `Topology.points()[i]` in GoCube.

`current_player`
: the player to act next. Initial player is Black.

`previous_board`
: board belonging to the immediately previous **accepted action snapshot**, not the previous distinct board. This distinction is normative for simple ko. Because a pass creates a new accepted snapshot with an unchanged board, a pass breaks immediate-ko repetition exactly as it does in GoCube.

`consecutive_passes`
: number of immediately preceding accepted pass actions since the last accepted stone placement. A stone placement resets it to zero.

`captures`
: two counters. `captures.black` is the number of White stones captured by Black during normal play; `captures.white` is the number of Black stones captured by White.

`ruleset`
: `chinese` or `japanese`. In V1 this affects scoring, not ordinary move legality.

`komi`
: finite numeric komi added to White's score.

### 4.2 Previous-position hashes

An implementation MAY cache a position hash instead of repeatedly comparing full boards, but the semantic rule is exact board equality. A hash optimization MUST NOT create observable false-ko or missed-ko behaviour. A collision-safe fallback to exact comparison is the reference-compatible implementation.

Full history is **not** part of the V1 legality state because GoCube implements simple ko rather than positional or situational superko.

### 4.3 Framework bookkeeping

The AlphaZero `_turns` counter MUST equal the number of accepted game actions, including pass, corresponding to GoCube `moveNumber`.

The AlphaZero player mapping is fixed as:

```text
_player == 0  <=>  black
_player == 1  <=>  white
```

`clone()` and equality/state keys used by MCTS MUST preserve every semantic component that can affect future legality or terminal reward. Immutable environment configuration may be stored at class/environment level instead of duplicated in every instance, but it remains part of the semantic identity of the state.

## 5. Point identity contract

`PointId` remains the application-boundary identity. The training engine uses a dense integer point index.

For an environment, construct once:

```text
points = exact GoCube Topology.points() sequence
point_index[PointId] = position in points
point_id[index]       = points[index]
```

The mapping MUST be a bijection.

For every board point:

```text
PointId ↔ integer point index ↔ neural action index
```

For stone-placement actions, `point index == action index` in V1. `PASS` is the single non-point action.

Renderer coordinates, duplicated visual torus regions, cube orientation in the UI, and presentation aliases MUST NOT create additional logical points or actions.

## 6. ActionSpace V1

There is no resign action in V1.

### 6.1 Torus

Canonical GoCube point IDs are `"x,y"` and `Topology.points()` is row-major in `y`, then `x`:

```text
index(x, y) = y * N + x
0 <= x,y < N
```

Action space:

```text
0 ... N²-1   stone placement at points[action]
N²           PASS
```

Therefore:

```text
action_size = N² + 1
PASS_ACTION = N²
```

### 6.2 Cube

Canonical face order is fixed forever for ActionSpace V1:

```text
0 front
1 back
2 left
3 right
4 top
5 bottom
```

Within each face, points are row-major:

```text
index(face, row, column)
  = face_ordinal * N² + row * N + column
```

Canonical PointId is `"<face>:<row>:<column>"`.

Action space:

```text
0 ... 6N²-1  stone placement at points[action]
6N²          PASS
```

Therefore:

```text
action_size = 6N² + 1
PASS_ACTION = 6N²
```

Changing face order, row/column meaning, or pass index is a breaking compatibility change and requires a new ActionSpace version and new model checkpoints.

## 7. Topology and neighbour semantics

Rules MUST operate only through the logical topology graph. No rule implementation may infer planar edges, corners, off-board liberties, or renderer geometry.

### 7.1 Torus

For point `(x,y)`, neighbour order at the source anchor is:

```text
(x-1 mod N, y)
(x+1 mod N, y)
(x, y-1 mod N)
(x, y+1 mod N)
```

This corresponds to left, right, up, down. Every edge wraps.

### 7.2 Cube

Within a face, neighbour direction order is:

```text
top, right, bottom, left
```

Cross-face transitions MUST match `CubeTopology.ts` exactly:

| From | top | right | bottom | left |
| --- | --- | --- | --- | --- |
| front | top.bottom | right.left | bottom.top | left.right |
| back | top.top reversed | left.left | bottom.bottom reversed | right.right |
| left | top.left | front.left | bottom.left reversed | back.right |
| right | top.right reversed | back.left | bottom.right | front.right |
| top | back.top reversed | right.top reversed | front.top | left.top |
| bottom | front.bottom | right.bottom | back.bottom reversed | left.bottom reversed |

`reversed` means that the edge coordinate is mapped to `N - 1 - index`.

Neighbour order is observable and SHOULD match GoCube exactly; group/liberty semantics use the resulting neighbour set.

## 8. Move and rule semantics

### 8.1 Group

A stone group is the maximal connected component of same-colour stones under `Topology.neighbors()`.

### 8.2 Liberties

The liberties of a group are the **unique** empty logical points adjacent to at least one stone in the group.

### 8.3 Stone placement

For a requested placement by `current_player` at point `p`, the reference transition order is:

1. reject if the game is not in playing phase;
2. reject if `p` is not empty;
3. tentatively place the current player's stone at `p`;
4. inspect adjacent opponent groups;
5. capture every such opponent group whose liberties are now zero;
6. remove all captured stones;
7. recompute the placed stone's own group;
8. reject as suicide if that group has zero liberties after captures;
9. construct the candidate board/state and update capture counters;
10. reject as simple-ko repetition if the candidate board exactly equals `previous_board`;
11. accept the move, switch player, increment turn/move number, and set `consecutive_passes = 0`.

A capture that creates liberties for the newly placed stone is legal; suicide is tested only after opponent captures are removed.

Any rejected move MUST leave board, player, counters, previous-board context, and pass count unchanged.

### 8.4 Simple ko

V1 implements **simple ko only**:

```text
candidate_board is illegal iff candidate_board == previous_board
```

The comparison is board occupancy only. No older position in history is consulted.

Because `previous_board` is the board from the immediately previous accepted action snapshot, a pass creates a repeated snapshot and therefore ends the immediate-ko prohibition in the same way as GoCube `LinearHistory.simpleKoContext()`.

### 8.5 Pass

`PASS` is legal whenever the state is in playing phase.

An accepted pass:

- leaves board unchanged;
- leaves captures unchanged;
- switches the current player;
- increments turn/move number;
- increments `consecutive_passes` by one.

The first consecutive pass leaves the game in playing phase. The second consecutive pass ends normal play and triggers terminal adjudication. Any accepted stone placement between passes resets `consecutive_passes` to zero.

### 8.6 Player switching

Every accepted stone placement and every accepted pass switches to the opponent. Rejected actions do not switch player.

## 9. Two-pass lifecycle semantics

Two passes are **not equivalent to a scored terminal result**.

Production GoCube semantics are:

```text
playing
  -- PASS, PASS -->
endgame
  --> EndgameClassifier proposal
  --> alive / dead / seki / unresolved review state
  --> complete classification
  --> ruleset scoring
  --> finished
```

The state immediately after the second pass therefore represents **end of normal play, pending adjudication**. It MUST NOT be treated as a normal already-scored terminal state.

For training, the implementation MAY collapse the pending-adjudication lifecycle into the second-pass `play_action()` call for compatibility with the synchronous AlphaZero loop, but it MUST preserve the adjudication semantics in section 11.

## 10. Scoring contract

Scoring consumes a complete classification of every logical stone group as `alive`, `dead`, or `seki`. `unresolved` is not a scoring status.

### 10.1 Effective scoring position

Before territory is computed:

- stones classified `dead` are **virtually removed** from the scoring board;
- dead stones remain attributable to their original colour for prisoner/dead-stone accounting;
- `seki` stones remain on the board;
- an empty connected region touching a classified seki group is counted as seki-neutral territory;
- otherwise an empty connected region bounded by exactly one colour belongs to that colour;
- a region touching both colours, or no owning colour, is neutral.

The played board itself MUST NOT be destructively rewritten merely to score it.

### 10.2 Chinese scoring

```text
black_score = black_stones_remaining + black_territory
white_score = white_stones_remaining + white_territory + komi
```

Normal-play capture counters do not directly add points under Chinese scoring.

### 10.3 Japanese scoring

Prisoners are:

```text
black_prisoners = captures.black + dead_white_stones
white_prisoners = captures.white + dead_black_stones
```

Scores are:

```text
black_score = black_territory + black_prisoners
white_score = white_territory + white_prisoners + komi
```

### 10.4 Winner and margin

```text
black_score > white_score  => black wins
white_score > black_score  => white wins
black_score == white_score => draw
margin = abs(black_score - white_score)
```

## 11. Automatic terminal adjudication contract for self-play

This is a separate contract because production GoCube may hand unresolved groups to a human endgame review, while self-play has no human actor.

### 11.1 Input

The automatic adjudicator receives the immutable state immediately after the second consecutive pass plus the bound environment configuration:

```text
board
current_player
captures
topology_kind
size
ruleset
komi
```

The normal-play state MUST NOT be modified by classification.

### 11.2 Required classifier semantics

The V1 automatic adjudicator MUST use semantics equivalent to the GoCube `AssistedEndgameClassifier` at the source anchor:

1. enumerate all complete logical stone groups using the topology graph;
2. produce exactly one proposal per complete group;
3. automatically prove only statuses supported by the conservative GoCube automatic proof logic;
4. permit the explicit result `unresolved` for any group that is not proven `alive`, `dead`, or `seki`.

A Python implementation does not have to call TypeScript at runtime, but it MUST be behaviourally equivalent on the compatibility fixture suite. Re-implementing the classifier with a different heuristic and calling the result “compatible” is not permitted.

### 11.3 Result type

The semantic result is:

```text
RESOLVED {
  classification: every group -> alive | dead | seki,
  score: FinalScore,
  winner: black | white | draw
}

or

UNRESOLVED {
  proposal: every group -> alive | dead | seki | unresolved,
  unresolved_groups: non-empty set
}
```

### 11.4 Resolved path

If every group is automatically resolved:

1. convert the proposal to a complete automatic classification;
2. score with the bound `ruleset` and `komi` using section 10;
3. expose the winner through AlphaZero `win_state()` as:

```text
black win -> [1, 0, 0]
white win -> [0, 1, 0]
draw      -> [0, 0, 1]
```

for the two-player framework value vector.

### 11.5 Unresolved path — normative safety rule

If at least one group is `unresolved`:

- the adjudicator MUST NOT call normal scoring;
- `unresolved` MUST NOT be silently converted to `alive`, `dead`, or `seki`;
- the position MUST NOT be labelled a draw;
- no win/loss/draw value target may be emitted for the episode;
- all training samples accumulated from that episode MUST be discarded.

This is the V1 fail-closed rule. It preserves semantic correctness instead of injecting fabricated terminal labels.

The training orchestration MUST therefore provide an explicit **unresolved-episode abort/reset path** outside the existing `win_state()` value representation. The current upstream-style `GameState.win_state()` cannot encode “normal play is over but terminal reward is unknown”, and `SelfPlayAgent` currently assumes every completed episode has a win/draw target. The integration MUST handle this condition before samples are enqueued.

An unresolved episode SHOULD be counted in dedicated telemetry such as `terminal_adjudication_unresolved`, but MUST NOT be counted as a successfully scored training game. Implementations SHOULD include a run-level guard/threshold so a high unresolved rate fails visibly instead of causing an endless attempt loop.

A future stronger **sound** automatic adjudicator may reduce the unresolved rate. Changing the fail-closed meaning of `unresolved` requires a compatibility review.

## 12. Mapping to `alphazero.Game.GameState`

The existing framework API remains authoritative. The Go environment maps onto it as follows:

| AlphaZero API | GoCube-compatible meaning |
| --- | --- |
| `num_players()` | `2` |
| `action_size()` | point count + one `PASS` action, section 6 |
| `valid_moves()` | binary vector in ActionSpace V1 order; legal stone placements plus `PASS` while playing |
| `play_action(a)` | apply the exact transition in section 8; second pass invokes/enters adjudication |
| `player` / `_player` | `0 = black`, `1 = white` |
| `turns` / `_turns` | accepted action count / GoCube `moveNumber` |
| `win_state()` | all-zero while normal play continues; resolved terminal vector only after successful adjudication |
| `clone()` | independent clone preserving all semantic state |
| `__eq__` / MCTS state identity | must distinguish states whenever future legality or reward can differ |
| `observation()` | encoding is outside V1, but must describe the bound game state consistently |

`valid_moves()` for a playing state MUST always include `PASS`. Illegal stone placements are masked out for occupancy, suicide, and simple ko.

No action may be presented to MCTS after a resolved terminal result. An unresolved terminal-adjudication outcome is intercepted by the self-play orchestration as specified above rather than exposed as a fake draw or a new neural action.

## 13. Observation and symmetry constraints for Stage 2

Although the exact neural planes are not fixed here, two constraints follow from this semantic contract:

1. observation design MUST account for state information needed by policy/value learning, including side to move and immediate-ko context;
2. symmetry augmentation MUST remain disabled until every proposed Torus/Cube transformation is proven to preserve both topology and ActionSpace V1 indexing under an explicit action permutation.

A renderer transform is not automatically a legal training symmetry.

## 14. Conformance requirements

The training environment is not “GoCube-compatible V1” until automated differential/golden tests establish the following.

### 14.1 Point/action mapping

For every supported training size:

- `points` is byte-for-byte/element-for-element equivalent to GoCube `Topology.points()`;
- PointId -> point index -> PointId round-trips;
- point action indices equal point indices;
- `PASS` is exactly the final action;
- action vector length is exact.

### 14.2 Topology

For every logical point:

- Torus neighbours equal GoCube neighbours;
- Cube neighbours equal GoCube neighbours, including every edge/corner transition and reversal.

### 14.3 Rule transitions

Golden fixtures MUST cover at least:

- ordinary single-stone capture;
- multi-stone capture;
- capture crossing a Torus wrap;
- capture crossing every Cube seam orientation;
- suicide rejection;
- capture-that-prevents-suicide;
- simple-ko rejection;
- legal recapture after an intervening pass/action consistent with GoCube simple-ko history semantics;
- rejected move leaves state unchanged;
- stone move resets consecutive passes;
- first pass switches player and remains playing;
- second pass switches player and triggers adjudication;
- capture counters match exactly.

### 14.4 Scoring

Golden fixtures MUST compare full score semantics for both rulesets, including:

- dead-stone virtual removal;
- seki-neutral regions;
- ordinary neutral/dame regions;
- Chinese stones + territory;
- Japanese captures + dead-stone prisoners + territory;
- komi;
- draw and margin.

### 14.5 Terminal adjudication

Fixtures MUST include:

- fully automatically resolved endgame -> exact final score and `win_state()`;
- at least one unresolved group -> `UNRESOLVED`, no score, no value target;
- mixed automatic statuses plus unresolved groups -> still `UNRESOLVED`;
- proof behaviour on topology seams so classification does not accidentally use planar-edge assumptions.

## 15. Compatibility invariants

The following are V1 invariants and should be treated as checkpoint/data compatibility boundaries:

1. `Topology.points()` order defines point identity order.
2. Stone action index equals canonical point index.
3. `PASS` is the final action.
4. Cube face order is `front, back, left, right, top, bottom`.
5. Rules use the topology graph; there are no physical board edges.
6. Suicide is forbidden after opponent captures are applied.
7. Ko is simple immediate board repetition only.
8. Pass participates in accepted-action history and breaks immediate ko in the same way as GoCube.
9. Two consecutive passes end normal play but do not by themselves produce reward.
10. Scoring requires a complete `alive/dead/seki` classification.
11. `unresolved` is a first-class adjudication failure and never a draw/default-alive shortcut.
12. Chinese/Japanese scoring and capture accounting match GoCube exactly.

## 16. Stage 2 implementation boundary

With this specification fixed, Stage 2 may implement the Go environment inside `gocube-alphazero` without inventing game semantics. The next implementation work should be limited to:

- topology tables/mapping matching this V1 contract;
- game-state transition logic;
- `GameState` adapter methods required by the existing framework;
- automatic endgame classifier/scoring parity or an equivalent verified bridge;
- unresolved-episode orchestration support;
- cross-engine conformance fixtures/tests;
- a separately reviewed neural observation encoding.

Until those conformance tests pass, generated self-play data MUST NOT be described as semantically equivalent to production GoCube.