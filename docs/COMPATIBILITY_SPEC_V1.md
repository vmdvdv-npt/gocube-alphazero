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

The AlphaZero framework source of truth is `alphazero/Game.py` plus the existing MCTS/self-play code that consumes that class.

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
8. the contract that a self-play terminal adjudicator must satisfy;
9. the mapping to the existing AlphaZero `GameState` lifecycle.

V1 intentionally does **not** fix the neural observation-plane layout, network architecture, symmetry augmentation, resign action, superko, handicap, or time controls. Those may be specified separately without changing the game semantics fixed here.

## 3. Environment identity

A training environment is bound to one immutable configuration:

```text
topology_kind ∈ {torus, cube}
size          = N
ruleset       ∈ {chinese, japanese}
komi          = finite number
terminal_adjudicator_id = versioned self-play adjudicator implementation
```

`terminal_adjudicator_id` is training metadata, not a field in production GoCube `GameState`. It is required because terminal classification affects reward semantics. Changing the terminal adjudicator implementation or its decision rules is a training-semantics change and MUST produce a new identifier.

This binding is important because `alphazero.Game.GameState.action_size()` and `observation_size()` are static methods. A single runtime `Game` class MUST NOT mix board sizes or topology kinds whose action spaces differ. Separate bound classes/configurations or separate training runs MAY be used.

Current production size constraints at the source anchor are:

- Torus: `N ∈ {9, 13, 19}`.
- Cube topology: any safe integer `N >= 2`; the product UI currently exposes `N ∈ {2, 3, 4, 5, 6, 7}`.

## 4. Semantic state contract

The minimum semantic game state required by the training engine is:

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
  --> AssistedEndgameClassifier proposal
  --> alive / dead / seki / unresolved review state
  --> user may resolve/override review statuses
  --> complete classification
  --> ruleset scoring
  --> finished
```

Both current Torus and Cube production controllers instantiate `AssistedEndgameClassifier` and keep proposal statuses editable until endgame review is explicitly completed.

The state immediately after the second pass therefore represents **end of normal play, pending adjudication**. It MUST NOT be treated as a normal already-scored terminal state.

For training, the implementation MAY collapse the pending-adjudication lifecycle into the second-pass `play_action()` call for compatibility with the synchronous AlphaZero loop, but it MUST preserve the terminal-adjudication contract in section 11.

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

This is a separate contract because production GoCube permits `unresolved` groups and hands them to human endgame review, while AlphaZero MCTS has no human actor and requires a terminal value for every terminal node it reaches.

### 11.1 Why the self-play adjudicator must be total

In the current framework, MCTS reaches a leaf by calling `play_action()`, then uses `win_state()` to decide whether the node is terminal and to backpropagate its result. `win_state()` can represent only Black win, White win, draw, or “not terminal yet”.

Therefore an `unresolved` two-pass position cannot be represented safely as an ordinary MCTS terminal node:

- mapping it to draw fabricates a reward;
- leaving `win_state()` all-zero while exposing no legal moves creates an invalid leaf;
- discarding only the final played episode is insufficient because MCTS can encounter the same two-pass state inside search simulations before the real game reaches it.

**Normative consequence:** the self-play terminal adjudicator MUST be **total for every two-pass state reachable by MCTS in an enabled training environment**. A normal self-play run may not expose `unresolved` as a terminal outcome to MCTS.

### 11.2 Input

The adjudicator receives the immutable state immediately after the second consecutive pass plus the bound environment configuration:

```text
board
current_player
captures
topology_kind
size
ruleset
komi
terminal_adjudicator_id
```

Classification MUST NOT mutate the played board or normal-play capture counters.

### 11.3 Stage A — production-parity proposal

The adjudicator MUST first reproduce semantics equivalent to GoCube `AssistedEndgameClassifier` at the source anchor:

1. enumerate all complete logical stone groups using the topology graph;
2. produce exactly one proposal per complete group;
3. automatically prove only statuses supported by the conservative GoCube automatic proof logic;
4. preserve explicit `unresolved` for groups that the production automatic proof does not resolve.

A Python implementation does not have to call TypeScript at runtime, but Stage A MUST be behaviourally equivalent on the compatibility fixture suite. Replacing the production classifier with a different heuristic and calling it Stage-A-compatible is not permitted.

### 11.4 Stage B — total self-play resolver

If Stage A contains any `unresolved` group, self-play requires a second, training-specific resolver.

The Stage B resolver MUST:

- receive the complete Stage-A proposal and the same immutable end-of-play position;
- leave every Stage-A status already proven `alive`, `dead`, or `seki` unchanged;
- assign `alive`, `dead`, or `seki` to every remaining unresolved group;
- operate on the logical topology graph, never on planar-edge assumptions or renderer coordinates;
- be deterministic for a fixed position, configuration, and `terminal_adjudicator_id`;
- return evidence/diagnostic data sufficient to reproduce or audit the decision;
- never use a silent fallback such as “unresolved = alive”, “unresolved = seki”, or “unresolved = draw” unless such a rule is explicitly adopted as a separately reviewed adjudicator version.

GoCube production currently does **not** define this automatic Stage-B decision: production uses human review for the remaining ambiguity. Consequently Stage B is a declared self-play policy, not something that can be inferred from the existing production engine.

The exact Stage-B algorithm MUST be selected, versioned, implemented, and covered by fixtures before full self-play is enabled. Until then, the environment is implementation-incomplete even though the GoCube compatibility boundary is defined by this document.

### 11.5 Successful result

Normal self-play terminal adjudication returns only a complete resolved result:

```text
RESOLVED {
  classification: every logical stone group -> alive | dead | seki,
  score: FinalScore,
  winner: black | white | draw,
  terminal_adjudicator_id,
  evidence
}
```

Then:

1. score the position with section 10 and the bound `ruleset`/`komi`;
2. expose the winner through AlphaZero `win_state()` as:

```text
black win -> [1, 0, 0]
white win -> [0, 1, 0]
draw      -> [0, 0, 1]
```

for the two-player framework value vector.

Every two-pass node that is allowed to behave as terminal inside MCTS MUST reach this resolved result.

### 11.6 Adjudication failure is a hard capability failure, not a game result

An implementation MAY detect an exceptional condition in which its configured Stage-B resolver cannot produce a complete classification. That condition is:

```text
ADJUDICATION_FAILURE
```

It is **not** a fourth game result and MUST NOT be surfaced to MCTS as draw/non-terminal/zero-value.

On such a failure:

- no score may be produced;
- no win/loss/draw target may be produced;
- no samples from the affected search/game may enter the training dataset;
- the worker/run MUST abort or fail closed before continuing MCTS with that node;
- the failure MUST be visible in telemetry/logging.

The system SHOULD include a pre-training conformance gate that exercises terminal fixtures so adjudication failure is discovered before expensive self-play begins.

### 11.7 Compatibility claim boundary

Normal-play topology, move legality, ko, captures, passes, and scoring in this specification are direct GoCube parity requirements.

Stage A is also direct production parity. Stage B is necessarily a self-play extension until GoCube itself has a total automatic endgame resolver. Therefore training artifacts SHOULD identify their rules semantics as, for example:

```text
GoCube Compatibility V1 + terminal_adjudicator_id=<id>
```

A training run MUST NOT claim bit-for-bit equivalence with the production human-review endgame unless its Stage-B outcomes have an independent basis for that stronger claim.

This explicit boundary is preferable to hiding an arbitrary dead/alive convention inside scoring.

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
| `win_state()` | all-zero while normal play continues; resolved terminal vector only after successful total adjudication |
| `clone()` | independent clone preserving all semantic state |
| `__eq__` / MCTS state identity | must distinguish states whenever future legality or reward can differ |
| `observation()` | encoding is outside V1, but must describe the bound game state consistently |

`valid_moves()` for a playing state MUST always include `PASS`. Illegal stone placements are masked out for occupancy, suicide, and simple ko.

No action may be presented to MCTS after a resolved terminal result. An adjudication failure is a hard integration error, not a state with zero valid moves and an all-zero `win_state()`.

## 13. Observation and symmetry constraints for Stage 2

Although the exact neural planes are not fixed here, two constraints follow from this semantic contract:

1. observation design MUST account for state information needed by policy/value learning, including side to move and immediate-ko context;
2. symmetry augmentation MUST remain disabled until every proposed Torus/Cube transformation is proven to preserve both topology and ActionSpace V1 indexing under an explicit action permutation.

A renderer transform is not automatically a legal training symmetry.

## 14. Conformance requirements

The training environment is not “GoCube-compatible V1” until automated differential/golden tests establish the following.

### 14.1 Point/action mapping

For every supported training size:

- `points` is element-for-element equivalent to GoCube `Topology.points()`;
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

- Stage-A fully resolved endgame -> exact production-equivalent classification and score;
- Stage-A unresolved groups -> Stage B fills only unresolved statuses and returns a complete classification;
- mixed automatic statuses plus unresolved groups -> proven Stage-A statuses remain unchanged;
- repeated adjudication of the same state/config/id -> identical result;
- topology-seam positions -> no planar-edge assumptions;
- deliberately injected resolver failure -> hard abort/no samples, never draw;
- MCTS integration -> a second-pass leaf never becomes “all-zero `win_state()` + zero valid moves”.

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
11. Production `unresolved` is preserved by Stage A and is never silently coerced inside scoring.
12. Self-play requires a versioned **total** Stage-B resolver before a two-pass state may be terminal in MCTS.
13. Chinese/Japanese scoring and capture accounting match GoCube exactly.

## 16. Stage 2 implementation boundary

With this specification fixed, Stage 2 may implement the Go environment inside `gocube-alphazero` without inventing ordinary game semantics. The next implementation work is bounded to:

- topology tables/mapping matching this V1 contract;
- game-state transition logic;
- `GameState` adapter methods required by the existing framework;
- Stage-A `AssistedEndgameClassifier` parity;
- selection and implementation of a versioned total Stage-B self-play resolver;
- exact Chinese/Japanese scoring parity;
- MCTS/self-play hard-failure handling for adjudication capability failures;
- cross-engine conformance fixtures/tests;
- a separately reviewed neural observation encoding.

Full self-play MUST NOT be enabled until a configured terminal adjudicator is total on the conformance suite and MCTS never sees an unresolved two-pass node. Until all conformance tests pass, generated self-play data MUST NOT be described as semantically equivalent to production GoCube.