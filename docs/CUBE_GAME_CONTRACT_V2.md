# Cube Game Contract V2

Status: Stage 1 game-semantics contract  
Base: `b08e63090cec3b07fa1cb18f78b6b04005dab186` (`origin/main` after PR #174)

## Scope

This contract defines the game played by the new Cube series before any new
network, replay format, trainer, production profile, or launch configuration is
implemented.

The family is `cube2` through `cube7`. Each size has separate future weights,
M0, replay, configuration, and checkpoints. The first future trainable line is
`cube4` from scratch. Joint multi-size training and Torus/Cube weight transfer
are outside this plan.

Historical Cube training profiles, network architecture, replay, checkpoints,
search settings, and the historical 1920-ply watchdog are not requirements of
this series. Existing verified Cube geometry and the shared Golden
`graph-area-v1` rules core may be reused where their semantics match this
contract.

Stage 1 does **not** integrate Cube into Orchestrator V2, self-play, training,
Arena, supervisor/runtime paths, notifications, or production defaults.

## Geometry family

For integer face size `n` in `{2,3,4,5,6,7}`:

- six faces each contain `n x n` distinct game points;
- `P = 6n^2`;
- points on different faces are never merged;
- a physical cube vertex is not an additional game point;
- the game graph is undirected, connected, loop-free, duplicate-free, and
  every point has exactly four game neighbors;
- edge relations are `SAME_FACE` and `CROSS_FACE_SEAM`; both have identical
  semantics for groups, liberties, captures, connectivity, and scoring;
- there are 12 physical seams, each joining `n` pairs, for `12n` undirected
  cross-face game edges;
- each of the 8 physical vertices is represented by three face-corner points;
  those three points form a triangle of game edges;
- future `CornerContext` is neural metadata and adds no game edges.

Point classes are:

- 24 face-corner points;
- `24(n-2)` edge points excluding corners;
- `6(n-2)^2` face-interior points.

For `cube2`, every point is a face-corner point.

| Profile | Points `P` | Policy actions including PASS |
|---|---:|---:|
| cube2 | 24 | 25 |
| cube3 | 54 | 55 |
| cube4 | 96 | 97 |
| cube5 | 150 | 151 |
| cube6 | 216 | 217 |
| cube7 | 294 | 295 |

Only the existing independently derived `cube4` topology is executable evidence
for geometry in Stage 1. The all-size generator, independent seam derivation for
every size, degenerate `cube2` normalization, and all 24 proper rotations for
`n=2..7` belong to Stage 2.

## Actions and PASS adapter

The canonical action space is integer `0..P`:

- points are `0..P-1`;
- PASS is integer action `P`;
- resign is not an action.

The existing Golden rules core internally uses the sentinel string `"PASS"`.
That is an implementation detail, not a new action-space definition. Stage 1
provides an isolated one-to-one adapter `P <-> "PASS"` and does not change the
shared rules core.

`P+1` is the number of choices at one position. It is not a game-length limit.

## Rules

1. The initial board is empty. Black moves first and colors alternate.
2. A point move first places the stone, then removes adjacent opponent groups
   with no liberties, then checks the moving player's resulting group for
   suicide.
3. Occupied-point moves and suicide are illegal. A move that gains liberties by
   capturing is legal even if the placement would have had no liberty before
   the capture was removed.
4. Liberties are the set of distinct empty neighbors of the whole connected
   group, including neighbors across seams. A shared liberty counts once.
5. A captured point becomes an ordinary empty point and may be played again
   later whenever the resulting move is legal.
6. Ko is positional superko. A point move is illegal when its resulting stone
   arrangement equals any earlier stone arrangement in the game. Side to move
   is not part of the repetition key. The initial empty arrangement is included.
7. PASS is legal in every nonterminal state. It is exempt from repetition,
   does not append a board to superko history, changes side to move, and
   increments consecutive PASS count. A point move resets that count.
8. Two consecutive PASS actions produce the only formal terminal condition in
   this contract. No move is legal after terminal, and terminal states are not
   sent to a network for move selection.
9. Scoring is exact graph area: stones plus empty connected components whose
   colored boundary contains exactly one color. A component touching both
   colors, or no colored boundary, is neutral.
10. Prisoners score no separate points. There is no automatic dead-stone
    removal, cleanup phase, life detection, or life-based termination.
11. Initial komi is explicitly `0.5` for every Stage-1 family entry. It is a
    baseline, not a claim of balance at every size.
12. Black margin is `black_area - white_area - komi`. Positive is a black win,
    negative is a white win, and zero is a draw.

With integer area and komi `0.5`, a formal draw cannot occur under this baseline,
but DRAW remains part of the WDL schema.

## State and histories

A complete game state includes:

- stones;
- side to move;
- positional-superko history;
- consecutive PASS count;
- topology identity;
- rules identity;
- komi.

Future neural observation history is the history of real moves/positions needed
by the observation schema. It is separate from positional-superko history.
Superko history omits PASS and cannot substitute for neural move history.

## Final results and future targets

Stage 1 defines meanings only; it does not implement replay or losses.

**Policy** is the root MCTS visit distribution over `P+1` canonical actions,
recorded before the chosen move. Illegal actions have target probability zero.

**WDL** is `[WIN, DRAW, LOSS]` from the perspective of the player to move in the
specific saved training position.

**Ownership** is `[OWN, OPPONENT, NEUTRAL]` per point from that same saved
position's perspective. It is the actual final graph-area ownership, not a
life-status estimate.

**Score** is the exact final komi-adjusted margin from that same saved
position's perspective. Loss normalization is a later target-schema decision.

The perspective is **not** taken from `side_to_move` of the final terminal
state. The absolute BLACK/WHITE final result is computed once and projected
separately for every saved training position.

Captures can make a game longer than `P`; action count does not constrain move
count.

## Technical termination

A watchdog is an execution safeguard, not a rule of victory. Stage 1 deliberately
chooses no numeric watchdog. The historical Cube value `1920` is not inherited.

`MOVE_LIMIT`, `TIMEOUT`, and `WORKER_ERROR` are technical/invalid completions.
They are not converted into WDL, score, or ownership, and later must be excluded
from training replay and formal win statistics.

Stage 1 implements only pure result classification. It does not place a
watchdog in the rules core or modify self-play/generation runners.

If the last permitted action itself creates the formal second consecutive PASS,
`DOUBLE_PASS` is the formal result even when the execution boundary is reached
on that same action. Technical classification applies only when there is no
formal terminal result.

## Identity and fingerprints

`cube_game_contract_v2.json` has a canonical SHA-256 family-contract
fingerprint. That fingerprint proves the family description; it is **not** a
concrete board identity.

A concrete game identity includes at least:

- game family;
- face size;
- point count and action count;
- topology ID and topology fingerprint;
- rules fingerprint;
- komi;
- family-contract fingerprint.

Stage 1 can instantiate this identity for the verified existing `cube4`.
Complete generated point mappings and adjacency fingerprints for `cube2..cube7`
are Stage 2 work.

Human-readable names are not compatibility checks. Historical Cube checkpoint
or architecture identities must not be accepted as identities of the future
network merely because they use the same board size.

The shared rules ID `graph-area-v1` remains valid where semantics are unchanged.
This contract does not rename or mutate existing Golden/Torus rules
fingerprints.

## Correspondence to the existing rules core

| Contract requirement | Existing implementation at the Stage-1 base | Stage-1 treatment |
|---|---|---|
| Empty board, black first, initial board in superko history | `gocube_golden.state.initial_state` / historical Cube initial-state wrapper | Reuse semantics |
| Place, capture opponent, then suicide check | `gocube_golden.rules._probe_point` | Reuse semantics |
| Unique group liberties across topology edges | `group_from_board`, `liberties_from_board` | Reuse semantics |
| Positional superko by stone arrangement | `board_key`, `superko_membership`, `_probe_point` | Reuse semantics |
| PASS does not append superko board | `apply_action` | Reuse semantics |
| Two PASS terminal / no post-terminal move | `GoldenState.is_terminal`, `apply_action` | Reuse semantics |
| Canonical integer PASS action `P` | Existing topology/action masks use last action while rules core uses `"PASS"` | New isolated adapter only |
| Exact graph-area / no dead-stone cleanup | `gocube_golden.scoring.score_terminal` | Reuse semantics |
| Winner from black margin | `gocube_golden.result.result_from_terminal` | Reuse semantics |
| Topology+komi participate in rules identity | `gocube_golden.state.rules_fingerprint_for` for non-frozen research identities | Preserve existing identities |
| Double-pass priority over historical Cube watchdog boundary | Historical `cube_post_action_termination` already checks formal terminal first | Contract only; historical numeric limit not inherited |
| Target perspective per saved position | Not a complete Stage-1 family abstraction in historical Cube profile | Pure projection helpers added |
| Technical result is not a game target | Historical Cube training code excludes technical games | Contract/pure classifier only; no runtime integration |

Private implementation functions in this table are references for semantic
correspondence, not new API commitments.

## Stage-1 verification

`tests/test_cube_game_contract_v2.py` verifies:

- strict integer size validation (`bool`, fractions, and out-of-range rejected);
- point/action counts and PASS mapping;
- existing `cube4` degree, seams, and physical-corner triangles;
- seam capture/group/liberty semantics, capture-before-suicide behavior, and
  reusability of captured points;
- positional superko and PASS history behavior;
- graph-area territory, mixed boundary, empty board, and no implicit dead-stone
  removal;
- WDL/ownership/score perspective projection;
- technical completion classification and double-PASS boundary priority;
- canonical contract fingerprint and rejection of semantic drift;
- preservation of frozen Torus and historical Cube identity values.

The tests do not claim verified geometry for `cube2`, `cube3`, `cube5`,
`cube6`, or `cube7`.

## Deferred work

Stage 2: generate and independently verify `cube2..cube7`, distances,
`cube2` degeneracies, seam mappings, and all 24 rotations.

Later stages: observations/features and the new network; target/replay/trainer;
integration into the shared Orchestrator V2 and Arena; full trial cycle;
target-hardware measurements; clean `cube4` M0.

`128 x 12` with global context remains an architecture candidate only. This
contract fixes no hidden width/block count, LR, batch size, MCTS simulations,
replay window, worker count, or historical Cube training default.
