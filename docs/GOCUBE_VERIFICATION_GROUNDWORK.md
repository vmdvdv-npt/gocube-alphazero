# GoCube verification groundwork

This change set prepares an independent verification layer. It does not
change production rules, scoring, observation channels, model contracts,
search budgets, cleanup semantics, or the TypeScript product.

## Method

The verification stack has three deliberately separate inputs/results:

| Layer | Authoritative input | Independently derived result |
| --- | --- | --- |
| rectangular differential | pinned native KataGo and rectangular topology | upstream rule compatibility |
| Cube graph | canonical `PointId` adjacency only | groups, unique liberties, captures, regions, graph triangles |
| product boundary | canonical fixture and MAIN action sequence | captures and board snapshots before/after the second PASS |

`tests/support/independent_graph.py` never imports a production group,
liberty, capture, scoring, or ko helper. It accepts only an occupancy vector
and an adjacency list. Production topology may supply that raw adjacency, but
the derived graph result is calculated independently.

## Existing inventory

Before adding this layer, the repository already had:

* canonical Cube/Torus topology in `alphazero/envs/gocube/core.py`;
* production group, empty-component, pass-alive, and V3 capture logic in
  `alphazero/envs/gocube/katago_v3.py`;
* the rectangular test bridge in `tests/gocube_reference_topology.py`;
* native KataGo fixtures and runner under `tests/reference/katago/` and
  `tests/katago_reference_runner.py`;
* canonical face/row/column `PointId` conversion through `Topology`;
* existing production topology and ko contract tests.

The new support layer intentionally does not duplicate the production
helpers: it uses the topology as raw graph input and derives its own answers.

## Support API

* `find_group` / `find_groups`: connected stones plus a set of unique
  liberties.
* `apply_move`: placement, separate opponent-group discovery, capture, then
  own-suicide validation. Optional exact previous-board comparison is only
  for small simple-ko proofs.
* `empty_regions`: empty flood-fill with bordering black/white group sets.
* `graph_triangles` / `triangle_membership`: renderer-free vertex-triangle
  discovery.
* `prove_positional_restoration`: independent true/false simple-ko proof.
* `ExhaustiveSolver`: deterministic bounded continuation search with pass and
  previous-position context. Node/depth exhaustion is always `unknown`.
* `cube_rotations`: exactly 24 orientation-preserving PointId permutations;
  `rotate_fixture` moves all point-bearing fixture state it knows about.
* `find_observation_collisions`: stable raw-byte/digest grouping for H1;
  `bounded_reachable_states` accepts only callback-approved legal transitions.
* `export_product_boundary_fixture`: V2 JSON-ready MAIN sequence, captures,
  snapshots after the first and second PASS, and a separate cleanup section.

## Rotation result

The rotation machinery uses signed permutation matrices with determinant +1 and
the canonical face frames implied by the topology edge table. It does not use
the renderer, camera, cube net, or face drawing coordinates.

* count: exactly 24 unique permutations;
* sizes checked: Cube 2, 3, 4, 5, 6, and 7;
* properties checked: bijection, identity, inverse, composition closure,
  adjacency preservation, and vertex-triangle preservation;
* fixture checks: group/liberty, capture, and true/false-ko invariance;
* torus smoke: generic graph helpers work on torus adjacency and discover no
  artificial vertex triangles.

Rotation variants are metamorphic derivatives, not independent samples. A
future train/holdout split must assign the source fixture first and keep all
24 variants in the same split; `assert_rotation_split_consistency` checks this
rule. Fixtures carrying history-sensitive cleanup state are marked
`rotation_safe: false` and `history-aware-only`.

## Corpus

The source corpus is the typed, readable registry returned by
`cube_verification_fixtures()` in `tests/support/fixtures.py`. Every entry has
an ID, family, topology, size, initial colors, side to move, actions, expected
payload, provenance, rotation policy, notes, and optional history/cleanup/
external-context metadata. It currently contains:

| Family | Count | Verification status |
| --- | ---: | --- |
| vertex groups | 4 | independent graph |
| captures | 4 | independent graph |
| eyes | 3 | structural-only |
| seki/dame | 1 | structural-only |
| ko | 2 | independent graph |
| seam tactics | 1 | independent graph / control pair |
| global connectivity | 2 | independent graph |
| cleanup | 2 | product fixture pending |
| S1 intruder support | 1 | product fixture pending |
| early termination | 1 | product fixture pending |

Eyes, seki/dame, the S1 intruder, cleanup, and early-termination entries do
not claim a life/death, scorer, or product result. Their expected values are
structural-only, pending, or unknown by design.

## M1 diagnostic pair

`cube4_false_simple_ko_001` is the reviewed false-positive shape:

```text
black: front:0:1, front:1:0, front:1:2, front:2:0
white: front:1:1
black plays: front:2:1
```

The independent checker captures the white point, then rejects white's
apparent recapture as suicide. Therefore there is no positional restoration
and this is not simple ko, regardless of any production heuristic result.

`cube4_true_simple_ko_001` independently proves the contrasting pattern:
one point is captured, the immediate recapture is locally legal without a ko
restriction, and it restores the entire initial coloring. Production ko
policy now consumes the exact rule-derived check; the M1 change and its
root-effect regression are documented in
`docs/GOCUBE_V3_STATE_OBSERVATION_AUDIT.md`.

## H1 state-field audit

The table distinguishes an absent observation field from a proven reachable
semantic ambiguity. Absence alone is not a bug classification.

| State field | Legality | Phase | Score | Root/search | Technical stop | Observation | Reconstructible |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `board` | yes | yes | yes | yes | no | planes 0/1 | no |
| `current_player` | yes | yes | no | yes | no | plane 4 | no |
| `turns` | no direct | no | no | no | move cap context | no | partly from replay |
| `consecutive_passes` | yes | yes | no | yes | yes | plane 5 | yes for immediate state |
| `captures` | no direct | no | yes (Japanese) | target context | no | planes 6/7 | no, unless replayed |
| `white_bonus_score` | no | no | yes | yes | no | no | no |
| `previous_board` | yes (simple ko) | no | no | yes | no | planes 2/3 | no |
| `ko_recap_blocked` | yes (cleanup) | yes | no | yes | no | plane 10/mask 16 | no |
| `phase_history` | yes (cycle checks) | yes | no | yes | yes | not direct | no |
| `history_since_pass` | yes (cycle checks) | yes | no | yes | yes | not direct | no |
| `black_pass_states` | yes (PASS transition) | yes | no | yes | yes | not direct | no |
| `white_pass_states` | yes (PASS transition) | yes | no | yes | yes | not direct | no |
| `ko_capture_history` | yes (cleanup repeat) | yes | no | yes | no | mask/pressure projection | no |
| `second_cleanup_start_colors` | no | yes (CLEANUP_2) | yes | yes | no | planes 11/12 | no |
| `cleanup2_moves` | no | yes | no | search context | no | planes 13/14 | no |
| `main_moves` | no | phase accounting | score setup context | yes | no | no | replay only |
| `cleanup1_moves` | no | phase accounting | no | yes | no | no | replay only |
| `terminal_kind` / `no_result_reason` | no | yes | yes | yes | yes | not direct | terminal API |
| `pass_alive_early_end` | no | yes | no | root stop | yes | not direct | replay only |
| `entered_cleanup1/2` | no | yes | no | yes | no | not direct | replay only |
| `cleanup_captures` | no | yes | no | no | no | diagnostic only | replay only |
| `ko_unblock_actions` | no | yes | no | yes | no | diagnostic only | replay only |

The pinned self-play wrapper adds configuration and accumulator state around a
game object (auto-end probability, root-pruning choice, seki-fork choice,
histories, temperatures, and MCTS reset state). Those controls are not
`V3State` fields and are therefore treated as separate integration metadata.

The H1 probe searches legal transitions supplied by the production adapter,
serializes observations using raw bytes plus shape/dtype, and compares legal
mask, phase/PASS transition, terminal behavior, and score-offset signatures.
No collision found in a bounded search means only “not found within search.”

The checked probe run used legal Cube 2 V3 transitions, seed `20260908`,
depth 2, and a 48-state cap: 48 samples, 48 observation groups, 0 collisions.
This is recorded as “not found within bounded search,” not as proof of full
Markov sufficiency. The synthetic probe test also demonstrates that the same
observation with different legal/phase semantics is reported as a semantic
collision.

## Applicability matrix

| Family | Native rectangular | Independent Cube graph | Exhaustive | Rotation | Product future |
| --- | --- | --- | --- | --- | --- |
| basic capture | local | yes | small cases | yes | yes |
| vertex triangle | no | yes | possible | yes | yes |
| vertex groups | no | yes | possible | yes | yes |
| eyes | local patterns only | structural | limited | yes when history-free | later |
| seki/dame | local patterns only | structural | limited | yes when history-free | later |
| ko | yes locally | yes | limited | yes for safe fixtures | yes |
| seam tactics | no | yes | possible | yes | yes |
| global connectivity | no | yes | possible | yes | yes |
| cleanup | no | boundary metadata | only tiny contexts | history-aware only | yes |
| early termination | no | boundary metadata | future | history-aware only | yes |

This is groundwork. It does not establish full Cube rule correctness, prove
all life-and-death outcomes, claim KataGo proves Cube geometry, or claim full
GoCube product compatibility.
