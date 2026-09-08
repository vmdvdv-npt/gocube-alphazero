# GoCube V1: pinned KataGo rule mapping

Status: V1 verification contract, based on the merged F0/PR55 state
`25b9f7f063acd0fa1921ab8f3c5a2adb8e8c8bad`.

The native oracle is built only from KataGo commit
`f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.  All V1 native cases use Rules
Version 3 with `KoRule=SIMPLE`, `ScoringRule=TERRITORY`, `TaxRule=SEKI`,
`multiStoneSuicide=false`, `button=false`, `whiteHandicapBonus=0`, and
`komi=0.5`.

The Python runner compares normalized semantics.  It does not compare
internal C++/Python history bytes, and it never creates an expected value by
calling the production function under test.

| ID | Pinned KataGo source/function | GoCube production function | Independent/differential mechanism | Expected equivalence | Intentional difference | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | `cpp/game/board.h`, `Board::colors`, `Board::adj` | `core.Topology.neighbor_indices`, `katago_v3.V3State.board` | native snapshot + `independent_graph.find_groups` | occupancy and graph groups/liberties match after coordinate normalization | point IDs/action indices are adapter-normalized | PASS |
| A2 | `cpp/game/board.cpp`, `Board::playMove`, `Board::removeChain` | `katago_v3._pseudolegal_candidate`, `apply_v3_action` | native fixture and independent `apply_move` capture transition | placement, removed points, capture counts, resulting board match | none in rectangular bridge | PASS |
| A3 | `cpp/game/board.cpp`, `Board::isIllegalSuicide` | `katago_v3._pseudolegal_candidate` | suicide and capture-before-suicide fixtures | ordinary suicide is rejected; capture-before-suicide is legal | none in rectangular bridge | PASS |
| B1 | `cpp/game/boardhistory.cpp`, `BoardHistory::isLegal` | `katago_v3.v3_valid_moves`, `_legal_placement` | static native fixtures + generated legal-mask diff | legal action set matches at every compared state | PASS action index is normalized | PASS |
| B2 | `cpp/game/boardhistory.cpp`, `BoardHistory::makeBoardMoveAssumeLegal` | `katago_v3.apply_v3_action`, `_placement`, `_pass` | per-step semantic snapshot diff | board, player, captures, previous-board ko meaning match | history containers differ internally | PASS |
| B3 | `cpp/game/boardhistory.cpp`, `BoardHistory::passWouldEndPhase`, `encorePhase` | `_pass`, `_finish_phase_after_pass`, `_phase_reset` | native pass/phase fixtures | PASS and MAIN→CLEANUP_1→CLEANUP_2→SCORED boundaries match | phase strings are normalized | PASS |
| B4 | `cpp/game/boardhistory.cpp`, `koRecapBlocked`, ko history | `_pass_for_ko_unblock_target`, `_unblock`, `ko_repeat_forbidden_mask` | cleanup-ko and both PASS-for-ko forms | block/release/repeat restriction matches point-for-point | hash tables become explicit point tuples/masks | PASS |
| B5 | `cpp/game/boardhistory.cpp`, `BoardHistory::clear`, `whiteBonusScore`, `secondEncoreStartColors` | `_boardhistory_clear_white_bonus`, `v3_state_from_board`, `final_v3_score` | S1 setup cases in MAIN/CLEANUP_1/CLEANUP_2 | setup stones/captures/start colors produce the same score offset | storage order for captures is normalized | PASS |
| C1 | `BoardHistory::isGameFinished`, `isNoResult`, `winner` | `terminal_from_state`, `V3Terminal` | terminal native fixtures + cycle fixture | scored terminal, NO_RESULT, winner and reason match | `NO_RESULT` is normalized to terminal kind | PASS |
| C2 | `BoardHistory::finalWhiteMinusBlackScore` | `final_v3_score`, `normalized_score_target_v3` | final score differential | canonical `white_minus_black` and winner match | training score head retains black-minus-white convention | PASS |
| D1 | `BoardHistory::endAndScoreGameNow`, `Board::calculateArea` | `independent_life_analysis`, `final_v3_score` | scoring/setup, territory, prisoner and seki-tax fixtures | territory, captures, komi, score, winner, ownership and masks match | Cube/Torus use graph adjacency only where native rectangle is inapplicable | PASS |
| D2 | Rules V3 area/cleanup accounting in `boardhistory.cpp` | `pass_alive_analysis`, `final_v3_score`, cleanup state fields | pass-alive, cleanup capture and cleanup-2 compensation fixtures | pass-alive and cleanup scoring semantics match where representable | product assisted/manual cleanup is V2 scope | PASS |
| E1 | `BoardHistory::encorePhase`, `koRecapBlocked`, `makeBoardMoveAssumeLegal` | `_unblock`, `_pass_for_ko_unblock_target`, `_finish_phase_after_pass` | native cleanup family and Cube/Torus graph controls | cleanup ko, pass-for-ko, repeated-ko prevention and phase state match | no KataGo rectangle is claimed as a Cube geometry oracle | PASS |
| T1 | `Board::adj`, `Location::getLoc/getX/getY` | `tests/gocube_reference_topology.rectangular_test_topology` | pinned rectangular bridge | row-major adjacency and action normalization match | test-only topology is not production topology | PASS |
| R1 | Cube topology has no KataGo rectangular equivalent | `independent_graph`, `core.cube_topology` | Cube2–Cube7 degree/triangle invariants, Cube4 graph fixtures and 24 rotations | graph consequences, seams, vertices, captures, eyes, seki/dame and ko match independent evidence | Cube geometry itself is not independently sourced by KataGo | PASS |
| T2 | Torus topology has no KataGo rectangular equivalent | `independent_graph`, `core.torus_topology` | Torus9 wrap group/capture/ko and no-triangle invariants | wrap-around graph consequences match | no Cube vertex features on Torus | PASS |

## Boundary rule

`episode_move_limit` is a `GameRunner` policy.  It is verified only by the
S3 boundary test (`episode_move_limit(topology)` and the formal transition
near that value); it is not part of `apply_v3_action` rules differential and
cannot create a formal `NO_RESULT`.

The source mapping intentionally distinguishes:

* native pinned KataGo, used for ordinary rectangular Rules V3 semantics;
* independent graph verification, used for Cube/Torus consequences;
* metamorphic rotations, which preserve `source_fixture_id` and do not count
  as independent corpus rows.
The mapping is reviewed by the V1 differential tests and is not a substitute
for their native/independent execution.
