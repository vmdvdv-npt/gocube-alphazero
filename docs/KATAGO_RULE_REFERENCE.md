KataGo reference commit:
f6bc4b19a1686caa2d088b56251e8c11c8be6d51

# Independent rule reference

The reference profile is KataGo Rules V3 with `KoRule = SIMPLE`,
`ScoringRule = TERRITORY`, `TaxRule = SEKI`, multi-stone suicide disabled,
button disabled, white handicap bonus zero, and komi `0.5`.  The oracle is a
small JSONL adapter around KataGo's `Board` and `BoardHistory`; it does not
reimplement liberties, captures, legality, ko, phase transitions, or scoring.

| ID | KataGo source | KataGo function/field | GoCube source | GoCube function | Проверяемая семантика | Допустимое отличие | Test ID |
| -- | ------------- | --------------------- | ------------- | --------------- | --------------------- | ------------------ | ------- |
| A1 | `cpp/game/board.h`, `cpp/game/board.cpp` | `Board::colors`, `Board::adj`, `Board::calculateArea` | `alphazero/envs/gocube/core.py`, `alphazero/envs/gocube/katago_v3.py` | `Topology.neighbor_indices`, `_collect_group` | occupancy, graph adjacency, groups, liberties | KataGo `Loc` and GoCube integer point IDs are converted through row-major `[x,y]`; edges must remain identical | `test_gocube_katago_reference_rect.py::test_static_fixture_matches_pinned_katago` |
| A2 | `cpp/game/board.h`, `cpp/game/board.cpp` | `Board::playMove`, `Board::removeChain` | `alphazero/envs/gocube/katago_v3.py` | `_pseudolegal_candidate`, `_placement` | stone placement and captured-group removal | only representation differs; capture outcome and resulting occupancy may not differ | `single-stone-capture`, `multi-stone-capture`, `capture-not-suicide` |
| A3 | `cpp/game/board.h`, `cpp/game/board.cpp` | `Board::isIllegalSuicide` | `alphazero/envs/gocube/katago_v3.py` | `_pseudolegal_candidate`, `_legal_placement` | single/multi-group suicide legality | none; both use the same four-neighbor graph in the planar bridge | `suicide-illegal`, `capture-not-suicide`, `test_gocube_katago_reference_random.py` |
| B1 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | `BoardHistory::isLegal` | `alphazero/envs/gocube/katago_v3.py` | `v3_valid_moves`, `_legal_placement` | occupied, suicide, simple-ko and pass legality | KataGo's `Loc` is normalized to action index `y*width+x`; pass is the final action | `test_gocube_katago_reference_random.py` |
| B2 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | `BoardHistory::makeBoardMoveAssumeLegal` | `alphazero/envs/gocube/katago_v3.py` | `apply_v3_action`, `_placement`, `_pass` | history update, previous board, captures and player-to-move | GoCube stores immutable NumPy snapshots instead of KataGo's board-history vector | `simple-ko`, `legal-after-ko-threat`, `test_gocube_katago_reference_random.py` |
| B3 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | `BoardHistory::passWouldEndPhase`, `encorePhase` | `alphazero/envs/gocube/katago_v3.py` | `_finish_phase_after_pass`, `_phase_reset`, `phase_history`, `history_since_pass` | pass, MAIN→CLEANUP_1→CLEANUP_2 and finish | external phase strings are `MAIN`, `CLEANUP_1`, `CLEANUP_2`, `SCORED`, `NO_RESULT` | `main-to-cleanup-1`, `cleanup-1-to-cleanup-2`, `two-ending-passes` |
| B4 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | ko hash history and `koRecapBlocked` | `alphazero/envs/gocube/katago_v3.py` | `ko_capture_history`, `ko_recap_blocked`, `ko_repeat_forbidden_mask` | normal simple ko, cleanup ko, recap restrictions, pass-for-ko and repeated recapture prevention | KataGo hash tables become explicit point masks/history tuples in GoCube; legal outcomes are still compared point-for-point | `simple-ko`, `cleanup-ko`, `ko-recap-block`, `pass-for-ko-form-1`, `pass-for-ko-form-2`, `repeated-ko-prevention` |
| C1 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | `isGameFinished`, `isNoResult`, `winner` | `alphazero/envs/gocube/katago_v3.py` | `terminal_from_state`, `terminal_kind`, `V3Terminal` | scored terminal, no-result terminal, winner and phase | KataGo booleans are normalized to one `terminal_kind`; framework's third transport slot is not treated as a scored draw | `pass-alive-terminal`, `no-result-cycle-repetition`, `test_gocube_katago_reference_random.py` |
| C2 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | `finalWhiteMinusBlackScore` | `alphazero/envs/gocube/katago_v3.py` | `final_v3_score`, `normalized_score_target_v3` | final score and terminal score sign | oracle and comparison adapter expose canonical `white_minus_black`; the training head keeps its existing black-minus-white normalization internally | `territory-scoring`, `seki-tax`, `prisoner-capture-contribution`, `cleanup-2-compensation` |
| D1 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | `BoardHistory::endAndScoreGameNow` | `alphazero/envs/gocube/katago_v3.py` | `final_v3_score`, `independent_life_analysis` | territory, independent life, seki/tax, prisoners and komi | KataGo `Loc` arrays are mapped to row-major point arrays; no score field is ignored | `territory-scoring`, `seki-tax`, `prisoner-capture-contribution` |
| D2 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | Rules V3 scoring and cleanup accounting | `alphazero/envs/gocube/katago_v3.py` | `pass_alive_analysis`, `cleanup2_moves`, `second_cleanup_start_colors` | pass-alive terminal, cleanup captures and Cleanup 2 compensation | GoCube's graph is substituted only in Cube/Torus metamorphic tests; planar expected outcomes come from KataGo | `pass-alive-terminal`, `cleanup-capture`, `cleanup-2-compensation` |
| E1 | `cpp/game/boardhistory.h`, `cpp/game/boardhistory.cpp` | `encorePhase`, `koRecapBlocked`, `makeBoardMoveAssumeLegal` | `alphazero/envs/gocube/katago_v3.py` | `_unblock`, `_pass_for_ko_unblock_target`, `_finish_phase_after_pass` | cleanup ko, ko recap blocks, pass-for-ko, repeated ko and phase transition | KataGo's internal recap bookkeeping is represented by explicit point IDs; legal mask and semantic block set are compared | `cleanup-ko`, `ko-recap-block`, `pass-for-ko-form-1`, `pass-for-ko-form-2`, `repeated-ko-prevention` |
| T1 | `cpp/game/board.h`, `cpp/game/boardhistory.h` | rectangular coordinate graph and `Loc` conversion | `tests/gocube_reference_topology.py` | `rectangular_test_topology` | independent planar bridge for 3×3, 5×3, 5×5 and 7×4 | test-only topology uses row-major integer points and a final pass action; it is not a production topology | `test_gocube_reference_topology.py` |

## Unavailable direct comparisons

KataGo's rectangular `Board` is not a Cube or Torus oracle, and its neural
network/search code is outside this rule harness.  Those differences are not
allowlisted as semantic mismatches.  Cube/Torus seam and wrap behavior is
tested by graph isomorphism and symmetry/metamorphic tests in
`tests/test_gocube_topology_reference_bridge.py` and
`tests/test_gocube_topology_symmetry_rules.py`; the planar neighborhood used
as the source fixture is still evaluated by the pinned KataGo oracle first.

The machine-readable allowlist in
`tests/reference/katago/allowed_differences.json` contains only coordinate
encoding, action indexing, graph-neighbor source, and score-sign conversion.
It does not permit board, legal, capture, ko, phase, terminal, winner, or
score mismatches.
