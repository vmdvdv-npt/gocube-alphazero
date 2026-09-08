from __future__ import annotations

import numpy as np
import pytest

from alphazero.envs.gocube import (
    CLEANUP_1,
    Cube4JapaneseGame,
    cube_topology,
    is_simple_ko_state,
    v3_state_from_board,
)
from alphazero.envs.gocube.katago_v3 import (
    V3IllegalMove,
    _pseudolegal_candidate,
    apply_v3_action,
    v3_valid_moves,
)
from alphazero.search_contract import (
    _simple_ko_likely_active,
    root_ending_white_score_bonuses,
)
from gocube_reference_topology import rectangular_test_topology
from tests.support.fixtures import cube_verification_fixtures


def _cube4_game_after_capture(fixture: str) -> tuple[Cube4JapaneseGame, int, int]:
    topology = cube_topology(4)
    if fixture == "false_single_capture":
        fixture_id = "cube4_false_simple_ko_001"
    elif fixture == "true_simple_ko":
        fixture_id = "cube4_true_simple_ko_001"
    else:
        raise AssertionError(fixture)
    source = next(item for item in cube_verification_fixtures() if item.id == fixture_id)
    board = source.board(topology.index_by_id)
    state = v3_state_from_board(
        topology,
        black=np.flatnonzero(np.asarray(board) == 1),
        white=np.flatnonzero(np.asarray(board) == 2),
        current_player=0,
    )
    game = Cube4JapaneseGame(state)
    capture = topology.point_index(source.actions[0])
    recapture = topology.point_index(source.actions[1])
    game.play_action(capture)
    return game, capture, recapture


def _legacy_changed_points_predicate(game: Cube4JapaneseGame) -> bool:
    state = game.semantic_state
    return state.previous_board is not None and int(
        np.count_nonzero(np.asarray(state.board) != np.asarray(state.previous_board))
    ) == 2


@pytest.mark.parametrize(
    ("fixture", "expected"),
    (("false_single_capture", False), ("true_simple_ko", True)),
)
def test_two_changed_points_are_not_sufficient_for_simple_ko(fixture, expected):
    game, _capture, _recapture = _cube4_game_after_capture(fixture)

    # This is the pre-M1 reproduction: both transitions change two points.
    assert _legacy_changed_points_predicate(game)
    assert is_simple_ko_state(game.semantic_state, game.logical_topology()) is expected
    assert _simple_ko_likely_active(game) is expected


def test_false_single_capture_recapture_is_suicide_not_ko():
    game, _capture, recapture = _cube4_game_after_capture("false_single_capture")
    assert v3_valid_moves(game.semantic_state, game.logical_topology())[recapture] == 0
    with pytest.raises(V3IllegalMove, match="suicide"):
        apply_v3_action(game.semantic_state, recapture, game.logical_topology())


def test_true_immediate_recapture_restores_board_but_simple_ko_rejects_it():
    game, _capture, recapture = _cube4_game_after_capture("true_simple_ko")
    state = game.semantic_state
    assert is_simple_ko_state(state, game.logical_topology())
    candidate, captured_groups = _pseudolegal_candidate(
        state.board, state.current_player, recapture, game.logical_topology()
    )
    assert sum(len(group) for group in captured_groups) == 1
    assert np.array_equal(candidate, state.previous_board)
    assert v3_valid_moves(state, game.logical_topology())[recapture] == 0
    with pytest.raises(V3IllegalMove, match="simple-ko"):
        apply_v3_action(state, recapture, game.logical_topology())


def test_root_ending_bonus_is_disabled_only_for_true_simple_ko():
    false_game, _false_capture, _false_recapture = _cube4_game_after_capture("false_single_capture")
    true_game, _true_capture, _true_recapture = _cube4_game_after_capture("true_simple_ko")
    action = false_game.logical_topology().point_index("front:0:3")
    assert v3_valid_moves(false_game.semantic_state, false_game.logical_topology())[action]
    assert v3_valid_moves(true_game.semantic_state, true_game.logical_topology())[action]
    ownership = np.zeros((false_game.logical_topology().point_count, 3), dtype=np.float32)
    ownership[action, 1] = 1.0

    false_bonus = root_ending_white_score_bonuses(false_game, ownership, 0.5)
    true_bonus = root_ending_white_score_bonuses(true_game, ownership, 0.5)
    assert false_bonus[action] == pytest.approx(-0.5)
    assert true_bonus[action] == pytest.approx(0.0)


def _cleanup_ko_fixture():
    topology = rectangular_test_topology(5, 5)
    black = tuple(topology.point_index(point) for point in ("2,1", "1,2", "3,2"))
    white = tuple(topology.point_index(point) for point in ("2,2", "1,3", "3,3", "2,4"))
    state = v3_state_from_board(
        topology,
        black=black,
        white=white,
        phase=CLEANUP_1,
        current_player=0,
    )
    return topology, state, topology.point_index("2,3"), topology.point_index("2,2")


@pytest.mark.parametrize("cleanup_phase", ("cleanup1", "cleanup2"))
def test_cleanup_pass_for_ko_is_distinct_from_ordinary_capture(cleanup_phase):
    topology, state, capture, recapture = _cleanup_ko_fixture()
    if cleanup_phase == "cleanup2":
        state = v3_state_from_board(
            topology,
            black=np.flatnonzero(state.board == 1),
            white=np.flatnonzero(state.board == 2),
            phase="cleanup2",
            current_player=0,
            second_cleanup_start_colors=bytes(state.board.tolist()),
        )

    captured = apply_v3_action(state, capture, topology)
    assert is_simple_ko_state(captured, topology)
    assert captured.ko_recap_blocked == (capture,)
    # The empty ko point is the explicit cleanup PASS-for-ko action. It leaves
    # occupancy and captures unchanged, unlike an ordinary one-stone capture.
    unblocked = apply_v3_action(captured, recapture, topology)
    assert np.array_equal(unblocked.board, captured.board)
    assert unblocked.captures == captured.captures
    assert unblocked.ko_recap_blocked == ()
    assert not is_simple_ko_state(unblocked, topology)


def test_rectangular_true_ko_uses_graph_transition_and_not_renderer_coordinates():
    topology = rectangular_test_topology(5, 5)
    capture = topology.point_index("2,3")
    recapture = topology.point_index("2,2")
    state = v3_state_from_board(
        topology,
        black=(topology.point_index(point) for point in ("2,1", "1,2", "3,2")),
        white=(topology.point_index(point) for point in ("2,2", "1,3", "3,3", "2,4")),
        current_player=0,
    )
    captured = apply_v3_action(state, capture, topology)
    assert is_simple_ko_state(captured, topology)
    assert v3_valid_moves(captured, topology)[recapture] == 0
