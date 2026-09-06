from dataclasses import replace

import numpy as np
import pytest

from alphazero.envs.gocube import (
    CLEANUP_1, CLEANUP_2, NO_RESULT, SCORED, Torus9JapaneseGame,
    Topology, apply_v3_action, initial_v3_state, v3_state_from_board, v3_valid_moves,
)
from alphazero.envs.gocube.katago_v3 import (
    EMERGENCY_MOVE_CAP_BASE, EMERGENCY_MOVE_CAP_FACTOR, _cycle_check_and_record,
    terminal_from_state,
)


def rect_topology(width=5, height=5):
    ids = tuple(f"{x},{y}" for y in range(height) for x in range(width))
    index = {p: i for i, p in enumerate(ids)}
    neighbors = []
    for y in range(height):
        for x in range(width):
            row = []
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    row.append(index[f"{nx},{ny}"])
            neighbors.append(tuple(row))
    return Topology("rect-test", width, ids, tuple(neighbors), index)


def test_two_passes_each_phase_reach_scored():
    topology = Torus9JapaneseGame.logical_topology()
    state = initial_v3_state(topology)
    state = apply_v3_action(state, topology.pass_action, topology)
    state = apply_v3_action(state, topology.pass_action, topology)
    assert state.phase == CLEANUP_1 and state.entered_cleanup1
    state = apply_v3_action(state, topology.pass_action, topology)
    state = apply_v3_action(state, topology.pass_action, topology)
    assert state.phase == CLEANUP_2 and state.entered_cleanup2
    assert state.second_cleanup_start_colors == bytes(state.board.tolist())
    state = apply_v3_action(state, topology.pass_action, topology)
    state = apply_v3_action(state, topology.pass_action, topology)
    assert state.phase == SCORED and state.terminal_kind == SCORED


def test_cleanup_capture_is_real_and_counted():
    t = rect_topology(3, 3)
    center = t.point_index("1,1")
    black = [t.point_index(p) for p in ("1,0", "0,1", "2,1")]
    state = v3_state_from_board(t, black=black, white=(center,), phase=CLEANUP_1, current_player=0)
    action = t.point_index("1,2")
    next_state = apply_v3_action(state, action, t)
    assert next_state.board[center] == 0
    assert next_state.captures == (1, 0)
    assert next_state.cleanup_captures == 1
    assert next_state.cleanup1_moves == (1, 0)
    assert next_state.cleanup2_moves == (0, 0)


def test_cleanup2_move_gets_compensation_counter_but_cleanup1_does_not():
    t = rect_topology(3, 3)
    action = t.point_index("1,1")
    c1 = v3_state_from_board(t, phase=CLEANUP_1)
    c2 = v3_state_from_board(t, phase=CLEANUP_2, second_cleanup_start_colors=bytes([0] * t.point_count))
    assert apply_v3_action(c1, action, t).cleanup2_moves == (0, 0)
    assert apply_v3_action(c2, action, t).cleanup2_moves == (1, 0)


def ko_fixture():
    t = rect_topology(5, 5)
    black = [t.point_index(p) for p in ("2,1", "1,2", "3,2")]
    white = [t.point_index(p) for p in ("2,2", "1,3", "3,3", "2,4")]
    state = v3_state_from_board(t, black=black, white=white, phase=CLEANUP_1, current_player=0)
    return t, state, t.point_index("2,3"), t.point_index("2,2")


def test_ko_recap_block_and_unblock_use_existing_point_action():
    t, state, capture, recapture = ko_fixture()
    captured = apply_v3_action(state, capture, t)
    assert capture in captured.ko_recap_blocked
    assert v3_valid_moves(captured, t)[capture] == 1
    unblocked = apply_v3_action(captured, capture, t)
    assert unblocked.board[capture] == 1
    assert capture not in unblocked.ko_recap_blocked
    assert unblocked.ko_unblock_actions == 1
    after_pass = apply_v3_action(unblocked, t.pass_action, t)
    assert after_pass.current_player == 1
    assert v3_valid_moves(after_pass, t)[recapture] in (0, 1)
    assert t.action_size == t.point_count + 1


def test_ko_recap_block_can_be_lifted_from_empty_ko_capture_point():
    t, state, capture, recapture = ko_fixture()
    captured = apply_v3_action(state, capture, t)
    assert capture in captured.ko_recap_blocked
    assert captured.board[recapture] == 0
    assert v3_valid_moves(captured, t)[recapture] == 1

    # KataGo's second pass-for-ko form: playing the empty ko-capture point
    # removes the recap block but does not place a stone or perform a capture.
    unblocked = apply_v3_action(captured, recapture, t)
    assert np.array_equal(unblocked.board, captured.board)
    assert unblocked.captures == captured.captures
    assert capture not in unblocked.ko_recap_blocked
    assert unblocked.current_player == 0
    assert unblocked.ko_unblock_actions == captured.ko_unblock_actions + 1

    # After the opponent takes a turn, the actual ko recapture is available.
    after_pass = apply_v3_action(unblocked, t.pass_action, t)
    assert after_pass.current_player == 1
    assert v3_valid_moves(after_pass, t)[recapture] == 1
    recaptured = apply_v3_action(after_pass, recapture, t)
    assert recaptured.board[recapture] == 2
    assert recaptured.board[capture] == 0


def test_repeated_state_cycle_is_no_result():
    t = rect_topology(3, 3)
    state = v3_state_from_board(t, phase=CLEANUP_1)
    key = state.history_since_pass[0]
    state = replace(state, history_since_pass=(key, key, key))
    state = _cycle_check_and_record(state, after_pass=False)
    assert state.terminal_kind == NO_RESULT
    assert state.no_result_reason == "cycle"


def test_no_result_is_framework_draw_utility_but_not_training_target():
    game = Torus9JapaneseGame()
    game._state = replace(game.semantic_state, phase=NO_RESULT, terminal_kind=NO_RESULT, no_result_reason="cycle")
    game._terminal = terminal_from_state(game._state, game.logical_topology(), game.KOMI)
    assert np.array_equal(game.win_state(), np.array([0, 0, 1], dtype=np.uint8))
    assert not game.has_training_result()
    with pytest.raises(ValueError):
        game.training_targets()


def test_emergency_move_cap_is_no_result_not_forced_score():
    t = rect_topology(3, 3)
    cap = EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * t.point_count
    state = replace(initial_v3_state(t), turns=cap - 1)
    state = apply_v3_action(state, t.pass_action, t)
    assert state.terminal_kind == NO_RESULT
    assert state.no_result_reason == "move-cap"
