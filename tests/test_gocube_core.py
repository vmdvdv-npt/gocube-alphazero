import numpy as np
import pytest

from alphazero.envs.gocube import (
    BLACK,
    ENDGAME,
    PLAYING,
    WHITE,
    GroupClassification,
    IllegalMove,
    apply_action,
    cube_topology,
    initial_state,
    score_position,
    state_from_point_ids,
    stone_groups,
    torus_topology,
    valid_moves,
)


def test_torus_point_order_action_space_and_wrapping_match_gocube():
    topology = torus_topology(9)

    assert topology.point_ids[:4] == ("0,0", "1,0", "2,0", "3,0")
    assert topology.point_ids[9] == "0,1"
    assert topology.point_ids[-1] == "8,8"
    assert topology.point_index("4,3") == 3 * 9 + 4
    assert topology.pass_action == 81
    assert topology.action_size == 82
    assert topology.neighbor_ids("0,0") == ("8,0", "1,0", "0,8", "0,1")


def test_cube_point_order_and_action_space_match_gocube_v1():
    topology = cube_topology(3)

    assert topology.point_ids[:4] == (
        "front:0:0",
        "front:0:1",
        "front:0:2",
        "front:1:0",
    )
    assert topology.point_ids[9] == "back:0:0"
    assert topology.point_ids[18] == "left:0:0"
    assert topology.point_ids[27] == "right:0:0"
    assert topology.point_ids[36] == "top:0:0"
    assert topology.point_ids[45] == "bottom:0:0"
    assert topology.point_ids[-1] == "bottom:2:2"
    assert topology.pass_action == 54
    assert topology.action_size == 55


def test_cube_seam_transitions_and_reversals_match_gocube():
    topology = cube_topology(3)

    assert topology.neighbor_ids("front:0:0") == (
        "top:2:0",
        "front:0:1",
        "front:1:0",
        "left:0:2",
    )
    assert topology.neighbor_ids("back:0:0")[0] == "top:0:2"
    assert topology.neighbor_ids("right:0:2")[1] == "back:0:0"
    assert topology.neighbor_ids("bottom:2:0")[2] == "back:2:2"
    assert topology.neighbor_ids("left:2:0")[2] == "bottom:2:0"


def test_initial_valid_moves_include_every_point_and_pass():
    topology = torus_topology(9)
    state = initial_state(topology)

    moves = valid_moves(state, topology)

    assert moves.dtype == np.uint8
    assert moves.shape == (82,)
    assert np.all(moves == 1)


def test_capture_updates_board_counter_player_and_previous_board():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        black=("0,1", "2,1", "1,0"),
        white=("1,1",),
        current_player=0,
    )

    next_state = apply_action(state, topology.point_index("1,2"), topology)

    assert next_state.board[topology.point_index("1,1")] == 0
    assert next_state.board[topology.point_index("1,2")] == BLACK
    assert next_state.captures == (1, 0)
    assert next_state.current_player == 1
    assert next_state.turns == 1
    assert next_state.consecutive_passes == 0
    assert np.array_equal(next_state.previous_board, state.board)


def test_suicide_is_rejected_without_changing_state():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        black=("0,1", "2,1", "1,0", "1,2"),
        current_player=1,
    )

    with pytest.raises(IllegalMove, match="suicide") as error:
        apply_action(state, topology.point_index("1,1"), topology)

    assert error.value.reason == "suicide"
    assert state.board[topology.point_index("1,1")] == 0
    assert state.current_player == 1
    assert state.turns == 0


def test_simple_ko_is_exact_comparison_with_immediately_previous_board():
    topology = torus_topology(9)
    previous = np.zeros(topology.point_count, dtype=np.uint8)
    previous[topology.point_index("4,4")] = BLACK
    state = state_from_point_ids(topology, current_player=0, previous_board=previous)

    moves = valid_moves(state, topology)

    assert moves[topology.point_index("4,4")] == 0
    with pytest.raises(IllegalMove, match="repetition") as error:
        apply_action(state, topology.point_index("4,4"), topology)
    assert error.value.reason == "repetition"


def test_pass_uses_accepted_action_snapshot_for_simple_ko_context():
    topology = torus_topology(9)
    state = state_from_point_ids(topology, black=("3,3",), current_player=0)

    after_pass = apply_action(state, topology.pass_action, topology)

    assert after_pass.phase == PLAYING
    assert after_pass.current_player == 1
    assert after_pass.turns == 1
    assert after_pass.consecutive_passes == 1
    assert after_pass.board is state.board
    assert after_pass.previous_board is state.board


def test_second_consecutive_pass_ends_normal_play_but_does_not_score():
    topology = torus_topology(9)
    first = apply_action(initial_state(topology), topology.pass_action, topology)
    second = apply_action(first, topology.pass_action, topology)

    assert second.phase == ENDGAME
    assert second.current_player == 0
    assert second.turns == 2
    assert second.consecutive_passes == 2
    assert np.count_nonzero(valid_moves(second, topology)) == 0

    with pytest.raises(IllegalMove, match="not-playing"):
        apply_action(second, topology.pass_action, topology)


def test_stone_move_resets_consecutive_pass_count():
    topology = torus_topology(9)
    state = apply_action(initial_state(topology), topology.pass_action, topology)

    next_state = apply_action(state, topology.point_index("2,2"), topology)

    assert next_state.consecutive_passes == 0
    assert next_state.current_player == 0
    assert next_state.turns == 2


def _complete_classification(state, topology, dead_point_ids=()):
    dead = {topology.point_index(point_id) for point_id in dead_point_ids}
    result = []
    for group in stone_groups(state, topology):
        status = "dead" if dead.intersection(group) else "alive"
        result.append(GroupClassification(group, status))
    return tuple(result)


def test_chinese_scoring_virtually_removes_dead_stones_and_counts_area():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        black=("0,1", "2,1", "1,0"),
        white=("1,1",),
    )
    classification = _complete_classification(state, topology, dead_point_ids=("1,1",))

    score = score_position(state, topology, classification, "chinese", 0.0)

    assert score.dead_stones.black == 0
    assert score.dead_stones.white == 1
    assert score.stones_on_board.black == 3
    assert score.stones_on_board.white == 0
    assert score.territory.black == 78
    assert score.black == 81.0
    assert score.white == 0.0
    assert score.prisoners is None
    assert score.winner == "black"
    assert score.margin == 81.0


def test_japanese_scoring_adds_dead_stones_to_prisoners():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        black=("0,1", "2,1", "1,0"),
        white=("1,1",),
    )
    classification = _complete_classification(state, topology, dead_point_ids=("1,1",))

    score = score_position(state, topology, classification, "japanese", 0.0)

    assert score.territory.black == 78
    assert score.prisoners == (1, 0)
    assert score.black == 79.0
    assert score.white == 0.0
    assert score.winner == "black"


def test_scoring_rejects_incomplete_group_classification():
    topology = torus_topology(9)
    state = state_from_point_ids(topology, black=("0,0",), white=("4,4",))
    black_group = next(group for group in stone_groups(state, topology) if state.board[group[0]] == BLACK)

    with pytest.raises(ValueError, match="complete classification"):
        score_position(
            state,
            topology,
            (GroupClassification(black_group, "alive"),),
            "chinese",
            7.5,
        )


def test_empty_torus_is_neutral_and_komi_decides_score():
    topology = torus_topology(9)
    state = initial_state(topology)

    score = score_position(state, topology, (), "chinese", 7.5)

    assert score.territory.neutral == 81
    assert score.black == 0.0
    assert score.white == 7.5
    assert score.winner == "white"
