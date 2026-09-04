import pytest

from alphazero.envs.gocube import (
    TORUS_SIZES,
    IllegalMove,
    apply_action,
    state_from_point_ids,
    torus_topology,
)


def test_classic_immediate_ko_recapture_is_rejected():
    topology = torus_topology(9)
    before_capture = state_from_point_ids(
        topology,
        white=("4,4", "3,5", "5,5", "4,6"),
        black=("3,4", "5,4", "4,3"),
        current_player=0,
    )

    black_capture = apply_action(before_capture, topology.point_index("4,5"), topology)
    assert black_capture.board[topology.point_index("4,4")] == 0
    assert black_capture.captures == (1, 0)

    with pytest.raises(IllegalMove, match="repetition") as error:
        apply_action(black_capture, topology.point_index("4,4"), topology)
    assert error.value.reason == "repetition"


def test_simple_ko_compares_post_capture_candidate_board():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        white=("4,4",),
        black=("3,4", "5,4", "4,3"),
        current_player=0,
    )
    candidate = apply_action(state, topology.point_index("4,5"), topology)

    repeated_context = state_from_point_ids(
        topology,
        white=("4,4",),
        black=("3,4", "5,4", "4,3"),
        current_player=0,
        previous_board=candidate.board,
    )

    with pytest.raises(IllegalMove, match="repetition"):
        apply_action(repeated_context, topology.point_index("4,5"), topology)


def test_simple_ko_does_not_implement_superko_against_older_position():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        white=("4,4",),
        black=("3,4", "5,4", "4,3"),
        current_player=0,
    )
    candidate = apply_action(state, topology.point_index("4,5"), topology)
    unrelated_immediate = candidate.board.copy()
    unrelated_immediate[topology.point_index("0,0")] = 1

    with_context = state_from_point_ids(
        topology,
        white=("4,4",),
        black=("3,4", "5,4", "4,3"),
        current_player=0,
        previous_board=unrelated_immediate,
    )

    accepted = apply_action(with_context, topology.point_index("4,5"), topology)
    assert accepted.captures == (1, 0)


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_immediate_ko_recapture_across_left_right_torus_seam(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    captured_point = f"0,{mid}"
    capture_point = f"{last},{mid}"
    before_capture = state_from_point_ids(
        topology,
        white=(
            captured_point,
            f"{last - 1},{mid}",
            f"{last},{mid - 1}",
            f"{last},{mid + 1}",
        ),
        black=(f"1,{mid}", f"0,{mid - 1}", f"0,{mid + 1}"),
        current_player=0,
    )

    black_capture = apply_action(before_capture, topology.point_index(capture_point), topology)

    with pytest.raises(IllegalMove, match="repetition"):
        apply_action(black_capture, topology.point_index(captured_point), topology)


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_immediate_ko_recapture_across_top_bottom_torus_seam(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    captured_point = f"{mid},0"
    capture_point = f"{mid},{last}"
    before_capture = state_from_point_ids(
        topology,
        white=(
            captured_point,
            f"{mid - 1},{last}",
            f"{mid + 1},{last}",
            f"{mid},{last - 1}",
        ),
        black=(f"{mid - 1},0", f"{mid + 1},0", f"{mid},1"),
        current_player=0,
    )

    black_capture = apply_action(before_capture, topology.point_index(capture_point), topology)

    with pytest.raises(IllegalMove, match="repetition"):
        apply_action(black_capture, topology.point_index(captured_point), topology)
