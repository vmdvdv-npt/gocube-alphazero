import pytest

from alphazero.envs.gocube import (
    BLACK,
    TORUS_SIZES,
    WHITE,
    IllegalMove,
    apply_action,
    state_from_point_ids,
    torus_topology,
)


def ids_with_stone(state, topology, stone):
    return {
        topology.point_id(index)
        for index, value in enumerate(state.board)
        if int(value) == stone
    }


def test_captures_connected_group_matching_gocube_fixture():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        white=("3,4", "4,4"),
        black=("2,4", "3,3", "3,5", "5,4", "4,3"),
        current_player=0,
    )

    result = apply_action(state, topology.point_index("4,5"), topology)

    assert "3,4" not in ids_with_stone(result, topology, WHITE)
    assert "4,4" not in ids_with_stone(result, topology, WHITE)
    assert result.captures == (2, 0)


def test_captures_multiple_neighboring_groups_matching_gocube_fixture():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        white=("3,4", "5,4"),
        black=("2,4", "3,3", "3,5", "6,4", "5,3", "5,5"),
        current_player=0,
    )

    result = apply_action(state, topology.point_index("4,4"), topology)

    white = ids_with_stone(result, topology, WHITE)
    assert "3,4" not in white
    assert "5,4" not in white
    assert result.captures == (2, 0)


def test_capture_happens_before_suicide_evaluation_matching_gocube_fixture():
    topology = torus_topology(9)
    state = state_from_point_ids(
        topology,
        white=("3,4", "5,4", "4,3", "4,5"),
        black=(
            "2,4", "3,3", "3,5",
            "6,4", "5,3", "5,5",
            "4,2", "4,6",
        ),
        current_player=0,
    )

    result = apply_action(state, topology.point_index("4,4"), topology)

    assert result.board[topology.point_index("4,4")] == BLACK
    assert result.captures == (4, 0)
    for captured in ("3,4", "5,4", "4,3", "4,5"):
        assert result.board[topology.point_index(captured)] == 0


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_capture_group_crossing_left_right_torus_seam(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    final_liberty = f"{last},{mid + 1}"
    state = state_from_point_ids(
        topology,
        white=(f"0,{mid}", f"{last},{mid}"),
        black=(
            f"1,{mid}",
            f"0,{mid - 1}",
            f"0,{mid + 1}",
            f"{last - 1},{mid}",
            f"{last},{mid - 1}",
        ),
        current_player=0,
    )

    result = apply_action(state, topology.point_index(final_liberty), topology)

    assert result.captures == (2, 0)
    assert result.board[topology.point_index(f"0,{mid}")] == 0
    assert result.board[topology.point_index(f"{last},{mid}")] == 0


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_capture_group_crossing_top_bottom_torus_seam(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    final_liberty = f"{mid + 1},{last}"
    state = state_from_point_ids(
        topology,
        white=(f"{mid},0", f"{mid},{last}"),
        black=(
            f"{mid - 1},0",
            f"{mid + 1},0",
            f"{mid},1",
            f"{mid - 1},{last}",
            f"{mid},{last - 1}",
        ),
        current_player=0,
    )

    result = apply_action(state, topology.point_index(final_liberty), topology)

    assert result.captures == (2, 0)
    assert result.board[topology.point_index(f"{mid},0")] == 0
    assert result.board[topology.point_index(f"{mid},{last}")] == 0


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_rejects_suicide_enclosed_across_left_right_torus_seam(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    point = f"0,{mid}"
    state = state_from_point_ids(
        topology,
        white=(f"{last},{mid}", f"1,{mid}", f"0,{mid - 1}", f"0,{mid + 1}"),
        current_player=0,
    )

    with pytest.raises(IllegalMove, match="suicide"):
        apply_action(state, topology.point_index(point), topology)


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_rejects_suicide_enclosed_across_top_bottom_torus_seam(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    point = f"{mid},0"
    state = state_from_point_ids(
        topology,
        white=(f"{mid},{last}", f"{mid},1", f"{mid - 1},0", f"{mid + 1},0"),
        current_player=0,
    )

    with pytest.raises(IllegalMove, match="suicide"):
        apply_action(state, topology.point_index(point), topology)


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_capture_across_left_right_seam_can_prevent_suicide(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    point = f"0,{mid}"
    captured = f"{last},{mid}"
    state = state_from_point_ids(
        topology,
        white=(captured, f"1,{mid}", f"0,{mid - 1}", f"0,{mid + 1}"),
        black=(f"{last - 1},{mid}", f"{last},{mid - 1}", f"{last},{mid + 1}"),
        current_player=0,
    )

    result = apply_action(state, topology.point_index(point), topology)

    assert result.board[topology.point_index(point)] == BLACK
    assert result.board[topology.point_index(captured)] == 0
    assert result.captures == (1, 0)


@pytest.mark.parametrize("size", TORUS_SIZES)
def test_capture_across_top_bottom_seam_can_prevent_suicide(size):
    topology = torus_topology(size)
    last = size - 1
    mid = size // 2
    point = f"{mid},0"
    captured = f"{mid},{last}"
    state = state_from_point_ids(
        topology,
        white=(captured, f"{mid},1", f"{mid - 1},0", f"{mid + 1},0"),
        black=(f"{mid - 1},{last}", f"{mid + 1},{last}", f"{mid},{last - 1}"),
        current_player=0,
    )

    result = apply_action(state, topology.point_index(point), topology)

    assert result.board[topology.point_index(point)] == BLACK
    assert result.board[topology.point_index(captured)] == 0
    assert result.captures == (1, 0)
