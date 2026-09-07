from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from alphazero.envs.gocube.core import BLACK, WHITE, cube_topology, torus_topology
from alphazero.envs.gocube.katago_v3 import (
    CLEANUP_1,
    CLEANUP_2,
    _all_groups,
    _collect_group,
    _pseudolegal_candidate,
    all_points_pass_alive,
    apply_v3_action,
    final_v3_score,
    independent_life_analysis,
    pass_alive_analysis,
    v3_state_from_board,
    v3_valid_moves,
)
from katago_reference_runner import KatagoOracleProcess


pytestmark = pytest.mark.katago_reference


def _topology(kind):
    return cube_topology(4) if kind == "cube" else torus_topology(9)


def _seam_edge(topology):
    if topology.kind == "cube":
        return topology.point_index("front:0:2"), topology.point_index("top:3:2")
    return topology.point_index("0,0"), topology.point_index("8,0")


def _ko_shape(topology, *, phase="main"):
    capture_point, recapture_point = _seam_edge(topology)
    black = {
        neighbor
        for neighbor in topology.neighbor_indices(recapture_point)
        if neighbor != capture_point
    }
    white = {
        recapture_point,
        *(
            neighbor
            for neighbor in topology.neighbor_indices(capture_point)
            if neighbor != recapture_point
        ),
    }
    assert black.isdisjoint(white)
    state = v3_state_from_board(
        topology,
        black=black,
        white=white,
        current_player=0,
        phase=phase,
    )
    _, liberties = _collect_group(state.board, recapture_point, WHITE, topology)
    assert liberties == {capture_point}
    return state, capture_point, recapture_point


def _multi_capture_state(topology):
    target = _seam_edge(topology)[0]
    neighbors = topology.neighbor_indices(target)
    for first_index, first in enumerate(neighbors):
        for second in neighbors[first_index + 1:]:
            white = {first, second}
            black = {
                neighbor
                for point in white
                for neighbor in topology.neighbor_indices(point)
                if neighbor != target
            }
            black -= white
            state = v3_state_from_board(topology, black=black, white=white)
            if all(
                _collect_group(state.board, point, WHITE, topology) == ({point}, {target})
                for point in white
            ):
                return state, target, white
    raise AssertionError(f"No two independent seam capture groups for {topology.kind}")


def _assert_reference_single_capture():
    with KatagoOracleProcess(x_size=5, y_size=5, komi=0.5) as oracle:
        oracle.setup({
            "black": [[0, 1], [1, 0], [1, 2]],
            "white": [[1, 1]],
            "next_player": "B",
        })
        response = oracle.play([2, 1])
    assert response["ok"] is True
    snapshot = response.get("snapshot", response)
    assert snapshot["captures"] == {"black": 0, "white": 1}
    assert snapshot["board"][1 + 1 * 5] == 0


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_group_connectivity_crosses_seam_or_wrap(kind):
    topology = _topology(kind)
    first, second = _seam_edge(topology)
    state = v3_state_from_board(topology, black=(first, second))
    groups = _all_groups(state.board, topology, BLACK)
    assert any({first, second}.issubset(group) for group in groups)


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_liberty_crosses_seam_or_wrap(kind):
    topology = _topology(kind)
    target, black_point = _seam_edge(topology)
    state = v3_state_from_board(topology, black=(black_point,))
    group, liberties = _collect_group(state.board, black_point, BLACK, topology)
    assert group == {black_point}
    assert target in liberties
    assert v3_valid_moves(state, topology)[target] == 1


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_single_capture_crosses_seam_or_wrap_and_matches_katago(kind):
    topology = _topology(kind)
    state, capture_point, recapture_point = _ko_shape(topology)
    next_state = apply_v3_action(state, capture_point, topology)
    assert next_state.board[recapture_point] == 0
    assert next_state.captures == (1, 0)
    _assert_reference_single_capture()


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_multi_capture_crosses_seam_or_wrap(kind):
    topology = _topology(kind)
    state, target, captured = _multi_capture_state(topology)
    next_state = apply_v3_action(state, target, topology)
    assert all(next_state.board[point] == 0 for point in captured)
    assert next_state.captures == (len(captured), 0)


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_suicide_across_seam_or_wrap_is_illegal(kind):
    topology = _topology(kind)
    target, _ = _seam_edge(topology)
    state = v3_state_from_board(topology, white=topology.neighbor_indices(target))
    assert v3_valid_moves(state, topology)[target] == 0


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_simple_ko_across_seam_or_wrap_is_enforced(kind):
    topology = _topology(kind)
    state, capture_point, recapture_point = _ko_shape(topology)
    captured = apply_v3_action(state, capture_point, topology)
    candidate, _ = _pseudolegal_candidate(
        captured.board, captured.current_player, recapture_point, topology
    )
    assert np.array_equal(candidate, state.board)
    assert v3_valid_moves(captured, topology)[recapture_point] == 0


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_cleanup_ko_across_seam_or_wrap_creates_recap_block(kind):
    topology = _topology(kind)
    state, capture_point, recapture_point = _ko_shape(topology, phase=CLEANUP_1)
    captured = apply_v3_action(state, capture_point, topology)
    assert captured.phase == CLEANUP_1
    assert captured.ko_recap_blocked == (capture_point,)
    assert v3_valid_moves(captured, topology)[recapture_point] == 1


@pytest.mark.parametrize("kind", ("cube", "torus"))
@pytest.mark.parametrize("form", ("blocked-stone", "empty-capture-point"))
def test_pass_for_ko_across_seam_or_wrap_clears_block_without_board_change(kind, form):
    topology = _topology(kind)
    state, capture_point, recapture_point = _ko_shape(topology, phase=CLEANUP_1)
    captured = apply_v3_action(state, capture_point, topology)
    action = capture_point if form == "blocked-stone" else recapture_point
    unblocked = apply_v3_action(captured, action, topology)
    assert np.array_equal(unblocked.board, captured.board)
    assert unblocked.captures == captured.captures
    assert unblocked.ko_recap_blocked == ()
    assert unblocked.ko_unblock_actions == captured.ko_unblock_actions + 1


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_pass_alive_across_seam_or_wrap_has_two_vital_regions(kind):
    topology = cube_topology(5) if kind == "cube" else torus_topology(9)
    if kind == "cube":
        eyes = (
            topology.point_index("front:0:2"),
            topology.point_index("top:4:2"),
            topology.point_index("back:2:2"),
        )
    else:
        eyes = (
            topology.point_index("0,0"),
            topology.point_index("8,0"),
            topology.point_index("4,4"),
        )
    state = v3_state_from_board(
        topology,
        black=(point for point in range(topology.point_count) if point not in eyes),
    )
    analysis = pass_alive_analysis(state.board, topology)
    assert len(analysis.pass_alive_black_groups) == 1
    assert set(eyes).issubset(set(analysis.pass_alive_black_territory))
    assert all_points_pass_alive(state.board, topology)


@pytest.mark.parametrize("kind", ("cube", "torus"))
def test_scoring_region_across_faces_or_wrap_is_counted(kind):
    topology = cube_topology(4) if kind == "cube" else torus_topology(9)
    eyes = (
        (topology.point_index("front:0:0"), topology.point_index("back:2:2"))
        if kind == "cube"
        else (topology.point_index("0,0"), topology.point_index("4,4"))
    )
    state = v3_state_from_board(
        topology,
        black=(point for point in range(topology.point_count) if point not in eyes),
        phase=CLEANUP_2,
    )
    state = replace(state, second_cleanup_start_colors=bytes(state.board.tolist()))
    score, _, _ = final_v3_score(state, topology, 0.0)
    life = independent_life_analysis(state.board, topology)
    assert score.territory.black == 2
    assert score.territory.white == 0
    assert set(eyes) == set(life.black_territory)
