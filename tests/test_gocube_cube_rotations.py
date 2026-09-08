from __future__ import annotations

from dataclasses import replace

import pytest

from alphazero.envs.gocube.core import BLACK, cube_topology

from tests.support.fixtures import cube_verification_fixtures
from tests.support.independent_graph import apply_move, find_group, graph_triangles
from tests.support.ko import prove_positional_restoration
from tests.support.rotations import cube_rotations, rotate_board, rotate_board_state, rotate_fixture, rotate_point_set


def _fixture(fixture_id):
    return next(item for item in cube_verification_fixtures() if item.id == fixture_id)


@pytest.mark.parametrize("size", (2, 3, 4, 5, 6, 7))
def test_cube_has_exactly_24_bijective_adjacency_preserving_rotations(size):
    topology = cube_topology(size)
    rotations = cube_rotations(topology)
    assert len(rotations) == 24
    assert len({rotation.permutation for rotation in rotations}) == 24
    identity = tuple(range(topology.point_count))
    assert identity in {rotation.permutation for rotation in rotations}
    for rotation in rotations:
        assert sorted(rotation.permutation) == list(range(topology.point_count))
        inverse = rotation.inverse()
        assert rotation.compose(inverse) == identity
        assert inverse.compose(rotation) == identity
        for point, neighbors in enumerate(topology.neighbors_by_index):
            assert {
                rotation.apply_point(neighbor) for neighbor in neighbors
            } == set(topology.neighbors_by_index[rotation.apply_point(point)])


def test_cube4_rotations_form_a_group_and_preserve_vertex_triangles():
    topology = cube_topology(4)
    rotations = cube_rotations(topology)
    permutations = {rotation.permutation for rotation in rotations}
    triangles = {frozenset(triangle) for triangle in graph_triangles(topology.neighbors_by_index)}
    for left in rotations:
        for right in rotations:
            assert left.compose(right) in permutations
        mapped = {
            frozenset(left.apply_point(point) for point in triangle)
            for triangle in triangles
        }
        assert mapped == triangles


def test_rotation_maps_group_liberties_and_capture_fixture():
    topology = cube_topology(4)
    fixture = _fixture("cube4_vertex_capture_001")
    board = fixture.board(topology.index_by_id)
    action = topology.point_index(fixture.actions[0])
    original = apply_move(board, BLACK, action, topology.neighbors_by_index)
    for rotation in cube_rotations(topology):
        rotated_fixture = rotate_fixture(fixture, topology, rotation)
        rotated_board = rotated_fixture.board(topology.index_by_id)
        rotated_action = topology.point_index(rotated_fixture.actions[0])
        rotated = apply_move(rotated_board, BLACK, rotated_action, topology.neighbors_by_index)
        assert rotate_point_set(original.captured_points, rotation) == rotated.captured_points
        assert rotate_point_set(original.own_group.stones, rotation) == rotated.own_group.stones
        assert rotate_point_set(original.own_group.liberties, rotation) == rotated.own_group.liberties


def test_rotation_maps_true_and_false_ko_proofs_without_partial_state_rotation():
    topology = cube_topology(4)
    for fixture_id, expected in (
        ("cube4_false_simple_ko_001", False),
        ("cube4_true_simple_ko_001", True),
    ):
        fixture = _fixture(fixture_id)
        before = fixture.board(topology.index_by_id)
        capture_action = topology.point_index(fixture.actions[0])
        recapture_action = topology.point_index(fixture.actions[1])
        baseline = prove_positional_restoration(
            before, BLACK, capture_action, recapture_action, topology.neighbors_by_index
        )
        assert baseline.is_simple_ko is expected
        for rotation in cube_rotations(topology):
            rotated = rotate_fixture(fixture, topology, rotation)
            proof = prove_positional_restoration(
                rotated.board(topology.index_by_id),
                BLACK,
                topology.point_index(rotated.actions[0]),
                topology.point_index(rotated.actions[1]),
                topology.neighbors_by_index,
            )
            assert proof.is_simple_ko is expected


def test_state_rotation_moves_previous_board_masks_and_cleanup_start_colors():
    topology = cube_topology(4)
    fixture = _fixture("cube4_cleanup2_pass_for_ko_001")
    board = fixture.board(topology.index_by_id)
    state = replace(
        fixture,
        previous_board=board,
        second_cleanup_start_colors=board,
        ko_state={"blocked_points": ["front:0:0"]},
    )
    rotation = cube_rotations(topology)[7]
    rotated = rotate_fixture(state, topology, rotation)
    assert rotated.previous_board == rotate_board(board, rotation)
    expected_blocked = topology.point_id(rotation.apply_point(topology.point_index("front:0:0")))
    assert rotated.ko_state["blocked_points"] == [expected_blocked]
    assert rotated.second_cleanup_start_colors == rotate_board_state(
        board=board,
        rotation=rotation,
        start_colors=board,
    )["start_colors"]
