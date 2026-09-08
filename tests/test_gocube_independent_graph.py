from __future__ import annotations

import ast
from pathlib import Path

import pytest

from alphazero.envs.gocube.core import cube_topology, torus_topology

from tests.support.fixtures import cube_verification_fixtures, fixture_counts
from tests.support.independent_graph import (
    BLACK,
    WHITE,
    IndependentIllegalMove,
    apply_move,
    empty_regions,
    find_group,
    find_groups,
    graph_triangles,
    triangle_membership,
)


def _fixture(fixture_id):
    return next(item for item in cube_verification_fixtures() if item.id == fixture_id)


def test_independent_group_counts_unique_vertex_and_shared_liberties():
    topology = cube_topology(4)
    vertex = _fixture("cube4_vertex_single_group_001")
    group = find_group(
        topology.point_index("front:0:0"),
        vertex.board(topology.index_by_id),
        topology.neighbors_by_index,
    )
    assert group.color == BLACK
    assert {topology.point_id(point) for point in group.stones} == set(vertex.expected["group"])
    assert len(group.liberties) == vertex.expected["liberty_count"] == 5

    inner = _fixture("cube4_inner_shared_liberty_001")
    inner_group = find_group(
        topology.point_index("front:1:1"),
        inner.board(topology.index_by_id),
        topology.neighbors_by_index,
    )
    assert len(inner_group.stones) == 2
    assert len(inner_group.liberties) == inner.expected["liberty_count"] == 6


def test_independent_groups_distinguish_seam_neighbor_from_visual_near_miss():
    topology = cube_topology(4)
    positive = _fixture("cube4_vertex_three_face_group_001")
    negative = _fixture("cube4_vertex_near_miss_001")
    assert len(find_groups(positive.board(topology.index_by_id), topology.neighbors_by_index)) == 1
    assert len(find_groups(negative.board(topology.index_by_id), topology.neighbors_by_index)) == 2
    front = topology.point_index("front:0:0")
    left = topology.point_index("left:0:3")
    assert left in topology.neighbors_by_index[front]
    assert topology.point_index("left:0:2") not in topology.neighbors_by_index[topology.point_index("front:0:1")]


def test_independent_capture_is_capture_before_suicide():
    topology = cube_topology(4)
    fixture = _fixture("cube4_vertex_capture_before_suicide_001")
    result = apply_move(
        fixture.board(topology.index_by_id),
        BLACK,
        topology.point_index(fixture.actions[0]),
        topology.neighbors_by_index,
    )
    assert result.capture_count == fixture.expected["capture_count"] == 4
    assert len(result.captured_groups) == fixture.expected["captured_group_count"] == 3
    assert len(result.own_group.liberties) == fixture.expected["own_liberties_after"] == 4


def test_independent_capture_handles_multiple_neighbor_groups_once():
    topology = cube_topology(4)
    fixture = _fixture("cube4_vertex_multiple_neighbor_capture_001")
    result = apply_move(
        fixture.board(topology.index_by_id),
        BLACK,
        topology.point_index(fixture.actions[0]),
        topology.neighbors_by_index,
    )
    assert result.capture_count == 4
    assert len(result.captured_groups) == 4
    assert len(set().union(*(set(group.stones) for group in result.captured_groups))) == 4


def test_capture_negative_control_is_suicide():
    topology = cube_topology(4)
    fixture = _fixture("cube4_vertex_suicide_control_001")
    with pytest.raises(IndependentIllegalMove, match="suicide"):
        apply_move(
            fixture.board(topology.index_by_id),
            WHITE,
            topology.point_index(fixture.actions[0]),
            topology.neighbors_by_index,
        )


def test_empty_regions_are_graph_components_and_work_on_torus():
    cube = cube_topology(4)
    regions = empty_regions((0,) * cube.point_count, cube.neighbors_by_index)
    assert len(regions) == 1
    assert len(regions[0].points) == cube.point_count

    torus = torus_topology(9)
    torus_regions = empty_regions((0,) * torus.point_count, torus.neighbors_by_index)
    assert len(torus_regions) == 1
    assert not graph_triangles(torus.neighbors_by_index)

    board = list((0,) * cube.point_count)
    board[cube.point_index("front:1:1")] = BLACK
    board[cube.point_index("back:1:1")] = WHITE
    bordered = empty_regions(board, cube.neighbors_by_index)[0]
    assert bordered.bordering_black_groups
    assert bordered.bordering_white_groups


def test_cube4_has_eight_graph_vertex_triangles_and_24_participating_points():
    topology = cube_topology(4)
    triangles = graph_triangles(topology.neighbors_by_index)
    membership = triangle_membership(triangles)
    assert len(triangles) == 8
    assert sum(membership) == 24
    assert sum(value > 0 for value in membership) == 24
    assert set(value for value in membership if value) == {1}


def test_support_layer_has_no_renderer_camera_or_ui_imports():
    forbidden = {"renderer", "camera", "AlphaZeroGUI", "pygame", "cv2"}
    support_root = Path(__file__).parent / "support"
    for path in (support_root / "independent_graph.py", support_root / "rotations.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert imported.isdisjoint(forbidden)


def test_corpus_has_stable_schema_and_multiple_families():
    fixtures = cube_verification_fixtures()
    assert len(fixtures) >= 18
    counts = fixture_counts(fixtures)
    assert {"vertex_groups", "captures", "eyes", "seki_dame", "ko", "seam_tactics", "global_connectivity", "cleanup", "early_termination"} <= set(counts)
    for fixture in fixtures:
        data = fixture.to_dict()
        for field in ("id", "family", "topology_kind", "size", "initial_position", "to_move", "actions", "expected", "oracle/source", "rotation_policy", "notes"):
            assert field in data
        assert fixture.oracle in {"native_katago", "independent_graph", "exhaustive_solver", "metamorphic", "product_fixture_pending", "hand_proved_invariant", "structural-only"}
