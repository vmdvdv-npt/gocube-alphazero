from __future__ import annotations

import pytest

from alphazero.envs.gocube.core import cube_topology as production_cube_topology
from gocube_golden.cube_topology import (
    CROSS_FACE_SEAM,
    CUBE4_ROTATION_PERMUTATIONS,
    CUBE4_TOPOLOGY,
    FACE_CORNER,
    FACE_EDGE,
    FACE_INTERIOR,
    SAME_FACE,
    geometry_class_counts,
)


def test_golden_cube_has_independent_canonical_identity_and_invariants():
    topology = CUBE4_TOPOLOGY
    assert topology.topology_id == "cube4x4x6-golden-topology-v1"
    assert topology.point_count == 96
    assert topology.action_size == 97
    assert all(len(row) == 4 for row in topology.adjacency)
    assert geometry_class_counts(topology) == {
        FACE_INTERIOR: 24,
        FACE_EDGE: 48,
        FACE_CORNER: 24,
    }
    assert topology.audit()["degree_distribution"] == {"4": 96}
    assert topology.audit()["cross_face_adjacency_pairs"] == 48
    assert len(topology.seams) == 12
    assert topology.fingerprint.startswith("sha256:")
    assert topology.geometry_fingerprint.startswith("sha256:")
    assert topology.point_ordering_fingerprint.startswith("sha256:")


def test_production_cube_is_only_a_diagnostic_cross_check():
    golden = CUBE4_TOPOLOGY
    production = production_cube_topology(4)
    assert tuple(production.point_ids) == golden.point_ids
    assert tuple(sorted(row) for row in production.neighbors_by_index) == tuple(sorted(row) for row in golden.adjacency)
    assert all(
        relation in (SAME_FACE, CROSS_FACE_SEAM)
        for row in golden.relation_types
        for relation in row
    )


def test_physical_corner_mapping_covers_all_face_corner_cells_once():
    topology = CUBE4_TOPOLOGY
    corners = topology.physical_corners
    assert len(corners) == 8
    assert all(len(corner) == 3 and len(set(corner)) == 3 for corner in corners)
    assert all(len({topology.geometry(point).face_id for point in corner}) == 3 for corner in corners)
    corner_points = [point.point_id for point in topology.points if point.geometry_class == FACE_CORNER]
    assert sorted(point for corner in corners for point in corner) == sorted(corner_points)
    assert all(topology.geometry(point).physical_corner_id is not None for point in corner_points)


def test_every_seam_maps_four_points_reciprocally_without_diagonals():
    topology = CUBE4_TOPOLOGY
    assert all(len(seam.point_pairs) == 4 for seam in topology.seams)
    for seam in topology.seams:
        for left, right in seam.point_pairs:
            assert topology.relation(left, right) == CROSS_FACE_SEAM
            assert topology.relation(right, left) == CROSS_FACE_SEAM
            assert left in topology.adjacency[right]
            assert right in topology.adjacency[left]


def test_corner_distance_is_independently_observable_and_bounded():
    topology = CUBE4_TOPOLOGY
    assert all(topology.geometry(point).corner_distance == 0 for corner in topology.physical_corners for point in corner)
    assert {topology.geometry(point).corner_distance_bucket for point in range(96)} <= {
        "corner_distance_0", "corner_distance_1", "corner_distance_2", "corner_distance_3_plus"
    }
    assert all(topology.geometry(point).num_cross_face_neighbors == sum(
        relation == CROSS_FACE_SEAM for relation in topology.relation_types[point]
    ) for point in range(96))


def test_cube_rotation_permutations_are_adjacency_and_corner_consistent():
    topology = CUBE4_TOPOLOGY
    assert len(CUBE4_ROTATION_PERMUTATIONS) == 24
    for permutation in CUBE4_ROTATION_PERMUTATIONS:
        assert sorted(permutation) == list(range(96))
        for point, neighbors in enumerate(topology.adjacency):
            assert {permutation[neighbor] for neighbor in neighbors} == set(topology.adjacency[permutation[point]])
        for corner in topology.physical_corners:
            mapped = {permutation[point] for point in corner}
            assert mapped in [set(item) for item in topology.physical_corners]


@pytest.mark.parametrize("point", (0, 3, 12, 15, 16, 19, 31, 48, 64, 80))
def test_point_metadata_is_total_and_has_valid_identity(point):
    metadata = CUBE4_TOPOLOGY.geometry(point)
    assert metadata.point_id == point
    assert metadata.face_id in {"front", "back", "left", "right", "top", "bottom"}
    assert metadata.geometry_class in {FACE_INTERIOR, FACE_EDGE, FACE_CORNER}
    assert metadata.corner_distance >= 0
    assert metadata.num_cross_face_neighbors in {0, 1, 2}
