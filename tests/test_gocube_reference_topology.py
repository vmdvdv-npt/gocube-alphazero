from __future__ import annotations

import pytest

from gocube_reference_topology import rectangular_test_topology


@pytest.mark.parametrize("width,height", [(3, 3), (5, 3), (5, 5), (7, 4)])
def test_rectangular_topology_is_planar_four_neighbor_graph(width, height):
    topology = rectangular_test_topology(width, height)
    assert topology.point_count == width * height
    assert topology.pass_action == width * height
    for point in range(topology.point_count):
        neighbors = topology.neighbor_indices(point)
        assert len(neighbors) == len(set(neighbors))
        assert point not in neighbors
        assert all(0 <= neighbor < topology.point_count for neighbor in neighbors)
        for neighbor in neighbors:
            assert point in topology.neighbor_indices(neighbor)


def test_rectangular_topology_uses_row_major_indices_without_wrap():
    topology = rectangular_test_topology(5, 3)
    assert topology.point_index("0,0") == 0
    assert topology.point_index("4,0") == 4
    assert topology.point_index("0,1") == 5
    assert topology.neighbor_indices(0) == (1, 5)
    assert topology.neighbor_indices(4) == (3, 9)
    assert topology.neighbor_indices(14) == (13, 9)
