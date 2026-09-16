import pytest

from alphazero.envs.gocube import CUBE_FACES, cube_topology


EDGE_EXPECTATIONS = (
    ("front", "left", "left", "right", False),
    ("front", "right", "right", "left", False),
    ("front", "top", "top", "bottom", False),
    ("front", "bottom", "bottom", "top", False),
    ("back", "left", "right", "right", False),
    ("back", "right", "left", "left", False),
    ("back", "top", "top", "top", True),
    ("back", "bottom", "bottom", "bottom", True),
    ("left", "top", "top", "left", False),
    ("left", "bottom", "bottom", "left", True),
    ("right", "top", "top", "right", True),
    ("right", "bottom", "bottom", "right", False),
)


def point_on_edge(face, edge, index, last):
    if edge == "top":
        return f"{face}:0:{index}"
    if edge == "right":
        return f"{face}:{index}:{last}"
    if edge == "bottom":
        return f"{face}:{last}:{index}"
    if edge == "left":
        return f"{face}:{index}:0"
    raise AssertionError(edge)


def physical_corners(last):
    return (
        (f"front:0:0", f"left:0:{last}", f"top:{last}:0"),
        (f"front:0:{last}", f"right:0:0", f"top:{last}:{last}"),
        (f"front:{last}:0", f"left:{last}:{last}", f"bottom:0:0"),
        (f"front:{last}:{last}", f"right:{last}:0", f"bottom:0:{last}"),
        (f"back:0:{last}", f"left:0:0", f"top:0:0"),
        (f"back:0:0", f"right:0:{last}", f"top:0:{last}"),
        (f"back:{last}:{last}", f"left:{last}:0", f"bottom:{last}:0"),
        (f"back:{last}:0", f"right:{last}:{last}", f"bottom:{last}:{last}"),
    )


@pytest.mark.parametrize("size", (2, 3, 4, 5, 6, 7, 8, 10))
def test_cube_topology_matches_gocube_generic_contract(size):
    topology = cube_topology(size)

    assert len(topology.point_ids) == 6 * size * size
    assert len(set(topology.point_ids)) == 6 * size * size
    for face in CUBE_FACES:
        assert sum(point.startswith(f"{face}:") for point in topology.point_ids) == size * size

    for point_id in topology.point_ids:
        neighbors = topology.neighbor_ids(point_id)
        assert len(neighbors) == 4
        assert len(set(neighbors)) == 4
        assert point_id not in neighbors
        for neighbor in neighbors:
            assert point_id in topology.neighbor_ids(neighbor)


@pytest.mark.parametrize("size", (2, 3, 4, 5, 6, 7, 8, 10))
def test_cube_every_physical_edge_matches_gocube_contract(size):
    topology = cube_topology(size)
    last = size - 1

    for from_face, from_edge, to_face, to_edge, reverse in EDGE_EXPECTATIONS:
        for index in range(size):
            source = point_on_edge(from_face, from_edge, index, last)
            target_index = last - index if reverse else index
            target = point_on_edge(to_face, to_edge, target_index, last)
            assert target in topology.neighbor_ids(source)
            assert source in topology.neighbor_ids(target)


@pytest.mark.parametrize("size", (2, 3, 4, 5, 6, 7, 8, 10))
def test_cube_physical_corner_triplets_match_gocube_contract(size):
    topology = cube_topology(size)
    last = size - 1

    for corner in physical_corners(last):
        for point in corner:
            neighbors = topology.neighbor_ids(point)
            other_corner_points = tuple(candidate for candidate in corner if candidate != point)
            assert len(other_corner_points) == 2
            assert all(candidate in neighbors for candidate in other_corner_points)
            assert sum(candidate in corner for candidate in neighbors) == 2
