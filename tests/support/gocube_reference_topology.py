"""Test-only planar topology used as the KataGo rectangular bridge."""

from __future__ import annotations

from alphazero.envs.gocube.core import Topology


def rectangular_test_topology(width: int, height: int) -> Topology:
    width = int(width)
    height = int(height)
    if width < 1 or height < 1:
        raise ValueError("rectangular test topology dimensions must be positive")
    point_ids = tuple(f"{x},{y}" for y in range(height) for x in range(width))
    index_by_id = {point_id: index for index, point_id in enumerate(point_ids)}
    neighbors = []
    for y in range(height):
        for x in range(width):
            adjacent = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    adjacent.append(ny * width + nx)
            neighbors.append(tuple(adjacent))
    return Topology(
        kind="rectangular-test",
        size=width,
        point_ids=point_ids,
        neighbors_by_index=tuple(neighbors),
        index_by_id=index_by_id,
    )
