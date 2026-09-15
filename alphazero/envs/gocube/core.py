"""Protocol-only GoCube geometry.

The Golden rules/state implementation lives in :mod:`gocube_golden`.  This
module contains no game state, move legality, scoring, or execution code; it
only describes the stable PointId adjacency used to audit the Protocol V1
mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


TORUS_SIZES = (9, 13, 19)
CUBE_FACES = ("front", "back", "left", "right", "top", "bottom")


@dataclass(frozen=True)
class Topology:
    kind: str
    size: int
    point_ids: tuple[str, ...]
    neighbors_by_index: tuple[tuple[int, ...], ...]
    index_by_id: Mapping[str, int]

    @property
    def point_count(self) -> int:
        return len(self.point_ids)

    @property
    def pass_action(self) -> int:
        return self.point_count

    @property
    def action_size(self) -> int:
        return self.point_count + 1

    def point_index(self, point_id: str) -> int:
        return self.index_by_id[point_id]

    def point_id(self, point_index: int) -> str:
        return self.point_ids[point_index]

    def neighbor_indices(self, point_index: int) -> tuple[int, ...]:
        return self.neighbors_by_index[point_index]

    def neighbor_ids(self, point_id: str) -> tuple[str, ...]:
        return tuple(
            self.point_ids[index]
            for index in self.neighbor_indices(self.point_index(point_id))
        )


def torus_topology(size: int) -> Topology:
    if size not in TORUS_SIZES:
        raise ValueError(f"Unsupported torus size: {size}; expected one of {TORUS_SIZES}")

    point_ids = tuple(f"{x},{y}" for y in range(size) for x in range(size))
    index_by_id = {point_id: index for index, point_id in enumerate(point_ids)}

    def index(x: int, y: int) -> int:
        return (y % size) * size + (x % size)

    neighbors = []
    for y in range(size):
        for x in range(size):
            neighbors.append(
                (
                    index(x - 1, y),
                    index(x + 1, y),
                    index(x, y - 1),
                    index(x, y + 1),
                )
            )
    return Topology("torus", size, point_ids, tuple(neighbors), index_by_id)


_EDGE_TRANSITIONS = {
    "front": {
        "top": ("top", "bottom", False),
        "right": ("right", "left", False),
        "bottom": ("bottom", "top", False),
        "left": ("left", "right", False),
    },
    "back": {
        "top": ("top", "top", True),
        "right": ("left", "left", False),
        "bottom": ("bottom", "bottom", True),
        "left": ("right", "right", False),
    },
    "left": {
        "top": ("top", "left", False),
        "right": ("front", "left", False),
        "bottom": ("bottom", "left", True),
        "left": ("back", "right", False),
    },
    "right": {
        "top": ("top", "right", True),
        "right": ("back", "left", False),
        "bottom": ("bottom", "right", False),
        "left": ("front", "right", False),
    },
    "top": {
        "top": ("back", "top", True),
        "right": ("right", "top", True),
        "bottom": ("front", "top", False),
        "left": ("left", "top", False),
    },
    "bottom": {
        "top": ("front", "bottom", False),
        "right": ("right", "bottom", False),
        "bottom": ("back", "bottom", True),
        "left": ("left", "bottom", True),
    },
}


def cube_topology(size: int) -> Topology:
    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        raise ValueError(f"Cube size must be an integer >= 2, got {size!r}")

    point_ids = tuple(
        f"{face}:{row}:{column}"
        for face in CUBE_FACES
        for row in range(size)
        for column in range(size)
    )
    index_by_id = {point_id: index for index, point_id in enumerate(point_ids)}
    last = size - 1

    def point_id(face: str, row: int, column: int) -> str:
        return f"{face}:{row}:{column}"

    def point_on_edge(face: str, edge: str, edge_index: int) -> str:
        if edge == "top":
            return point_id(face, 0, edge_index)
        if edge == "right":
            return point_id(face, edge_index, last)
        if edge == "bottom":
            return point_id(face, last, edge_index)
        if edge == "left":
            return point_id(face, edge_index, 0)
        raise AssertionError(edge)

    def cross(face: str, edge: str, edge_index: int) -> str:
        target_face, target_edge, reverse = _EDGE_TRANSITIONS[face][edge]
        target_index = last - edge_index if reverse else edge_index
        return point_on_edge(target_face, target_edge, target_index)

    neighbors = []
    for face in CUBE_FACES:
        for row in range(size):
            for column in range(size):
                top = point_id(face, row - 1, column) if row > 0 else cross(face, "top", column)
                right = point_id(face, row, column + 1) if column < last else cross(face, "right", row)
                bottom = point_id(face, row + 1, column) if row < last else cross(face, "bottom", column)
                left = point_id(face, row, column - 1) if column > 0 else cross(face, "left", row)
                neighbors.append(tuple(index_by_id[item] for item in (top, right, bottom, left)))
    return Topology("cube", size, point_ids, tuple(neighbors), index_by_id)


def make_topology(kind: str, size: int) -> Topology:
    if kind == "torus":
        return torus_topology(size)
    if kind == "cube":
        return cube_topology(size)
    raise ValueError(f"Unknown topology kind: {kind}")


__all__ = [
    "CUBE_FACES",
    "TORUS_SIZES",
    "Topology",
    "cube_topology",
    "make_topology",
    "torus_topology",
]
