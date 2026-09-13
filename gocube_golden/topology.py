from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence


TORUS_5X5_TOPOLOGY_ID = "torus-5x5-row-major-v1"
TORUS_5X5_TOPOLOGY_FINGERPRINT = (
    "sha256:b4097c32d4ab5034b84300fa41f951b353a5fcf0d8226e83922889b5552289ef"
)

TORUS_9X9_TOPOLOGY_ID = "torus-9x9-row-major-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class GoldenTopology:
    """Small immutable graph topology used only by the Golden reference line."""

    topology_id: str
    adjacency: tuple[tuple[int, ...], ...]
    fingerprint: str
    kind: str = "research-graph"
    dimensions: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        point_count = len(self.adjacency)
        if not self.topology_id:
            raise ValueError("Golden topology_id must be non-empty")
        if point_count == 0:
            raise ValueError("Golden topology must contain at least one point")
        for point, neighbors in enumerate(self.adjacency):
            if len(set(neighbors)) != len(neighbors):
                raise ValueError(f"Point {point} contains duplicate neighbors")
            if point in neighbors:
                raise ValueError(f"Point {point} contains a self-loop")
            for neighbor in neighbors:
                if isinstance(neighbor, bool) or not isinstance(neighbor, int):
                    raise ValueError(f"Neighbor {neighbor!r} of point {point} is not an integer PointId")
                if not 0 <= neighbor < point_count:
                    raise ValueError(f"Neighbor {neighbor} of point {point} is outside the topology")
        for point, neighbors in enumerate(self.adjacency):
            for neighbor in neighbors:
                if point not in self.adjacency[neighbor]:
                    raise ValueError(f"Golden topology edge {point}<->{neighbor} is not reciprocal")

    @property
    def point_count(self) -> int:
        return len(self.adjacency)

    def neighbors(self, point: int) -> tuple[int, ...]:
        return self.adjacency[point]

    @property
    def width(self) -> int | None:
        return self.dimensions[0] if self.dimensions is not None else None

    @property
    def height(self) -> int | None:
        return self.dimensions[1] if self.dimensions is not None else None

    def coordinate_to_point(self, x: int, y: int) -> int:
        if self.dimensions is None:
            raise ValueError("This research topology has no coordinate system")
        width, height = self.dimensions
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, int) or not isinstance(y, int):
            raise ValueError("Torus coordinates must be integer values")
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"Coordinate ({x},{y}) is outside 0..{width - 1} x 0..{height - 1}")
        return y * width + x

    def point_to_coordinate(self, point: int) -> tuple[int, int]:
        if self.dimensions is None:
            raise ValueError("This research topology has no coordinate system")
        if isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < self.point_count:
            raise ValueError(f"PointId {point!r} is outside the topology")
        width, _ = self.dimensions
        return point % width, point // width


def _torus_5x5_adjacency() -> tuple[tuple[int, int, int, int], ...]:
    width = height = 5
    rows: list[tuple[int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            north = ((y - 1) % height) * width + x
            east = y * width + ((x + 1) % width)
            south = ((y + 1) % height) * width + x
            west = y * width + ((x - 1) % width)
            rows.append((north, east, south, west))
    return tuple(rows)


def _stage0_torus_identity_payload() -> dict[str, object]:
    return {
        "topology_id": TORUS_5X5_TOPOLOGY_ID,
        "kind": "torus",
        "width": 5,
        "height": 5,
        "number_of_points": 25,
        "wrap_x": True,
        "wrap_y": True,
        "undirected": True,
        "degree": 4,
        "self_loops": False,
        "duplicate_neighbors": False,
        "point_order": "row-major-yx:point_id=y*width+x",
        "neighbor_order": ["N", "E", "S", "W"],
        "neighbors_by_point_id": [list(row) for row in _torus_5x5_adjacency()],
    }


def torus_5x5() -> GoldenTopology:
    """Return the one canonical Golden Torus 5x5 topology from Stage 0.

    This is deliberately *not* a general Torus factory.  Stage 1 does not
    extend or call the production topology factory.
    """

    computed = _fingerprint(_stage0_torus_identity_payload())
    if computed != TORUS_5X5_TOPOLOGY_FINGERPRINT:
        raise RuntimeError(
            "Golden Torus 5x5 topology drifted from the Stage-0 fingerprint: "
            f"expected {TORUS_5X5_TOPOLOGY_FINGERPRINT}, got {computed}"
        )
    return GoldenTopology(
        topology_id=TORUS_5X5_TOPOLOGY_ID,
        adjacency=_torus_5x5_adjacency(),
        fingerprint=computed,
        kind="torus",
        dimensions=(5, 5),
    )


def _torus_adjacency(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    if width < 3 or height < 3:
        raise ValueError("A simple four-neighbor torus requires width and height >= 3")
    rows: list[tuple[int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            rows.append(
                (
                    ((y - 1) % height) * width + x,
                    y * width + ((x + 1) % width),
                    ((y + 1) % height) * width + x,
                    y * width + ((x - 1) % width),
                )
            )
    return tuple(rows)


def _torus_identity_payload(topology_id: str, width: int, height: int) -> dict[str, object]:
    return {
        "topology_id": topology_id,
        "kind": "torus",
        "width": width,
        "height": height,
        "number_of_points": width * height,
        "wrap_x": True,
        "wrap_y": True,
        "undirected": True,
        "degree": 4,
        "self_loops": False,
        "duplicate_neighbors": False,
        "point_order": "row-major-yx:point_id=y*width+x",
        "neighbor_order": ["N", "E", "S", "W"],
        "neighbors_by_point_id": [list(row) for row in _torus_adjacency(width, height)],
    }


def torus_9x9() -> GoldenTopology:
    """Return the standalone Golden Torus 9×9 topology."""
    adjacency = _torus_adjacency(9, 9)
    fingerprint = _fingerprint(_torus_identity_payload(TORUS_9X9_TOPOLOGY_ID, 9, 9))
    return GoldenTopology(
        topology_id=TORUS_9X9_TOPOLOGY_ID,
        adjacency=adjacency,
        fingerprint=fingerprint,
        kind="torus",
        dimensions=(9, 9),
    )


def research_topology(
    adjacency: Sequence[Iterable[int]], *, topology_id: str = "golden-research-graph"
) -> GoldenTopology:
    """Construct a tiny explicit graph for tests/research only.

    This helper exists for Stage-1 fixtures and graph-isomorphism checks.  It
    is intentionally outside every production topology registry/factory.
    """

    canonical = tuple(tuple(int(neighbor) for neighbor in row) for row in adjacency)
    payload = {
        "topology_id": topology_id,
        "kind": "research-graph",
        "adjacency": [list(row) for row in canonical],
    }
    return GoldenTopology(
        topology_id=topology_id,
        adjacency=canonical,
        fingerprint=_fingerprint(payload),
        kind="research-graph",
    )


def permute_topology(
    topology: GoldenTopology,
    old_to_new: Sequence[int],
    *, topology_id: str = "golden-permuted-research-graph",
) -> GoldenTopology:
    if len(old_to_new) != topology.point_count:
        raise ValueError("Permutation length must equal topology point count")
    if set(old_to_new) != set(range(topology.point_count)):
        raise ValueError("old_to_new must be a PointId permutation")

    adjacency: list[tuple[int, ...] | None] = [None] * topology.point_count
    for old_point, old_neighbors in enumerate(topology.adjacency):
        new_point = old_to_new[old_point]
        adjacency[new_point] = tuple(old_to_new[n] for n in old_neighbors)
    return research_topology(
        tuple(row for row in adjacency if row is not None), topology_id=topology_id
    )


TORUS_5X5 = torus_5x5()
TORUS_9X9 = torus_9x9()
TORUS_9X9_TOPOLOGY_FINGERPRINT = TORUS_9X9.fingerprint
