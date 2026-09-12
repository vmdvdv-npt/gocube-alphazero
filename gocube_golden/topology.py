from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence


TORUS_5X5_TOPOLOGY_ID = "torus-5x5-row-major-v1"
TORUS_5X5_TOPOLOGY_FINGERPRINT = (
    "sha256:b4097c32d4ab5034b84300fa41f951b353a5fcf0d8226e83922889b5552289ef"
)


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
