"""Deterministic graph-only structural features for GoCube observations.

The feature calculation deliberately knows nothing about faces, rendering, or
the meaning of a PointId.  A topology is the complete input: its canonical
point order and its canonical adjacency table.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


STRUCTURAL_FEATURE_SCHEMA = "gocube-structural-features-v1"
STRUCTURAL_FEATURE_CHANNELS = 2
TRIANGLE_MEMBERSHIP_CHANNEL = 0
DISTANCE_TO_TRIANGLE_CHANNEL = 1


@dataclass(frozen=True)
class StructuralFeatureMetadata:
    """Cached, immutable result of graph structural analysis."""

    triangles: tuple[tuple[int, int, int], ...]
    max_distance: int
    matrix: np.ndarray


def graph_triangles(topology: Any) -> tuple[tuple[int, int, int], ...]:
    """Return each undirected graph triangle once in canonical order."""

    return _cached_metadata(
        topology.kind,
        int(topology.size),
        tuple(topology.point_ids),
        tuple(tuple(int(neighbor) for neighbor in neighbors) for neighbors in topology.neighbors_by_index),
    ).triangles


def structural_feature_metadata(topology: Any) -> StructuralFeatureMetadata:
    """Return cached structural metadata for ``topology``.

    The cache key contains the complete topology rather than only kind/size,
    so a distinct graph cannot accidentally reuse another graph's features.
    """

    return _cached_metadata(
        topology.kind,
        int(topology.size),
        tuple(topology.point_ids),
        tuple(tuple(int(neighbor) for neighbor in neighbors) for neighbors in topology.neighbors_by_index),
    )


def structural_feature_matrix(topology: Any) -> np.ndarray:
    """Return ``(2, point_count, 1)`` float32 structural channels.

    The public result is a copy so callers cannot mutate the topology-level
    cache.  The cached matrix itself is read-only and deterministic.
    """

    return structural_feature_metadata(topology).matrix.copy()


def triangle_membership(topology: Any) -> np.ndarray:
    """Return binary graph-triangle membership in canonical point order."""

    return structural_feature_matrix(topology)[TRIANGLE_MEMBERSHIP_CHANNEL, :, 0]


def distance_to_vertex_triangle(topology: Any) -> np.ndarray:
    """Return normalized shortest-path distance to the triangle point set."""

    return structural_feature_matrix(topology)[DISTANCE_TO_TRIANGLE_CHANNEL, :, 0]


def _topology_adjacency(point_ids, neighbors):
    point_count = len(point_ids)
    if len(neighbors) != point_count:
        raise ValueError(
            "Structural feature topology point count mismatch: "
            f"point_ids={point_count}, adjacency={len(neighbors)}"
        )
    adjacency = []
    for point, point_neighbors in enumerate(neighbors):
        values = tuple(int(neighbor) for neighbor in point_neighbors)
        if point in values:
            raise ValueError(f"Structural feature adjacency contains self-edge at point {point}")
        if len(values) != len(set(values)):
            raise ValueError(f"Structural feature adjacency contains duplicate neighbor at point {point}")
        if any(neighbor < 0 or neighbor >= point_count for neighbor in values):
            raise ValueError(f"Structural feature adjacency contains out-of-range neighbor at point {point}")
        adjacency.append(frozenset(values))
    for point, point_neighbors in enumerate(adjacency):
        for neighbor in point_neighbors:
            if point not in adjacency[neighbor]:
                raise ValueError(
                    "Structural feature adjacency must be undirected: "
                    f"{point} -> {neighbor} has no reverse edge"
                )
    return tuple(adjacency)


@lru_cache(maxsize=None)
def _cached_metadata(kind, size, point_ids, neighbors) -> StructuralFeatureMetadata:
    del kind, size  # They remain in the cache key by design.
    adjacency = _topology_adjacency(point_ids, neighbors)
    point_count = len(adjacency)

    triangles: list[tuple[int, int, int]] = []
    for first in range(point_count):
        for second in sorted(adjacency[first]):
            if second <= first:
                continue
            for third in sorted(adjacency[first].intersection(adjacency[second])):
                if third > second:
                    triangles.append((first, second, third))
    triangle_tuple = tuple(triangles)

    matrix = np.zeros((STRUCTURAL_FEATURE_CHANNELS, point_count, 1), dtype=np.float32)
    if not triangle_tuple:
        matrix.flags.writeable = False
        return StructuralFeatureMetadata(triangle_tuple, 0, matrix)

    triangle_points = sorted({point for triangle in triangle_tuple for point in triangle})
    matrix[TRIANGLE_MEMBERSHIP_CHANNEL, triangle_points, 0] = 1.0

    distances = [-1] * point_count
    pending = deque(triangle_points)
    for point in triangle_points:
        distances[point] = 0
    while pending:
        point = pending.popleft()
        for neighbor in sorted(adjacency[point]):
            if distances[neighbor] >= 0:
                continue
            distances[neighbor] = distances[point] + 1
            pending.append(neighbor)

    if any(distance < 0 for distance in distances):
        raise ValueError("Structural feature topology has points disconnected from triangle set")
    max_distance = max(distances, default=0)
    denominator = max(1, max_distance)
    matrix[DISTANCE_TO_TRIANGLE_CHANNEL, :, 0] = np.asarray(
        distances, dtype=np.float32
    ) / float(denominator)
    if not np.isfinite(matrix).all():
        raise ValueError("Structural feature matrix contains non-finite values")
    matrix.flags.writeable = False
    return StructuralFeatureMetadata(triangle_tuple, max_distance, matrix)


# Names used by callers that prefer the shorter feature terminology.
compute_structural_features = structural_feature_matrix
find_graph_triangles = graph_triangles
