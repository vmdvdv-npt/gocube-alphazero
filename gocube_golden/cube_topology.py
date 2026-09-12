"""Independent, geometry-derived Golden topology for Cube 4x4.

The production GoCube topology is intentionally not imported here.  A face is
described by an outward normal and two oriented in-face axes.  Seam mappings
are derived by folding those frames over a cube edge; the resulting graph is
then audited before it is exposed to Golden rules or neural code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

Vector = tuple[int, int, int]
PointClass = str

FACE_INTERIOR = "FACE_INTERIOR"
FACE_EDGE = "FACE_EDGE"
FACE_CORNER = "FACE_CORNER"
SAME_FACE = "SAME_FACE"
CROSS_FACE_SEAM = "CROSS_FACE_SEAM"

CUBE_FACES = ("front", "back", "left", "right", "top", "bottom")
EDGE_NAMES = ("top", "right", "bottom", "left")
TOPOLOGY_VERSION = 1
GEOMETRY_SCHEMA_ID = "cube4x4x6-golden-geometry-v1"
CUBE4_TOPOLOGY_ID = "cube4x4x6-golden-topology-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _neg(vector: Vector) -> Vector:
    return tuple(-value for value in vector)  # type: ignore[return-value]


def _add(*vectors: Vector) -> Vector:
    return tuple(sum(vector[index] for vector in vectors) for index in range(3))  # type: ignore[return-value]


def _determinant(a: Vector, b: Vector, c: Vector) -> int:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@dataclass(frozen=True)
class CubeFaceFrame:
    face_id: str
    normal: Vector
    row_axis: Vector
    col_axis: Vector


# Local row/column orientation is part of the canonical point ordering.  The
# orientation is chosen so the independent derivation agrees with the
# conventional front/back/left/right/top/bottom naming, while the builder
# below never reads the production transition table.
FACE_FRAMES: tuple[CubeFaceFrame, ...] = (
    CubeFaceFrame("front", (0, 0, 1), (0, 1, 0), (1, 0, 0)),
    CubeFaceFrame("back", (0, 0, -1), (0, 1, 0), (-1, 0, 0)),
    CubeFaceFrame("left", (-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    CubeFaceFrame("right", (1, 0, 0), (0, 1, 0), (0, 0, -1)),
    CubeFaceFrame("top", (0, -1, 0), (0, 0, 1), (1, 0, 0)),
    CubeFaceFrame("bottom", (0, 1, 0), (0, 0, -1), (1, 0, 0)),
)
_FRAME_BY_FACE = {frame.face_id: frame for frame in FACE_FRAMES}


@dataclass(frozen=True)
class CubePointGeometry:
    point_id: int
    face_id: str
    local_row: int
    local_col: int
    geometry_class: PointClass
    physical_corner_id: int | None
    incident_physical_cube_edges: tuple[str, ...]
    corner_distance: int
    corner_distance_bucket: str
    has_cross_face_neighbor: bool
    num_cross_face_neighbors: int


@dataclass(frozen=True)
class CubeSeam:
    seam_id: str
    face_a: str
    edge_a: str
    face_b: str
    edge_b: str
    reversed_index: bool
    point_pairs: tuple[tuple[int, int], ...]


class CubeGoldenTopology:
    """Immutable Cube surface graph plus independently derived geometry."""

    def __init__(
        self,
        *,
        size: int,
        topology_id: str,
        adjacency: tuple[tuple[int, ...], ...],
        relation_types: tuple[tuple[str, ...], ...],
        points: tuple[CubePointGeometry, ...],
        seams: tuple[CubeSeam, ...],
        physical_corners: tuple[tuple[int, int, int], ...],
        face_frames: tuple[CubeFaceFrame, ...],
    ) -> None:
        self.size = int(size)
        self.topology_id = topology_id
        self.adjacency = adjacency
        self.relation_types = relation_types
        self.points = points
        self.seams = seams
        self.physical_corners = physical_corners
        self.face_frames = face_frames
        self.point_ids = tuple(
            f"{point.face_id}:{point.local_row}:{point.local_col}" for point in points
        )
        self.index_by_id = {point_id: index for index, point_id in enumerate(self.point_ids)}
        self._validate()
        self.fingerprint = _fingerprint(self._topology_payload())
        self.geometry_fingerprint = _fingerprint(self._geometry_payload())
        self.point_ordering_fingerprint = _fingerprint(
            {"point_order": list(self.point_ids), "action_order": list(range(self.action_size))}
        )

    @property
    def topology_version(self) -> int:
        return TOPOLOGY_VERSION

    @property
    def geometry_schema_id(self) -> str:
        return GEOMETRY_SCHEMA_ID

    @property
    def point_count(self) -> int:
        return len(self.adjacency)

    @property
    def pass_action(self) -> int:
        return self.point_count

    @property
    def action_size(self) -> int:
        return self.point_count + 1

    def neighbors(self, point: int) -> tuple[int, ...]:
        return self.adjacency[point]

    def point_id(self, point: int) -> str:
        return self.point_ids[point]

    def point_index(self, point_id: str) -> int:
        return self.index_by_id[point_id]

    def relation(self, point: int, neighbor: int) -> str:
        try:
            index = self.adjacency[point].index(neighbor)
        except ValueError as exc:
            raise ValueError(f"{neighbor} is not a neighbor of {point}") from exc
        return self.relation_types[point][index]

    def geometry(self, point: int) -> CubePointGeometry:
        return self.points[point]

    def _topology_payload(self) -> dict[str, object]:
        return {
            "topology_id": self.topology_id,
            "topology_version": TOPOLOGY_VERSION,
            "kind": "cube-surface",
            "size": self.size,
            "point_order": list(self.point_ids),
            "adjacency": [list(row) for row in self.adjacency],
        }

    def _geometry_payload(self) -> dict[str, object]:
        return {
            "geometry_schema_id": GEOMETRY_SCHEMA_ID,
            "topology_fingerprint": _fingerprint(self._topology_payload()),
            "face_frames": [
                {
                    "face_id": frame.face_id,
                    "normal": list(frame.normal),
                    "row_axis": list(frame.row_axis),
                    "col_axis": list(frame.col_axis),
                }
                for frame in self.face_frames
            ],
            "relation_types": [list(row) for row in self.relation_types],
            "points": [
                {
                    "point_id": point.point_id,
                    "face_id": point.face_id,
                    "local_row": point.local_row,
                    "local_col": point.local_col,
                    "geometry_class": point.geometry_class,
                    "physical_corner_id": point.physical_corner_id,
                    "incident_physical_cube_edges": list(point.incident_physical_cube_edges),
                    "corner_distance": point.corner_distance,
                    "corner_distance_bucket": point.corner_distance_bucket,
                    "has_cross_face_neighbor": point.has_cross_face_neighbor,
                    "num_cross_face_neighbors": point.num_cross_face_neighbors,
                }
                for point in self.points
            ],
            "physical_corners": [list(corner) for corner in self.physical_corners],
            "seams": [
                {
                    "seam_id": seam.seam_id,
                    "face_a": seam.face_a,
                    "edge_a": seam.edge_a,
                    "face_b": seam.face_b,
                    "edge_b": seam.edge_b,
                    "reversed_index": seam.reversed_index,
                    "point_pairs": [list(pair) for pair in seam.point_pairs],
                }
                for seam in self.seams
            ],
        }

    def audit(self) -> dict[str, object]:
        class_counts: dict[str, int] = {}
        degree_counts: dict[str, int] = {}
        relation_counts: dict[str, int] = {}
        for point in self.points:
            class_counts[point.geometry_class] = class_counts.get(point.geometry_class, 0) + 1
        for point, neighbors in enumerate(self.adjacency):
            degree = str(len(neighbors))
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
            for relation in self.relation_types[point]:
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
        return {
            "topology_id": self.topology_id,
            "topology_version": TOPOLOGY_VERSION,
            "topology_fingerprint": self.fingerprint,
            "geometry_schema_id": GEOMETRY_SCHEMA_ID,
            "geometry_fingerprint": self.geometry_fingerprint,
            "point_ordering_fingerprint": self.point_ordering_fingerprint,
            "point_count": self.point_count,
            "action_count": self.action_size,
            "degree_distribution": degree_counts,
            "geometry_class_counts": class_counts,
            "relation_directed_counts": relation_counts,
            "cross_face_adjacency_pairs": sum(
                relation == CROSS_FACE_SEAM for row in self.relation_types for relation in row
            ) // 2,
            "seam_count": len(self.seams),
            "physical_corner_count": len(self.physical_corners),
            "physical_corners": [list(corner) for corner in self.physical_corners],
            "independent_derivation": {
                "source": "face-normal,row-axis,column-axis seam folding",
                "production_topology_runtime_dependency": False,
            },
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            **self.audit(),
            "points": [
                {
                    "point_id": point.point_id,
                    "face_id": point.face_id,
                    "local_row": point.local_row,
                    "local_col": point.local_col,
                    "geometry_class": point.geometry_class,
                    "physical_corner_id": point.physical_corner_id,
                    "incident_physical_cube_edges": list(point.incident_physical_cube_edges),
                    "corner_distance": point.corner_distance,
                    "corner_distance_bucket": point.corner_distance_bucket,
                    "has_cross_face_neighbor": point.has_cross_face_neighbor,
                    "num_cross_face_neighbors": point.num_cross_face_neighbors,
                    "game_neighbors": list(self.adjacency[point.point_id]),
                    "relation_types": list(self.relation_types[point.point_id]),
                    "degree": len(self.adjacency[point.point_id]),
                    "same_face_neighbors": [
                        neighbor
                        for neighbor, relation in zip(
                            self.adjacency[point.point_id], self.relation_types[point.point_id]
                        )
                        if relation == SAME_FACE
                    ],
                    "cross_face_neighbors": [
                        neighbor
                        for neighbor, relation in zip(
                            self.adjacency[point.point_id], self.relation_types[point.point_id]
                        )
                        if relation == CROSS_FACE_SEAM
                    ],
                }
                for point in self.points
            ],
            "seams": [
                {
                    "seam_id": seam.seam_id,
                    "face_a": seam.face_a,
                    "edge_a": seam.edge_a,
                    "face_b": seam.face_b,
                    "edge_b": seam.edge_b,
                    "reversed_index": seam.reversed_index,
                    "point_pairs": [list(pair) for pair in seam.point_pairs],
                }
                for seam in self.seams
            ],
        }

    def _validate(self) -> None:
        if self.size != 4:
            raise ValueError("The canonical Golden Cube topology is fixed to Cube 4x4x6")
        if len(self.adjacency) != 96 or len(self.points) != 96:
            raise ValueError("Golden Cube topology must contain exactly 96 points")
        if len(self.relation_types) != len(self.adjacency):
            raise ValueError("Cube relation metadata length mismatch")
        for point, neighbors in enumerate(self.adjacency):
            if len(neighbors) != 4:
                raise ValueError(f"Cube point {point} does not have degree four")
            if len(set(neighbors)) != len(neighbors):
                raise ValueError(f"Cube point {point} has duplicate game neighbors")
            if point in neighbors:
                raise ValueError(f"Cube point {point} has a self-loop")
            if any(neighbor < 0 or neighbor >= 96 for neighbor in neighbors):
                raise ValueError(f"Cube point {point} has an invalid neighbor")
            if len(self.relation_types[point]) != len(neighbors):
                raise ValueError(f"Cube point {point} relation count mismatch")
            if any(relation not in (SAME_FACE, CROSS_FACE_SEAM) for relation in self.relation_types[point]):
                raise ValueError(f"Cube point {point} has an unknown edge relation")
            for neighbor, relation in zip(neighbors, self.relation_types[point]):
                if point not in self.adjacency[neighbor]:
                    raise ValueError(f"Cube adjacency {point}<->{neighbor} is not reciprocal")
                if relation != self.relation(neighbor, point):
                    raise ValueError(f"Cube relation {point}<->{neighbor} is not reciprocal")
        seen = {0}
        queue = deque([0])
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(seen) != 96:
            raise ValueError("Golden Cube game graph is not connected")
        if len(self.seams) != 12:
            raise ValueError("Golden Cube must expose exactly 12 physical seams")
        if len(self.physical_corners) != 8:
            raise ValueError("Golden Cube must expose exactly 8 physical corners")
        corner_points = [point for point in self.points if point.geometry_class == FACE_CORNER]
        if len(corner_points) != 24:
            raise ValueError("Golden Cube must contain exactly 24 face-corner cells")
        if len({point.point_id for point in corner_points}) != 24:
            raise ValueError("Cube face-corner point identities are not unique")
        memberships: list[int] = []
        for corner_id, corner in enumerate(self.physical_corners):
            if len(corner) != 3 or len(set(corner)) != 3:
                raise ValueError(f"Physical corner {corner_id} must contain three distinct points")
            if any(self.points[point].physical_corner_id != corner_id for point in corner):
                raise ValueError(f"Physical corner {corner_id} membership is inconsistent")
            if len({self.points[point].face_id for point in corner}) != 3:
                raise ValueError(f"Physical corner {corner_id} must span three faces")
            memberships.extend(corner)
        if sorted(memberships) != sorted(point.point_id for point in corner_points):
            raise ValueError("The 24 face-corner cells are not covered exactly once")
        if sum(relation == CROSS_FACE_SEAM for row in self.relation_types for relation in row) != 96:
            raise ValueError("Golden Cube must contain 48 undirected cross-face seam pairs")


def _point_id(face: str, row: int, col: int, size: int) -> int:
    face_index = CUBE_FACES.index(face)
    return face_index * size * size + row * size + col


def _edge_point(face: str, edge: str, index: int, size: int) -> int:
    last = size - 1
    if edge == "top":
        return _point_id(face, 0, index, size)
    if edge == "right":
        return _point_id(face, index, last, size)
    if edge == "bottom":
        return _point_id(face, last, index, size)
    if edge == "left":
        return _point_id(face, index, 0, size)
    raise ValueError(edge)


def _edge_tangent(frame: CubeFaceFrame, edge: str) -> Vector:
    return frame.col_axis if edge in ("top", "bottom") else frame.row_axis


def _edge_outward(frame: CubeFaceFrame, edge: str) -> Vector:
    if edge == "top":
        return _neg(frame.row_axis)
    if edge == "right":
        return frame.col_axis
    if edge == "bottom":
        return frame.row_axis
    if edge == "left":
        return _neg(frame.col_axis)
    raise ValueError(edge)


def _target_edge(frame: CubeFaceFrame, outward: Vector) -> str:
    candidates = [edge for edge in EDGE_NAMES if _edge_outward(frame, edge) == outward]
    if len(candidates) != 1:
        raise ValueError(f"Could not derive unique target edge for normal {outward}")
    return candidates[0]


def _derive_seams(size: int) -> tuple[CubeSeam, ...]:
    seams: list[CubeSeam] = []
    frame_rank = {face: index for index, face in enumerate(CUBE_FACES)}
    for face in CUBE_FACES:
        source_frame = _FRAME_BY_FACE[face]
        for edge in EDGE_NAMES:
            target_face = next(
                candidate.face_id
                for candidate in FACE_FRAMES
                if candidate.normal == _edge_outward(source_frame, edge)
            )
            target_frame = _FRAME_BY_FACE[target_face]
            # The target face's edge points back toward the source face's
            # normal.  (The source edge's outward direction points toward
            # the target face normal.)
            target_edge = _target_edge(target_frame, source_frame.normal)
            if (frame_rank[face], EDGE_NAMES.index(edge)) >= (
                frame_rank[target_face], EDGE_NAMES.index(target_edge)
            ):
                continue
            tangent = _edge_tangent(source_frame, edge)
            target_tangent = _edge_tangent(target_frame, target_edge)
            if target_tangent == tangent:
                reversed_index = False
            elif target_tangent == _neg(tangent):
                reversed_index = True
            else:
                raise ValueError(f"Seam tangent is not aligned: {face}:{edge} -> {target_face}:{target_edge}")
            pairs = tuple(
                (
                    _edge_point(face, edge, index, size),
                    _edge_point(target_face, target_edge, size - 1 - index if reversed_index else index, size),
                )
                for index in range(size)
            )
            seams.append(
                CubeSeam(
                    seam_id=f"{face}:{edge}--{target_face}:{target_edge}",
                    face_a=face,
                    edge_a=edge,
                    face_b=target_face,
                    edge_b=target_edge,
                    reversed_index=reversed_index,
                    point_pairs=pairs,
                )
            )
    if len(seams) != 12:
        raise ValueError(f"Expected 12 derived cube seams, got {len(seams)}")
    return tuple(seams)


def _physical_corner_vector(frame: CubeFaceFrame, row: int, col: int, last: int) -> Vector:
    row_sign = -1 if row == 0 else 1
    col_sign = -1 if col == 0 else 1
    return _add(frame.normal, tuple(row_sign * value for value in frame.row_axis), tuple(col_sign * value for value in frame.col_axis))  # type: ignore[arg-type]


def _build_cube4() -> CubeGoldenTopology:
    size = 4
    seams = _derive_seams(size)
    seam_by_directed_edge = {
        (seam.face_a, seam.edge_a): seam for seam in seams
    }
    seam_by_directed_edge.update(
        {
            (seam.face_b, seam.edge_b): CubeSeam(
                seam_id=seam.seam_id,
                face_a=seam.face_b,
                edge_a=seam.edge_b,
                face_b=seam.face_a,
                edge_b=seam.edge_a,
                reversed_index=seam.reversed_index,
                point_pairs=tuple(
                    (
                        _edge_point(seam.face_b, seam.edge_b, index, size),
                        _edge_point(
                            seam.face_a,
                            seam.edge_a,
                            size - 1 - index if seam.reversed_index else index,
                            size,
                        ),
                    )
                    for index in range(size)
                ),
            )
            for seam in seams
        }
    )

    corner_vectors = sorted(
        {
            _physical_corner_vector(frame, row, col, size - 1)
            for frame in FACE_FRAMES
            for row in (0, size - 1)
            for col in (0, size - 1)
        }
    )
    if len(corner_vectors) != 8:
        raise ValueError("Face-frame geometry did not produce eight cube corners")
    corner_id_by_vector = {vector: index for index, vector in enumerate(corner_vectors)}
    physical_corner_points: list[list[int]] = [[] for _ in corner_vectors]
    for frame in FACE_FRAMES:
        for row in range(size):
            for col in range(size):
                if row not in (0, size - 1) or col not in (0, size - 1):
                    continue
                vector = _physical_corner_vector(frame, row, col, size - 1)
                physical_corner_points[corner_id_by_vector[vector]].append(
                    _point_id(frame.face_id, row, col, size)
                )
    physical_corners = tuple(tuple(sorted(points)) for points in physical_corner_points)

    adjacency: list[list[int]] = [[] for _ in range(6 * size * size)]
    relations: list[list[str]] = [[] for _ in adjacency]

    def add_edge(left: int, right: int, relation: str) -> None:
        if right not in adjacency[left]:
            adjacency[left].append(right)
            relations[left].append(relation)
        elif relations[left][adjacency[left].index(right)] != relation:
            raise ValueError(f"Conflicting relation for edge {left}<->{right}")
        if left not in adjacency[right]:
            adjacency[right].append(left)
            relations[right].append(relation)
        elif relations[right][adjacency[right].index(left)] != relation:
            raise ValueError(f"Conflicting reciprocal relation for edge {left}<->{right}")

    for frame in FACE_FRAMES:
        for row in range(size):
            for col in range(size):
                point = _point_id(frame.face_id, row, col, size)
                if row > 0:
                    add_edge(point, _point_id(frame.face_id, row - 1, col, size), SAME_FACE)
                if col > 0:
                    add_edge(point, _point_id(frame.face_id, row, col - 1, size), SAME_FACE)
                if row == 0:
                    seam = seam_by_directed_edge[(frame.face_id, "top")]
                    add_edge(point, seam.point_pairs[col][1], CROSS_FACE_SEAM)
                elif row == size - 1:
                    seam = seam_by_directed_edge[(frame.face_id, "bottom")]
                    add_edge(point, seam.point_pairs[col][1], CROSS_FACE_SEAM)
                if col == 0:
                    seam = seam_by_directed_edge[(frame.face_id, "left")]
                    add_edge(point, seam.point_pairs[row][1], CROSS_FACE_SEAM)
                elif col == size - 1:
                    seam = seam_by_directed_edge[(frame.face_id, "right")]
                    add_edge(point, seam.point_pairs[row][1], CROSS_FACE_SEAM)

    distances = [None] * (6 * size * size)
    queue = deque()
    for corner in physical_corners:
        for point in corner:
            if distances[point] is None:
                distances[point] = 0
                queue.append(point)
    while queue:
        point = queue.popleft()
        assert distances[point] is not None
        for neighbor in adjacency[point]:
            if distances[neighbor] is None:
                distances[neighbor] = distances[point] + 1
                queue.append(neighbor)

    point_metadata: list[CubePointGeometry] = []
    for frame in FACE_FRAMES:
        for row in range(size):
            for col in range(size):
                point = _point_id(frame.face_id, row, col, size)
                if row in (0, size - 1) and col in (0, size - 1):
                    geometry_class = FACE_CORNER
                    corner_id = corner_id_by_vector[_physical_corner_vector(frame, row, col, size - 1)]
                elif row in (0, size - 1) or col in (0, size - 1):
                    geometry_class = FACE_EDGE
                    corner_id = None
                else:
                    geometry_class = FACE_INTERIOR
                    corner_id = None
                incident_edges = tuple(
                    seam_by_directed_edge[(frame.face_id, edge)].seam_id
                    for edge, present in (
                        ("top", row == 0),
                        ("right", col == size - 1),
                        ("bottom", row == size - 1),
                        ("left", col == 0),
                    )
                    if present
                )
                distance = int(distances[point])
                point_metadata.append(
                    CubePointGeometry(
                        point_id=point,
                        face_id=frame.face_id,
                        local_row=row,
                        local_col=col,
                        geometry_class=geometry_class,
                        physical_corner_id=corner_id,
                        incident_physical_cube_edges=tuple(sorted(incident_edges)),
                        corner_distance=distance,
                        corner_distance_bucket=(
                            f"corner_distance_{distance}" if distance <= 2 else "corner_distance_3_plus"
                        ),
                        has_cross_face_neighbor=any(
                            relation == CROSS_FACE_SEAM for relation in relations[point]
                        ),
                        num_cross_face_neighbors=sum(
                            relation == CROSS_FACE_SEAM for relation in relations[point]
                        ),
                    )
                )
    return CubeGoldenTopology(
        size=size,
        topology_id=CUBE4_TOPOLOGY_ID,
        adjacency=tuple(tuple(row) for row in adjacency),
        relation_types=tuple(tuple(row) for row in relations),
        points=tuple(point_metadata),
        seams=seams,
        physical_corners=physical_corners,
        face_frames=FACE_FRAMES,
    )


CUBE4_TOPOLOGY = _build_cube4()
CUBE4_TOPOLOGY_FINGERPRINT = CUBE4_TOPOLOGY.fingerprint
CUBE4_GEOMETRY_FINGERPRINT = CUBE4_TOPOLOGY.geometry_fingerprint


def cube4x4x6() -> CubeGoldenTopology:
    """Return the immutable canonical Golden Cube 4x4x6 topology."""

    return CUBE4_TOPOLOGY


# ---------------------------------------------------------------------------
# Diagnostic-only proper rotation construction.  The Golden runtime does not
# use rotations to reject evaluation starts or to alter game semantics.
# ---------------------------------------------------------------------------

Matrix = tuple[Vector, Vector, Vector]


def _mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def cube_rotation_matrices() -> tuple[Matrix, ...]:
    matrices: list[Matrix] = []
    axes = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for x_axis in axes:
        for y_axis in axes:
            if sum(x_axis[index] * y_axis[index] for index in range(3)) != 0:
                continue
            z_axis = (
                x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
                x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
                x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
            )
            matrix = (x_axis, y_axis, z_axis)
            if _determinant(*matrix) == 1 and matrix not in matrices:
                matrices.append(matrix)
    if len(matrices) != 24:
        raise RuntimeError(f"Expected 24 proper cube rotations, got {len(matrices)}")
    return tuple(matrices)


def cube_rotation_permutations(
    topology: CubeGoldenTopology = CUBE4_TOPOLOGY,
) -> tuple[tuple[int, ...], ...]:
    frame_by_normal = {frame.normal: frame for frame in topology.face_frames}
    permutations: list[tuple[int, ...]] = []
    for matrix in cube_rotation_matrices():
        mapped: list[int] = []
        for point in topology.points:
            source = _FRAME_BY_FACE[point.face_id]
            target_normal = _mat_vec(matrix, source.normal)
            target = frame_by_normal[target_normal]
            target_row_axis = _mat_vec(matrix, source.row_axis)
            target_col_axis = _mat_vec(matrix, source.col_axis)
            if target_row_axis in (target.row_axis, _neg(target.row_axis)) and target_col_axis in (target.col_axis, _neg(target.col_axis)):
                row = point.local_row if target_row_axis == target.row_axis else topology.size - 1 - point.local_row
                col = point.local_col if target_col_axis == target.col_axis else topology.size - 1 - point.local_col
            elif target_row_axis in (target.col_axis, _neg(target.col_axis)) and target_col_axis in (target.row_axis, _neg(target.row_axis)):
                row = point.local_col if target_col_axis == target.row_axis else topology.size - 1 - point.local_col
                col = point.local_row if target_row_axis == target.col_axis else topology.size - 1 - point.local_row
            else:
                raise RuntimeError("Rotation did not preserve a face frame")
            mapped.append(_point_id(target.face_id, row, col, topology.size))
        permutation = tuple(mapped)
        if permutation in permutations:
            continue
        if sorted(permutation) != list(range(topology.point_count)):
            raise RuntimeError("Cube rotation is not a point permutation")
        for point, neighbors in enumerate(topology.adjacency):
            if {permutation[neighbor] for neighbor in neighbors} != set(topology.adjacency[permutation[point]]):
                raise RuntimeError("Cube rotation does not preserve game adjacency")
        permutations.append(permutation)
    if len(permutations) != 24:
        raise RuntimeError(f"Expected 24 cube point rotations, got {len(permutations)}")
    return tuple(permutations)


CUBE4_ROTATION_PERMUTATIONS = cube_rotation_permutations()


def geometry_class_counts(topology: CubeGoldenTopology = CUBE4_TOPOLOGY) -> Mapping[str, int]:
    result: dict[str, int] = {}
    for point in topology.points:
        result[point.geometry_class] = result.get(point.geometry_class, 0) + 1
    return result
