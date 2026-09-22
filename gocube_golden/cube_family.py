"""Parameterized Cube v2 geometry/state primitives for sizes 2..7."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from typing import Iterable, Mapping, Sequence, TypeVar

from .cube_game_contract_v2 import SUPPORTED_SIZES, validate_cube_size

Vector = tuple[int, int, int]
Matrix = tuple[Vector, Vector, Vector]
T = TypeVar("T")

FACE_INTERIOR = "FACE_INTERIOR"
FACE_EDGE = "FACE_EDGE"
FACE_CORNER = "FACE_CORNER"
SAME_FACE = "SAME_FACE"
CROSS_FACE_SEAM = "CROSS_FACE_SEAM"
CUBE_FACES = ("front", "back", "left", "right", "top", "bottom")
EDGE_NAMES = ("top", "right", "bottom", "left")
GAME_GRAPH_SCHEMA_ID = "cube-family-game-graph-v2"
GEOMETRY_SCHEMA_ID = "cube-family-geometry-v2"
STATE_SCHEMA_VERSION = 1


def _fp(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _neg(v: Vector) -> Vector:
    return tuple(-x for x in v)  # type: ignore[return-value]


def _add(*vectors: Vector) -> Vector:
    return tuple(sum(v[i] for v in vectors) for i in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class CubeFaceFrame:
    face_id: str
    normal: Vector
    row_axis: Vector
    col_axis: Vector


FACE_FRAMES = (
    CubeFaceFrame("front", (0, 0, 1), (0, 1, 0), (1, 0, 0)),
    CubeFaceFrame("back", (0, 0, -1), (0, 1, 0), (-1, 0, 0)),
    CubeFaceFrame("left", (-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    CubeFaceFrame("right", (1, 0, 0), (0, 1, 0), (0, 0, -1)),
    CubeFaceFrame("top", (0, -1, 0), (0, 0, 1), (1, 0, 0)),
    CubeFaceFrame("bottom", (0, 1, 0), (0, 0, -1), (1, 0, 0)),
)
_FRAME = {frame.face_id: frame for frame in FACE_FRAMES}


@dataclass(frozen=True)
class CubeSeam:
    seam_id: str
    face_a: str
    edge_a: str
    face_b: str
    edge_b: str
    reversed_index: bool
    point_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CubePointGeometry:
    point_id: int
    face_id: str
    row: int
    column: int
    geometry_class: str
    physical_corner_id: int | None
    incident_seam_ids: tuple[str, ...]
    num_cross_face_neighbors: int
    has_cross_face_neighbor: bool
    corner_distance: int
    seam_distance: int

    @property
    def local_row(self) -> int:
        return self.row

    @property
    def local_col(self) -> int:
        return self.column


@dataclass(frozen=True)
class CubeRotation:
    index: int
    matrix: Matrix
    point_permutation: tuple[int, ...]
    action_permutation: tuple[int, ...]

    def point(self, point_id: int) -> int:
        return self.point_permutation[point_id]

    def action(self, action_index: int) -> int:
        return self.action_permutation[action_index]

    def permute_points(self, values: Sequence[T]) -> tuple[T, ...]:
        if len(values) != len(self.point_permutation):
            raise ValueError("Point array length does not match Cube topology")
        result = [values[0]] * len(values)
        for old, new in enumerate(self.point_permutation):
            result[new] = values[old]
        return tuple(result)


class CubeFamilyTopology:
    def __init__(self, *, size: int, adjacency, relations, points, seams, physical_corners):
        self.size = validate_cube_size(size)
        self.topology_id = f"cube{self.size}-family-topology-v2"
        self.kind = "cube-surface"
        self.dimensions = None
        self.adjacency = tuple(tuple(row) for row in adjacency)
        self.relation_types = tuple(tuple(row) for row in relations)
        self.points = tuple(points)
        self.seams = tuple(seams)
        self.physical_corners = tuple(tuple(corner) for corner in physical_corners)
        self.face_frames = FACE_FRAMES
        self.point_ids = tuple(f"{p.face_id}:{p.row}:{p.column}" for p in self.points)
        self.index_by_id = {name: i for i, name in enumerate(self.point_ids)}
        _validate_topology(self)
        self.game_graph_fingerprint = _fp(_graph_payload(self))
        self.fingerprint = self.game_graph_fingerprint
        self.geometry_fingerprint = _fp(_geometry_payload(self))
        self.rotations = _build_rotations(self)
        _validate_rotations(self)

    @property
    def point_count(self) -> int:
        return len(self.adjacency)

    @property
    def pass_action(self) -> int:
        return self.point_count

    @property
    def action_size(self) -> int:
        return self.point_count + 1

    @property
    def action_count(self) -> int:
        return self.action_size

    @property
    def topology_fingerprint(self) -> str:
        return self.game_graph_fingerprint

    def neighbors(self, point: int) -> tuple[int, ...]:
        return self.adjacency[point]

    def relation(self, point: int, neighbor: int) -> str:
        try:
            return self.relation_types[point][self.adjacency[point].index(neighbor)]
        except ValueError as exc:
            raise ValueError(f"{neighbor} is not a neighbor of {point}") from exc

    def geometry(self, point: int) -> CubePointGeometry:
        return self.points[point]

    def point_address(self, point: int) -> tuple[str, int, int]:
        p = self.geometry(point)
        return p.face_id, p.row, p.column

    def point_index(self, face_id: str, row: int, column: int) -> int:
        if face_id not in CUBE_FACES or type(row) is not int or type(column) is not int:
            raise ValueError("Invalid Cube face/row/column")
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise ValueError("Cube row/column outside topology")
        return _point_id(face_id, row, column, self.size)

    def audit(self) -> dict[str, object]:
        return {
            "topology_id": self.topology_id,
            "size": self.size,
            "point_count": self.point_count,
            "action_count": self.action_count,
            "game_graph_fingerprint": self.game_graph_fingerprint,
            "geometry_fingerprint": self.geometry_fingerprint,
            "seam_count": len(self.seams),
            "physical_corner_count": len(self.physical_corners),
            "rotation_count": len(self.rotations),
            "geometry_class_counts": geometry_class_counts(self),
        }


def _seam_payload(seam: CubeSeam) -> dict[str, object]:
    return {
        "seam_id": seam.seam_id,
        "face_a": seam.face_a,
        "edge_a": seam.edge_a,
        "face_b": seam.face_b,
        "edge_b": seam.edge_b,
        "reversed_index": seam.reversed_index,
        "point_pairs": [list(pair) for pair in seam.point_pairs],
    }


def _graph_payload(t: CubeFamilyTopology) -> dict[str, object]:
    return {
        "schema_id": GAME_GRAPH_SCHEMA_ID,
        "topology_id": t.topology_id,
        "size": t.size,
        "point_order": list(t.point_ids),
        "adjacency": [list(row) for row in t.adjacency],
        "relation_types": [list(row) for row in t.relation_types],
        "seams": [_seam_payload(s) for s in t.seams],
    }


def _geometry_payload(t: CubeFamilyTopology) -> dict[str, object]:
    return {
        "schema_id": GEOMETRY_SCHEMA_ID,
        "game_graph_fingerprint": t.game_graph_fingerprint,
        "face_frames": [
            {"face_id": f.face_id, "normal": list(f.normal), "row_axis": list(f.row_axis), "col_axis": list(f.col_axis)}
            for f in FACE_FRAMES
        ],
        "physical_corners": [list(c) for c in t.physical_corners],
        "seams": [_seam_payload(s) for s in t.seams],
        "points": [
            {
                "point_id": p.point_id,
                "face_id": p.face_id,
                "row": p.row,
                "column": p.column,
                "geometry_class": p.geometry_class,
                "physical_corner_id": p.physical_corner_id,
                "incident_seam_ids": list(p.incident_seam_ids),
                "num_cross_face_neighbors": p.num_cross_face_neighbors,
                "has_cross_face_neighbor": p.has_cross_face_neighbor,
                "corner_distance": p.corner_distance,
                "seam_distance": p.seam_distance,
            }
            for p in t.points
        ],
    }


def _point_id(face: str, row: int, col: int, n: int) -> int:
    return CUBE_FACES.index(face) * n * n + row * n + col


def _edge_point(face: str, edge: str, i: int, n: int) -> int:
    last = n - 1
    if edge == "top": return _point_id(face, 0, i, n)
    if edge == "right": return _point_id(face, i, last, n)
    if edge == "bottom": return _point_id(face, last, i, n)
    if edge == "left": return _point_id(face, i, 0, n)
    raise ValueError(edge)


def _edge_outward(frame: CubeFaceFrame, edge: str) -> Vector:
    if edge == "top": return _neg(frame.row_axis)
    if edge == "right": return frame.col_axis
    if edge == "bottom": return frame.row_axis
    if edge == "left": return _neg(frame.col_axis)
    raise ValueError(edge)


def _edge_tangent(frame: CubeFaceFrame, edge: str) -> Vector:
    return frame.col_axis if edge in ("top", "bottom") else frame.row_axis


def _target_edge(frame: CubeFaceFrame, outward: Vector) -> str:
    matches = [edge for edge in EDGE_NAMES if _edge_outward(frame, edge) == outward]
    if len(matches) != 1: raise ValueError("Cube frame has ambiguous target edge")
    return matches[0]


def _derive_seams(n: int) -> tuple[CubeSeam, ...]:
    seams = []
    rank = {face: i for i, face in enumerate(CUBE_FACES)}
    for face in CUBE_FACES:
        source = _FRAME[face]
        for edge in EDGE_NAMES:
            target_face = next(f.face_id for f in FACE_FRAMES if f.normal == _edge_outward(source, edge))
            target = _FRAME[target_face]
            target_edge = _target_edge(target, source.normal)
            if (rank[face], EDGE_NAMES.index(edge)) >= (rank[target_face], EDGE_NAMES.index(target_edge)):
                continue
            tangent, target_tangent = _edge_tangent(source, edge), _edge_tangent(target, target_edge)
            if target_tangent == tangent: reversed_index = False
            elif target_tangent == _neg(tangent): reversed_index = True
            else: raise ValueError("Cube seam tangent mismatch")
            pairs = tuple(
                (_edge_point(face, edge, i, n), _edge_point(target_face, target_edge, n - 1 - i if reversed_index else i, n))
                for i in range(n)
            )
            seams.append(CubeSeam(f"{face}:{edge}--{target_face}:{target_edge}", face, edge, target_face, target_edge, reversed_index, pairs))
    if len(seams) != 12: raise ValueError("Cube must have 12 seams")
    return tuple(seams)


def _corner_vector(frame: CubeFaceFrame, row: int, col: int, last: int) -> Vector:
    if row not in (0, last) or col not in (0, last): raise ValueError("Not a face corner")
    rs, cs = (-1 if row == 0 else 1), (-1 if col == 0 else 1)
    return _add(frame.normal, tuple(rs * x for x in frame.row_axis), tuple(cs * x for x in frame.col_axis))  # type: ignore[arg-type]


def _distances(adjacency, sources: Iterable[int]) -> tuple[int, ...]:
    d = [-1] * len(adjacency); q = deque()
    for source in sources:
        if d[source] == -1: d[source] = 0; q.append(source)
    while q:
        point = q.popleft()
        for neighbor in adjacency[point]:
            if d[neighbor] == -1: d[neighbor] = d[point] + 1; q.append(neighbor)
    if -1 in d: raise ValueError("Cube graph is disconnected")
    return tuple(d)


def _directed_seams(seams, n):
    result = {(s.face_a, s.edge_a): s for s in seams}
    for s in seams:
        pairs = tuple(
            (_edge_point(s.face_b, s.edge_b, i, n), _edge_point(s.face_a, s.edge_a, n - 1 - i if s.reversed_index else i, n))
            for i in range(n)
        )
        result[(s.face_b, s.edge_b)] = CubeSeam(s.seam_id, s.face_b, s.edge_b, s.face_a, s.edge_a, s.reversed_index, pairs)
    return result


def _build_topology(size: int) -> CubeFamilyTopology:
    n = validate_cube_size(size); last = n - 1
    seams = _derive_seams(n); directed = _directed_seams(seams, n)
    vectors = sorted({_corner_vector(f, r, c, last) for f in FACE_FRAMES for r in (0, last) for c in (0, last)})
    corner_id = {v: i for i, v in enumerate(vectors)}
    corners = [[] for _ in range(8)]
    for f in FACE_FRAMES:
        for r in (0, last):
            for c in (0, last):
                corners[corner_id[_corner_vector(f, r, c, last)]].append(_point_id(f.face_id, r, c, n))
    physical_corners = tuple(tuple(sorted(c)) for c in corners)

    adjacency = [[] for _ in range(6 * n * n)]; relations = [[] for _ in adjacency]
    def add(a: int, b: int, relation: str) -> None:
        for left, right in ((a, b), (b, a)):
            if right not in adjacency[left]: adjacency[left].append(right); relations[left].append(relation)
            elif relations[left][adjacency[left].index(right)] != relation: raise ValueError("Conflicting Cube relation")

    for f in FACE_FRAMES:
        for r in range(n):
            for c in range(n):
                p = _point_id(f.face_id, r, c, n)
                if r > 0: add(p, _point_id(f.face_id, r - 1, c, n), SAME_FACE)
                if c > 0: add(p, _point_id(f.face_id, r, c - 1, n), SAME_FACE)
                if r == 0: add(p, directed[(f.face_id, "top")].point_pairs[c][1], CROSS_FACE_SEAM)
                elif r == last: add(p, directed[(f.face_id, "bottom")].point_pairs[c][1], CROSS_FACE_SEAM)
                if c == 0: add(p, directed[(f.face_id, "left")].point_pairs[r][1], CROSS_FACE_SEAM)
                elif c == last: add(p, directed[(f.face_id, "right")].point_pairs[r][1], CROSS_FACE_SEAM)

    corner_distance = _distances(adjacency, (p for corner in physical_corners for p in corner))
    seam_sources = (p for p, row in enumerate(relations) if CROSS_FACE_SEAM in row)
    seam_distance = _distances(adjacency, seam_sources)
    points = []
    for f in FACE_FRAMES:
        for r in range(n):
            for c in range(n):
                p = _point_id(f.face_id, r, c, n)
                is_corner = r in (0, last) and c in (0, last)
                is_edge = r in (0, last) or c in (0, last)
                cls = FACE_CORNER if is_corner else FACE_EDGE if is_edge else FACE_INTERIOR
                pc = corner_id[_corner_vector(f, r, c, last)] if is_corner else None
                incident = tuple(sorted(directed[(f.face_id, edge)].seam_id for edge, present in (("top", r == 0), ("right", c == last), ("bottom", r == last), ("left", c == 0)) if present))
                cross = sum(rel == CROSS_FACE_SEAM for rel in relations[p])
                points.append(CubePointGeometry(p, f.face_id, r, c, cls, pc, incident, cross, cross > 0, corner_distance[p], seam_distance[p]))
    return CubeFamilyTopology(size=n, adjacency=adjacency, relations=relations, points=points, seams=seams, physical_corners=physical_corners)


def _validate_topology(t: CubeFamilyTopology) -> None:
    n = t.size; expected = 6 * n * n
    if t.point_count != expected or len(t.points) != expected or len(t.relation_types) != expected: raise ValueError("Cube point count mismatch")
    for p, neighbors in enumerate(t.adjacency):
        if len(neighbors) != 4 or len(set(neighbors)) != 4 or p in neighbors: raise ValueError("Cube degree/self-loop invariant failed")
        if len(t.relation_types[p]) != 4: raise ValueError("Cube relation count mismatch")
        for neighbor, relation in zip(neighbors, t.relation_types[p]):
            if relation not in (SAME_FACE, CROSS_FACE_SEAM) or p not in t.adjacency[neighbor]: raise ValueError("Cube reciprocal edge invariant failed")
            if t.relation(neighbor, p) != relation: raise ValueError("Cube reciprocal relation invariant failed")
    if len(t.seams) != 12 or any(len(s.point_pairs) != n for s in t.seams): raise ValueError("Cube seam invariant failed")
    if sum(rel == CROSS_FACE_SEAM for row in t.relation_types for rel in row) // 2 != 12 * n: raise ValueError("Cube seam edge count mismatch")
    if len(t.physical_corners) != 8 or any(len(set(c)) != 3 for c in t.physical_corners): raise ValueError("Cube physical corner invariant failed")
    corner_points = [p.point_id for p in t.points if p.geometry_class == FACE_CORNER]
    if sorted(p for corner in t.physical_corners for p in corner) != sorted(corner_points): raise ValueError("Cube corner membership mismatch")
    if geometry_class_counts(t) != {FACE_CORNER: 24, FACE_EDGE: 24 * (n - 2), FACE_INTERIOR: 6 * (n - 2) ** 2}: raise ValueError("Cube geometry class counts mismatch")
    seen = {0}; q = deque([0])
    while q:
        p = q.popleft()
        for neighbor in t.adjacency[p]:
            if neighbor not in seen: seen.add(neighbor); q.append(neighbor)
    if len(seen) != expected: raise ValueError("Cube graph disconnected")


def _det(a: Vector, b: Vector, c: Vector) -> int:
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def _mat_vec(m: Matrix, v: Vector) -> Vector:
    return tuple(sum(m[r][c] * v[c] for c in range(3)) for r in range(3))  # type: ignore[return-value]


def cube_rotation_matrices() -> tuple[Matrix, ...]:
    axes = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)); result = []
    for x in axes:
        for y in axes:
            if sum(x[i] * y[i] for i in range(3)): continue
            z = (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]); m = (x, y, z)
            if _det(*m) == 1 and m not in result: result.append(m)
    if len(result) != 24: raise RuntimeError("Expected 24 proper Cube rotations")
    return tuple(result)


def _build_rotations(t: CubeFamilyTopology) -> tuple[CubeRotation, ...]:
    by_normal = {f.normal: f for f in FACE_FRAMES}; result = []; seen = set()
    for matrix in cube_rotation_matrices():
        mapped = []
        for p in t.points:
            source = _FRAME[p.face_id]; target = by_normal[_mat_vec(matrix, source.normal)]
            ra, ca = _mat_vec(matrix, source.row_axis), _mat_vec(matrix, source.col_axis)
            if ra in (target.row_axis, _neg(target.row_axis)) and ca in (target.col_axis, _neg(target.col_axis)):
                row = p.row if ra == target.row_axis else t.size - 1 - p.row; col = p.column if ca == target.col_axis else t.size - 1 - p.column
            elif ra in (target.col_axis, _neg(target.col_axis)) and ca in (target.row_axis, _neg(target.row_axis)):
                row = p.column if ca == target.row_axis else t.size - 1 - p.column; col = p.row if ra == target.col_axis else t.size - 1 - p.row
            else: raise RuntimeError("Cube rotation frame mismatch")
            mapped.append(_point_id(target.face_id, row, col, t.size))
        permutation = tuple(mapped)
        if permutation not in seen:
            seen.add(permutation); result.append(CubeRotation(len(result), matrix, permutation, permutation + (t.pass_action,)))
    if len(result) != 24: raise RuntimeError("Expected 24 Cube point permutations")
    return tuple(result)


def _validate_rotations(t: CubeFamilyTopology) -> None:
    corners = {frozenset(c) for c in t.physical_corners}
    seams = {frozenset(frozenset(pair) for pair in s.point_pairs) for s in t.seams}
    for rotation in t.rotations:
        perm = rotation.point_permutation
        if sorted(perm) != list(range(t.point_count)) or rotation.action_permutation[-1] != t.pass_action: raise ValueError("Invalid Cube rotation permutation")
        for p, neighbors in enumerate(t.adjacency):
            mapped = perm[p]
            if {perm[n] for n in neighbors} != set(t.adjacency[mapped]): raise ValueError("Rotation adjacency mismatch")
            a, b = t.points[p], t.points[mapped]
            if (a.geometry_class, a.corner_distance, a.seam_distance, a.num_cross_face_neighbors) != (b.geometry_class, b.corner_distance, b.seam_distance, b.num_cross_face_neighbors): raise ValueError("Rotation geometry mismatch")
            for n in neighbors:
                if t.relation(p, n) != t.relation(mapped, perm[n]): raise ValueError("Rotation relation mismatch")
        if any(frozenset(perm[p] for p in c) not in corners for c in t.physical_corners): raise ValueError("Rotation corner mismatch")
        for seam in t.seams:
            mapped_pairs = frozenset(frozenset((perm[a], perm[b])) for a, b in seam.point_pairs)
            if mapped_pairs not in seams: raise ValueError("Rotation seam mismatch")


@lru_cache(maxsize=len(SUPPORTED_SIZES))
def cube_family_topology(size: int) -> CubeFamilyTopology:
    return _build_topology(validate_cube_size(size))


def geometry_class_counts(t: CubeFamilyTopology) -> dict[str, int]:
    result = {FACE_CORNER: 0, FACE_EDGE: 0, FACE_INTERIOR: 0}
    for p in t.points: result[p.geometry_class] += 1
    return result


def family_fingerprints() -> Mapping[int, tuple[str, str]]:
    return {n: (cube_family_topology(n).fingerprint, cube_family_topology(n).geometry_fingerprint) for n in SUPPORTED_SIZES}


def initial_cube_state(*, size: int, komi: float = 0.5):
    from .state import initial_state
    return initial_state(topology=cube_family_topology(size), komi=komi)


def cube_state_from_stones(stones: Sequence[int], *, size: int, side_to_move=None, komi: float = 0.5, superko_history: Iterable[Sequence[int]] | None = None, consecutive_passes: int = 0):
    from .state import BLACK, research_state_from_stones
    return research_state_from_stones(stones, side_to_move=BLACK if side_to_move is None else side_to_move, topology=cube_family_topology(size), komi=komi, superko_history=superko_history, consecutive_passes=consecutive_passes)


def serialize_cube_state(state) -> dict[str, object]:
    from .state import GoldenState
    if not isinstance(state, GoldenState) or not isinstance(state.topology, CubeFamilyTopology): raise ValueError("Cube v2 serialization requires CubeFamilyTopology")
    t = state.topology
    return {
        "schema_version": STATE_SCHEMA_VERSION, "size": t.size,
        "stones": [int(s) for s in state.stones], "side_to_move": int(state.side_to_move),
        "consecutive_passes": state.consecutive_passes, "komi": state.komi,
        "superko_history": [list(p) for p in state.superko_history], "history_provenance": state.history_provenance,
        "topology_id": t.topology_id, "topology_fingerprint": t.fingerprint, "geometry_fingerprint": t.geometry_fingerprint,
        "rules_identity": {"rules_id": state.rules_id, "rules_fingerprint": state.rules_fingerprint},
    }


def deserialize_cube_state(payload: Mapping[str, object]):
    from .state import GoldenState, RULES_PROFILE_ID, Stone, rules_fingerprint_for, validate_komi
    if not isinstance(payload, Mapping) or payload.get("schema_version") != STATE_SCHEMA_VERSION: raise ValueError("Invalid Cube state schema")
    t = cube_family_topology(validate_cube_size(payload.get("size")))
    if payload.get("topology_id") != t.topology_id or payload.get("topology_fingerprint") != t.fingerprint or payload.get("geometry_fingerprint") != t.geometry_fingerprint: raise ValueError("Serialized Cube topology identity mismatch")
    komi = validate_komi(payload.get("komi"), context="Cube v2 state")
    rules = payload.get("rules_identity"); expected_rules = rules_fingerprint_for(t, komi)
    if not isinstance(rules, Mapping) or rules.get("rules_id") != RULES_PROFILE_ID or rules.get("rules_fingerprint") != expected_rules: raise ValueError("Serialized Cube rules identity mismatch")
    stones, history = payload.get("stones"), payload.get("superko_history")
    if not isinstance(stones, list) or len(stones) != t.point_count or not isinstance(history, list): raise ValueError("Serialized Cube board/history shape mismatch")
    try:
        return GoldenState(stones=tuple(Stone(x) for x in stones), side_to_move=Stone(payload.get("side_to_move")), superko_history=tuple(tuple(int(x) for x in row) for row in history), consecutive_passes=payload.get("consecutive_passes"), topology=t, rules_id=RULES_PROFILE_ID, rules_fingerprint=expected_rules, komi=komi, history_provenance=payload.get("history_provenance"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Serialized Cube state payload is invalid") from exc


def rotate_cube_state(state, rotation: CubeRotation):
    from .state import GoldenState, Stone
    t = state.topology
    if not isinstance(t, CubeFamilyTopology) or len(rotation.point_permutation) != t.point_count: raise ValueError("Rotation/topology mismatch")
    return GoldenState(stones=tuple(Stone(x) for x in rotation.permute_points(state.stones)), side_to_move=state.side_to_move, superko_history=tuple(rotation.permute_points(p) for p in state.superko_history), consecutive_passes=state.consecutive_passes, topology=t, rules_id=state.rules_id, rules_fingerprint=state.rules_fingerprint, komi=state.komi, history_provenance=state.history_provenance)


def rotate_action_index(t: CubeFamilyTopology, rotation: CubeRotation, action: int) -> int:
    if type(action) is not int or not 0 <= action <= t.pass_action: raise ValueError("Invalid Cube action index")
    return rotation.action(action)


def rotate_rules_action(t: CubeFamilyTopology, rotation: CubeRotation, action):
    from .state import PASS
    if action == PASS: return PASS
    if type(action) is not int or not 0 <= action < t.point_count: raise ValueError("Invalid Cube rules action")
    return rotation.point(action)


__all__ = [name for name in (
    "CROSS_FACE_SEAM", "CUBE_FACES", "CubeFamilyTopology", "CubePointGeometry", "CubeRotation", "CubeSeam",
    "FACE_CORNER", "FACE_EDGE", "FACE_INTERIOR", "FACE_FRAMES", "GAME_GRAPH_SCHEMA_ID", "GEOMETRY_SCHEMA_ID", "SAME_FACE",
    "cube_family_topology", "cube_rotation_matrices", "cube_state_from_stones", "deserialize_cube_state", "family_fingerprints",
    "geometry_class_counts", "initial_cube_state", "rotate_action_index", "rotate_cube_state", "rotate_rules_action", "serialize_cube_state",
)]
