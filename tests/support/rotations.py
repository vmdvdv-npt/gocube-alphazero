"""The 24 orientation-preserving cube rotations for canonical PointIds.

Rotations are derived from face frames, not from the renderer or a net.  A
face frame records its outward normal, column direction, and row direction;
rotating that frame identifies the destination face and local coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations, product
from typing import Any, Iterable, Sequence

Vector = tuple[int, int, int]
Matrix = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


def _dot(left: Vector, right: Vector) -> int:
    return sum(a * b for a, b in zip(left, right))


def _mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(_dot(row, vector) for row in matrix)  # type: ignore[return-value]


def _determinant(matrix: Matrix) -> int:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _signed_permutation_matrices() -> tuple[Matrix, ...]:
    matrices = []
    for axes in permutations(range(3)):
        parity = 1 if sum(axes[i] > axes[j] for i in range(3) for j in range(i + 1, 3)) % 2 == 0 else -1
        for signs in product((-1, 1), repeat=3):
            matrix = tuple(
                tuple(signs[row] if column == axes[row] else 0 for column in range(3))
                for row in range(3)
            )
            if parity * signs[0] * signs[1] * signs[2] == 1:
                matrices.append(matrix)  # type: ignore[arg-type]
    return tuple(sorted(set(matrices)))


# (outward normal, local column direction, local row direction).  These are
# the canonical orientations implied by core.cube_topology's edge table.
_FRAMES: dict[str, tuple[Vector, Vector, Vector]] = {
    "front": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "back": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
    "left": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "right": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "top": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "bottom": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
}
_FACE_BY_NORMAL = {frame[0]: face for face, frame in _FRAMES.items()}


@dataclass(frozen=True)
class CubeRotation:
    """A permutation mapping each source point index to its destination."""

    index: int
    matrix: Matrix
    permutation: tuple[int, ...]

    def apply_point(self, point: int) -> int:
        return self.permutation[point]

    def inverse(self) -> "CubeRotation":
        inverse = [0] * len(self.permutation)
        for source, target in enumerate(self.permutation):
            inverse[target] = source
        inverse_matrix = tuple(tuple(self.matrix[column][row] for column in range(3)) for row in range(3))
        return CubeRotation(-1, inverse_matrix, tuple(inverse))  # type: ignore[arg-type]

    def compose(self, other: "CubeRotation") -> tuple[int, ...]:
        """Return the permutation for applying ``self`` and then ``other``."""

        return tuple(other.permutation[self.permutation[point]] for point in range(len(self.permutation)))


def cube_rotations(topology: Any) -> tuple[CubeRotation, ...]:
    """Build exactly 24 rotations for a canonical cube topology."""

    if getattr(topology, "kind", None) != "cube":
        raise ValueError("Cube rotations require a topology with kind='cube'")
    size = int(topology.size)
    last = size - 1
    rotations = []
    for index, matrix in enumerate(_signed_permutation_matrices()):
        permutation = []
        for point_id in topology.point_ids:
            face, row_text, column_text = point_id.split(":")
            row = int(row_text)
            column = int(column_text)
            normal, u_axis, v_axis = _FRAMES[face]
            # Twice-centered coordinates avoid fractional centers for even n.
            centered = tuple(
                normal[axis] * last
                + u_axis[axis] * (2 * column - last)
                + v_axis[axis] * (2 * row - last)
                for axis in range(3)
            )
            transformed_normal = _mat_vec(matrix, normal)
            target_face = _FACE_BY_NORMAL[transformed_normal]
            target_frame = _FRAMES[target_face]
            transformed = _mat_vec(matrix, centered)
            target_column = (_dot(transformed, target_frame[1]) + last) // 2
            target_row = (_dot(transformed, target_frame[2]) + last) // 2
            if not (0 <= target_row < size and 0 <= target_column < size):
                raise AssertionError(f"Rotation produced an invalid point: {point_id}")
            permutation.append(topology.point_index(f"{target_face}:{target_row}:{target_column}"))
        rotations.append(CubeRotation(index, matrix, tuple(permutation)))
    rotations.sort(key=lambda rotation: rotation.permutation)
    return tuple(replace(rotation, index=index) for index, rotation in enumerate(rotations))


def rotate_point_set(points: Iterable[int], rotation: CubeRotation) -> frozenset[int]:
    return frozenset(rotation.apply_point(int(point)) for point in points)


def rotate_board(board: Sequence[int], rotation: CubeRotation) -> tuple[int, ...]:
    if len(board) != len(rotation.permutation):
        raise ValueError("Board and rotation have different point counts")
    rotated = [0] * len(board)
    for source, value in enumerate(board):
        rotated[rotation.apply_point(source)] = int(value)
    return tuple(rotated)


def rotate_action(action: int, rotation: CubeRotation, pass_action: int | None = None) -> int:
    if pass_action is not None and action == pass_action:
        return action
    return rotation.apply_point(int(action))


def _rotate_payload(value: Any, rotation: CubeRotation, topology: Any, key: str | None = None) -> Any:
    point_id_set = set(topology.point_ids)
    if isinstance(value, str) and value in point_id_set:
        return topology.point_id(rotation.apply_point(topology.point_index(value)))
    if isinstance(value, dict):
        rotated = {}
        for item_key, item in value.items():
            rotated_key = (
                topology.point_id(rotation.apply_point(topology.point_index(item_key)))
                if isinstance(item_key, str) and item_key in point_id_set
                else item_key
            )
            rotated[rotated_key] = _rotate_payload(item, rotation, topology)
        return rotated
    if isinstance(value, tuple):
        return tuple(_rotate_payload(item, rotation, topology, key) for item in value)
    if isinstance(value, list):
        return [_rotate_payload(item, rotation, topology, key) for item in value]
    if isinstance(value, frozenset):
        return frozenset(_rotate_payload(item, rotation, topology, key) for item in value)
    if isinstance(value, set):
        return {_rotate_payload(item, rotation, topology, key) for item in value}
    return value


def rotate_fixture(fixture: Any, topology: Any, rotation: CubeRotation) -> Any:
    """Rotate all point-bearing state carried by a VerificationFixture.

    The fixture object is duck-typed to keep this machinery usable by small
    test-only fixture dataclasses without creating a production dependency.
    """

    point_ids = lambda values: tuple(
        topology.point_id(rotation.apply_point(topology.point_index(value))) for value in values
    )
    changes = {
        "id": f"{fixture.id}__r{rotation.index:02d}",
        "source_fixture_id": getattr(fixture, "source_fixture_id", None) or fixture.id,
        "rotation_index": rotation.index,
        "black": point_ids(fixture.black),
        "white": point_ids(fixture.white),
        "actions": tuple(
            action
            if action in ("PASS", "pass", None)
            else topology.point_id(rotation.apply_point(topology.point_index(action)))
            for action in fixture.actions
        ),
        "expected": _rotate_payload(fixture.expected, rotation, topology),
    }
    if getattr(fixture, "previous_board", None) is not None:
        changes["previous_board"] = rotate_board(fixture.previous_board, rotation)
    for field_name in ("ko_state", "cleanup_metadata", "external_context"):
        if hasattr(fixture, field_name):
            changes[field_name] = _rotate_payload(getattr(fixture, field_name), rotation, topology)
    if getattr(fixture, "second_cleanup_start_colors", None) is not None:
        changes["second_cleanup_start_colors"] = rotate_board(fixture.second_cleanup_start_colors, rotation)
    return replace(fixture, **changes)


def rotate_board_state(
    *,
    board: Sequence[int],
    rotation: CubeRotation,
    previous_board: Sequence[int] | None = None,
    point_masks: Iterable[int] = (),
    start_colors: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Rotate a complete point-indexed state fragment for metamorphic tests."""

    return {
        "board": rotate_board(board, rotation),
        "previous_board": None if previous_board is None else rotate_board(previous_board, rotation),
        "point_mask": rotate_point_set(point_masks, rotation),
        "start_colors": None if start_colors is None else rotate_board(start_colors, rotation),
    }
