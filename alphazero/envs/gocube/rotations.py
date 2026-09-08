"""Canonical 24 orientation-preserving Cube rotations.

This is the production counterpart of the verification-groundwork rotation
convention. It uses canonical PointIds and fixed cube face frames; structural
features themselves do not depend on this module and remain adjacency-only.
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


def _signed_permutation_matrices() -> tuple[Matrix, ...]:
    matrices = []
    for axes in permutations(range(3)):
        parity = 1 if sum(
            axes[i] > axes[j] for i in range(3) for j in range(i + 1, 3)
        ) % 2 == 0 else -1
        for signs in product((-1, 1), repeat=3):
            matrix = tuple(
                tuple(signs[row] if column == axes[row] else 0 for column in range(3))
                for row in range(3)
            )
            if parity * signs[0] * signs[1] * signs[2] == 1:
                matrices.append(matrix)  # type: ignore[arg-type]
    return tuple(sorted(set(matrices)))


# (outward normal, local column direction, local row direction). These are
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
        inverse_matrix = tuple(
            tuple(self.matrix[column][row] for column in range(3))
            for row in range(3)
        )
        return CubeRotation(-1, inverse_matrix, tuple(inverse))  # type: ignore[arg-type]

    def compose(self, other: "CubeRotation") -> tuple[int, ...]:
        """Return the permutation for applying ``self`` and then ``other``."""

        return tuple(
            other.permutation[self.permutation[point]]
            for point in range(len(self.permutation))
        )


def cube_rotations(topology: Any) -> tuple[CubeRotation, ...]:
    """Build exactly 24 rotations for a canonical Cube topology."""

    if getattr(topology, "kind", None) != "cube":
        raise ValueError("Cube rotations require a topology with kind='cube'")
    size = int(topology.size)
    if size < 2:
        raise ValueError("Cube rotations require size >= 2")
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
            permutation.append(
                topology.point_index(f"{target_face}:{target_row}:{target_column}")
            )
        rotations.append(CubeRotation(index, matrix, tuple(permutation)))
    rotations.sort(key=lambda rotation: rotation.permutation)
    result = tuple(replace(rotation, index=index) for index, rotation in enumerate(rotations))
    if len(result) != 24 or len({rotation.permutation for rotation in result}) != 24:
        raise AssertionError("Cube rotation convention must contain 24 unique rotations")
    return result


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


def permute_point_axis(values, rotation_or_permutation, axis: int = -1):
    """Apply a source-to-target point permutation to an array or tensor."""

    permutation = getattr(rotation_or_permutation, "permutation", rotation_or_permutation)
    permutation = tuple(int(value) for value in permutation)
    try:
        import torch
    except ImportError:  # pragma: no cover - PyTorch is a project dependency.
        torch = None
    if torch is not None and isinstance(values, torch.Tensor):
        axis = axis if axis >= 0 else values.ndim + axis
        if axis < 0 or axis >= values.ndim or values.shape[axis] != len(permutation):
            raise ValueError("Point permutation axis does not match permutation size")
        inverse = [0] * len(permutation)
        for source, target in enumerate(permutation):
            inverse[target] = source
        indices = torch.as_tensor(inverse, dtype=torch.long, device=values.device)
        return torch.index_select(values, axis, indices)

    import numpy as np

    array = np.asarray(values)
    axis = axis if axis >= 0 else array.ndim + axis
    if axis < 0 or axis >= array.ndim or array.shape[axis] != len(permutation):
        raise ValueError("Point permutation axis does not match permutation size")
    result = np.empty_like(array)
    for source, target in enumerate(permutation):
        source_slice = [slice(None)] * array.ndim
        target_slice = [slice(None)] * array.ndim
        source_slice[axis] = source
        target_slice[axis] = target
        result[tuple(target_slice)] = array[tuple(source_slice)]
    return result


# Friendly aliases for symmetry callers.
cube_rotation_permutations = cube_rotations
all_cube_rotations = cube_rotations
