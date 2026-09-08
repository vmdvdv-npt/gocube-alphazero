"""Small, production-independent graph rules used by verification tests.

The module intentionally accepts only an occupancy vector and an adjacency
list.  A caller may obtain the adjacency list from the canonical GoCube
topology, but group, liberty, region, and capture derivation lives here rather
than in the production engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

EMPTY = 0
BLACK = 1
WHITE = 2
PASS = -1
_COLORS = frozenset((BLACK, WHITE))


class IndependentIllegalMove(ValueError):
    """A move rejected by the independent local rules checker."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _normalized_board(board: Sequence[int], point_count: int | None = None) -> tuple[int, ...]:
    result = tuple(int(value) for value in board)
    if point_count is not None and len(result) != point_count:
        raise ValueError(f"Expected {point_count} points, got {len(result)}")
    if any(value not in (EMPTY, BLACK, WHITE) for value in result):
        raise ValueError("Board contains an occupancy value other than 0, 1, or 2")
    return result


def _check_point(point: int, point_count: int) -> None:
    if not isinstance(point, int) or isinstance(point, bool) or not 0 <= point < point_count:
        raise ValueError(f"Invalid point index: {point!r}")


def _adjacency(adjacency: Sequence[Sequence[int]], point_count: int) -> tuple[tuple[int, ...], ...]:
    if len(adjacency) != point_count:
        raise ValueError("Adjacency length must equal board length")
    normalized = []
    for point, neighbors in enumerate(adjacency):
        values = tuple(int(neighbor) for neighbor in neighbors)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate neighbor at point {point}")
        if any(neighbor < 0 or neighbor >= point_count for neighbor in values):
            raise ValueError(f"Out-of-range neighbor at point {point}")
        if point in values:
            raise ValueError(f"Self-neighbor at point {point}")
        normalized.append(values)
    return tuple(normalized)


@dataclass(frozen=True)
class IndependentGroup:
    color: int
    stones: frozenset[int]
    liberties: frozenset[int]

    @property
    def points(self) -> frozenset[int]:
        """Compatibility name for fixtures that call stones ``points``."""

        return self.stones


@dataclass(frozen=True)
class EmptyRegion:
    points: frozenset[int]
    bordering_black_groups: tuple[frozenset[int], ...]
    bordering_white_groups: tuple[frozenset[int], ...]

    @property
    def bordering_colors(self) -> frozenset[int]:
        colors = set()
        if self.bordering_black_groups:
            colors.add(BLACK)
        if self.bordering_white_groups:
            colors.add(WHITE)
        return frozenset(colors)


@dataclass(frozen=True)
class IndependentMoveResult:
    board: tuple[int, ...]
    player: int
    action: int
    captured_groups: tuple[IndependentGroup, ...]
    own_group: IndependentGroup

    @property
    def captured_points(self) -> frozenset[int]:
        points: set[int] = set()
        for group in self.captured_groups:
            points.update(group.stones)
        return frozenset(points)

    @property
    def capture_count(self) -> int:
        return len(self.captured_points)


def find_group(
    start_point: int,
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> IndependentGroup:
    """Return one connected stone group and its *unique* liberties."""

    values = _normalized_board(board)
    graph = _adjacency(adjacency, len(values))
    _check_point(start_point, len(values))
    color = values[start_point]
    if color not in _COLORS:
        raise ValueError("find_group requires an occupied starting point")
    stones = {start_point}
    liberties: set[int] = set()
    pending = [start_point]
    while pending:
        point = pending.pop()
        for neighbor in graph[point]:
            occupancy = values[neighbor]
            if occupancy == EMPTY:
                liberties.add(neighbor)
            elif occupancy == color and neighbor not in stones:
                stones.add(neighbor)
                pending.append(neighbor)
    return IndependentGroup(color, frozenset(stones), frozenset(liberties))


def find_groups(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    color: int | None = None,
) -> tuple[IndependentGroup, ...]:
    """Enumerate each logical group exactly once, in point order."""

    values = _normalized_board(board)
    graph = _adjacency(adjacency, len(values))
    if color is not None and color not in _COLORS:
        raise ValueError("color must be BLACK, WHITE, or None")
    visited: set[int] = set()
    result = []
    for point, occupancy in enumerate(values):
        if occupancy == EMPTY or point in visited or (color is not None and occupancy != color):
            continue
        group = find_group(point, values, graph)
        visited.update(group.stones)
        result.append(group)
    return tuple(result)


def empty_regions(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> tuple[EmptyRegion, ...]:
    """Flood-fill empty components and attach their bordering stone groups."""

    values = _normalized_board(board)
    graph = _adjacency(adjacency, len(values))
    groups = find_groups(values, graph)
    group_by_stone = {
        stone: group
        for group in groups
        for stone in group.stones
    }
    visited: set[int] = set()
    regions = []
    for start, occupancy in enumerate(values):
        if occupancy != EMPTY or start in visited:
            continue
        visited.add(start)
        points = {start}
        pending = [start]
        bordering: dict[int, set[frozenset[int]]] = {BLACK: set(), WHITE: set()}
        while pending:
            point = pending.pop()
            for neighbor in graph[point]:
                neighbor_value = values[neighbor]
                if neighbor_value == EMPTY and neighbor not in visited:
                    visited.add(neighbor)
                    points.add(neighbor)
                    pending.append(neighbor)
                elif neighbor_value in _COLORS:
                    bordering[neighbor_value].add(group_by_stone[neighbor].stones)
        regions.append(
            EmptyRegion(
                points=frozenset(points),
                bordering_black_groups=tuple(sorted(bordering[BLACK], key=lambda group: min(group))),
                bordering_white_groups=tuple(sorted(bordering[WHITE], key=lambda group: min(group))),
            )
        )
    return tuple(regions)


def graph_triangles(adjacency: Sequence[Sequence[int]]) -> tuple[tuple[int, int, int], ...]:
    """Find all 3-cliques using only graph adjacency."""

    graph = tuple(frozenset(int(neighbor) for neighbor in neighbors) for neighbors in adjacency)
    triangles = []
    for a in range(len(graph)):
        for b in sorted(neighbor for neighbor in graph[a] if neighbor > a):
            for c in sorted(neighbor for neighbor in graph[a] & graph[b] if neighbor > b):
                triangles.append((a, b, c))
    return tuple(triangles)


def triangle_membership(
    triangles: Iterable[Sequence[int]],
) -> tuple[int, ...]:
    triangles = tuple(tuple(int(point) for point in triangle) for triangle in triangles)
    point_count = max((point for triangle in triangles for point in triangle), default=-1) + 1
    membership = [0] * point_count
    for triangle in triangles:
        if len(triangle) != 3 or len(set(triangle)) != 3:
            raise ValueError("A graph triangle must contain three distinct points")
        for point in triangle:
            membership[point] += 1
    return tuple(membership)


def apply_move(
    board: Sequence[int],
    player: int,
    action: int,
    adjacency: Sequence[Sequence[int]],
    *,
    previous_board: Sequence[int] | None = None,
    enforce_simple_ko: bool = False,
) -> IndependentMoveResult:
    """Apply one placement with capture-before-suicide ordering.

    The function deliberately has no cleanup, pass-alive, or production-ko
    policy.  ``previous_board`` is an optional exact positional comparison for
    small simple-ko proofs only.
    """

    values = _normalized_board(board)
    graph = _adjacency(adjacency, len(values))
    if player not in _COLORS:
        raise ValueError("player must be BLACK or WHITE")
    _check_point(action, len(values))
    if values[action] != EMPTY:
        raise IndependentIllegalMove("occupied")

    opponent = WHITE if player == BLACK else BLACK
    candidate = list(values)
    candidate[action] = player
    captured: list[IndependentGroup] = []
    visited_opponent: set[int] = set()
    for neighbor in graph[action]:
        if candidate[neighbor] != opponent or neighbor in visited_opponent:
            continue
        group = find_group(neighbor, candidate, graph)
        visited_opponent.update(group.stones)
        if not group.liberties:
            captured.append(group)
            for point in group.stones:
                candidate[point] = EMPTY

    own_group = find_group(action, candidate, graph)
    if not own_group.liberties:
        raise IndependentIllegalMove("suicide")
    if enforce_simple_ko and previous_board is not None:
        if tuple(candidate) == _normalized_board(previous_board, len(values)):
            raise IndependentIllegalMove("simple-ko")
    return IndependentMoveResult(
        board=tuple(candidate),
        player=player,
        action=action,
        captured_groups=tuple(captured),
        own_group=own_group,
    )


def board_from_colors(
    point_count: int,
    black: Iterable[int] = (),
    white: Iterable[int] = (),
) -> tuple[int, ...]:
    """Build a checked immutable occupancy vector for compact fixtures."""

    board = [EMPTY] * point_count
    for point in black:
        _check_point(int(point), point_count)
        if board[point] != EMPTY:
            raise ValueError(f"Overlapping or duplicate point: {point}")
        board[point] = BLACK
    for point in white:
        _check_point(int(point), point_count)
        if board[point] != EMPTY:
            raise ValueError(f"Overlapping or duplicate point: {point}")
        board[point] = WHITE
    return tuple(board)
