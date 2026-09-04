from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

PLAYING = "playing"
ENDGAME = "endgame"

TORUS_SIZES = (9, 13, 19)
CUBE_FACES = ("front", "back", "left", "right", "top", "bottom")

_GROUP_STATUSES = frozenset(("alive", "dead", "seki"))


class IllegalMove(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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
        return tuple(self.point_ids[i] for i in self.neighbor_indices(self.point_index(point_id)))


def torus_topology(size: int) -> Topology:
    if size not in TORUS_SIZES:
        raise ValueError(f"Unsupported torus size: {size}; expected one of {TORUS_SIZES}")

    point_ids = tuple(f"{x},{y}" for y in range(size) for x in range(size))
    index_by_id = {point_id: index for index, point_id in enumerate(point_ids)}

    def idx(x: int, y: int) -> int:
        return (y % size) * size + (x % size)

    neighbors = []
    for y in range(size):
        for x in range(size):
            neighbors.append((
                idx(x - 1, y),
                idx(x + 1, y),
                idx(x, y - 1),
                idx(x, y + 1),
            ))

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
                neighbors.append(tuple(index_by_id[p] for p in (top, right, bottom, left)))

    return Topology("cube", size, point_ids, tuple(neighbors), index_by_id)


def make_topology(kind: str, size: int) -> Topology:
    if kind == "torus":
        return torus_topology(size)
    if kind == "cube":
        return cube_topology(size)
    raise ValueError(f"Unknown topology kind: {kind}")


def _readonly_board(values: Sequence[int] | np.ndarray, expected_size: int) -> np.ndarray:
    board = np.asarray(values, dtype=np.uint8).reshape(-1).copy()
    if board.shape != (expected_size,):
        raise ValueError(f"Expected board of length {expected_size}, got {board.shape}")
    if not np.isin(board, (EMPTY, BLACK, WHITE)).all():
        raise ValueError("Board contains invalid occupancy values")
    board.flags.writeable = False
    return board


@dataclass(frozen=True, eq=False)
class GoState:
    board: np.ndarray
    current_player: int
    turns: int
    consecutive_passes: int
    captures: tuple[int, int]
    previous_board: np.ndarray | None
    phase: str = PLAYING

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GoState):
            return NotImplemented
        return (
            self.current_player == other.current_player
            and self.turns == other.turns
            and self.consecutive_passes == other.consecutive_passes
            and self.captures == other.captures
            and self.phase == other.phase
            and np.array_equal(self.board, other.board)
            and (
                (self.previous_board is None and other.previous_board is None)
                or (
                    self.previous_board is not None
                    and other.previous_board is not None
                    and np.array_equal(self.previous_board, other.previous_board)
                )
            )
        )


def initial_state(topology: Topology) -> GoState:
    return GoState(
        board=_readonly_board(np.zeros(topology.point_count, dtype=np.uint8), topology.point_count),
        current_player=0,
        turns=0,
        consecutive_passes=0,
        captures=(0, 0),
        previous_board=None,
        phase=PLAYING,
    )


def state_from_point_ids(
    topology: Topology,
    *,
    black: Iterable[str] = (),
    white: Iterable[str] = (),
    current_player: int = 0,
    turns: int = 0,
    consecutive_passes: int = 0,
    captures: tuple[int, int] = (0, 0),
    previous_board: np.ndarray | None = None,
    phase: str = PLAYING,
) -> GoState:
    board = np.zeros(topology.point_count, dtype=np.uint8)
    for point_id in black:
        index = topology.point_index(point_id)
        if board[index] != EMPTY:
            raise ValueError(f"Duplicate occupied point: {point_id}")
        board[index] = BLACK
    for point_id in white:
        index = topology.point_index(point_id)
        if board[index] != EMPTY:
            raise ValueError(f"Duplicate occupied point: {point_id}")
        board[index] = WHITE
    previous = None if previous_board is None else _readonly_board(previous_board, topology.point_count)
    return GoState(
        board=_readonly_board(board, topology.point_count),
        current_player=current_player,
        turns=turns,
        consecutive_passes=consecutive_passes,
        captures=captures,
        previous_board=previous,
        phase=phase,
    )


def _stone_for_player(player: int) -> int:
    if player == 0:
        return BLACK
    if player == 1:
        return WHITE
    raise ValueError(f"Invalid player index: {player}")


def _opponent_stone(stone: int) -> int:
    return WHITE if stone == BLACK else BLACK


def _collect_group(board: np.ndarray, start: int, stone: int, topology: Topology) -> tuple[set[int], set[int]]:
    points: set[int] = {start}
    liberties: set[int] = set()
    pending = [start]
    while pending:
        point = pending.pop()
        for neighbor in topology.neighbor_indices(point):
            occupancy = int(board[neighbor])
            if occupancy == EMPTY:
                liberties.add(neighbor)
            elif occupancy == stone and neighbor not in points:
                points.add(neighbor)
                pending.append(neighbor)
    return points, liberties


def stone_groups(state: GoState, topology: Topology) -> tuple[tuple[int, ...], ...]:
    visited: set[int] = set()
    groups: list[tuple[int, ...]] = []
    for point in range(topology.point_count):
        stone = int(state.board[point])
        if stone == EMPTY or point in visited:
            continue
        group, _ = _collect_group(state.board, point, stone, topology)
        visited.update(group)
        groups.append(tuple(sorted(group)))
    return tuple(groups)


def _placement_candidate(state: GoState, action: int, topology: Topology) -> tuple[np.ndarray, int]:
    if action < 0 or action >= topology.point_count:
        raise IllegalMove("invalid-action")
    if int(state.board[action]) != EMPTY:
        raise IllegalMove("occupied")

    board = np.asarray(state.board).copy()
    stone = _stone_for_player(state.current_player)
    opponent = _opponent_stone(stone)
    board[action] = stone

    captured: set[int] = set()
    visited_opponent: set[int] = set()
    for neighbor in topology.neighbor_indices(action):
        if int(board[neighbor]) != opponent or neighbor in visited_opponent:
            continue
        group, liberties = _collect_group(board, neighbor, opponent, topology)
        visited_opponent.update(group)
        if not liberties:
            captured.update(group)

    for point in captured:
        board[point] = EMPTY

    _, own_liberties = _collect_group(board, action, stone, topology)
    if not own_liberties:
        raise IllegalMove("suicide")

    if state.previous_board is not None and np.array_equal(board, state.previous_board):
        raise IllegalMove("repetition")

    return board, len(captured)


def apply_action(state: GoState, action: int, topology: Topology) -> GoState:
    if state.phase != PLAYING:
        raise IllegalMove("not-playing")

    if action == topology.pass_action:
        consecutive_passes = state.consecutive_passes + 1
        return GoState(
            board=state.board,
            current_player=1 - state.current_player,
            turns=state.turns + 1,
            consecutive_passes=consecutive_passes,
            captures=state.captures,
            previous_board=state.board,
            phase=ENDGAME if consecutive_passes >= 2 else PLAYING,
        )

    board, captured = _placement_candidate(state, action, topology)
    captures = list(state.captures)
    captures[state.current_player] += captured
    return GoState(
        board=_readonly_board(board, topology.point_count),
        current_player=1 - state.current_player,
        turns=state.turns + 1,
        consecutive_passes=0,
        captures=(captures[0], captures[1]),
        previous_board=state.board,
        phase=PLAYING,
    )


def valid_moves(state: GoState, topology: Topology) -> np.ndarray:
    result = np.zeros(topology.action_size, dtype=np.uint8)
    if state.phase != PLAYING:
        return result

    for action in range(topology.point_count):
        if int(state.board[action]) != EMPTY:
            continue
        try:
            _placement_candidate(state, action, topology)
        except IllegalMove:
            continue
        result[action] = 1
    result[topology.pass_action] = 1
    return result


@dataclass(frozen=True)
class GroupClassification:
    points: tuple[int, ...]
    status: str


@dataclass(frozen=True)
class TerritoryBreakdown:
    black: int
    white: int
    neutral: int
    seki: int


@dataclass(frozen=True)
class TerritoryPoints:
    black: tuple[int, ...]
    white: tuple[int, ...]
    neutral: tuple[int, ...]
    seki: tuple[int, ...]


@dataclass(frozen=True)
class StoneBreakdown:
    black: int
    white: int


@dataclass(frozen=True)
class FinalScore:
    ruleset: str
    black: float
    white: float
    komi: float
    territory: TerritoryBreakdown
    territory_points: TerritoryPoints
    stones_on_board: StoneBreakdown
    captures: tuple[int, int]
    prisoners: tuple[int, int] | None
    dead_stones: StoneBreakdown
    winner: str
    margin: float


def _classification_statuses(
    state: GoState,
    topology: Topology,
    classification: Sequence[GroupClassification],
) -> dict[int, str]:
    expected_groups = {frozenset(group) for group in stone_groups(state, topology)}
    supplied_groups: set[frozenset[int]] = set()
    statuses: dict[int, str] = {}

    for item in classification:
        if item.status not in _GROUP_STATUSES:
            raise ValueError(f"Invalid group status: {item.status}")
        group = frozenset(item.points)
        if not group:
            raise ValueError("Classification group must not be empty")
        if group not in expected_groups:
            raise ValueError("Classification must contain complete logical stone groups")
        if group in supplied_groups:
            raise ValueError("Duplicate classified group")
        supplied_groups.add(group)
        for point in group:
            statuses[point] = item.status

    if supplied_groups != expected_groups:
        raise ValueError("Scoring requires a complete classification of every stone group")
    return statuses


def score_position(
    state: GoState,
    topology: Topology,
    classification: Sequence[GroupClassification],
    ruleset: str,
    komi: float,
) -> FinalScore:
    if ruleset not in ("chinese", "japanese"):
        raise ValueError(f"Unsupported ruleset: {ruleset}")
    if not np.isfinite(komi):
        raise ValueError("Komi must be finite")

    statuses = _classification_statuses(state, topology, classification)
    effective = np.asarray(state.board).copy()
    dead = [0, 0]
    seki_stones: set[int] = set()

    for point, status in statuses.items():
        occupancy = int(state.board[point])
        if status == "dead":
            dead[0 if occupancy == BLACK else 1] += 1
            effective[point] = EMPTY
        elif status == "seki":
            seki_stones.add(point)

    stones = [int(np.count_nonzero(effective == BLACK)), int(np.count_nonzero(effective == WHITE))]
    territory_points = [[], [], [], []]  # black, white, neutral, seki
    visited: set[int] = set()

    for start in range(topology.point_count):
        if int(effective[start]) != EMPTY or start in visited:
            continue
        region: list[int] = []
        boundary_colors: set[int] = set()
        touches_seki = False
        pending = [start]
        visited.add(start)
        while pending:
            point = pending.pop()
            region.append(point)
            for neighbor in topology.neighbor_indices(point):
                occupancy = int(effective[neighbor])
                if occupancy == EMPTY:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        pending.append(neighbor)
                else:
                    boundary_colors.add(occupancy)
                    if neighbor in seki_stones:
                        touches_seki = True

        if touches_seki:
            territory_points[3].extend(region)
        elif boundary_colors == {BLACK}:
            territory_points[0].extend(region)
        elif boundary_colors == {WHITE}:
            territory_points[1].extend(region)
        else:
            territory_points[2].extend(region)

    territory = TerritoryBreakdown(*(len(points) for points in territory_points))
    frozen_territory_points = TerritoryPoints(*(tuple(points) for points in territory_points))
    stones_breakdown = StoneBreakdown(stones[0], stones[1])
    dead_breakdown = StoneBreakdown(dead[0], dead[1])

    if ruleset == "chinese":
        prisoners = None
        black_score = float(stones[0] + territory.black)
        white_score = float(stones[1] + territory.white + komi)
    else:
        prisoners = (state.captures[0] + dead[1], state.captures[1] + dead[0])
        black_score = float(territory.black + prisoners[0])
        white_score = float(territory.white + prisoners[1] + komi)

    winner = "draw" if black_score == white_score else ("black" if black_score > white_score else "white")
    return FinalScore(
        ruleset=ruleset,
        black=black_score,
        white=white_score,
        komi=float(komi),
        territory=territory,
        territory_points=frozen_territory_points,
        stones_on_board=stones_breakdown,
        captures=state.captures,
        prisoners=prisoners,
        dead_stones=dead_breakdown,
        winner=winner,
        margin=abs(black_score - white_score),
    )
