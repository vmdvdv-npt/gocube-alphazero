"""Production-independent endgame proofs used by the V1 corpus.

The helpers in this module deliberately consume only an occupancy vector and
an adjacency list.  Production Benson/life/scoring functions are called by
the tests only after these graph facts have been established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .exhaustive_solver import PROVED_DRAW, ExhaustiveSolver, SolverState
from .independent_graph import (
    BLACK,
    EMPTY,
    WHITE,
    IndependentIllegalMove,
    apply_move,
    empty_regions,
    find_groups,
)


@dataclass(frozen=True)
class VitalRegionProof:
    """A graph-only two-vital-region proof for one stone group."""

    color: int
    group: frozenset[int]
    vital_regions: tuple[frozenset[int], ...]


@dataclass(frozen=True)
class PlacementExhaustionProof:
    """Proof that the opponent has no legal placement in a closed board."""

    color: int
    target_group: frozenset[int]
    opponent: int
    opponent_legal_actions: tuple[int, ...]

    @property
    def proved(self) -> bool:
        return not self.opponent_legal_actions


@dataclass(frozen=True)
class SekiReplyLine:
    """One independently checked first-placement/answer continuation."""

    player: int
    first_action: int
    defender: int
    defensive_action: int
    captured_first_group: frozenset[int]


@dataclass(frozen=True)
class SettledSekiProof:
    """Bounded exhaustive proof for a closed two-shared-liberty seki."""

    status: str
    search_status: str
    nodes: int
    max_depth: int
    max_depth_reached: int
    black_group: frozenset[int]
    white_group: frozenset[int]
    shared_liberties: tuple[int, ...]
    legal_first_actions: tuple[tuple[int, tuple[int, ...]], ...]
    reply_lines: tuple[SekiReplyLine, ...]


def mixed_border_regions(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> tuple[frozenset[int], ...]:
    """Return empty regions independently classified as dame."""

    return tuple(
        region.points
        for region in empty_regions(board, adjacency)
        if region.bordering_colors == frozenset((BLACK, WHITE))
    )


def prove_two_vital_regions(
    board: Sequence[int],
    color: int,
    adjacency: Sequence[Sequence[int]],
) -> VitalRegionProof:
    """Prove Benson-style unconditional life from two exclusive regions.

    A vital region must be empty, border exactly one group, and have no
    opposing-color boundary.  This is a graph fact; it does not call the
    production pass-alive implementation.
    """

    if color not in (BLACK, WHITE):
        raise ValueError("color must be BLACK or WHITE")
    groups = tuple(group for group in find_groups(board, adjacency) if group.color == color)
    if len(groups) != 1:
        raise ValueError("two-vital-region proof requires exactly one target group")
    target = groups[0]
    vital = []
    for region in empty_regions(board, adjacency):
        if region.bordering_colors != frozenset((color,)):
            continue
        bordering = (
            region.bordering_black_groups
            if color == BLACK
            else region.bordering_white_groups
        )
        if bordering == (target.stones,):
            vital.append(region.points)
    if len(vital) < 2:
        raise ValueError("fewer than two exclusive vital regions")
    return VitalRegionProof(color, target.stones, tuple(vital))


def prove_opponent_placement_exhaustion(
    board: Sequence[int],
    color: int,
    adjacency: Sequence[Sequence[int]],
) -> PlacementExhaustionProof:
    """Enumerate all opponent placements in a closed pass-alive control."""

    if color not in (BLACK, WHITE):
        raise ValueError("color must be BLACK or WHITE")
    groups = tuple(group for group in find_groups(board, adjacency) if group.color == color)
    if len(groups) != 1:
        raise ValueError("placement-exhaustion proof requires exactly one target group")
    opponent = WHITE if color == BLACK else BLACK
    legal = []
    for action, value in enumerate(tuple(int(value) for value in board)):
        if value != EMPTY:
            continue
        try:
            apply_move(board, opponent, action, adjacency)
        except IndependentIllegalMove:
            continue
        legal.append(action)
    return PlacementExhaustionProof(color, groups[0].stones, opponent, tuple(legal))


def prove_settled_seki(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    *,
    max_depth: int = 3,
    max_nodes: int = 512,
) -> SettledSekiProof:
    """Prove a closed two-shared-liberty seki by bounded exhaustive search.

    The root has exactly two opposite-color groups and exactly two empty
    points.  Every legal first placement is therefore one of the shared
    liberties.  The explicit reply table checks that the other liberty is a
    legal response which captures the first player's group.  The independent
    minimax search then exhausts all placements and PASS branches to depth 3;
    ``proved_draw`` means neither player can force a capture against a
    defender who chooses the safe continuation.
    """

    groups = find_groups(board, adjacency)
    black_groups = tuple(group for group in groups if group.color == BLACK)
    white_groups = tuple(group for group in groups if group.color == WHITE)
    if len(black_groups) != 1 or len(white_groups) != 1:
        raise ValueError("settled seki proof requires exactly one group of each color")
    black_group, white_group = black_groups[0], white_groups[0]
    if black_group.liberties != white_group.liberties or len(black_group.liberties) != 2:
        raise ValueError("groups must have exactly the same two liberties")
    shared = tuple(sorted(black_group.liberties))
    occupied = set(black_group.stones) | set(white_group.stones)
    if set(range(len(tuple(board)))) - occupied != set(shared):
        raise ValueError("settled seki proof requires a closed board with two empty liberties")
    if not any(set(region.points) == set(shared) for region in empty_regions(board, adjacency)):
        raise ValueError("the two shared liberties must form one empty graph region")

    legal_first_actions = []
    reply_lines = []
    for player, target in ((BLACK, black_group), (WHITE, white_group)):
        legal = []
        for action in shared:
            first = apply_move(board, player, action, adjacency)
            if first.captured_groups:
                raise ValueError("first placement unexpectedly captures in seki root")
            legal.append(action)
            defender = WHITE if player == BLACK else BLACK
            defensive_action = next(point for point in shared if point != action)
            response = apply_move(first.board, defender, defensive_action, adjacency)
            captured = response.captured_points
            expected_group = target.stones | {action}
            if not expected_group.issubset(captured):
                raise ValueError("the defensive reply does not capture the first group")
            reply_lines.append(
                SekiReplyLine(player, action, defender, defensive_action, frozenset(captured))
            )
        legal_first_actions.append((player, tuple(legal)))

    def terminal(state: SolverState) -> int | str | None:
        if any(state.board[point] != BLACK for point in black_group.stones):
            return WHITE
        if any(state.board[point] != WHITE for point in white_group.stones):
            return BLACK
        if state.consecutive_passes >= 2:
            return "draw"
        return None

    solver = ExhaustiveSolver(
        adjacency,
        max_nodes=max_nodes,
        max_depth=max_depth,
        include_pass=True,
        enforce_simple_ko=True,
    )
    result = solver.solve(
        SolverState(board=tuple(int(value) for value in board), to_move=BLACK),
        terminal,
    )
    if result.status != PROVED_DRAW or result.exhausted:
        raise ValueError(
            f"settled seki search did not prove draw: {result.status}, exhausted={result.exhausted}"
        )
    return SettledSekiProof(
        status="proved_settled_seki",
        search_status=result.status,
        nodes=result.nodes,
        max_depth=max_depth,
        max_depth_reached=result.max_depth_reached,
        black_group=black_group.stones,
        white_group=white_group.stones,
        shared_liberties=shared,
        legal_first_actions=tuple(legal_first_actions),
        reply_lines=tuple(reply_lines),
    )
