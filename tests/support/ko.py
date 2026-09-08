"""Independent positional-restoration checks for simple-ko fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .independent_graph import (
    BLACK,
    WHITE,
    IndependentIllegalMove,
    IndependentMoveResult,
    apply_move,
)


@dataclass(frozen=True)
class KoProof:
    capture_legal: bool
    captured_points: frozenset[int]
    recapture_legal_without_ko: bool
    recapture_reason: str | None
    restores_initial_board: bool
    is_simple_ko: bool
    after_capture: tuple[int, ...] | None


def prove_positional_restoration(
    before_board: Sequence[int],
    capturing_player: int,
    capture_action: int,
    recapture_action: int,
    adjacency: Sequence[Sequence[int]],
) -> KoProof:
    """Prove a true/false simple-ko pattern without consulting production code."""

    try:
        capture = apply_move(before_board, capturing_player, capture_action, adjacency)
    except IndependentIllegalMove as error:
        return KoProof(False, frozenset(), False, error.reason, False, False, None)
    opponent = WHITE if capturing_player == BLACK else BLACK
    try:
        recapture = apply_move(capture.board, opponent, recapture_action, adjacency)
    except IndependentIllegalMove as error:
        return KoProof(True, capture.captured_points, False, error.reason, False, False, capture.board)
    return KoProof(
        capture_legal=True,
        captured_points=capture.captured_points,
        recapture_legal_without_ko=True,
        recapture_reason=None,
        restores_initial_board=recapture.board == tuple(int(value) for value in before_board),
        is_simple_ko=(
            len(capture.captured_points) == 1
            and recapture.board == tuple(int(value) for value in before_board)
        ),
        after_capture=capture.board,
    )


def apply_fixture_actions(
    board: Sequence[int],
    actions: Sequence[int],
    to_move: int,
    adjacency: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[frozenset[int], ...]]:
    """Apply placement-only actions and return the board plus captured groups."""

    current = tuple(int(value) for value in board)
    player = to_move
    captures = []
    for action in actions:
        result: IndependentMoveResult = apply_move(current, player, int(action), adjacency)
        current = result.board
        captures.extend(group.stones for group in result.captured_groups)
        player = WHITE if player == BLACK else BLACK
    return current, tuple(captures)
