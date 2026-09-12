from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .state import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    BoardKey,
    GoldenState,
    Stone,
    board_key,
    opponent,
)


class IllegalMoveReason(str, Enum):
    OCCUPIED = "occupied"
    SUICIDE = "suicide"
    SUPERKO = "superko"
    INVALID_ACTION_ID = "invalid_action_id"
    MOVE_AFTER_TERMINAL = "move_after_terminal"


class IllegalMoveError(ValueError):
    def __init__(self, reason: IllegalMoveReason, action: object, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.action = action


@dataclass(frozen=True)
class Transition:
    before: GoldenState
    action: int | str
    captured: tuple[int, ...]
    after: GoldenState


def group_from_board(state: GoldenState, point: int) -> frozenset[int]:
    if isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < state.topology.point_count:
        raise ValueError(f"Invalid PointId {point!r}")
    color = state.stones[point]
    if color == EMPTY:
        return frozenset()
    found = {point}
    stack = [point]
    while stack:
        current = stack.pop()
        for neighbor in state.topology.neighbors(current):
            if neighbor not in found and state.stones[neighbor] == color:
                found.add(neighbor)
                stack.append(neighbor)
    return frozenset(found)


def _group_on_stones(
    stones: list[Stone] | tuple[Stone, ...], topology, point: int
) -> frozenset[int]:
    color = stones[point]
    if color == EMPTY:
        return frozenset()
    found = {point}
    stack = [point]
    while stack:
        current = stack.pop()
        for neighbor in topology.neighbors(current):
            if neighbor not in found and stones[neighbor] == color:
                found.add(neighbor)
                stack.append(neighbor)
    return frozenset(found)


def liberties_from_board(state: GoldenState, group: Iterable[int]) -> frozenset[int]:
    liberties: set[int] = set()
    for point in group:
        for neighbor in state.topology.neighbors(point):
            if state.stones[neighbor] == EMPTY:
                liberties.add(neighbor)
    return frozenset(liberties)


def _liberties_on_stones(stones, topology, group: Iterable[int]) -> frozenset[int]:
    liberties: set[int] = set()
    for point in group:
        for neighbor in topology.neighbors(point):
            if stones[neighbor] == EMPTY:
                liberties.add(neighbor)
    return frozenset(liberties)


def _next_state(
    state: GoldenState,
    *,
    stones: tuple[Stone, ...],
    side_to_move: Stone,
    history: tuple[BoardKey, ...],
    consecutive_passes: int,
) -> GoldenState:
    return GoldenState(
        stones=stones,
        side_to_move=side_to_move,
        superko_history=history,
        consecutive_passes=consecutive_passes,
        topology=state.topology,
        rules_id=state.rules_id,
        rules_fingerprint=state.rules_fingerprint,
        komi=state.komi,
    )


def apply_action(state: GoldenState, action: int | str) -> Transition:
    if state.is_terminal:
        raise IllegalMoveError(
            IllegalMoveReason.MOVE_AFTER_TERMINAL,
            action,
            "Golden rules reject every action after formal DOUBLE_PASS terminal",
        )

    if action == PASS:
        passes = state.consecutive_passes + 1
        after = _next_state(
            state,
            stones=state.stones,
            side_to_move=opponent(state.side_to_move),
            history=state.superko_history,
            consecutive_passes=passes,
        )
        return Transition(before=state, action=PASS, captured=(), after=after)

    if isinstance(action, bool) or not isinstance(action, int):
        raise IllegalMoveError(
            IllegalMoveReason.INVALID_ACTION_ID,
            action,
            f"Golden action must be PointId 0..{state.topology.point_count - 1} or PASS",
        )
    if not 0 <= action < state.topology.point_count:
        raise IllegalMoveError(
            IllegalMoveReason.INVALID_ACTION_ID,
            action,
            f"PointId {action} is outside 0..{state.topology.point_count - 1}",
        )
    if state.stones[action] != EMPTY:
        raise IllegalMoveError(
            IllegalMoveReason.OCCUPIED,
            action,
            f"PointId {action} is occupied",
        )

    moving_color = state.side_to_move
    opponent_color = opponent(moving_color)
    provisional = list(state.stones)
    provisional[action] = moving_color

    # Capture decisions are made from one common post-placement snapshot.  This
    # avoids making one opponent group survive merely because another group was
    # removed first.
    captured: set[int] = set()
    checked: set[int] = set()
    for neighbor in state.topology.neighbors(action):
        if provisional[neighbor] != opponent_color or neighbor in checked:
            continue
        group = _group_on_stones(provisional, state.topology, neighbor)
        checked.update(group)
        if not _liberties_on_stones(provisional, state.topology, group):
            captured.update(group)

    for point in captured:
        provisional[point] = EMPTY

    own_group = _group_on_stones(provisional, state.topology, action)
    if not _liberties_on_stones(provisional, state.topology, own_group):
        raise IllegalMoveError(
            IllegalMoveReason.SUICIDE,
            action,
            f"PointId {action} leaves the new {moving_color.name} group with zero liberties",
        )

    new_stones = tuple(provisional)
    new_key = board_key(new_stones)
    # Exact tuple equality is the oracle.  No hash-only acceptance/rejection.
    if new_key in state.superko_history:
        raise IllegalMoveError(
            IllegalMoveReason.SUPERKO,
            action,
            f"PointId {action} recreates an earlier board arrangement under positional superko",
        )

    after = _next_state(
        state,
        stones=new_stones,
        side_to_move=opponent(moving_color),
        history=state.superko_history + (new_key,),
        consecutive_passes=0,
    )
    return Transition(
        before=state,
        action=action,
        captured=tuple(sorted(captured)),
        after=after,
    )


def legal_actions(state: GoldenState) -> tuple[int | str, ...]:
    if state.is_terminal:
        return ()
    legal: list[int | str] = []
    for action in range(state.topology.point_count):
        try:
            apply_action(state, action)
        except IllegalMoveError:
            continue
        legal.append(action)
    legal.append(PASS)
    return tuple(legal)
