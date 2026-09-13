from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .diagnostics import increment
from .state import (
    BLACK, EMPTY, PASS, WHITE, BoardKey, GoldenState, Stone, board_key, opponent,
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


@dataclass(frozen=True)
class ActionProbe:
    """Exact point-move result without constructing a child GoldenState."""

    action: int
    captured: tuple[int, ...]
    stones: tuple[Stone, ...]
    board_key: BoardKey


@dataclass(frozen=True)
class LegalActionContext:
    """One exact legality calculation reusable by observation and search."""

    state_key: tuple[object, ...]
    actions: tuple[int | str, ...]
    action_mask: tuple[bool, ...]

    def assert_compatible(self, state: GoldenState) -> None:
        if state.state_key != self.state_key:
            raise ValueError("Prepared legal-action context belongs to another state")

def group_from_board(state: GoldenState, point: int) -> frozenset[int]:
    if isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < state.topology.point_count:
        raise ValueError(f"Invalid PointId {point!r}")
    color = state.stones[point]
    if color == EMPTY:
        return frozenset()
    found = {point}; stack=[point]
    while stack:
        current=stack.pop()
        for neighbor in state.topology.neighbors(current):
            if neighbor not in found and state.stones[neighbor] == color:
                found.add(neighbor); stack.append(neighbor)
    return frozenset(found)

def _group_on_stones(stones, topology, point: int) -> frozenset[int]:
    color=stones[point]
    if color == EMPTY: return frozenset()
    found={point}; stack=[point]
    while stack:
        current=stack.pop()
        for neighbor in topology.neighbors(current):
            if neighbor not in found and stones[neighbor] == color:
                found.add(neighbor); stack.append(neighbor)
    return frozenset(found)

def liberties_from_board(state: GoldenState, group: Iterable[int]) -> frozenset[int]:
    liberties=set()
    for point in group:
        for neighbor in state.topology.neighbors(point):
            if state.stones[neighbor] == EMPTY: liberties.add(neighbor)
    return frozenset(liberties)

def _liberties_on_stones(stones, topology, group: Iterable[int]) -> frozenset[int]:
    liberties=set()
    for point in group:
        for neighbor in topology.neighbors(point):
            if stones[neighbor] == EMPTY: liberties.add(neighbor)
    return frozenset(liberties)

def _next_state(state: GoldenState, *, stones: tuple[Stone,...], side_to_move: Stone,
                append_board_key: BoardKey | None, consecutive_passes: int) -> GoldenState:
    return GoldenState._from_trusted_transition(
        state,
        stones=stones,
        side_to_move=side_to_move,
        append_board_key=append_board_key,
        consecutive_passes=consecutive_passes,
    )

def _probe_point(state: GoldenState, action: int) -> ActionProbe:
    increment("action_probe_calls")
    if state.stones[action] != EMPTY:
        raise IllegalMoveError(IllegalMoveReason.OCCUPIED, action, f"PointId {action} is occupied")
    moving_color=state.side_to_move; opponent_color=opponent(moving_color)
    provisional=list(state.stones); provisional[action]=moving_color
    captured=set(); checked=set()
    for neighbor in state.topology.neighbors(action):
        if provisional[neighbor] != opponent_color or neighbor in checked: continue
        group=_group_on_stones(provisional,state.topology,neighbor)
        checked.update(group)
        if not _liberties_on_stones(provisional,state.topology,group): captured.update(group)
    for point in captured: provisional[point]=EMPTY
    own_group=_group_on_stones(provisional,state.topology,action)
    if not _liberties_on_stones(provisional,state.topology,own_group):
        raise IllegalMoveError(IllegalMoveReason.SUICIDE,action,
            f"PointId {action} leaves the new {moving_color.name} group with zero liberties")
    new_stones=tuple(provisional); new_key=board_key(new_stones)
    if new_key in state.superko_membership:
        raise IllegalMoveError(IllegalMoveReason.SUPERKO, action,
            f"PointId {action} recreates an earlier board arrangement under positional superko")
    increment("successful_action_probes")
    return ActionProbe(action, tuple(sorted(captured)), new_stones, new_key)


def probe_action(state: GoldenState, action: int | str) -> ActionProbe:
    """Apply exact local rules for a point without allocating a child state."""

    if state.is_terminal:
        raise IllegalMoveError(IllegalMoveReason.MOVE_AFTER_TERMINAL, action,
            "Golden rules reject every action after formal DOUBLE_PASS terminal")
    if action == PASS:
        raise IllegalMoveError(IllegalMoveReason.INVALID_ACTION_ID, action,
            "PASS has no hypothetical point transition")
    if isinstance(action,bool) or not isinstance(action,int):
        raise IllegalMoveError(IllegalMoveReason.INVALID_ACTION_ID, action,
            f"Golden action must be PointId 0..{state.topology.point_count-1} or PASS")
    if not 0 <= action < state.topology.point_count:
        raise IllegalMoveError(IllegalMoveReason.INVALID_ACTION_ID, action,
            f"PointId {action} is outside 0..{state.topology.point_count-1}")
    return _probe_point(state, action)


def apply_action(state: GoldenState, action: int | str) -> Transition:
    increment("apply_action_calls")
    if state.is_terminal:
        raise IllegalMoveError(IllegalMoveReason.MOVE_AFTER_TERMINAL, action,
            "Golden rules reject every action after formal DOUBLE_PASS terminal")
    if action == PASS:
        after=_next_state(state, stones=state.stones, side_to_move=opponent(state.side_to_move),
                          append_board_key=None,
                          consecutive_passes=state.consecutive_passes+1)
        return Transition(state, PASS, (), after)
    if isinstance(action,bool) or not isinstance(action,int):
        raise IllegalMoveError(IllegalMoveReason.INVALID_ACTION_ID, action,
            f"Golden action must be PointId 0..{state.topology.point_count-1} or PASS")
    if not 0 <= action < state.topology.point_count:
        raise IllegalMoveError(IllegalMoveReason.INVALID_ACTION_ID, action,
            f"PointId {action} is outside 0..{state.topology.point_count-1}")
    probe = _probe_point(state, action)
    after=_next_state(state, stones=probe.stones, side_to_move=opponent(state.side_to_move),
                      append_board_key=probe.board_key, consecutive_passes=0)
    return Transition(state,action,probe.captured,after)

def prepare_legal_actions(state: GoldenState) -> LegalActionContext:
    increment("legal_calculations")
    if state.is_terminal:
        return LegalActionContext(
            state.state_key, (), (False,) * (state.topology.point_count + 1)
        )
    legal=[]
    for action in range(state.topology.point_count):
        try: probe_action(state,action)
        except IllegalMoveError: continue
        legal.append(action)
    legal.append(PASS)
    mask = [False] * (state.topology.point_count + 1)
    for action in legal:
        mask[state.topology.point_count if action == PASS else int(action)] = True
    return LegalActionContext(state.state_key, tuple(legal), tuple(mask))


def legal_actions(state: GoldenState) -> tuple[int|str,...]:
    increment("legal_actions_calls")
    return prepare_legal_actions(state).actions


def reference_apply_action(state: GoldenState, action: int | str) -> Transition:
    """Slow, fully validated semantic oracle retained for equivalence tests."""

    increment("apply_action_calls")
    if state.is_terminal:
        raise IllegalMoveError(IllegalMoveReason.MOVE_AFTER_TERMINAL, action,
            "Golden rules reject every action after formal DOUBLE_PASS terminal")
    if action == PASS:
        after = GoldenState(
            stones=state.stones, side_to_move=opponent(state.side_to_move),
            superko_history=state.superko_history,
            consecutive_passes=state.consecutive_passes + 1,
            topology=state.topology, rules_id=state.rules_id,
            rules_fingerprint=state.rules_fingerprint, komi=state.komi,
            history_provenance=state.history_provenance,
        )
        return Transition(state, PASS, (), after)
    if isinstance(action, bool) or not isinstance(action, int):
        raise IllegalMoveError(IllegalMoveReason.INVALID_ACTION_ID, action,
            f"Golden action must be PointId 0..{state.topology.point_count-1} or PASS")
    if not 0 <= action < state.topology.point_count:
        raise IllegalMoveError(IllegalMoveReason.INVALID_ACTION_ID, action,
            f"PointId {action} is outside 0..{state.topology.point_count-1}")
    if state.stones[action] != EMPTY:
        raise IllegalMoveError(IllegalMoveReason.OCCUPIED, action, f"PointId {action} is occupied")
    moving_color = state.side_to_move
    opponent_color = opponent(moving_color)
    provisional = list(state.stones)
    provisional[action] = moving_color
    captured = set()
    checked = set()
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
        raise IllegalMoveError(IllegalMoveReason.SUICIDE, action,
            f"PointId {action} leaves the new {moving_color.name} group with zero liberties")
    new_stones = tuple(provisional)
    new_key = board_key(new_stones)
    if new_key in state.superko_history:
        raise IllegalMoveError(IllegalMoveReason.SUPERKO, action,
            f"PointId {action} recreates an earlier board arrangement under positional superko")
    after = GoldenState(
        stones=new_stones, side_to_move=opponent(moving_color),
        superko_history=state.superko_history + (new_key,), consecutive_passes=0,
        topology=state.topology, rules_id=state.rules_id,
        rules_fingerprint=state.rules_fingerprint, komi=state.komi,
        history_provenance=state.history_provenance,
    )
    return Transition(state, action, tuple(sorted(captured)), after)


def reference_legal_actions(state: GoldenState) -> tuple[int | str, ...]:
    """Reference counterpart of legal_actions for fixed-corpus gates."""

    increment("legal_actions_calls")
    if state.is_terminal:
        return ()
    legal = []
    for action in range(state.topology.point_count):
        try:
            reference_apply_action(state, action)
        except IllegalMoveError:
            continue
        legal.append(action)
    legal.append(PASS)
    return tuple(legal)
