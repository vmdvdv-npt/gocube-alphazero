"""A small, deterministic rule state used as a second verification oracle.

This module intentionally does not import GoCube production code.  It consumes
only a raw occupancy vector and a raw adjacency list.  It is not intended to
replace the native KataGo oracle: it covers graph consequences and the basic
normal-play transition on Cube/Torus, while the native oracle remains the
authority for rectangular Rules V3 phase/scoring semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .independent_graph import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    IndependentIllegalMove,
    apply_move,
    find_groups,
)

MAIN = "main"
CLEANUP_1 = "cleanup1"
CLEANUP_2 = "cleanup2"
SCORED = "scored"
NO_RESULT = "no_result"


@dataclass(frozen=True)
class IndependentRuleState:
    """Minimal state needed by the independent graph transition."""

    board: tuple[int, ...]
    current_player: int = BLACK
    previous_board: tuple[int, ...] | None = None
    consecutive_passes: int = 0
    captures: tuple[int, int] = (0, 0)
    phase: str = MAIN
    history_since_pass: tuple[tuple[int, tuple[int, ...]], ...] = ()
    terminal_kind: str | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if self.current_player not in (BLACK, WHITE):
            raise ValueError("current_player must be BLACK or WHITE")
        if self.phase not in (MAIN, CLEANUP_1, CLEANUP_2, SCORED, NO_RESULT):
            raise ValueError(f"unsupported independent phase: {self.phase!r}")
        if len(self.captures) != 2 or any(value < 0 for value in self.captures):
            raise ValueError("captures must contain two non-negative values")


def initial_rule_state(
    board: Sequence[int],
    *,
    current_player: int = BLACK,
    previous_board: Sequence[int] | None = None,
    captures: tuple[int, int] = (0, 0),
    phase: str = MAIN,
) -> IndependentRuleState:
    """Create an immutable independent state from raw values."""

    normalized = tuple(int(value) for value in board)
    if any(value not in (EMPTY, BLACK, WHITE) for value in normalized):
        raise ValueError("board contains an invalid occupancy value")
    previous = None if previous_board is None else tuple(int(value) for value in previous_board)
    if previous is not None and len(previous) != len(normalized):
        raise ValueError("previous_board must have the same length as board")
    key = (int(current_player), normalized)
    return IndependentRuleState(
        board=normalized,
        current_player=int(current_player),
        previous_board=previous,
        captures=(int(captures[0]), int(captures[1])),
        phase=phase,
        history_since_pass=(key,),
    )


def legal_actions(
    state: IndependentRuleState,
    adjacency: Sequence[Sequence[int]],
    *,
    pass_action: int | None = None,
) -> tuple[int, ...]:
    """Return legal point indices plus PASS in deterministic point order."""

    if state.terminal_kind is not None or state.phase not in (MAIN, CLEANUP_1, CLEANUP_2):
        return ()
    actions = []
    for action, value in enumerate(state.board):
        if value != EMPTY:
            continue
        try:
            apply_move(
                state.board,
                state.current_player,
                action,
                adjacency,
                previous_board=state.previous_board,
                enforce_simple_ko=True,
            )
        except IndependentIllegalMove:
            continue
        actions.append(action)
    if pass_action is not None:
        actions.append(int(pass_action))
    return tuple(actions)


def _phase_after_two_passes(phase: str) -> str:
    if phase == MAIN:
        return CLEANUP_1
    if phase == CLEANUP_1:
        return CLEANUP_2
    if phase == CLEANUP_2:
        return SCORED
    return phase


def apply_action(
    state: IndependentRuleState,
    action: int,
    adjacency: Sequence[Sequence[int]],
    *,
    pass_action: int | None = None,
) -> IndependentRuleState:
    """Apply one independent graph action.

    PASS changes turn and history but not occupancy.  Two consecutive PASS
    actions advance the three Rules V3 phases.  Cleanup ko recap bookkeeping
    and scoring deliberately remain outside this small graph oracle; those
    semantics are compared to pinned KataGo by the rectangular differential.
    """

    if state.terminal_kind is not None or state.phase not in (MAIN, CLEANUP_1, CLEANUP_2):
        raise IndependentIllegalMove("not-playing")
    if pass_action is not None and int(action) == int(pass_action):
        next_state = replace(
            state,
            current_player=WHITE if state.current_player == BLACK else BLACK,
            previous_board=state.board,
            consecutive_passes=state.consecutive_passes + 1,
        )
        if next_state.consecutive_passes >= 2:
            phase = _phase_after_two_passes(state.phase)
            return replace(
                next_state,
                phase=phase,
                consecutive_passes=0 if phase != SCORED else next_state.consecutive_passes,
                terminal_kind=SCORED if phase == SCORED else None,
                termination_reason="formal_pass" if phase == SCORED else None,
                history_since_pass=((next_state.current_player, next_state.board),),
            )
        return replace(
            next_state,
            history_since_pass=((next_state.current_player, next_state.board),),
        )

    try:
        result = apply_move(
            state.board,
            state.current_player,
            int(action),
            adjacency,
            previous_board=state.previous_board,
            enforce_simple_ko=True,
        )
    except IndependentIllegalMove:
        raise
    captures = list(state.captures)
    captures[0 if state.current_player == BLACK else 1] += result.capture_count
    next_player = WHITE if state.current_player == BLACK else BLACK
    key = (next_player, result.board)
    if key in state.history_since_pass:
        return replace(
            state,
            board=result.board,
            current_player=next_player,
            previous_board=state.board,
            consecutive_passes=0,
            captures=(captures[0], captures[1]),
            terminal_kind=NO_RESULT,
            termination_reason="cycle",
            history_since_pass=state.history_since_pass + (key,),
        )
    return replace(
        state,
        board=result.board,
        current_player=next_player,
        previous_board=state.board,
        consecutive_passes=0,
        captures=(captures[0], captures[1]),
        history_since_pass=state.history_since_pass + (key,),
    )


def semantic_snapshot(
    state: IndependentRuleState,
    adjacency: Sequence[Sequence[int]],
    *,
    pass_action: int | None = None,
) -> dict[str, object]:
    """Normalize the independent state for semantic, not byte-level, diffs."""

    groups = find_groups(state.board, adjacency)
    return {
        "board": state.board,
        "current_player": state.current_player,
        "legal_actions": legal_actions(state, adjacency, pass_action=pass_action),
        "captures": state.captures,
        "phase": state.phase,
        "consecutive_passes": state.consecutive_passes,
        "group_count": len(groups),
        "terminal_kind": state.terminal_kind,
        "termination_reason": state.termination_reason,
    }
