from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .result import GoldenResult, result_from_terminal
from .rules import IllegalMoveError, IllegalMoveReason, Transition, apply_action
from .state import GoldenState


@dataclass(frozen=True)
class ReplayReport:
    initial_state: GoldenState
    final_state: GoldenState
    transitions: tuple[Transition, ...]
    result: GoldenResult | None
    illegal_action_index: int | None = None
    illegal_action: object | None = None
    illegal_reason: IllegalMoveReason | None = None
    illegal_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.illegal_reason is None


def replay(initial: GoldenState, actions: Iterable[int | str]) -> ReplayReport:
    state = initial
    transitions: list[Transition] = []
    for index, action in enumerate(actions):
        try:
            transition = apply_action(state, action)
        except IllegalMoveError as exc:
            return ReplayReport(
                initial_state=initial,
                final_state=state,
                transitions=tuple(transitions),
                result=result_from_terminal(state) if state.is_terminal else None,
                illegal_action_index=index,
                illegal_action=action,
                illegal_reason=exc.reason,
                illegal_message=str(exc),
            )
        transitions.append(transition)
        state = transition.after

    return ReplayReport(
        initial_state=initial,
        final_state=state,
        transitions=tuple(transitions),
        result=result_from_terminal(state) if state.is_terminal else None,
    )
