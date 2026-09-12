from __future__ import annotations

from dataclasses import dataclass

from .result import GoldenResult, Winner, result_from_terminal
from .rules import apply_action, legal_actions
from .state import PASS, GoldenState

class GoldenSearchBoundaryError(RuntimeError):
    pass

@dataclass(frozen=True)
class GoldenSearchAdapter:
    """The only boundary between Golden rules and Stage-2 search.

    The adapter exposes rule-owned legal/apply/terminal operations.  Search is
    never allowed to score a position or decide a winner itself.
    """

    def legal_actions(self, state: GoldenState) -> tuple[int | str, ...]:
        return legal_actions(state)

    def apply_action(self, state: GoldenState, action: int | str) -> GoldenState:
        return apply_action(state, action).after

    def is_terminal(self, state: GoldenState) -> bool:
        return state.is_terminal

    def terminal_result(self, state: GoldenState) -> GoldenResult:
        if not state.is_terminal:
            raise GoldenSearchBoundaryError("Exact terminal result requested for nonterminal state")
        return result_from_terminal(state)

    def terminal_utility(self, state: GoldenState) -> float:
        """Exact utility from the terminal state's side-to-move perspective."""
        result = self.terminal_result(state)
        if result.winner == Winner.DRAW:
            return 0.0
        winner_color = 1 if result.winner == Winner.BLACK else 2
        return 1.0 if int(state.side_to_move) == winner_color else -1.0

    def action_index(self, state: GoldenState, action: int | str) -> int:
        if action == PASS:
            return state.topology.point_count
        if isinstance(action, bool) or not isinstance(action, int):
            raise GoldenSearchBoundaryError(f"Invalid Golden action {action!r}")
        if not 0 <= action < state.topology.point_count:
            raise GoldenSearchBoundaryError(f"Invalid Golden PointId {action!r}")
        return action

    def action_space(self, state: GoldenState) -> tuple[int | str, ...]:
        return tuple(range(state.topology.point_count)) + (PASS,)
