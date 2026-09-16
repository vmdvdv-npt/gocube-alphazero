from __future__ import annotations

from tests.support.exhaustive_solver import (
    PROVED_WIN,
    UNKNOWN,
    ExhaustiveSolver,
    SolverState,
)
from tests.support.independent_graph import BLACK, WHITE


def _tiny_terminal(state):
    if state.consecutive_passes >= 2:
        return "draw"
    if WHITE not in state.board and BLACK in state.board:
        return BLACK
    if all(value != 0 for value in state.board):
        return BLACK if state.board.count(BLACK) > state.board.count(WHITE) else WHITE
    return None


def test_limited_solver_proves_a_small_capture_continuation_and_supports_pass():
    # A one-point race: Black captures at 1, then can fill the last point.
    adjacency = ((1,), (0, 2), (1,))
    initial = SolverState((BLACK, 0, WHITE), BLACK)
    assert any(action == -1 for action, _child in ExhaustiveSolver(adjacency).children(initial))
    result = ExhaustiveSolver(adjacency, max_nodes=100, max_depth=8).solve(initial, _tiny_terminal)
    assert result.status == PROVED_WIN
    assert result.nodes <= 100
    assert result.max_depth_reached <= 8


def test_exhausted_solver_is_fail_safe_unknown():
    adjacency = ((1,), (0, 2), (1,))
    initial = SolverState((BLACK, 0, WHITE), BLACK)
    result = ExhaustiveSolver(adjacency, max_nodes=1, max_depth=0).solve(initial, _tiny_terminal)
    assert result.status == UNKNOWN
    assert result.exhausted
