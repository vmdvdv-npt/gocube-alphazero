"""Bounded exhaustive continuation solver for independent graph positions.

This is intentionally a small research tool.  It proves only what the caller
supplies as a terminal evaluator; node/depth exhaustion always returns
``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

from .independent_graph import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    IndependentIllegalMove,
    apply_move,
)

PROVED_WIN = "proved_win"
PROVED_LOSS = "proved_loss"
PROVED_DRAW = "proved_draw"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class SolverState:
    board: tuple[int, ...]
    to_move: int
    previous_board: tuple[int, ...] | None = None
    consecutive_passes: int = 0
    position_history: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.to_move not in (BLACK, WHITE):
            raise ValueError("to_move must be BLACK or WHITE")


@dataclass(frozen=True)
class SolverResult:
    status: str
    nodes: int
    max_depth_reached: int
    exhausted: bool
    principal_variation: tuple[int, ...] = ()


TerminalEvaluator = Callable[[SolverState], Optional[Union[int, str]]]


class ExhaustiveSolver:
    def __init__(
        self,
        adjacency: Sequence[Sequence[int]],
        *,
        max_nodes: int = 10_000,
        max_depth: int = 12,
        include_pass: bool = True,
        enforce_simple_ko: bool = True,
    ) -> None:
        if max_nodes < 1 or max_depth < 0:
            raise ValueError("Solver limits must be positive (depth may be zero only as a root limit)")
        self.adjacency = tuple(tuple(int(point) for point in neighbors) for neighbors in adjacency)
        self.max_nodes = int(max_nodes)
        self.max_depth = int(max_depth)
        self.include_pass = include_pass
        self.enforce_simple_ko = enforce_simple_ko
        self.nodes = 0
        self.max_depth_reached = 0
        self.exhausted = False

    def children(self, state: SolverState) -> tuple[tuple[int, SolverState], ...]:
        result = []
        for action, board in self._placements(state):
            result.append(
                (
                    action,
                    SolverState(
                        board=board,
                        to_move=WHITE if state.to_move == BLACK else BLACK,
                        previous_board=state.board,
                        position_history=state.position_history + (state.board,),
                    ),
                )
            )
        if self.include_pass:
            result.append(
                (
                    PASS,
                    SolverState(
                        board=state.board,
                        to_move=WHITE if state.to_move == BLACK else BLACK,
                        previous_board=state.board,
                        consecutive_passes=state.consecutive_passes + 1,
                        position_history=state.position_history + (state.board,),
                    ),
                )
            )
        return tuple(result)

    def _placements(self, state: SolverState) -> tuple[tuple[int, tuple[int, ...]], ...]:
        result = []
        for action, occupancy in enumerate(state.board):
            if occupancy != EMPTY:
                continue
            try:
                child = apply_move(
                    state.board,
                    state.to_move,
                    action,
                    self.adjacency,
                    previous_board=state.previous_board,
                    enforce_simple_ko=self.enforce_simple_ko,
                )
            except IndependentIllegalMove:
                continue
            result.append((action, child.board))
        return tuple(result)

    def solve(self, initial: SolverState, terminal: TerminalEvaluator) -> SolverResult:
        self.nodes = 0
        self.max_depth_reached = 0
        self.exhausted = False
        root_player = initial.to_move

        def visit(state: SolverState, depth: int) -> tuple[str, tuple[int, ...], int | str | None]:
            self.max_depth_reached = max(self.max_depth_reached, depth)
            terminal_value = terminal(state)
            if terminal_value is not None:
                return self._relative_status(terminal_value, root_player), (), terminal_value
            if depth >= self.max_depth or self.nodes >= self.max_nodes:
                self.exhausted = True
                return UNKNOWN, (), None
            self.nodes += 1
            children = self.children(state)
            if not children:
                self.exhausted = True
                return UNKNOWN, (), None
            unknown_seen = False
            draw_seen = False
            losses = []
            for action, child in children:
                child_status, pv, value = visit(child, depth + 1)
                if child_status == UNKNOWN:
                    unknown_seen = True
                    continue
                if value == state.to_move:
                    return PROVED_WIN if state.to_move == root_player else PROVED_LOSS, (action,) + pv, value
                if value in ("draw", "no_result", 2):
                    draw_seen = True
                losses.append((action, pv, value))
            if unknown_seen:
                return UNKNOWN, (), None
            if draw_seen:
                # Every non-winning line was known, and at least one is draw.
                return PROVED_DRAW, (), "draw"
            if losses:
                # Every complete line is a win for the opposing side.
                return PROVED_LOSS if state.to_move == root_player else PROVED_WIN, losses[0][0:1] + losses[0][1], losses[0][2]
            return UNKNOWN, (), None

        status, pv, _ = visit(initial, 0)
        return SolverResult(status, self.nodes, self.max_depth_reached, self.exhausted, pv)

    @staticmethod
    def _relative_status(value: int | str, root_player: int) -> str:
        if value in ("draw", "no_result", 2):
            return PROVED_DRAW
        if value == root_player:
            return PROVED_WIN
        if value in (BLACK, WHITE):
            return PROVED_LOSS
        raise ValueError(f"Terminal evaluator returned unsupported value: {value!r}")
