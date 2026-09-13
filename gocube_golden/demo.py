from __future__ import annotations

from .replay import replay
from .scoring import score_terminal
from .state import BLACK, EMPTY, PASS, WHITE, initial_state


DEMO_ACTIONS: tuple[int | str, ...] = (0, 1, 2, 10, 6, 11, 21, 3, PASS, PASS)


def _board_lines(stones) -> tuple[str, ...]:
    symbol = {EMPTY: ".", BLACK: "B", WHITE: "W"}
    return tuple(
        "".join(symbol[stones[y * 5 + x]] for x in range(5)) for y in range(5)
    )


def demonstration_text() -> str:
    report = replay(initial_state(), DEMO_ACTIONS)
    if not report.ok or report.result is None:
        raise RuntimeError(f"Golden Stage-1 demonstration failed: {report}")
    score = score_terminal(report.final_state)
    captures = [
        {"action": transition.action, "captured": transition.captured}
        for transition in report.transitions
        if transition.captured
    ]
    lines = [
        "Golden Torus 5x5 Stage-1 deterministic demonstration",
        f"action trace: {list(DEMO_ACTIONS)}",
        f"captures: {captures}",
        "final stones:",
        *_board_lines(report.final_state.stones),
        f"black area: {score.black_area}",
        f"white area: {score.white_area}",
        f"neutral points: {score.neutral_points}",
        f"komi: {score.komi}",
        f"margin_black: {score.margin_black}",
        f"winner: {report.result.winner.value}",
        f"terminal reason: {report.result.terminal_reason}",
    ]
    return "\n".join(lines)
