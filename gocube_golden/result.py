from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .scoring import GoldenScore, score_terminal
from .state import GoldenState


DOUBLE_PASS = "DOUBLE_PASS"


class Winner(str, Enum):
    BLACK = "BLACK"
    WHITE = "WHITE"
    DRAW = "DRAW"


@dataclass(frozen=True)
class GoldenResult:
    winner: Winner
    black_area: int
    white_area: int
    komi: float
    margin_black: float
    terminal_reason: str
    rules_fingerprint: str
    topology_fingerprint: str


def result_from_terminal(state: GoldenState) -> GoldenResult:
    score: GoldenScore = score_terminal(state)
    if score.margin_black > 0:
        winner = Winner.BLACK
    elif score.margin_black < 0:
        winner = Winner.WHITE
    else:
        winner = Winner.DRAW
    return GoldenResult(
        winner=winner,
        black_area=score.black_area,
        white_area=score.white_area,
        komi=score.komi,
        margin_black=score.margin_black,
        terminal_reason=DOUBLE_PASS,
        rules_fingerprint=state.rules_fingerprint,
        topology_fingerprint=state.topology.fingerprint,
    )
