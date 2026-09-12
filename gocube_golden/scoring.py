from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state import BLACK, EMPTY, WHITE, GoldenState


class Ownership(str, Enum):
    BLACK = "BLACK"
    WHITE = "WHITE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class GoldenScore:
    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    neutral_points: int
    black_area: int
    white_area: int
    komi: float
    margin_black: float
    ownership: tuple[Ownership, ...]


def _empty_component(state: GoldenState, start: int) -> tuple[frozenset[int], frozenset[int]]:
    component = {start}
    boundary_colors: set[int] = set()
    stack = [start]
    while stack:
        point = stack.pop()
        for neighbor in state.topology.neighbors(point):
            stone = state.stones[neighbor]
            if stone == EMPTY and neighbor not in component:
                component.add(neighbor)
                stack.append(neighbor)
            elif stone in (BLACK, WHITE):
                boundary_colors.add(int(stone))
    return frozenset(component), frozenset(boundary_colors)


def score_terminal(state: GoldenState) -> GoldenScore:
    if not state.is_terminal:
        raise ValueError("Golden graph-area score is defined only after formal DOUBLE_PASS terminal")

    black_stones = sum(stone == BLACK for stone in state.stones)
    white_stones = sum(stone == WHITE for stone in state.stones)
    black_territory = 0
    white_territory = 0
    neutral_points = 0
    ownership: list[Ownership | None] = [None] * state.topology.point_count

    for point, stone in enumerate(state.stones):
        if stone == BLACK:
            ownership[point] = Ownership.BLACK
        elif stone == WHITE:
            ownership[point] = Ownership.WHITE

    seen_empty: set[int] = set()
    for point, stone in enumerate(state.stones):
        if stone != EMPTY or point in seen_empty:
            continue
        component, boundary = _empty_component(state, point)
        seen_empty.update(component)
        size = len(component)
        if boundary == {int(BLACK)}:
            owner = Ownership.BLACK
            black_territory += size
        elif boundary == {int(WHITE)}:
            owner = Ownership.WHITE
            white_territory += size
        else:
            owner = Ownership.NEUTRAL
            neutral_points += size
        for empty_point in component:
            ownership[empty_point] = owner

    resolved_ownership = tuple(
        item if item is not None else Ownership.NEUTRAL for item in ownership
    )
    black_area = black_stones + black_territory
    white_area = white_stones + white_territory
    margin_black = float(black_area - white_area) - state.komi
    return GoldenScore(
        black_stones=black_stones,
        white_stones=white_stones,
        black_territory=black_territory,
        white_territory=white_territory,
        neutral_points=neutral_points,
        black_area=black_area,
        white_area=white_area,
        komi=state.komi,
        margin_black=margin_black,
        ownership=resolved_ownership,
    )
