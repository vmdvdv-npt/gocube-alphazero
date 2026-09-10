"""Production-independent terminal scorer for the GoCube Golden Arena.

This module is deliberately boring. It uses only Python integers, tuples,
sets, and the raw graph adjacency supplied by a semantic game state. It does
not call the production KataGo-V3 life/scoring implementation.

The Golden Arena uses this implementation as a second adjudicator and refuses
to count a game unless the production scorer agrees with it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

EMPTY = 0
BLACK = 1
WHITE = 2

SCORED = "scored"
NO_RESULT = "no_result"

# Intentionally duplicated rather than imported from production_contract.
# A drift in either copy must fail tests instead of silently moving the oracle.
GOLDEN_KOMI = 0.5


class GoldenScoringError(RuntimeError):
    """The reference scorer cannot safely adjudicate the supplied state."""


class GoldenScoreMismatch(GoldenScoringError):
    """Production and reference terminal adjudication disagree."""


class GoldenOutcome(str, Enum):
    BLACK = "black"
    WHITE = "white"
    DRAW = "draw"
    NO_RESULT = "no_result"


@dataclass(frozen=True)
class GoldenArea:
    black_area: tuple[int, ...]
    white_area: tuple[int, ...]
    black_territory: tuple[int, ...]
    white_territory: tuple[int, ...]
    neutral: tuple[int, ...]
    seki: tuple[int, ...]


@dataclass(frozen=True)
class GoldenScore:
    black: float
    white: float
    komi: float
    margin: float
    outcome: GoldenOutcome
    captures: tuple[int, int]
    area: GoldenArea


@dataclass(frozen=True)
class GoldenAdjudication:
    terminal_kind: str
    outcome: GoldenOutcome
    score: GoldenScore | None


def _require_komi(komi: float) -> float:
    try:
        value = float(komi)
    except (TypeError, ValueError) as exc:
        raise GoldenScoringError(f"Golden Arena requires komi {GOLDEN_KOMI}") from exc
    if not math.isclose(value, GOLDEN_KOMI, rel_tol=0.0, abs_tol=1e-12):
        raise GoldenScoringError(
            f"Golden Arena requires komi {GOLDEN_KOMI}, got {komi!r}"
        )
    return GOLDEN_KOMI


def _normalize_graph(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    stones = tuple(int(value) for value in board)
    if not stones:
        raise GoldenScoringError("Golden scorer requires a non-empty board")
    if any(value not in (EMPTY, BLACK, WHITE) for value in stones):
        raise GoldenScoringError("Board contains an invalid occupancy value")
    if len(adjacency) != len(stones):
        raise GoldenScoringError(
            f"Adjacency length {len(adjacency)} does not match board length {len(stones)}"
        )

    graph: list[tuple[int, ...]] = []
    for point, raw_neighbors in enumerate(adjacency):
        neighbors = tuple(int(neighbor) for neighbor in raw_neighbors)
        if len(set(neighbors)) != len(neighbors):
            raise GoldenScoringError(f"Point {point} has duplicate neighbors")
        for neighbor in neighbors:
            if neighbor < 0 or neighbor >= len(stones):
                raise GoldenScoringError(
                    f"Point {point} has out-of-range neighbor {neighbor}"
                )
            if neighbor == point:
                raise GoldenScoringError(f"Point {point} has a self-edge")
        graph.append(neighbors)

    frozen_graph = tuple(graph)
    for point, neighbors in enumerate(frozen_graph):
        for neighbor in neighbors:
            if point not in frozen_graph[neighbor]:
                raise GoldenScoringError(
                    f"Adjacency is not reciprocal for edge {point}<->{neighbor}"
                )
    return stones, frozen_graph


def _components(
    values: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    predicate,
) -> tuple[tuple[int, ...], ...]:
    visited: set[int] = set()
    result: list[tuple[int, ...]] = []
    for start in range(len(values)):
        if start in visited or not predicate(int(values[start])):
            continue
        visited.add(start)
        pending = [start]
        component: list[int] = []
        while pending:
            point = pending.pop()
            component.append(point)
            for neighbor in adjacency[point]:
                if neighbor not in visited and predicate(int(values[neighbor])):
                    visited.add(neighbor)
                    pending.append(neighbor)
        result.append(tuple(sorted(component)))
    return tuple(result)


def _color_groups(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    color: int,
) -> tuple[tuple[int, ...], ...]:
    return _components(board, adjacency, lambda value: value == color)


def _group_liberties(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    group: Sequence[int],
) -> frozenset[int]:
    return frozenset(
        neighbor
        for point in group
        for neighbor in adjacency[point]
        if int(board[neighbor]) == EMPTY
    )


def _benson_alive_groups(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    color: int,
) -> tuple[tuple[int, ...], ...]:
    groups = _color_groups(board, adjacency, color)
    if not groups:
        return ()

    group_for_point = {
        point: group_index
        for group_index, group in enumerate(groups)
        for point in group
    }

    # Each region is a maximal connected component containing everything that
    # is not an own-color stone, exactly as the pinned Rules-V3 graph contract
    # defines Benson regions.
    region_facts: list[tuple[set[int], set[int]]] = []
    for region in _components(board, adjacency, lambda value: value != color):
        bordering_groups = {
            group_for_point[neighbor]
            for point in region
            for neighbor in adjacency[point]
            if neighbor in group_for_point
        }
        if not bordering_groups:
            continue

        vital_for = set(bordering_groups)
        for point in region:
            if int(board[point]) != EMPTY:
                continue
            adjacent_groups = {
                group_for_point[neighbor]
                for neighbor in adjacency[point]
                if neighbor in group_for_point
            }
            vital_for.intersection_update(adjacent_groups)
        region_facts.append((bordering_groups, vital_for))

    live_groups = set(range(len(groups)))
    live_regions = set(range(len(region_facts)))
    while True:
        doomed_groups = {
            group_index
            for group_index in live_groups
            if sum(
                group_index in region_facts[region_index][1]
                for region_index in live_regions
            )
            < 2
        }
        if doomed_groups:
            live_groups.difference_update(doomed_groups)

        doomed_regions = {
            region_index
            for region_index in live_regions
            if any(
                group_index not in live_groups
                for group_index in region_facts[region_index][0]
            )
        }
        if doomed_regions:
            live_regions.difference_update(doomed_regions)

        if not doomed_groups and not doomed_regions:
            break

    return tuple(groups[index] for index in sorted(live_groups))


def _pass_alive_territory(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    color: int,
    alive_groups: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    alive_points = {point for group in alive_groups for point in group}
    territory: set[int] = set()

    for region in _components(board, adjacency, lambda value: value != color):
        own_boundary: set[int] = set()
        every_boundary_group_alive = True
        for point in region:
            for neighbor in adjacency[point]:
                if int(board[neighbor]) != color:
                    continue
                own_boundary.add(neighbor)
                if neighbor not in alive_points:
                    every_boundary_group_alive = False

        if not own_boundary or not every_boundary_group_alive:
            continue

        points_touching_alive = sum(
            any(neighbor in alive_points for neighbor in adjacency[point])
            for point in region
        )
        if points_touching_alive >= len(region) - 1:
            territory.update(region)

    return tuple(sorted(territory))


def _reference_area(
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> GoldenArea:
    black_alive = _benson_alive_groups(board, adjacency, BLACK)
    white_alive = _benson_alive_groups(board, adjacency, WHITE)
    black_pass_territory = _pass_alive_territory(
        board, adjacency, BLACK, black_alive
    )
    white_pass_territory = _pass_alive_territory(
        board, adjacency, WHITE, white_alive
    )

    area_labels = [EMPTY] * len(board)
    for color, groups, territory in (
        (BLACK, black_alive, black_pass_territory),
        (WHITE, white_alive, white_pass_territory),
    ):
        for group in groups:
            for point in group:
                area_labels[point] = color
        for point in territory:
            area_labels[point] = color

    # Add single-color empty regions even when their surrounding group was not
    # proven pass-alive. The later independent-life pass decides whether such a
    # component is excluded as seki.
    for color in (BLACK, WHITE):
        opponent = WHITE if color == BLACK else BLACK
        for component in _components(
            board, adjacency, lambda value, own=color: value != own
        ):
            if any(int(board[point]) == opponent for point in component):
                continue
            if not any(
                int(board[neighbor]) == color
                for point in component
                for neighbor in adjacency[point]
            ):
                continue
            for point in component:
                if area_labels[point] == EMPTY:
                    area_labels[point] = color

    for point, occupancy in enumerate(board):
        if area_labels[point] == EMPTY and int(occupancy) != EMPTY:
            area_labels[point] = int(occupancy)

    liberty_count_by_stone: dict[int, int] = {}
    for color in (BLACK, WHITE):
        for group in _color_groups(board, adjacency, color):
            liberty_count = len(_group_liberties(board, adjacency, group))
            for point in group:
                liberty_count_by_stone[point] = liberty_count

    seki_components: set[frozenset[int]] = set()
    for color in (BLACK, WHITE):
        for component in _components(
            area_labels, adjacency, lambda value, own=color: value == own
        ):
            is_seki = False
            for point in component:
                if int(board[point]) == color and liberty_count_by_stone.get(point) == 1:
                    is_seki = True
                    break
                if any(
                    int(board[neighbor]) == EMPTY
                    and int(area_labels[neighbor]) == EMPTY
                    for neighbor in adjacency[point]
                ):
                    is_seki = True
                    break
            if is_seki:
                seki_components.add(frozenset(component))

    black_area: set[int] = set()
    white_area: set[int] = set()
    for color, sink in ((BLACK, black_area), (WHITE, white_area)):
        for component in _components(
            area_labels, adjacency, lambda value, own=color: value == own
        ):
            if frozenset(component) not in seki_components:
                sink.update(component)

    if black_area & white_area:
        raise GoldenScoringError("Reference area assigns a point to both colors")

    black_territory = {
        point for point in black_area if int(board[point]) == EMPTY
    }
    white_territory = {
        point for point in white_area if int(board[point]) == EMPTY
    }
    neutral = {
        point
        for point, occupancy in enumerate(board)
        if int(occupancy) == EMPTY and int(area_labels[point]) == EMPTY
    }
    seki = {
        point
        for component in seki_components
        for point in component
        if int(board[point]) == EMPTY
    }

    return GoldenArea(
        black_area=tuple(sorted(black_area)),
        white_area=tuple(sorted(white_area)),
        black_territory=tuple(sorted(black_territory)),
        white_territory=tuple(sorted(white_territory)),
        neutral=tuple(sorted(neutral)),
        seki=tuple(sorted(seki)),
    )


def adjudicate_v3_terminal(
    *,
    board: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    terminal_kind: str,
    white_bonus_score: float,
    second_cleanup_start_colors: Sequence[int] | bytes | None,
    captures: Sequence[int] = (0, 0),
    komi: float = GOLDEN_KOMI,
) -> GoldenAdjudication:
    """Independently adjudicate a raw KataGo-V3 terminal state.

    NO_RESULT is intentionally not a draw. Only SCORED states receive a
    numeric score and an absolute Black/White/Draw winner.
    """

    canonical_komi = _require_komi(komi)
    stones, graph = _normalize_graph(board, adjacency)

    if terminal_kind == NO_RESULT:
        return GoldenAdjudication(
            terminal_kind=NO_RESULT,
            outcome=GoldenOutcome.NO_RESULT,
            score=None,
        )
    if terminal_kind != SCORED:
        raise GoldenScoringError(
            f"Golden scorer requires terminal_kind='scored' or 'no_result', got {terminal_kind!r}"
        )

    try:
        bonus = float(white_bonus_score)
    except (TypeError, ValueError) as exc:
        raise GoldenScoringError("white_bonus_score must be finite") from exc
    if not math.isfinite(bonus):
        raise GoldenScoringError("white_bonus_score must be finite")

    try:
        raw_captures = tuple(captures)
        capture_pair = (int(raw_captures[0]), int(raw_captures[1]))
    except (IndexError, TypeError, ValueError) as exc:
        raise GoldenScoringError("captures must be a pair of non-negative integers") from exc
    if (
        len(raw_captures) != 2
        or tuple(raw_captures) != capture_pair
        or any(isinstance(value, bool) or value < 0 for value in raw_captures)
    ):
        raise GoldenScoringError("captures must be a pair of non-negative integers")

    start_colors: tuple[int, ...] | None
    if second_cleanup_start_colors is None:
        start_colors = None
    else:
        start_colors = tuple(int(value) for value in second_cleanup_start_colors)
        if len(start_colors) != len(stones):
            raise GoldenScoringError(
                "second_cleanup_start_colors length does not match board"
            )
        if any(value not in (EMPTY, BLACK, WHITE) for value in start_colors):
            raise GoldenScoringError(
                "second_cleanup_start_colors contains an invalid occupancy value"
            )

    area = _reference_area(stones, graph)
    black_area = set(area.black_area)
    white_area = set(area.white_area)

    black_score = float(len(black_area))
    white_score = float(len(white_area))
    encore2 = start_colors is not None

    for point, color in enumerate(stones):
        if (
            color == BLACK
            and point not in black_area
            and point not in white_area
            and (not encore2 or start_colors[point] == BLACK)
        ):
            black_score += 1.0
        elif (
            color == WHITE
            and point not in white_area
            and point not in black_area
            and (not encore2 or start_colors[point] == WHITE)
        ):
            white_score += 1.0

    white_score += bonus + canonical_komi

    if black_score == white_score:
        outcome = GoldenOutcome.DRAW
    elif black_score > white_score:
        outcome = GoldenOutcome.BLACK
    else:
        outcome = GoldenOutcome.WHITE

    return GoldenAdjudication(
        terminal_kind=SCORED,
        outcome=outcome,
        score=GoldenScore(
            black=black_score,
            white=white_score,
            komi=canonical_komi,
            margin=abs(black_score - white_score),
            outcome=outcome,
            captures=capture_pair,
            area=area,
        ),
    )


def assert_production_agreement(
    game_state,
    reference: GoldenAdjudication,
) -> None:
    """Fail unless production terminal scoring exactly matches the reference."""

    production = getattr(game_state, "terminal_adjudication", None)
    if production is None:
        raise GoldenScoreMismatch("Production terminal adjudication is missing")

    production_kind = getattr(production, "terminal_kind", None)
    if production_kind != reference.terminal_kind:
        raise GoldenScoreMismatch(
            "Terminal kind mismatch: "
            f"golden={reference.terminal_kind!r}, production={production_kind!r}"
        )

    production_score = getattr(production, "score", None)
    if reference.outcome is GoldenOutcome.NO_RESULT:
        if production_score is not None:
            raise GoldenScoreMismatch(
                "NO_RESULT must not carry a production numeric score"
            )
        return

    if reference.score is None or production_score is None:
        raise GoldenScoreMismatch("Scored terminal is missing a numeric score")

    checks = {
        "black": (reference.score.black, float(production_score.black)),
        "white": (reference.score.white, float(production_score.white)),
        "komi": (reference.score.komi, float(production_score.komi)),
        "margin": (reference.score.margin, float(production_score.margin)),
    }
    for name, (golden_value, production_value) in checks.items():
        if not math.isclose(
            golden_value, production_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise GoldenScoreMismatch(
                f"{name} mismatch: golden={golden_value}, production={production_value}"
            )

    production_winner = str(getattr(production_score, "winner", ""))
    if production_winner != reference.outcome.value:
        raise GoldenScoreMismatch(
            "Winner mismatch: "
            f"golden={reference.outcome.value!r}, production={production_winner!r}"
        )


def adjudicate_game_state(
    game_state,
    *,
    require_production_agreement: bool = True,
) -> GoldenAdjudication:
    """Adjudicate a finished GoCube V3 game from raw semantic state.

    This convenience path consumes score-relevant raw state fields. The Golden
    Arena runner is stricter: for games started from the empty board it
    reconstructs captures, whiteBonusScore, and the CLEANUP_2 start snapshot
    from the observed move transitions and passes those independent values to
    :func:`adjudicate_v3_terminal`.
    """

    semantic = getattr(game_state, "semantic_state", None)
    topology_getter = getattr(type(game_state), "logical_topology", None)
    if semantic is None or topology_getter is None:
        raise GoldenScoringError(
            "Golden Arena requires a GoCube V3 game with semantic_state/topology"
        )
    topology = topology_getter()

    reference = adjudicate_v3_terminal(
        board=tuple(int(value) for value in semantic.board),
        adjacency=topology.neighbors_by_index,
        terminal_kind=getattr(semantic, "terminal_kind", None),
        white_bonus_score=getattr(semantic, "white_bonus_score", float("nan")),
        second_cleanup_start_colors=getattr(
            semantic, "second_cleanup_start_colors", None
        ),
        captures=getattr(semantic, "captures", (0, 0)),
        komi=getattr(type(game_state), "KOMI", GOLDEN_KOMI),
    )

    if require_production_agreement:
        assert_production_agreement(game_state, reference)
    return reference
