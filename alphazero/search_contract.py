from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

LEGACY_SEARCH_UTILITY_MODE = "legacy"
GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE = "gocube-katago-v3"
GOCUBE_SEARCH_CONTRACT = "katago-v3-score-aware-v1"


@dataclass(frozen=True)
class SearchOutput:
    """Typed neural output consumed by search.

    ``score`` is normalized Black-minus-White score for GoCube. ``ownership``
    stores Black/White/neutral probabilities per logical point. Legacy games
    leave both optional heads as ``None``.
    """

    policy: Any
    value: Any
    score: Any | None = None
    ownership: Any | None = None


def score_value(points: float, center: float, scale: float, point_count: int) -> float:
    """KataGo smooth score value for a deterministic score mean.

    This is the zero-score-stdev specialization of KataGo's
    ``ScoreValue::expectedWhiteScoreValue`` / ``whiteScoreValueOfScoreSmooth``
    at pinned commit f6bc4b19a1686caa2d088b56251e8c11c8be6d51.
    """

    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if scale <= 0:
        raise ValueError("score value scale must be positive")
    return (2.0 / math.pi) * math.atan(
        (float(points) - float(center)) / (float(scale) * math.sqrt(float(point_count)))
    )


def recent_score_center(
    root_score_points: float,
    *,
    zero_weight: float,
    center_scale: float,
    point_count: int,
) -> float:
    """KataGo-style dynamic score center with the board-size cap."""

    if not 0.0 <= zero_weight <= 1.0:
        raise ValueError("dynamic score center zero weight must be within [0,1]")
    if center_scale <= 0:
        raise ValueError("dynamic score center scale must be positive")
    center = float(root_score_points) * (1.0 - float(zero_weight))
    cap = math.sqrt(float(point_count)) * float(center_scale)
    return max(-cap, min(cap, center))


def score_utility(
    score_points: float,
    *,
    recent_center: float,
    point_count: int,
    static_factor: float,
    dynamic_factor: float,
    dynamic_scale: float,
) -> float:
    static_value = score_value(score_points, 0.0, 2.0, point_count)
    dynamic_value = score_value(score_points, recent_center, dynamic_scale, point_count)
    return float(static_factor) * static_value + float(dynamic_factor) * dynamic_value


def score_utility_diff(
    score_points: float,
    delta_points: float,
    *,
    recent_center: float,
    point_count: int,
    static_factor: float,
    dynamic_factor: float,
    dynamic_scale: float,
) -> float:
    before = score_utility(
        score_points,
        recent_center=recent_center,
        point_count=point_count,
        static_factor=static_factor,
        dynamic_factor=dynamic_factor,
        dynamic_scale=dynamic_scale,
    )
    after = score_utility(
        score_points + delta_points,
        recent_center=recent_center,
        point_count=point_count,
        static_factor=static_factor,
        dynamic_factor=dynamic_factor,
        dynamic_scale=dynamic_scale,
    )
    return after - before


def player_score_points(normalized_black_minus_white: float, player: int, point_count: int) -> float:
    """Convert the GoCube normalized score head to the requested player view."""

    raw_black_minus_white = float(normalized_black_minus_white) * float(point_count)
    if int(player) == 0:
        return raw_black_minus_white
    if int(player) == 1:
        return -raw_black_minus_white
    raise ValueError(f"GoCube search only supports Black/White player ids, got {player}")


def exact_player_score_points(game: Any, player: int) -> float | None:
    """Return exact formal terminal score, or None for NO_RESULT/nonterminal."""

    terminal = getattr(game, "terminal_adjudication", None)
    score = getattr(terminal, "score", None) if terminal is not None else None
    if score is None:
        return None
    black_minus_white = float(score.black) - float(score.white)
    return black_minus_white if int(player) == 0 else -black_minus_white


def equivalent_win_probability(value: Any, player: int, num_players: int) -> float:
    """Win estimate with a draw treated as an equal fractional win.

    GoCube's framework historically stores ``args._num_players`` as the width
    of the result vector, i.e. Black, White, and draw (3), while some callers
    pass the actual number of players (2). Accept both forms so score-aware
    search cannot silently drop the draw probability when it receives the
    framework's result-vector width.
    """

    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    player = int(player)
    num_players = int(num_players)

    actual_players = num_players
    if arr.size == 3 and num_players == 3:
        actual_players = 2

    if player < 0 or player >= actual_players:
        raise ValueError("player index outside value vector")
    if arr.size < actual_players:
        raise ValueError("value vector is shorter than the player count")

    result = float(arr[player])
    if arr.size > actual_players:
        result += float(arr[actual_players]) / float(actual_players)
    return result


def win_loss_utility(value: Any, player: int, num_players: int) -> float:
    """KataGo-style [-1,1] win/loss value for the player to move."""

    return 2.0 * equivalent_win_probability(value, player, num_players) - 1.0


def combined_utility(
    value: Any,
    normalized_black_minus_white: float,
    *,
    player: int,
    num_players: int,
    point_count: int,
    recent_center: float,
    win_loss_factor: float,
    static_score_factor: float,
    dynamic_score_factor: float,
    dynamic_score_scale: float,
) -> tuple[float, float, float, float]:
    """Return (combined, win_prob, player_score_points, score_utility)."""

    win_prob = equivalent_win_probability(value, player, num_players)
    result_utility = win_loss_utility(value, player, num_players) * float(win_loss_factor)
    points = player_score_points(normalized_black_minus_white, player, point_count)
    score_util = score_utility(
        points,
        recent_center=recent_center,
        point_count=point_count,
        static_factor=static_score_factor,
        dynamic_factor=dynamic_score_factor,
        dynamic_scale=dynamic_score_scale,
    )
    return result_utility + score_util, win_prob, points, score_util
