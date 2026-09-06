from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

KATAGO_REFERENCE_COMMIT = "f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
LEGACY_SEARCH_UTILITY_MODE = "legacy"
KATAGO_PINNED_SEARCH_UTILITY_MODE = "katago-pinned-f6bc4b19"
KATAGO_SEARCH_CONTRACT = "katago-pinned-search-v1"

# Pinned from cpp/configs/training/selfplay8b20.cfg at KATAGO_REFERENCE_COMMIT.
KATAGO_SEARCH_DEFAULTS = {
    "win_loss_utility_factor": 1.0,
    "static_score_utility_factor": 0.0,
    "dynamic_score_utility_factor": 0.30,
    "dynamic_score_center_zero_weight": 0.25,
    "dynamic_score_center_scale": 0.50,
    "cpuct_exploration": 1.10,
    "cpuct_exploration_log": 0.0,
    "cpuct_exploration_base": 500.0,
    "fpu_reduction_max": 0.20,
    "root_fpu_reduction_max": 0.0,
    "fpu_parent_weight_by_visited_policy": True,
    "fpu_parent_weight_by_visited_policy_pow": 2.0,
    "root_ending_bonus_points": 0.50,
    "fill_dame_before_pass": True,
    "conservative_pass": True,
}


@dataclass(frozen=True)
class SearchOutput:
    """All neural heads consumed by GoCube search from one forward pass."""

    policy: Any
    value: Any
    score: Any | None = None
    ownership: Any | None = None


def white_win_loss_value(value: Any) -> float:
    """KataGo win/loss value in [-1,1], always from White's perspective."""

    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("value vector must contain Black and White probabilities")
    return float(arr[1] - arr[0])


def normalized_black_minus_white_to_white_score(
    normalized_black_minus_white: float,
    point_count: int,
) -> float:
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    return -float(normalized_black_minus_white) * float(point_count)


def score_value(score: float, center: float, scale: float, point_count: int) -> float:
    """Zero-stdev specialization of KataGo whiteScoreValueOfScoreSmoothNoDrawAdjust."""

    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if scale <= 0:
        raise ValueError("score value scale must be positive")
    return (2.0 / math.pi) * math.atan(
        (float(score) - float(center)) / (float(scale) * math.sqrt(float(point_count)))
    )


def recent_score_center(
    expected_score: float,
    *,
    zero_weight: float,
    center_scale: float,
    point_count: int,
) -> float:
    """Pinned KataGo dynamic score center.

    KataGo first moves the center toward zero, then caps that movement relative
    to the *expected score* by sqrt(board area) * dynamicScoreCenterScale.
    The cap is intentionally not centered on zero.
    """

    if not 0.0 <= zero_weight <= 1.0:
        raise ValueError("zero_weight must be within [0,1]")
    if center_scale <= 0:
        raise ValueError("center_scale must be positive")
    if point_count <= 0:
        raise ValueError("point_count must be positive")

    expected_score = float(expected_score)
    center = expected_score * (1.0 - float(zero_weight))
    cap = math.sqrt(float(point_count)) * float(center_scale)
    return min(expected_score + cap, max(expected_score - cap, center))


def score_utility(
    white_score: float,
    *,
    recent_center: float,
    point_count: int,
    static_factor: float,
    dynamic_factor: float,
    dynamic_scale: float,
) -> float:
    static_value = score_value(white_score, 0.0, 2.0, point_count)
    dynamic_value = score_value(white_score, recent_center, dynamic_scale, point_count)
    return float(static_factor) * static_value + float(dynamic_factor) * dynamic_value


def score_utility_diff(
    white_score: float,
    delta_white_score: float,
    *,
    recent_center: float,
    point_count: int,
    static_factor: float,
    dynamic_factor: float,
    dynamic_scale: float,
) -> float:
    before = score_utility(
        white_score,
        recent_center=recent_center,
        point_count=point_count,
        static_factor=static_factor,
        dynamic_factor=dynamic_factor,
        dynamic_scale=dynamic_scale,
    )
    after = score_utility(
        white_score + delta_white_score,
        recent_center=recent_center,
        point_count=point_count,
        static_factor=static_factor,
        dynamic_factor=dynamic_factor,
        dynamic_scale=dynamic_scale,
    )
    return after - before


def combined_white_utility(
    value: Any,
    white_score: float | None,
    *,
    recent_center: float,
    point_count: int,
    win_loss_factor: float,
    static_score_factor: float,
    dynamic_score_factor: float,
    dynamic_score_scale: float,
) -> float:
    utility = white_win_loss_value(value) * float(win_loss_factor)
    if white_score is None:
        return utility
    return utility + score_utility(
        white_score,
        recent_center=recent_center,
        point_count=point_count,
        static_factor=static_score_factor,
        dynamic_factor=dynamic_score_factor,
        dynamic_scale=dynamic_score_scale,
    )


def white_owner_map(ownership: Any) -> np.ndarray:
    """Convert GoCube [Black, White, Neutral] probabilities to KataGo-style White ownership."""

    arr = np.asarray(ownership, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"ownership must have shape (points,3), got {arr.shape}")
    return arr[:, 1] - arr[:, 0]


def player_ownership(white_ownership: np.ndarray, point: int, player: int) -> float:
    value = float(white_ownership[int(point)])
    if int(player) == 1:
        return value
    if int(player) == 0:
        return -value
    raise ValueError(f"unsupported player {player}")


def _group_from(board: np.ndarray, topology: Any, start: int, color: int) -> frozenset[int]:
    group = {int(start)}
    pending = [int(start)]
    while pending:
        point = pending.pop()
        for neighbor in topology.neighbor_indices(point):
            neighbor = int(neighbor)
            if neighbor not in group and int(board[neighbor]) == color:
                group.add(neighbor)
                pending.append(neighbor)
    return frozenset(group)


def _pass_alive_point_sets(game: Any, player: int) -> tuple[set[int], set[int]]:
    """Return own and opponent pass-alive areas using the production graph Benson analysis."""

    state = getattr(game, "semantic_state", None)
    if state is None or not hasattr(game, "logical_topology"):
        return set(), set()
    try:
        from alphazero.envs.gocube.katago_v3 import pass_alive_analysis
    except ImportError:
        return set(), set()

    analysis = pass_alive_analysis(state.board, game.logical_topology())
    black = set(analysis.pass_alive_black_territory)
    white = set(analysis.pass_alive_white_territory)
    for group in analysis.pass_alive_black_groups:
        black.update(group)
    for group in analysis.pass_alive_white_groups:
        white.update(group)
    return (black, white) if int(player) == 0 else (white, black)


def _is_non_pass_alive_self_connection(
    game: Any,
    action: int,
    player: int,
    own_pass_alive: set[int],
    opp_pass_alive: set[int],
) -> bool:
    """Graph-topology form of Board::isNonPassAliveSelfConnection intent."""

    state = getattr(game, "semantic_state", None)
    if state is None or not hasattr(game, "logical_topology"):
        return False
    topology = game.logical_topology()
    board = np.asarray(state.board)
    action = int(action)
    own_color = 1 if int(player) == 0 else 2
    if action < 0 or action >= int(topology.point_count) or int(board[action]) != 0:
        return False
    if action in own_pass_alive or action in opp_pass_alive:
        return False

    groups: list[frozenset[int]] = []
    for neighbor in topology.neighbor_indices(action):
        neighbor = int(neighbor)
        if int(board[neighbor]) != own_color:
            continue
        group = _group_from(board, topology, neighbor, own_color)
        if group not in groups:
            groups.append(group)
    if len(groups) < 2:
        return False

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a_alive = bool(set(groups[i]) & own_pass_alive)
            b_alive = bool(set(groups[j]) & own_pass_alive)
            if not (a_alive and b_alive):
                return True
    return False


def _simple_ko_likely_active(game: Any) -> bool:
    state = getattr(game, "semantic_state", None)
    board = getattr(state, "board", None) if state is not None else None
    previous = getattr(state, "previous_board", None) if state is not None else None
    if board is None or previous is None:
        return False
    # A simple one-stone ko transition changes exactly the played point and the
    # captured point. This mirrors the only ko shape relevant to the root bonus.
    return int(np.count_nonzero(np.asarray(board) != np.asarray(previous))) == 2


def _would_capture(game: Any, action: int, player: int) -> bool:
    """Detect whether a legal placement removes opponent stones without planar assumptions."""

    try:
        clone = game.clone()
        before = np.asarray(clone.semantic_state.board).copy()
        clone.play_action(int(action))
        after = np.asarray(clone.semantic_state.board)
    except Exception:
        return False
    opp_color = 2 if int(player) == 0 else 1
    return int(np.count_nonzero(before == opp_color)) > int(np.count_nonzero(after == opp_color))


def root_ending_white_score_bonuses(
    game: Any,
    ownership: Any,
    bonus_points: float,
) -> np.ndarray:
    """Precompute KataGo root-ending adjustments for all actions once per root."""

    action_size = int(game.action_size())
    result = np.zeros(action_size, dtype=np.float32)
    bonus_points = float(bonus_points)
    if bonus_points <= 0 or ownership is None:
        return result

    state = getattr(game, "semantic_state", None)
    if state is None or not hasattr(game, "logical_topology"):
        return result

    topology = game.logical_topology()
    board = np.asarray(state.board)
    pass_action = int(game.pass_action())
    root_player = int(game.player)
    phase = getattr(state, "phase", None)
    area_ish = phase == "cleanup2"
    extreme = 0.95
    tail = 0.05
    ko_active = _simple_ko_likely_active(game)
    owner = white_owner_map(ownership)
    own_pass_alive, opp_pass_alive = _pass_alive_point_sets(game, root_player)
    opp_color = 2 if root_player == 0 else 1

    valids = np.asarray(game.valid_moves()).reshape(-1)
    for action in range(action_size):
        if action >= valids.size or not bool(valids[action]):
            continue

        extra_root_points = 0.0
        if action == pass_action:
            if not area_ish:
                extra_root_points -= bonus_points * (2.0 / 3.0)
        elif not ko_active:
            pla_ownership = player_ownership(owner, action, root_player)
            adjacent_opponent = any(
                int(board[int(n)]) == opp_color for n in topology.neighbor_indices(action)
            )
            self_connection = _is_non_pass_alive_self_connection(
                game,
                action,
                root_player,
                own_pass_alive,
                opp_pass_alive,
            )

            if pla_ownership <= -extreme:
                if not area_ish or not _would_capture(game, action, root_player):
                    extra_root_points -= bonus_points * ((-extreme - pla_ownership) / tail)
            elif pla_ownership >= extreme and not adjacent_opponent and not self_connection:
                extra_root_points -= bonus_points * ((pla_ownership - extreme) / tail)

        result[action] = extra_root_points if root_player == 1 else -extra_root_points
    return result


def root_ending_white_score_bonus(
    game: Any,
    action: int,
    ownership: Any,
    bonus_points: float,
) -> float:
    """Scalar convenience wrapper around the once-per-root bonus computation."""

    return float(root_ending_white_score_bonuses(game, ownership, bonus_points)[int(action)])


def conservative_root_observation(game: Any, observation: np.ndarray) -> np.ndarray:
    """Map KataGo conservativePass root history hiding onto GoCube observation V3."""

    state = getattr(game, "semantic_state", None)
    if state is None:
        return observation
    if getattr(state, "phase", None) != "main" or int(getattr(state, "consecutive_passes", 0)) != 1:
        return observation
    result = np.asarray(observation).copy()
    if result.shape[0] >= 4:
        result[2:4] = 0.0
    if result.shape[0] >= 6:
        result[5] = 0.0
    if result.shape[0] >= 16:
        result[15] = 0.0
    if result.shape[0] >= 17:
        result[16] = 0.0
    return result
