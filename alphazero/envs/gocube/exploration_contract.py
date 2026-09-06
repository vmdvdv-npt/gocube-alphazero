from __future__ import annotations

import math

import numpy as np


KATAGO_PINNED_EXPLORATION_CONTRACT = "katago-pinned-exploration-v2"
KATAGO_PINNED_EXPLORATION_DEFAULTS = {
    # cpp/configs/training/selfplay8b20.cfg plus SearchParams defaults at the
    # pinned PR #32 commit.
    "root_dirichlet_noise_total_concentration": 10.83,
    "root_dirichlet_noise_weight": 0.25,
    "root_policy_temperature_early": 1.25,
    "root_policy_temperature": 1.10,
    "root_policy_temperature_halflife": 19.0,
    "root_desired_per_child_visits_coeff": 2.0,
    "chosen_move_temperature_early": 0.75,
    "chosen_move_temperature": 0.15,
    "chosen_move_temperature_halflife": 19.0,
    "chosen_move_subtract": 0.0,
    "chosen_move_prune": 1.0,
    "use_lcb_for_selection": True,
    "lcb_stdevs": 5.0,
    "min_visit_prop_for_lcb": 0.15,
    "value_weight_exponent": 0.5,
}


def interpolate_early(
    turn_number: int,
    point_count: int,
    *,
    early_value: float,
    value: float,
    halflife: float,
) -> float:
    """Pinned KataGo interpolateEarly using logical point count as board area."""

    point_count = int(point_count)
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if halflife <= 0.0:
        raise ValueError("temperature halflife must be positive")
    raw_halflives = max(0.0, float(turn_number)) / float(halflife)
    scaled_halflives = raw_halflives * 19.0 / math.sqrt(float(point_count))
    return float(value) + (float(early_value) - float(value)) * (0.5 ** scaled_halflives)


def root_policy_temperature(
    turn_number: int,
    point_count: int,
    *,
    early_temperature: float,
    temperature: float,
    halflife: float,
) -> float:
    return interpolate_early(
        turn_number,
        point_count,
        early_value=early_temperature,
        value=temperature,
        halflife=halflife,
    )


def chosen_move_temperature(
    turn_number: int,
    point_count: int,
    *,
    early_temperature: float,
    temperature: float,
    halflife: float,
) -> float:
    return interpolate_early(
        turn_number,
        point_count,
        early_value=early_temperature,
        value=temperature,
        halflife=halflife,
    )


def chosen_move_temperature_scaling(
    current_temperature: float,
    turn_number: int,
    _max_turns,
    *,
    point_count: int,
    early_temperature: float,
    temperature: float,
    halflife: float,
) -> float:
    if float(current_temperature) <= 0.0:
        return float(current_temperature)
    return chosen_move_temperature(
        turn_number,
        point_count,
        early_temperature=early_temperature,
        temperature=temperature,
        halflife=halflife,
    )


def shaped_dirichlet_alpha_distribution(policy: np.ndarray) -> np.ndarray:
    """Port KataGo computeDirichletAlphaDistribution for legal root moves."""

    probs = np.asarray(policy, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        raise ValueError("root policy must contain at least one legal move")
    if np.any(probs < 0.0) or not np.all(np.isfinite(probs)):
        raise ValueError("root policy must be finite and non-negative")

    logs = np.log(np.minimum(0.01, probs) + 1e-20)
    log_policy_mean = float(np.mean(logs))
    shaped = np.maximum(0.0, logs - log_policy_mean)
    shaped_sum = float(np.sum(shaped))
    uniform = np.full(probs.size, 1.0 / probs.size, dtype=np.float64)
    if shaped_sum <= 0.0:
        return uniform
    return 0.5 * (shaped / shaped_sum + uniform)


def retrospectively_reduce_root_visits(
    raw_counts: np.ndarray,
    policy: np.ndarray,
    child_utilities_white: np.ndarray,
    *,
    root_player: int,
    explore_scaling: float,
    legal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compatibility form of KataGo's pre-LCB reduced selection weight."""

    counts = np.asarray(raw_counts, dtype=np.float64).reshape(-1)
    probs = np.asarray(policy, dtype=np.float64).reshape(-1)
    utilities = np.asarray(child_utilities_white, dtype=np.float64).reshape(-1)
    if counts.size != probs.size or counts.size != utilities.size:
        raise ValueError("root count/policy/utility sizes must match")
    if root_player not in (0, 1):
        raise ValueError("root_player must be black (0) or white (1)")
    if legal_mask is None:
        legal = np.ones(counts.size, dtype=np.bool_)
    else:
        legal = np.asarray(legal_mask, dtype=np.bool_).reshape(-1)
        if legal.size != counts.size:
            raise ValueError("legal mask size must match root counts")

    result = np.asarray(np.maximum(0.0, counts), dtype=np.int32)
    visited = np.flatnonzero(legal & (counts > 0.0))
    if visited.size <= 1:
        return result

    goodness = np.full(counts.size, -np.inf, dtype=np.float64)
    for idx in visited:
        weight = counts[idx]
        goodness[idx] = weight * max(0.0, weight - 1.0) / max(1.0, weight) + 2.0 * probs[idx]
    best_idx = int(np.argmax(goodness))

    def selection_value(idx: int) -> float:
        value_component = utilities[idx] if root_player == 1 else -utilities[idx]
        return value_component + float(explore_scaling) * probs[idx] / (1.0 + counts[idx])

    best_selection = selection_value(best_idx)
    for idx in visited:
        if idx == best_idx:
            continue
        value_component = utilities[idx] if root_player == 1 else -utilities[idx]
        explore_component = best_selection - value_component
        if explore_component <= 0.0:
            wanted = 1e100
        else:
            wanted = max(0.0, float(explore_scaling) * probs[idx] / explore_component - 1.0)
        if counts[idx] > wanted:
            result[idx] = int(math.ceil(wanted))
    return result


def retrospectively_reduce_root_weights(
    raw_weights: np.ndarray,
    policy: np.ndarray,
    child_utilities_white: np.ndarray,
    *,
    root_player: int,
    explore_scaling: float,
    legal_mask: np.ndarray | None = None,
    edge_visits: np.ndarray | None = None,
) -> np.ndarray:
    """Pinned getReducedPlaySelectionWeight on KataGo child weights.

    KataGo keeps the non-LCB-best child's original floating weight. Every other
    child is passed through getReducedPlaySelectionWeight and then ``ceil`` is
    applied before LCB processing, even when no reduction was necessary.
    """

    weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    probs = np.asarray(policy, dtype=np.float64).reshape(-1)
    utilities = np.asarray(child_utilities_white, dtype=np.float64).reshape(-1)
    if weights.size != probs.size or weights.size != utilities.size:
        raise ValueError("root weight/policy/utility sizes must match")
    if root_player not in (0, 1):
        raise ValueError("root_player must be black (0) or white (1)")
    legal = np.ones(weights.size, dtype=np.bool_) if legal_mask is None else np.asarray(legal_mask, dtype=np.bool_).reshape(-1)
    if legal.size != weights.size:
        raise ValueError("legal mask size must match root weights")
    visits = weights if edge_visits is None else np.asarray(edge_visits, dtype=np.float64).reshape(-1)
    if visits.size != weights.size:
        raise ValueError("edge visit size must match root weights")

    result = np.maximum(weights, 0.0).copy()
    visited = np.flatnonzero(legal & (weights > 0.0))
    if visited.size <= 1:
        return result

    goodness = np.full(weights.size, -np.inf, dtype=np.float64)
    for idx in visited:
        edge = max(1.0, float(visits[idx]))
        goodness[idx] = weights[idx] * max(0.0, edge - 1.0) / edge + 2.0 * probs[idx]
    best_idx = int(np.argmax(goodness))

    def selection_value(idx: int) -> float:
        value_component = utilities[idx] if root_player == 1 else -utilities[idx]
        return value_component + float(explore_scaling) * probs[idx] / (1.0 + weights[idx])

    best_selection = selection_value(best_idx)
    for idx in visited:
        if idx == best_idx:
            continue
        value_component = utilities[idx] if root_player == 1 else -utilities[idx]
        explore_component = best_selection - value_component
        wanted = 1e100 if explore_component <= 0.0 else max(0.0, float(explore_scaling) * probs[idx] / explore_component - 1.0)
        reduced = min(float(weights[idx]), wanted)
        result[idx] = float(math.ceil(reduced)) if reduced > 0.0 else 0.0
    return result


def student_t3_cdf(z: float) -> float:
    """Closed-form CDF for Student-t with 3 degrees of freedom."""

    z = float(z)
    value = 0.5 + (
        math.atan(z / math.sqrt(3.0)) + math.sqrt(3.0) * z / (z * z + 3.0)
    ) / math.pi
    return min(1.0, max(0.0, value))


def kata_value_child_weights(
    raw_weights: np.ndarray,
    child_self_utilities: np.ndarray,
    *,
    exponent: float,
    subtract: float = 0.0,
    prune: float = 0.0,
) -> np.ndarray:
    """Pinned downweightBadChildrenAndNormalizeWeight for a tree node."""

    weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    utilities = np.asarray(child_self_utilities, dtype=np.float64).reshape(-1)
    if weights.size != utilities.size:
        raise ValueError("child weights/utilities size mismatch")
    if exponent < 0.0:
        raise ValueError("value weight exponent must be non-negative")
    desired_total = float(np.maximum(weights, 0.0).sum())
    if desired_total <= 0.0:
        return np.maximum(weights, 0.0)

    base = np.maximum(weights, 0.0).copy()
    simple_value = float(np.dot(base, utilities) / desired_total)
    result = np.zeros_like(base)
    total_new = 0.0
    for idx, weight in enumerate(base):
        if weight <= 0.0 or weight < float(prune):
            continue
        new_weight = weight - float(subtract)
        if new_weight <= 0.0:
            continue
        if exponent != 0.0:
            precision = 1.5 * math.sqrt(float(weight))
            stdev = math.sqrt(1e-8 + 1.0 / precision)
            z = (float(utilities[idx]) - simple_value) / stdev
            probability = student_t3_cdf(z) + 0.0001
            new_weight *= probability ** float(exponent)
        result[idx] = new_weight
        total_new += new_weight
    if total_new <= 0.0:
        raise ValueError("child pruning removed every positive child weight")
    result *= desired_total / total_new
    return result


def apply_lcb_play_selection(
    play_selection_values: np.ndarray,
    raw_edge_visits: np.ndarray,
    policy: np.ndarray,
    child_utilities_white: np.ndarray,
    child_utility_sq: np.ndarray,
    child_weight_sum: np.ndarray,
    child_weight_sq_sum: np.ndarray,
    ending_utility_diffs_white: np.ndarray,
    *,
    root_player: int,
    utility_range_radius: float,
    lcb_stdevs: float,
    min_visit_prop_for_lcb: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    """Pinned KataGo LCB adjustment after retrospective root-weight reduction."""

    weights = np.asarray(play_selection_values, dtype=np.float64).reshape(-1).copy()
    edge_visits = np.asarray(raw_edge_visits, dtype=np.float64).reshape(-1)
    probs = np.asarray(policy, dtype=np.float64).reshape(-1)
    utilities = np.asarray(child_utilities_white, dtype=np.float64).reshape(-1)
    utility_sq = np.asarray(child_utility_sq, dtype=np.float64).reshape(-1)
    weight_sum_stats = np.asarray(child_weight_sum, dtype=np.float64).reshape(-1)
    weight_sq_stats = np.asarray(child_weight_sq_sum, dtype=np.float64).reshape(-1)
    bonus_diffs = np.asarray(ending_utility_diffs_white, dtype=np.float64).reshape(-1)
    arrays = (edge_visits, probs, utilities, utility_sq, weight_sum_stats, weight_sq_stats, bonus_diffs)
    if any(arr.size != weights.size for arr in arrays):
        raise ValueError("LCB array sizes must match")
    if root_player not in (0, 1):
        raise ValueError("root_player must be black (0) or white (1)")
    if lcb_stdevs < 0.0 or not 0.0 <= min_visit_prop_for_lcb <= 1.0:
        raise ValueError("invalid LCB parameters")

    active = np.flatnonzero(weights > 0.0)
    lcbs = np.full(weights.size, -2.0 * utility_range_radius * lcb_stdevs, dtype=np.float64)
    radii = np.full(weights.size, 2.0 * utility_range_radius * lcb_stdevs, dtype=np.float64)
    if active.size == 0:
        return weights, lcbs, radii, None

    goodness = np.full(weights.size, -np.inf, dtype=np.float64)
    for idx in active:
        edge = edge_visits[idx]
        goodness[idx] = weights[idx] * max(0.0, edge - 1.0) / max(1.0, edge) + 2.0 * probs[idx]
    non_lcb_best_idx = int(np.argmax(goodness))
    non_lcb_best_weight = float(weights[non_lcb_best_idx])

    for idx in active:
        edge = float(edge_visits[idx])
        stats_sum = float(weight_sum_stats[idx])
        stats_sq = float(weight_sq_stats[idx])
        if edge <= 0.0 or stats_sum <= 0.0 or stats_sq <= 0.0:
            continue
        weight_sum = stats_sum
        weight_sq_sum = stats_sq
        ess = weight_sum * weight_sum / weight_sq_sum
        if ess <= 0.0:
            continue
        prior_weight = weight_sum / (ess * ess * ess)
        mean = float(utilities[idx])
        sq = max(float(utility_sq[idx]), mean * mean + 1e-8)
        sq = (sq * weight_sum + (sq + utility_range_radius * utility_range_radius) * prior_weight) / (
            weight_sum + prior_weight
        )
        weight_sum += prior_weight
        weight_sq_sum += prior_weight * prior_weight
        ess = weight_sum * weight_sum / weight_sq_sum
        self_utility = mean + float(bonus_diffs[idx])
        if root_player == 0:
            self_utility = -self_utility
        variance = max(0.0, sq - mean * mean)
        radius = math.sqrt(variance / ess) * float(lcb_stdevs)
        lcbs[idx] = self_utility - radius
        radii[idx] = radius

    eligible = [
        int(idx)
        for idx in active
        if weights[idx] >= float(min_visit_prop_for_lcb) * non_lcb_best_weight
    ]
    if not eligible:
        return weights, lcbs, radii, None
    best_lcb_idx = max(eligible, key=lambda idx: lcbs[idx])
    best_lcb = float(lcbs[best_lcb_idx])
    adjusted_weight = float(weights[best_lcb_idx])
    for idx in active:
        if int(idx) == best_lcb_idx:
            continue
        excess = best_lcb - float(lcbs[idx])
        if excess < 0.0:
            continue
        radius = float(radii[idx])
        denom = radius + 0.20 * excess
        radius_factor = (radius + excess) / denom if denom > 1e-30 else 5.0
        lbound = radius_factor * radius_factor * float(weights[idx])
        adjusted_weight = max(adjusted_weight, lbound)
    weights[best_lcb_idx] = adjusted_weight
    return weights, lcbs, radii, best_lcb_idx


def apply_chosen_move_pruning(
    play_selection_values: np.ndarray,
    *,
    subtract: float,
    prune: float,
) -> np.ndarray:
    """Pinned chosenMoveSubtract/chosenMovePrune final root-weight cleanup."""

    values = np.asarray(play_selection_values, dtype=np.float64).reshape(-1).copy()
    max_value = float(values.max(initial=0.0))
    if max_value <= 0.0:
        return values
    amount_to_subtract = min(float(subtract), max_value / 64.0)
    amount_to_prune = min(float(prune), max_value / 64.0)
    for idx in range(values.size):
        if values[idx] < amount_to_prune:
            values[idx] = 0.0
        else:
            values[idx] = max(0.0, values[idx] - amount_to_subtract)
    return values
