from __future__ import annotations

import math

import numpy as np


KATAGO_PINNED_EXPLORATION_CONTRACT = "katago-pinned-exploration-v1"
KATAGO_PINNED_EXPLORATION_DEFAULTS = {
    # cpp/configs/training/selfplay8b20.cfg at the pinned PR #32 commit.
    "root_dirichlet_noise_total_concentration": 10.83,
    "root_dirichlet_noise_weight": 0.25,
    "root_policy_temperature_early": 1.25,
    "root_policy_temperature": 1.10,
    "root_policy_temperature_halflife": 19.0,
    "root_desired_per_child_visits_coeff": 2.0,
}


def root_policy_temperature(
    turn_number: int,
    point_count: int,
    *,
    early_temperature: float,
    temperature: float,
    halflife: float,
) -> float:
    """Pinned KataGo interpolateEarly, using logical point count as board area.

    KataGo scales the number of elapsed halflives by ``19/sqrt(x*y)``. GoCube
    has no planar x/y rectangle, so the topology-preserving equivalent is the
    logical graph point count. This is the only topology-specific substitution.
    """

    point_count = int(point_count)
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if halflife <= 0.0:
        raise ValueError("root policy temperature halflife must be positive")
    raw_halflives = max(0.0, float(turn_number)) / float(halflife)
    scaled_halflives = raw_halflives * 19.0 / math.sqrt(float(point_count))
    return float(temperature) + (float(early_temperature) - float(temperature)) * (
        0.5 ** scaled_halflives
    )


def shaped_dirichlet_alpha_distribution(policy: np.ndarray) -> np.ndarray:
    """Port KataGo computeDirichletAlphaDistribution for legal root moves.

    The returned values are proportions summing to one. Callers multiply them
    by the total concentration before drawing Gamma/Dirichlet noise.
    """

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
    """Port KataGo's pre-LCB getReducedPlaySelectionWeight correction.

    GoCube uses integral visit counts as child weights. For each non-best root
    child, invert PUCT against the best child's selection value and cap the
    output at the weight that would retrospectively have been needed. This is
    what prevents root-forcing/exploration overspend from becoming ordinary
    policy supervision.
    """

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
        goodness[idx] = (
            weight * max(0.0, weight - 1.0) / max(1.0, weight) + 2.0 * probs[idx]
        )
    best_idx = int(np.argmax(goodness))

    def selection_value(idx: int) -> float:
        utility = utilities[idx]
        value_component = utility if root_player == 1 else -utility
        return value_component + float(explore_scaling) * probs[idx] / (1.0 + counts[idx])

    best_selection = selection_value(best_idx)
    for idx in visited:
        if idx == best_idx:
            continue
        utility = utilities[idx]
        value_component = utility if root_player == 1 else -utility
        explore_component = best_selection - value_component
        if explore_component <= 0.0:
            wanted = 1e100
        else:
            wanted = float(explore_scaling) * probs[idx] / explore_component - 1.0
            wanted = max(0.0, wanted)
        if counts[idx] > wanted:
            result[idx] = int(math.ceil(wanted))
    return result
