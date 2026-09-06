from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReplayTrainingPlan:
    new_selfplay_samples: int
    replay_window_samples: int
    train_samples_per_new_sample: float
    planned_training_samples: int
    planned_optimizer_steps: int
    planned_passes_over_replay_window: float


def build_replay_training_plan(
    *,
    new_selfplay_samples: int,
    replay_window_samples: int,
    train_samples_per_new_sample: float,
    batch_size: int,
) -> ReplayTrainingPlan:
    """Build the production sample budget from new data only.

    Replay history is deliberately only a sampling source. Growing the replay
    window must not increase the optimizer budget for an iteration.
    """

    new_selfplay_samples = int(new_selfplay_samples)
    replay_window_samples = int(replay_window_samples)
    batch_size = int(batch_size)
    ratio = float(train_samples_per_new_sample)
    if new_selfplay_samples < 0:
        raise ValueError("new self-play sample count cannot be negative")
    if replay_window_samples < 0:
        raise ValueError("replay window sample count cannot be negative")
    if batch_size < 1:
        raise ValueError("training batch size must be positive")
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("train_samples_per_new_sample must be finite and non-negative")

    # ceil makes the contract monotonic for fractional ratios and avoids a
    # silent zero budget for small but non-zero amounts of new data.
    planned_samples = int(math.ceil(new_selfplay_samples * ratio))
    planned_steps = int(math.ceil(planned_samples / batch_size)) if planned_samples else 0
    passes = planned_samples / replay_window_samples if replay_window_samples else 0.0
    return ReplayTrainingPlan(
        new_selfplay_samples=new_selfplay_samples,
        replay_window_samples=replay_window_samples,
        train_samples_per_new_sample=ratio,
        planned_training_samples=planned_samples,
        planned_optimizer_steps=planned_steps,
        planned_passes_over_replay_window=passes,
    )


def anchor_checkpoint_iteration(iteration: int, period: int) -> int:
    """Return the most recent periodic anchor strictly older than iteration."""

    iteration = int(iteration)
    period = int(period)
    if iteration < 1:
        raise ValueError("iteration must be at least 1")
    if period < 1:
        raise ValueError("anchor period must be positive")
    return ((iteration - 1) // period) * period


def summarize_arena_outcomes(outcomes: Iterable[tuple[str, str]]) -> dict[str, object]:
    """Summarize current-checkpoint outcomes, including the color split.

    ``outcomes`` entries are ``(current_color, result)`` where color is
    ``black``/``white`` and result is ``win``/``loss``/``draw``/``no_result``.
    No-results are reported but excluded from the scored win rate.
    """

    by_color = {
        "black": {"games": 0, "wins": 0, "losses": 0, "draws": 0, "no_results": 0},
        "white": {"games": 0, "wins": 0, "losses": 0, "draws": 0, "no_results": 0},
    }
    totals = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "no_results": 0}
    for color, result in outcomes:
        if color not in by_color:
            raise ValueError(f"unsupported Arena color: {color!r}")
        if result not in ("win", "loss", "draw", "no_result"):
            raise ValueError(f"unsupported Arena result: {result!r}")
        by_color[color]["games"] += 1
        totals["games"] += 1
        if result == "no_result":
            by_color[color]["no_results"] += 1
            totals["no_results"] += 1
        else:
            key = "draws" if result == "draw" else result + "s"
            by_color[color][key] += 1
            totals[key] += 1

    scored = totals["wins"] + totals["losses"] + totals["draws"]
    totals["scored_games"] = scored
    totals["win_rate"] = (
        (totals["wins"] + 0.5 * totals["draws"]) / scored if scored else 0.0
    )
    totals["by_color"] = by_color
    return totals


def arena_regression_signals(
    current_win_rate: float,
    previous_win_rates: Iterable[float],
    *,
    material_threshold: float = 0.45,
) -> dict[str, object]:
    """Observational regression flags only; never a training/model gate."""

    current = float(current_win_rate)
    threshold = float(material_threshold)
    if not 0.0 <= current <= 1.0:
        raise ValueError("Arena win rate must be within [0,1]")
    if not 0.0 <= threshold <= 0.5:
        raise ValueError("material regression threshold must be within [0,0.5]")
    history = [float(value) for value in previous_win_rates]
    if any(not 0.0 <= value <= 1.0 for value in history):
        raise ValueError("historical Arena win rates must be within [0,1]")
    recent = history[-3:]
    below_even_count = sum(value < 0.5 for value in recent)
    return {
        "below_even_against_previous": current < 0.5,
        "material_threshold": threshold,
        "material_regression": current < threshold,
        "recent_previous_win_rates": recent,
        "recent_below_even_count": below_even_count,
        "multi_checkpoint_regression_signal": len(recent) >= 2 and below_even_count >= 2,
    }
