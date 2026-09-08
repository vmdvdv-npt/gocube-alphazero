from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# These names are the public accounting vocabulary shared by the trainer,
# replay manifests, checkpoints, and experiment orchestration.  Per-iteration
# deltas use the same names; cumulative values are emitted under both the
# ``cumulative`` object and the explicit ``cumulative_<name>`` keys.
TRAINING_COUNTER_KEYS = (
    "selfplay_games_completed",
    "positions_generated",
    "saved_replay_samples",
    "new_samples_accepted",
    "optimizer_steps",
    "optimizer_examples_seen",
)
CUMULATIVE_TRAINING_COUNTER_KEYS = tuple(
    f"cumulative_{key}" for key in TRAINING_COUNTER_KEYS
)
TRAINING_PROGRESS_SCHEMA_VERSION = 1
TRAINING_PROGRESS_FILENAME = "training-progress.json"


def _non_negative_counter(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


@dataclass
class CumulativeTrainingCounters:
    """Canonical cumulative accounting for one training namespace.

    ``optimizer_steps`` and ``optimizer_examples_seen`` are normally sourced
    from the sample-clock scheduler after a train call.  The other fields are
    accumulated from completed self-play/replay iteration manifests.  Keeping
    the two sources separate is important on resume: a replay window is a
    sampling source and must never be mistaken for newly accepted data.
    """

    selfplay_games_completed: int = 0
    positions_generated: int = 0
    saved_replay_samples: int = 0
    new_samples_accepted: int = 0
    optimizer_steps: int = 0
    optimizer_examples_seen: int = 0

    def __post_init__(self) -> None:
        for key in TRAINING_COUNTER_KEYS:
            setattr(self, key, _non_negative_counter(getattr(self, key), key))

    def copy(self) -> "CumulativeTrainingCounters":
        return CumulativeTrainingCounters(**self.as_dict())

    def add_generation(
        self,
        *,
        selfplay_games_completed: int = 0,
        positions_generated: int = 0,
        saved_replay_samples: int = 0,
        new_samples_accepted: int = 0,
    ) -> None:
        for key, amount in (
            ("selfplay_games_completed", selfplay_games_completed),
            ("positions_generated", positions_generated),
            ("saved_replay_samples", saved_replay_samples),
            ("new_samples_accepted", new_samples_accepted),
        ):
            amount = _non_negative_counter(amount, key)
            setattr(self, key, getattr(self, key) + amount)

    def set_optimizer_totals(self, *, steps: int, examples_seen: int) -> None:
        steps = _non_negative_counter(steps, "optimizer_steps")
        examples_seen = _non_negative_counter(examples_seen, "optimizer_examples_seen")
        # A resumed optimizer is allowed to report its restored total, but a
        # later checkpoint must never move the cumulative clock backwards.
        self.optimizer_steps = max(self.optimizer_steps, steps)
        self.optimizer_examples_seen = max(self.optimizer_examples_seen, examples_seen)

    def as_dict(self) -> dict[str, int]:
        return {key: int(getattr(self, key)) for key in TRAINING_COUNTER_KEYS}

    def as_cumulative_dict(self) -> dict[str, int]:
        return {
            f"cumulative_{key}": int(getattr(self, key))
            for key in TRAINING_COUNTER_KEYS
        }

    def payload(self, *, deltas: dict[str, int] | None = None) -> dict[str, object]:
        """Return a JSON-safe payload with both delta and cumulative views."""

        normalized_deltas = {
            key: _non_negative_counter((deltas or {}).get(key, 0), key)
            for key in TRAINING_COUNTER_KEYS
        }
        cumulative = self.as_dict()
        return {
            "iteration": normalized_deltas,
            "cumulative": cumulative,
            "cumulative_counters": cumulative,
            **self.as_cumulative_dict(),
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> "CumulativeTrainingCounters":
        if not isinstance(mapping, Mapping):
            return cls()
        source = mapping.get("cumulative_counters")
        if not isinstance(source, Mapping):
            source = mapping.get("cumulative")
        if not isinstance(source, Mapping):
            source = mapping.get("counters")
        if not isinstance(source, Mapping):
            source = mapping
        values = {}
        for key in TRAINING_COUNTER_KEYS:
            explicit = f"cumulative_{key}"
            value = source.get(key, mapping.get(explicit, 0))
            values[key] = _non_negative_counter(value, key)
        return cls(**values)


def canonical_training_counters(
    counters: CumulativeTrainingCounters | dict[str, object],
    *,
    deltas: dict[str, int] | None = None,
) -> dict[str, object]:
    """Normalize counters for reports and machine-readable orchestration."""

    if not isinstance(counters, CumulativeTrainingCounters):
        counters = CumulativeTrainingCounters.from_mapping(counters)
    return counters.payload(deltas=deltas)


@dataclass(frozen=True)
class SampleBudgetTarget:
    """A scientific stopping target expressed on a cumulative sample clock."""

    kind: str
    target: int

    NEW_SAMPLES = "cumulative_new_samples"
    OPTIMIZER_EXAMPLES = "cumulative_optimizer_examples"

    def __post_init__(self) -> None:
        if self.kind not in (self.NEW_SAMPLES, self.OPTIMIZER_EXAMPLES):
            raise ValueError(
                f"unsupported sample budget target {self.kind!r}; "
                f"expected {self.NEW_SAMPLES!r} or {self.OPTIMIZER_EXAMPLES!r}"
            )
        if int(self.target) <= 0:
            raise ValueError("sample budget target must be positive")
        object.__setattr__(self, "target", int(self.target))

    @property
    def counter_key(self) -> str:
        return (
            "new_samples_accepted"
            if self.kind == self.NEW_SAMPLES
            else "optimizer_examples_seen"
        )

    def current(self, counters: CumulativeTrainingCounters | dict[str, object]) -> int:
        if not isinstance(counters, CumulativeTrainingCounters):
            counters = CumulativeTrainingCounters.from_mapping(counters)
        return int(getattr(counters, self.counter_key))

    def status(
        self,
        counters: CumulativeTrainingCounters | dict[str, object],
        *,
        before: int | None = None,
        generation_chunk_games: int = 0,
    ) -> dict[str, object]:
        after = self.current(counters)
        before_value = after if before is None else int(before)
        overshoot = max(0, after - self.target)
        return {
            "kind": self.kind,
            "target": int(self.target),
            "counter": self.counter_key,
            "before": before_value,
            "after": after,
            "remaining": max(0, self.target - after),
            "reached": after >= self.target,
            "overshot": overshoot > 0,
            "overshoot": overshoot,
            "generation_chunk_games": int(generation_chunk_games),
        }


# Friendly alias for callers that describe this as a training budget rather
# than a sample-clock target.
TrainingBudgetTarget = SampleBudgetTarget


def build_sample_budget_target(
    *,
    cumulative_new_samples_target: int | None = None,
    cumulative_optimizer_examples_target: int | None = None,
) -> SampleBudgetTarget | None:
    """Build the one allowed scientific stopping target from CLI/config values."""

    if cumulative_new_samples_target is not None and cumulative_optimizer_examples_target is not None:
        raise ValueError("choose either cumulative new samples or optimizer examples as the target")
    if cumulative_new_samples_target is not None:
        return SampleBudgetTarget(SampleBudgetTarget.NEW_SAMPLES, int(cumulative_new_samples_target))
    if cumulative_optimizer_examples_target is not None:
        return SampleBudgetTarget(
            SampleBudgetTarget.OPTIMIZER_EXAMPLES,
            int(cumulative_optimizer_examples_target),
        )
    return None


def sample_budget_reached(
    counters: CumulativeTrainingCounters | dict[str, object],
    target: SampleBudgetTarget | None,
) -> bool:
    return target is not None and target.current(counters) >= target.target


def training_progress_path(data_root: str | os.PathLike[str], run_name: str) -> Path:
    return Path(data_root) / str(run_name) / TRAINING_PROGRESS_FILENAME


def write_training_progress(
    data_root: str | os.PathLike[str],
    run_name: str,
    *,
    counters: CumulativeTrainingCounters | dict[str, object],
    latest_iteration: int,
    target_status: dict[str, object] | None = None,
    latest_iteration_metrics: dict[str, object] | None = None,
) -> Path:
    """Atomically publish cumulative training accounting for orchestration."""

    path = training_progress_path(data_root, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(counters, CumulativeTrainingCounters):
        counters = CumulativeTrainingCounters.from_mapping(counters)
    payload: dict[str, object] = {
        "schema_version": TRAINING_PROGRESS_SCHEMA_VERSION,
        "run_name": str(run_name),
        "latest_iteration": int(latest_iteration),
        "counters": counters.as_dict(),
        "cumulative": counters.as_dict(),
        "cumulative_counters": counters.as_dict(),
        **counters.as_cumulative_dict(),
    }
    if target_status is not None:
        payload["budget"] = dict(target_status)
    if latest_iteration_metrics is not None:
        payload["latest_iteration_metrics"] = dict(latest_iteration_metrics)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_training_progress(
    data_root: str | os.PathLike[str],
    run_name: str,
) -> dict[str, object] | None:
    path = training_progress_path(data_root, run_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid training progress artifact: {path}") from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != TRAINING_PROGRESS_SCHEMA_VERSION:
        raise ValueError(f"unsupported training progress schema: {path}")
    counters = CumulativeTrainingCounters.from_mapping(payload)
    payload["counters"] = counters.as_dict()
    payload["cumulative"] = counters.as_dict()
    payload["cumulative_counters"] = counters.as_dict()
    payload.update(counters.as_cumulative_dict())
    return payload


def _manifest_counter_delta(aggregate: object) -> dict[str, int]:
    """Read one iteration's generation delta without treating replay history as new."""

    if not isinstance(aggregate, dict):
        return {key: 0 for key in TRAINING_COUNTER_KEYS}
    sample = aggregate.get("sample_accounting")
    if not isinstance(sample, dict):
        sample = aggregate
    training = aggregate.get("training")
    if not isinstance(training, dict):
        training = {}
    return {
        "selfplay_games_completed": int(sample.get("selfplay_games_completed", sample.get("games", 0))),
        "positions_generated": int(sample.get("positions_generated", sample.get("base_positions", 0))),
        "saved_replay_samples": int(
            sample.get("saved_replay_samples", sample.get("saved_total", training.get("new_selfplay_samples", 0)))
        ),
        # Only the current iteration's accepted rows count here.  The replay
        # window's sum is deliberately absent from this expression.
        "new_samples_accepted": int(
            sample.get("new_samples_accepted", training.get("new_selfplay_samples", 0))
        ),
        "optimizer_steps": int(training.get("actual_optimizer_steps", training.get("optimizer_steps", 0))),
        "optimizer_examples_seen": int(
            training.get("actual_training_samples", training.get("examples_seen", 0))
        ),
    }


def recover_cumulative_training_counters(
    data_root: str | os.PathLike[str],
    run_name: str,
    *,
    optimizer_steps: int = 0,
    optimizer_examples_seen: int = 0,
) -> CumulativeTrainingCounters:
    """Recover generation counters from committed manifests for a resume.

    The progress artifact is preferred because it is written after the
    checkpoint and manifest commit.  The manifest scan is a conservative
    fallback for runs created before the artifact existed.
    """

    progress = load_training_progress(data_root, run_name)
    if progress is not None:
        counters = CumulativeTrainingCounters.from_mapping(progress)
        counters.set_optimizer_totals(steps=optimizer_steps, examples_seen=optimizer_examples_seen)
        return counters

    counters = CumulativeTrainingCounters()
    manifest_root = Path(data_root) / str(run_name) / "records"
    paths = sorted(manifest_root.glob("iteration-*/iteration-manifest.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        delta = _manifest_counter_delta((payload or {}).get("aggregate_metrics"))
        counters.add_generation(
            selfplay_games_completed=delta["selfplay_games_completed"],
            positions_generated=delta["positions_generated"],
            saved_replay_samples=delta["saved_replay_samples"],
            new_samples_accepted=delta["new_samples_accepted"],
        )
        # Older runs may not have a progress artifact yet.  Their manifests
        # still contain per-iteration optimizer deltas, so recover those too;
        # a checkpoint-provided absolute scheduler total below remains
        # authoritative when it is available.
        counters.optimizer_steps += _non_negative_counter(
            delta["optimizer_steps"], "optimizer_steps"
        )
        counters.optimizer_examples_seen += _non_negative_counter(
            delta["optimizer_examples_seen"], "optimizer_examples_seen"
        )
    counters.set_optimizer_totals(steps=optimizer_steps, examples_seen=optimizer_examples_seen)
    return counters


@dataclass(frozen=True)
class ReplayTrainingPlan:
    new_selfplay_samples: int
    replay_window_samples: int
    train_samples_per_new_sample: float
    planned_training_samples: int
    planned_optimizer_steps: int
    planned_passes_over_replay_window: float

    @property
    def sample_clock_increment(self) -> int:
        """Optimizer examples planned for this new-data delta."""

        return int(self.planned_training_samples)

    @property
    def new_samples(self) -> int:
        """Short orchestration-facing alias for the accepted data delta."""

        return int(self.new_selfplay_samples)

    def as_dict(self) -> dict[str, object]:
        return {
            "new_selfplay_samples": int(self.new_selfplay_samples),
            "replay_window_samples": int(self.replay_window_samples),
            "train_samples_per_new_sample": float(self.train_samples_per_new_sample),
            "planned_training_samples": int(self.planned_training_samples),
            "planned_optimizer_steps": int(self.planned_optimizer_steps),
            "planned_passes_over_replay_window": float(self.planned_passes_over_replay_window),
            "sample_clock_increment": int(self.sample_clock_increment),
        }


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
    result_keys = {"win": "wins", "loss": "losses", "draw": "draws"}
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
            key = result_keys[result]
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
