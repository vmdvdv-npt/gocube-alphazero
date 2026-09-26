"""Pure performance-tuning policy.

There are no process, filesystem, notifier, or supervisor calls in this
module.  It is intentionally usable against saved telemetry fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .contracts import (
    Decision,
    DecisionType,
    FailureCategory,
    Mode,
    Observation,
    Plan,
    SelectedProfile,
)


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: object) -> int | None:
    if type(value) is int:
        return value
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result == value else None


def _source_number(sources: Sequence[Mapping[str, object]], keys: Sequence[str]) -> float | None:
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            value = _number(source[key])
            if value is not None:
                return value
    return None


def _contract_value(contract: object, key: str, default: object = None) -> object:
    if isinstance(contract, Plan):
        if key == "expected_games":
            return None
        return getattr(contract, key, default)
    if isinstance(contract, Mapping):
        return contract.get(key, default)
    return getattr(contract, key, default)


def _expected_games(contract: object) -> int | None:
    value = _contract_value(contract, "expected_games")
    if value is None:
        value = _contract_value(contract, "games_per_generation")
    if value is None and isinstance(contract, Plan):
        # Plan deliberately does not carry the full scientific config.  A
        # standalone caller may omit this check and provide a contract map.
        return None
    return _int(value)


def _expected_execution(contract: object, mode: Mode) -> dict[str, int] | None:
    workers = _contract_value(contract, "workers")
    try:
        workers_int = int(workers)
    except (TypeError, ValueError):
        return None
    if workers_int <= 0:
        return None
    return {
        "workers": workers_int,
        "active_games_per_worker": mode.active_games_per_worker,
        "total_active_contexts": mode.total_active_contexts,
    }


def assess_observation(raw_metrics: Mapping[str, object], contract: object) -> Observation:
    """Normalize one generation's metrics and assess stability.

    ``raw_metrics`` may be the complete generation summary or the nested
    ``orchestrator_selfplay`` object.  Missing, NaN, and infinite values are
    retained as missing evidence; they never become zero-valued winners.
    """

    if not isinstance(raw_metrics, Mapping):
        raise ValueError("raw_metrics must be an object")
    nested = raw_metrics.get("orchestrator_selfplay")
    metrics = nested if isinstance(nested, Mapping) else raw_metrics
    inference = metrics.get("inference") if isinstance(metrics.get("inference"), Mapping) else {}
    timing = metrics.get("timing") if isinstance(metrics.get("timing"), Mapping) else {}
    mode_value = _contract_value(contract, "mode")
    if not isinstance(mode_value, Mode):
        mode_value = Mode.from_dict(mode_value) if isinstance(mode_value, Mapping) else None
    if mode_value is None:
        mode_value = Mode(
            str(_contract_value(contract, "label", "unknown")),
            int(_contract_value(contract, "active_games_per_worker", 1)),
            int(_contract_value(contract, "total_active_contexts", 1)),
        )
    generation_value = _contract_value(contract, "generation", 0)
    generation = int(generation_value)
    action_id = str(_contract_value(contract, "action_id", f"observation:{generation}:{mode_value.label}"))
    reasons: list[str] = []

    games = _int(metrics.get("games"))
    moves = _int(metrics.get("moves"))
    expected_games = _expected_games(contract)
    if games is None:
        reasons.append("games metric is missing or malformed")
    elif expected_games is not None and games != expected_games:
        reasons.append(f"games={games}, expected={expected_games}")

    wall = _number(metrics.get("selfplay_time_sec"))
    if wall is None or wall <= 0.0:
        reasons.append("missing self-play wall time")
    games_per_hour = _number(metrics.get("games_per_hour"))
    if games_per_hour is None or games_per_hour <= 0.0:
        reasons.append("missing games per hour")

    technical = _int(metrics.get("technical_games"))
    invalid = _int(metrics.get("invalid_games"))
    if technical not in (None, 0):
        reasons.append(f"technical_games={technical}")
    if invalid not in (None, 0):
        reasons.append(f"invalid_games={invalid}")

    stall_value: int | None = None
    for source in (metrics, timing, inference):
        for key in ("stall_count", "stalls", "execution_stalls"):
            if key in source:
                stall_value = _int(source[key])
                break
        if stall_value is not None:
            break
    if stall_value not in (None, 0):
        reasons.append(f"stalls={stall_value}")

    expected_execution = _expected_execution(contract, mode_value)
    actual_execution: dict[str, object] | None = None
    raw_execution = metrics.get("execution")
    if expected_execution is not None:
        if not isinstance(raw_execution, Mapping):
            reasons.append("self-play execution values are missing")
        else:
            actual_execution = {key: raw_execution.get(key) for key in expected_execution}
            if any(type(actual_execution[key]) is not int or actual_execution[key] <= 0 for key in expected_execution):
                reasons.append("self-play execution values are malformed")
            elif actual_execution != expected_execution:
                reasons.append(
                    "self-play execution values do not match the selected mode: "
                    f"actual={actual_execution!r}, expected={expected_execution!r}"
                )

    finite_metrics: dict[str, object] = {
        "games": games,
        "moves": moves,
        "selfplay_wall_time_sec": wall,
        "games_per_hour": games_per_hour,
        "technical_games": technical,
        "invalid_games": invalid,
        "stalls": stall_value,
        "mean_game_length": (moves / games if moves is not None and games else None),
        "moves_per_sec": _number(metrics.get("moves_per_sec")),
        "mean_inference_batch": _number(inference.get("mean_batch_rows")),
        "gpu_utilization_percent": _source_number((metrics, inference, timing), ("gpu_utilization_percent", "gpu_utilization")),
        "gpu_power_w": _source_number((metrics, inference, timing), ("gpu_power_w", "gpu_power", "power_w")),
        "cpu_utilization_percent": _source_number((metrics, inference, timing), ("cpu_utilization_percent", "cpu_utilization")),
    }
    if actual_execution is not None:
        finite_metrics["actual_execution"] = actual_execution
    timing_evidence = {
        key: timing[key]
        for key in (
            "restore_previous_state_wall_time_sec",
            "self_play_wall_time_sec",
            "replay_file_load_wall_time_sec",
            "optimizer_wall_time_sec",
        )
        if key in timing
    }
    finite_metrics["timing"] = timing_evidence
    return Observation(
        action_id=action_id,
        mode=mode_value,
        generation=generation,
        metrics=finite_metrics,
        stable=not reasons,
        stability_reasons=tuple(reasons),
        role=str(_contract_value(contract, "role", "measurement")),
        raw_metrics=dict(raw_metrics),
    )


def classify_failure(error: BaseException, *, explicit_category: FailureCategory | str | None = None) -> FailureCategory:
    """Classify only at the execution adapter boundary.

    A generic RuntimeError is unknown.  Only the existing, narrow supervisor
    failure wording is treated as a transient retry; no arbitrary exception
    is interpreted as OOM.
    """

    if explicit_category is not None:
        return explicit_category if isinstance(explicit_category, FailureCategory) else FailureCategory(str(explicit_category))
    name = error.__class__.__name__.lower()
    text = str(error).lower()
    if "integrity" in name or "artifact" in name and "integrity" in text:
        return FailureCategory.ARTIFACT_INTEGRITY_ERROR
    if isinstance(error, ValueError) or "configuration" in text or "invalid input" in text:
        return FailureCategory.CONFIGURATION_ERROR
    if "explicit stop" in text or "operator stop" in text:
        return FailureCategory.EXPLICIT_STOP
    if isinstance(error, MemoryError) or any(token in text for token in ("out of memory", "oom", "cuda out of memory")):
        return FailureCategory.RESOURCE_EXHAUSTED
    if isinstance(error, (OSError, TimeoutError, ChildProcessError)):
        return FailureCategory.TRANSIENT_EXECUTION_FAILURE
    if isinstance(error, RuntimeError) and any(
        token in text for token in ("production generation", "supervisor", "timed out", "timeout", "technical failure")
    ):
        return FailureCategory.TRANSIENT_EXECUTION_FAILURE
    return FailureCategory.UNKNOWN_FAILURE


def _as_observation(value: Observation | Mapping[str, object]) -> Observation:
    return value if isinstance(value, Observation) else Observation.from_dict(value)


def _metric(value: Observation, key: str) -> float | None:
    raw = value.metrics.get(key)
    return _number(raw)


def select_profile(
    plan: Plan,
    observations: Sequence[Observation | Mapping[str, object]],
    failures: Sequence[Mapping[str, object] | object] = (),
) -> SelectedProfile:
    """Select by the legacy rule: mean games/hour, then mean wall time."""

    selected, selected_values, source = select_mode(plan, observations, failures)
    normalized = [_as_observation(item) for item in observations]
    if selected_values is None:
        selected_values = [item for item in normalized if item.mode.label == selected.label]
    if source == "measured_selection":
        limitations = [
            "Measurements use the explicit plan and its recorded checkpoint sequence.",
        ]
    else:
        limitations = [
            "No stable measurement evidence was available; baseline was selected explicitly.",
        ]
    metric_values = {
        "games_per_hour_mean": (
            sum(_metric(item, "games_per_hour") or 0.0 for item in selected_values) / len(selected_values)
            if selected_values and all(_metric(item, "games_per_hour") is not None for item in selected_values)
            else None
        ),
        "selfplay_wall_time_sec_mean": (
            sum(_metric(item, "selfplay_wall_time_sec") or 0.0 for item in selected_values) / len(selected_values)
            if selected_values and all(_metric(item, "selfplay_wall_time_sec") is not None for item in selected_values)
            else None
        ),
        "stable_observations": len(selected_values),
    }
    refs = tuple(
        item.checkpoint_ref
        for item in selected_values
        if item.checkpoint_ref is not None
    )
    return SelectedProfile(
        profile_id=f"{plan.tuning_id}:{selected.label}",
        topology=plan.topology,
        mode=selected,
        execution_overrides=selected.execution_overrides,
        scientific_config_fingerprint=plan.scientific_config_fingerprint or "unspecified",
        parent_checkpoint=plan.parent_checkpoint.to_dict(),  # type: ignore[union-attr]
        workers=plan.workers,
        measurement_checkpoint_refs=refs,
        execution_code_commit=plan.execution_code_commit,
        device_characteristics=plan.device_characteristics,
        basis={"rule": "mean_games_per_hour_then_wall_time", "metrics": metric_values},
        limitations=tuple(limitations),
        source=source,
    )


def select_mode(
    plan: Plan,
    observations: Sequence[Observation | Mapping[str, object]],
    failures: Sequence[Mapping[str, object] | object] = (),
) -> tuple[Mode, list[Observation] | None, str]:
    """Return the policy result without constructing a provenance profile."""

    normalized = [_as_observation(item) for item in observations]
    failed_labels = {
        str(failure["label"])
        for failure in failures
        if isinstance(failure, Mapping) and failure.get("label") is not None
    }
    candidates: list[tuple[float, float, int, Mode, list[Observation]]] = []
    # Insertion order is deliberate: the old implementation's max() retained
    # the first mode on an exact tie, and the baseline was first.
    for order, mode in enumerate(plan.planned_modes):
        if mode.label in failed_labels:
            continue
        values = [item for item in normalized if item.mode.label == mode.label and item.stable]
        values = [item for item in values if _metric(item, "games_per_hour") is not None and _metric(item, "selfplay_wall_time_sec") is not None]
        if not values:
            continue
        hours = sum(_metric(item, "games_per_hour") or 0.0 for item in values) / len(values)
        walls = sum(_metric(item, "selfplay_wall_time_sec") or 0.0 for item in values) / len(values)
        candidates.append((hours, walls, -order, mode, values))
    if candidates:
        _hours, _wall, _order, selected, selected_values = max(
            candidates, key=lambda item: (item[0], -item[1], item[2])
        )
        return selected, selected_values, "measured_selection"
    return plan.baseline, [], "explicit_baseline_fallback"
def choose_next(plan: Plan, state: Mapping[str, object]) -> Decision:
    """Choose one persisted action without starting it."""

    selected = state.get("selected_profile")
    if selected is not None:
        return Decision(DecisionType.FINISH, reason="selected profile is already durable")
    raw_intent = state.get("fallback_intent")
    if isinstance(raw_intent, Mapping) and raw_intent.get("pending") is True:
        mode = plan.baseline
        action_id = raw_intent.get("action_id")
        return Decision(
            DecisionType.RETRY_BASELINE,
            action_id=None if action_id is None else str(action_id),
            mode=mode,
            reason="durable baseline recovery intent is pending",
            fallback_intent=dict(raw_intent),
        )
    raw_index = state.get("next_index", 0)
    try:
        index = int(raw_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("tuning state next_index is malformed") from exc
    actions = plan.action_profiles()
    if index < len(actions):
        action_id, mode, _role = actions[index]
        return Decision(DecisionType.RUN_MODE, action_id=action_id, mode=mode, reason="planned measurement")
    observations = state.get("observations", [])
    failures = state.get("failed_modes", [])
    if not isinstance(observations, list) or not isinstance(failures, list):
        raise ValueError("tuning state evidence is malformed")
    # Finish is the only decision that can create a selected profile; callers
    # perform the deterministic select_profile operation before persisting it.
    expected_labels = {mode.label for mode in plan.planned_modes}
    evidence_labels = {
        item.get("mode", {}).get("label")
        for item in observations
        if isinstance(item, Mapping) and isinstance(item.get("mode"), Mapping)
    }
    failed_labels = {
        item.get("label")
        for item in failures
        if isinstance(item, Mapping)
    }
    if expected_labels.issubset(evidence_labels | failed_labels):
        return Decision(DecisionType.FINISH, reason="all planned modes have terminal evidence")
    return Decision(DecisionType.STOP_WITH_ERROR, reason="measurement budget exhausted before all modes completed")


__all__ = ["assess_observation", "choose_next", "classify_failure", "select_mode", "select_profile"]
