"""Adapter for the historical inline self-play concurrency sweep.

The adapter keeps the old durable ``performance_sweep`` shape and report path
while delegating observation assessment and selection to the new pure policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from .contracts import FailureCategory, Mode, Plan, scientific_contract_fingerprint
from .policy import assess_observation, classify_failure, select_mode


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: object, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


CONCURRENCY_SWEEP_SCHEMA = "gocube-orchestrator-v2-self-play-concurrency-sweep-v1"


@dataclass(frozen=True)
class SelfPlayConcurrencySweep:
    """Legacy schedule retained as a compatibility input contract."""

    start_after_generation: int
    workers: int
    baseline: Mode
    modes: tuple[Mode, ...]
    continue_after_sweep: bool = True

    def __post_init__(self) -> None:
        if type(self.start_after_generation) is not int or self.start_after_generation < 0:
            raise ValueError("sweep start_after_generation must be a non-negative integer")
        if type(self.workers) is not int or self.workers <= 0:
            raise ValueError("sweep workers must be a positive integer")
        if not self.modes:
            raise ValueError("self-play concurrency sweep requires at least one test mode")
        labels = {self.baseline.label}
        for mode in self.modes:
            if mode.label in labels:
                raise ValueError(f"duplicate self-play concurrency mode: {mode.label}")
            labels.add(mode.label)
        if type(self.continue_after_sweep) is not bool:
            raise ValueError("sweep continue_after_sweep must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CONCURRENCY_SWEEP_SCHEMA,
            "start_after_generation": self.start_after_generation,
            "workers": self.workers,
            "baseline": self.baseline.to_dict(),
            "modes": [mode.to_dict() for mode in self.modes],
            "continue_after_sweep": self.continue_after_sweep,
        }

    def mode_for_index(self, index: int) -> Mode:
        return self.modes[index]


class LegacyConcurrencyAdapter:
    """Compatibility facade with no independent ranking implementation."""

    def __init__(self, sweep: object, config: object, *, lineage_root: Path, logger: logging.Logger | None = None) -> None:
        self.sweep = sweep
        self.config = config
        self.lineage_root = Path(lineage_root).resolve()
        self.logger = logger or logging.getLogger(__name__)
        self.continuous_config: object | None = None

    @property
    def baseline(self) -> Mode:
        return Mode(
            self.sweep.baseline.label,
            self.sweep.baseline.active_games_per_worker,
            self.sweep.baseline.total_active_contexts,
        )

    @property
    def modes(self) -> tuple[Mode, ...]:
        return tuple(Mode(item.label, item.active_games_per_worker, item.total_active_contexts) for item in self.sweep.modes)

    def plan(self, *, parent_checkpoint: object | None = None, lineage_id: str = "legacy") -> Plan | None:
        if parent_checkpoint is None:
            return None
        return Plan(
            tuning_id=f"legacy-{lineage_id}",
            topology=str(getattr(self.config, "topology", "torus9")),
            parent_checkpoint=parent_checkpoint,
            baseline=self.baseline,
            modes=self.modes,
            workers=int(self.sweep.workers),
            scientific_config_fingerprint=scientific_contract_fingerprint(self.config),
            measurement_budget={"max_actions": max(1, len(self.modes) + 1), "baseline_repetitions": 1},
            owner_id=lineage_id,
            owner_root=self.lineage_root,
            finish_behavior="continue_training",
        )

    def new_state(self) -> dict[str, object]:
        return {
            "schema": "gocube-orchestrator-v2-self-play-concurrency-sweep-v1-state-v1",
            "config": self.sweep.to_dict(),
            "status": "RUNNING",
            "observations": [],
            "failed_modes": [],
            "next_mode_index": 0,
            "retry_baseline_generation": None,
            "selected_mode": None,
            "selection_generation": None,
            "selected_profile": None,
            "fallback_intent": None,
        }

    def next_mode(self, raw: Mapping[str, object], generation: int) -> Mode:
        if raw.get("retry_baseline_generation") == generation:
            return self.baseline
        if generation <= self.sweep.start_after_generation:
            return self.baseline
        next_index = int(raw.get("next_mode_index", 0))
        if next_index < len(self.modes):
            return self.modes[next_index]
        raw_selected = raw.get("selected_mode")
        if isinstance(raw_selected, Mapping):
            return Mode.from_dict(raw_selected)
        return self.baseline

    def is_baseline_recovery(self, raw: Mapping[str, object], *, generation: int, mode: Mode) -> bool:
        return mode.label == self.baseline.label and raw.get("retry_baseline_generation") == generation

    def classify_failure(self, error: BaseException) -> FailureCategory:
        return classify_failure(error)

    def load_observation(self, generation: int, mode: Mode, *, role: str) -> dict[str, object] | None:
        summary_path = self.lineage_root / f"iter-{generation:02d}-summary.json"
        if not summary_path.is_file():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.logger.warning("Cannot read sweep summary for M%s: %s", generation, exc)
            return None
        if not isinstance(summary, Mapping):
            return None
        metrics = summary.get("orchestrator_selfplay")
        if not isinstance(metrics, Mapping):
            return None
        selfplay = getattr(self.config, "self_play", {})
        expected_games = selfplay.get("games_per_iteration", selfplay.get("games", 0))
        contract = {
            "mode": mode,
            "generation": generation,
            "action_id": f"legacy:measurement:{generation}:{mode.label}",
            "role": role,
            "expected_games": expected_games,
            "workers": int(self.sweep.workers),
        }
        assessed = assess_observation(metrics, contract)
        values = dict(assessed.metrics)
        request_path = self.lineage_root / "runtime" / "requests" / f"train-one-{generation:04d}.json"
        start_epoch = request_path.stat().st_mtime if request_path.is_file() else None
        raw_execution = values.get("actual_execution")
        observation: dict[str, object] = {
            "generation": generation,
            "role": role,
            "mode": mode.to_dict(),
            "actual_execution": raw_execution,
            "workers": self.sweep.workers,
            "selfplay_wall_time_sec": values.get("selfplay_wall_time_sec"),
            "cycle_wall_time_sec": None,
            "games_per_hour": values.get("games_per_hour"),
            "mean_game_length": values.get("mean_game_length"),
            "mcts_simulations": int(selfplay.get("mcts_simulations", selfplay.get("simulations", 0)) or 0),
            "mcts_simulations_per_sec_equivalent": (
                (int(values.get("moves") or 0) * int(selfplay.get("mcts_simulations", selfplay.get("simulations", 0)) or 0)
                 / float(values["selfplay_wall_time_sec"]))
                if values.get("selfplay_wall_time_sec") not in (None, 0) else None
            ),
            "moves_per_sec": values.get("moves_per_sec"),
            "mean_inference_batch": values.get("mean_inference_batch"),
            "gpu_utilization_percent": values.get("gpu_utilization_percent"),
            "gpu_power_w": values.get("gpu_power_w"),
            "cpu_utilization_percent": values.get("cpu_utilization_percent"),
            "technical_games": values.get("technical_games"),
            "invalid_games": values.get("invalid_games"),
            "stalls": values.get("stalls"),
            "stable": assessed.stable,
            "stability_reasons": list(assessed.stability_reasons),
            "request_path": str(request_path.relative_to(self.lineage_root)) if request_path.is_relative_to(self.lineage_root) else str(request_path),
            "request_start_at": datetime.fromtimestamp(start_epoch, timezone.utc).isoformat() if start_epoch is not None else None,
            "timing": values.get("timing", {}),
        }
        return observation

    def refresh_cycles(self, observations: list[object]) -> None:
        for item in observations:
            if not isinstance(item, dict):
                continue
            try:
                generation = int(item["generation"])
            except (KeyError, TypeError, ValueError):
                continue
            start = self.lineage_root / "runtime" / "requests" / f"train-one-{generation:04d}.json"
            following = self.lineage_root / "runtime" / "requests" / f"train-one-{generation + 1:04d}.json"
            if not start.is_file() or not following.is_file():
                continue
            elapsed = following.stat().st_mtime - start.stat().st_mtime
            if elapsed >= 0.0:
                item["cycle_wall_time_sec"] = elapsed
                item["cycle_end_at"] = datetime.fromtimestamp(following.stat().st_mtime, timezone.utc).isoformat()

    def select(self, raw: dict[str, object], *, generation: int) -> tuple[Mode, str] | None:
        observations = raw.get("observations")
        failures = raw.get("failed_modes")
        if not isinstance(observations, list) or not isinstance(failures, list):
            raise RuntimeError("continuous state self-play sweep records are malformed")
        expected = {mode.label for mode in self.modes}
        observed = {
            item.get("mode", {}).get("label")
            for item in observations
            if isinstance(item, Mapping) and isinstance(item.get("mode"), Mapping)
        }
        failed = {item.get("label") for item in failures if isinstance(item, Mapping)}
        if not expected.issubset(observed | failed):
            return None
        # A Plan is only needed for pure selection.  The legacy state already
        # has its owner/checkpoint information, so a lightweight object with
        # the same fields is sufficient and avoids a storage migration.
        class _PolicyPlan:
            baseline = self.baseline
            modes = self.modes

            @property
            def planned_modes(self):
                return (self.baseline, *self.modes)

        selected, _values, source = select_mode(_PolicyPlan(), observations, failures)  # type: ignore[arg-type]
        raw["selected_mode"] = selected.to_dict()
        raw["selection_generation"] = generation
        raw["status"] = "SELECTED"
        raw["selection_source"] = source
        return selected, source

    def report_payload(self, raw: Mapping[str, object]) -> dict[str, object]:
        observations = raw.get("observations", [])
        observations = observations if isinstance(observations, list) else []
        stable = [
            item for item in observations
            if isinstance(item, Mapping) and item.get("stable") is True
            and isinstance(item.get("mode"), Mapping)
            and item["mode"].get("label") == self.baseline.label
        ]
        baseline_wall = sum(float(item["selfplay_wall_time_sec"]) for item in stable if _number(item.get("selfplay_wall_time_sec")) is not None) / len(stable) if stable and all(_number(item.get("selfplay_wall_time_sec")) is not None for item in stable) else None
        baseline_hours = sum(float(item["games_per_hour"]) for item in stable if _number(item.get("games_per_hour")) is not None) / len(stable) if stable and all(_number(item.get("games_per_hour")) is not None for item in stable) else None
        copied: list[dict[str, object]] = []
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            out = dict(item)
            wall = _number(out.get("selfplay_wall_time_sec"))
            hours = _number(out.get("games_per_hour"))
            out["change_to_baseline_percent"] = (baseline_wall - wall) / baseline_wall * 100.0 if baseline_wall and wall is not None else None
            out["games_per_hour_change_to_baseline_percent"] = (hours - baseline_hours) / baseline_hours * 100.0 if baseline_hours and hours is not None else None
            copied.append(out)
        effective_config = getattr(self.continuous_config, "effective_config", self.config)
        selfplay = getattr(effective_config, "self_play", {})
        training = getattr(effective_config, "training", {})
        replay = getattr(effective_config, "replay", {})
        execution = getattr(effective_config, "execution", {})
        continuous = self.continuous_config
        return {
            "schema": "gocube-orchestrator-v2-self-play-concurrency-sweep-v1-report-v1",
            "lineage_id": getattr(continuous, "lineage_id", None),
            "sweep": self.sweep.to_dict(),
            "contract": {
                "games_per_generation": selfplay.get("games_per_iteration", selfplay.get("games")),
                "mcts_simulations": selfplay.get("mcts_simulations", selfplay.get("simulations")),
                "inference_batch_cap": execution.get("inference_batch_cap"),
                "inference_batch_wait_ms": execution.get("inference_batch_wait_ms"),
                "optimizer": training.get("optimizer"),
                "learning_rate": training.get("learning_rate"),
                "optimizer_steps": training.get("optimizer_steps", training.get("optimizer_steps_per_iteration")),
                "replay_generations": replay.get("generations", replay.get("window")),
                "replay_cap": replay.get("cap"),
                "arena_cadence": getattr(continuous, "arena_cadence", None),
                "arena_games": getattr(getattr(continuous, "arena_config", None), "games", None),
                "arena_diagnostic_only": True,
                "arena_gating": False,
            },
            "status": raw.get("status", "RUNNING"),
            "selected_mode": raw.get("selected_mode"),
            "selection_generation": raw.get("selection_generation"),
            "selection_source": raw.get("selection_source"),
            "baseline_reference": {"observations": len(stable), "mean_selfplay_wall_time_sec": baseline_wall, "mean_games_per_hour": baseline_hours},
            "observations": copied,
            "failed_modes": [dict(item) for item in raw.get("failed_modes", []) if isinstance(item, Mapping)],
        }

    def write_report(self, state: Mapping[str, object]) -> None:
        raw = state.get("performance_sweep") if isinstance(state, Mapping) else None
        if not isinstance(raw, Mapping):
            return
        path = self.lineage_root / "metrics" / "self-play-concurrency-sweep-v1.json"
        try:
            atomic_write_text(path, canonical_json(self.report_payload(raw)) + "\n")
        except (OSError, TypeError, ValueError):
            self.logger.warning("Could not persist self-play concurrency sweep report", exc_info=True)


__all__ = ["CONCURRENCY_SWEEP_SCHEMA", "LegacyConcurrencyAdapter", "SelfPlayConcurrencySweep"]
