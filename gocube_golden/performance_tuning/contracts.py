"""Immutable contracts for execution-profile measurement.

The tuning package is deliberately small.  It describes *which* execution
profile may be measured and records the evidence; process supervision and
generation publication remain in the existing Orchestrator V2 layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..artifact_graph import CheckpointRef, EffectiveConfig
from ..provenance import sha256_fingerprint


PERFORMANCE_TUNING_SCHEMA = "gocube-performance-tuning-v1"
MODE_SCHEMA = f"{PERFORMANCE_TUNING_SCHEMA}-mode-v1"
PLAN_SCHEMA = f"{PERFORMANCE_TUNING_SCHEMA}-plan-v1"
OBSERVATION_SCHEMA = f"{PERFORMANCE_TUNING_SCHEMA}-observation-v1"
SELECTED_PROFILE_SCHEMA = f"{PERFORMANCE_TUNING_SCHEMA}-selected-profile-v1"

# This is intentionally an allowlist, rather than a list of keys removed from
# an arbitrary effective-config dictionary.  The production generation layer
# uses the same names in ResolvedGenerationInput.
EXECUTION_OVERRIDE_ALLOWLIST = frozenset(
    {"active_games_per_worker", "total_active_contexts"}
)
EXECUTION_ALIAS_NAMES = frozenset(
    {"active_contexts", "contexts", "inference_batch_cap", "inference_batch_wait_ms", "inference_wait"}
)


class DecisionType(str, Enum):
    RUN_MODE = "RUN_MODE"
    RETRY_BASELINE = "RETRY_BASELINE"
    FINISH = "FINISH"
    STOP_WITH_ERROR = "STOP_WITH_ERROR"


DecisionKind = DecisionType


class FailureCategory(str, Enum):
    TRANSIENT_EXECUTION_FAILURE = "transient_execution_failure"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CONFIGURATION_ERROR = "configuration_error"
    ARTIFACT_INTEGRITY_ERROR = "artifact_integrity_error"
    EXPLICIT_STOP = "explicit_stop"
    UNKNOWN_FAILURE = "unknown_failure"


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, label: str) -> float:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _checkpoint_dict(value: CheckpointRef | Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, CheckpointRef):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint reference must be an object")
    return dict(value)


def scientific_contract_payload(config: object) -> dict[str, object]:
    """Return the experiment contract with execution knobs excluded.

    ``EffectiveConfig.fingerprint`` intentionally includes execution details.
    Tuning therefore records this separate fingerprint: applying an allowed
    runtime profile may change scheduling, but cannot change self-play,
    replay, training, arena, topology, or compatibility identity.
    """

    raw = config
    if hasattr(raw, "config"):
        raw = getattr(raw, "config")
    if isinstance(raw, EffectiveConfig):
        payload = raw.to_dict()
    elif hasattr(raw, "to_dict") and callable(getattr(raw, "to_dict")):
        value = raw.to_dict()
        if not isinstance(value, Mapping):
            raise ValueError("scientific config to_dict() must return an object")
        payload = dict(value)
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise TypeError("scientific contract requires an EffectiveConfig or object")
    payload.pop("execution", None)
    return payload


def scientific_contract_fingerprint(config: object) -> str:
    return sha256_fingerprint(scientific_contract_payload(config))


def validate_execution_overrides(value: Mapping[str, object] | None) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("execution profile must be an object")
    unknown = set(value) - EXECUTION_OVERRIDE_ALLOWLIST
    if unknown:
        raise ValueError(
            "execution profile contains scientific or unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    aliases = EXECUTION_ALIAS_NAMES.intersection(value)
    if aliases:
        raise ValueError("execution profile contains unsupported aliases: " + ", ".join(sorted(aliases)))
    result: dict[str, int] = {}
    for key in EXECUTION_OVERRIDE_ALLOWLIST:
        if key not in value:
            continue
        result[key] = _positive_int(value[key], f"execution profile.{key}")
    return result


@dataclass(frozen=True)
class Mode:
    """One execution-only profile.

    Worker count is intentionally not part of a mode.  It is a fixed plan
    constraint and is never sent as an execution override.
    """

    label: str
    active_games_per_worker: int
    total_active_contexts: int
    schema: str = MODE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _component(self.label, "mode label"))
        _positive_int(self.active_games_per_worker, "active_games_per_worker")
        _positive_int(self.total_active_contexts, "total_active_contexts")
        if self.schema != MODE_SCHEMA:
            raise ValueError("unsupported performance tuning mode schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "active_games_per_worker": self.active_games_per_worker,
            "total_active_contexts": self.total_active_contexts,
        }

    @property
    def execution_overrides(self) -> dict[str, int]:
        return {
            "active_games_per_worker": self.active_games_per_worker,
            "total_active_contexts": self.total_active_contexts,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], *, label: str = "mode") -> "Mode":
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
        allowed = {"schema", "label", "active_games_per_worker", "total_active_contexts"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"{label} contains unsupported fields: " + ", ".join(sorted(map(str, unknown))))
        if value.get("schema", MODE_SCHEMA) != MODE_SCHEMA:
            raise ValueError(f"{label} has an unsupported schema")
        missing = {"label", "active_games_per_worker", "total_active_contexts"} - set(value)
        if missing:
            raise ValueError(f"{label} is missing explicit fields: " + ", ".join(sorted(missing)))
        return cls(
            str(value["label"]),
            value["active_games_per_worker"],  # type: ignore[arg-type]
            value["total_active_contexts"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class MeasurementBudget:
    """Explicit count of new generation actions allowed to the tuner."""

    max_actions: int | None = None
    baseline_repetitions: int = 1
    repetitions_per_mode: int = 1
    baseline_recovery_actions: int = 1

    def __post_init__(self) -> None:
        if self.max_actions is not None:
            _positive_int(self.max_actions, "measurement_budget.max_actions")
        _positive_int(self.baseline_repetitions, "measurement_budget.baseline_repetitions")
        _positive_int(self.repetitions_per_mode, "measurement_budget.repetitions_per_mode")
        _nonnegative_int(self.baseline_recovery_actions, "measurement_budget.baseline_recovery_actions")

    @classmethod
    def from_value(cls, value: object, *, required_actions: int) -> "MeasurementBudget":
        if value is None:
            return cls(max_actions=required_actions)
        if type(value) is int:
            return cls(max_actions=_positive_int(value, "measurement_budget"))
        if not isinstance(value, Mapping):
            raise ValueError("measurement_budget must be a positive integer or object")
        allowed = {
            "max_actions", "max_generations", "baseline_repetitions",
            "repetitions_per_mode", "baseline_recovery_actions",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("measurement_budget contains unsupported fields: " + ", ".join(sorted(map(str, unknown))))
        max_actions = value.get("max_actions", value.get("max_generations"))
        return cls(
            max_actions=None if max_actions is None else _positive_int(max_actions, "measurement_budget.max_actions"),
            baseline_repetitions=_positive_int(value.get("baseline_repetitions", 1), "measurement_budget.baseline_repetitions"),
            repetitions_per_mode=_positive_int(value.get("repetitions_per_mode", 1), "measurement_budget.repetitions_per_mode"),
            baseline_recovery_actions=_nonnegative_int(value.get("baseline_recovery_actions", 1), "measurement_budget.baseline_recovery_actions"),
        )

    def required_actions(self, mode_count: int) -> int:
        return self.baseline_repetitions + mode_count * self.repetitions_per_mode

    def to_dict(self) -> dict[str, object]:
        return {
            "max_actions": self.max_actions,
            "baseline_repetitions": self.baseline_repetitions,
            "repetitions_per_mode": self.repetitions_per_mode,
            "baseline_recovery_actions": self.baseline_recovery_actions,
        }


@dataclass(frozen=True)
class Plan:
    tuning_id: str
    topology: str
    parent_checkpoint: CheckpointRef | Mapping[str, object]
    baseline: Mode
    modes: tuple[Mode, ...] = ()
    workers: int = 1
    scientific_config_fingerprint: str | None = None
    measurement_budget: MeasurementBudget | Mapping[str, object] | int | None = None
    measurement_contract: Mapping[str, object] = field(default_factory=dict)
    owner_id: str | None = None
    owner_root: str | Path | None = None
    execution_code_commit: str | None = None
    device_characteristics: Mapping[str, object] = field(default_factory=dict)
    finish_behavior: str = "export_profile"
    schema: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "tuning_id", _component(self.tuning_id, "tuning_id"))
        object.__setattr__(self, "topology", _component(self.topology, "topology"))
        parent = self.parent_checkpoint if isinstance(self.parent_checkpoint, CheckpointRef) else CheckpointRef.from_dict(self.parent_checkpoint)
        if parent.topology != self.topology:
            raise ValueError("plan topology does not match parent checkpoint")
        object.__setattr__(self, "parent_checkpoint", parent)
        if not isinstance(self.baseline, Mode):
            object.__setattr__(self, "baseline", Mode.from_dict(self.baseline))  # type: ignore[arg-type]
        modes = tuple(item if isinstance(item, Mode) else Mode.from_dict(item) for item in self.modes)
        labels = {self.baseline.label}
        for mode in modes:
            if mode.label in labels:
                raise ValueError(f"duplicate performance tuning mode: {mode.label}")
            labels.add(mode.label)
        object.__setattr__(self, "modes", modes)
        _positive_int(self.workers, "plan workers")
        for mode in (self.baseline, *modes):
            if mode.total_active_contexts > self.workers * mode.active_games_per_worker:
                raise ValueError(f"mode {mode.label} exceeds worker lane capacity")
        budget = self.measurement_budget
        required = 1 + len(modes)
        if budget is None:
            normalized_budget = MeasurementBudget(max_actions=required + 1)
        elif isinstance(budget, MeasurementBudget):
            normalized_budget = budget
        else:
            normalized_budget = MeasurementBudget.from_value(budget, required_actions=required)
        if normalized_budget.max_actions is None:
            normalized_budget = MeasurementBudget(
                max_actions=normalized_budget.required_actions(len(modes)),
                baseline_repetitions=normalized_budget.baseline_repetitions,
                repetitions_per_mode=normalized_budget.repetitions_per_mode,
                baseline_recovery_actions=normalized_budget.baseline_recovery_actions,
            )
        if normalized_budget.max_actions < normalized_budget.required_actions(len(modes)):
            raise ValueError("measurement budget is smaller than the explicitly planned measurements")
        object.__setattr__(self, "measurement_budget", normalized_budget)
        fingerprint = self.scientific_config_fingerprint
        if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
            raise ValueError("scientific_config_fingerprint must be a non-empty string")
        if self.owner_id is not None:
            object.__setattr__(self, "owner_id", _component(self.owner_id, "owner_id"))
        if not isinstance(self.measurement_contract, Mapping):
            raise ValueError("measurement_contract must be an object")
        object.__setattr__(self, "measurement_contract", dict(self.measurement_contract))
        if not isinstance(self.device_characteristics, Mapping):
            raise ValueError("device_characteristics must be an object")
        object.__setattr__(self, "device_characteristics", dict(self.device_characteristics))
        if self.owner_root is not None:
            object.__setattr__(self, "owner_root", str(Path(self.owner_root).resolve()))
        if self.finish_behavior not in {"export_profile", "continue_training"}:
            raise ValueError("finish_behavior must be export_profile or continue_training")
        if self.schema != PLAN_SCHEMA:
            raise ValueError("unsupported performance tuning plan schema")

    @property
    def plan_fingerprint(self) -> str:
        return sha256_fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def planned_modes(self) -> tuple[Mode, ...]:
        return (self.baseline, *self.modes)

    def action_profiles(self) -> tuple[tuple[str, Mode, str], ...]:
        actions: list[tuple[str, Mode, str]] = []
        index = 0
        for _ in range(self.measurement_budget.baseline_repetitions):  # type: ignore[union-attr]
            actions.append((f"{self.tuning_id}:measurement:{index:04d}:{self.baseline.label}", self.baseline, "baseline"))
            index += 1
        for mode in self.modes:
            for _ in range(self.measurement_budget.repetitions_per_mode):  # type: ignore[union-attr]
                actions.append((f"{self.tuning_id}:measurement:{index:04d}:{mode.label}", mode, "measurement"))
                index += 1
        return tuple(actions)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "tuning_id": self.tuning_id,
            "topology": self.topology,
            "parent_checkpoint": self.parent_checkpoint.to_dict(),  # type: ignore[union-attr]
            "baseline": self.baseline.to_dict(),
            "modes": [mode.to_dict() for mode in self.modes],
            "workers": self.workers,
            "scientific_config_fingerprint": self.scientific_config_fingerprint,
            "measurement_budget": self.measurement_budget.to_dict(),  # type: ignore[union-attr]
            "measurement_contract": _thaw(self.measurement_contract),
            "owner_id": self.owner_id,
            "owner_root": self.owner_root,
            "execution_code_commit": self.execution_code_commit,
            "device_characteristics": _thaw(self.device_characteristics),
            "finish_behavior": self.finish_behavior,
        }
        if include_fingerprint:
            payload["plan_fingerprint"] = self.plan_fingerprint
        return payload


@dataclass(frozen=True)
class Observation:
    action_id: str
    mode: Mode
    generation: int
    metrics: Mapping[str, object]
    stable: bool
    stability_reasons: tuple[str, ...] = ()
    checkpoint_ref: Mapping[str, object] | None = None
    evidence_refs: tuple[Mapping[str, object], ...] = ()
    role: str = "measurement"
    raw_metrics: Mapping[str, object] = field(default_factory=dict)
    schema: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        _component(self.action_id, "observation action_id")
        if not isinstance(self.mode, Mode):
            object.__setattr__(self, "mode", Mode.from_dict(self.mode))  # type: ignore[arg-type]
        _nonnegative_int(self.generation, "observation generation")
        if not isinstance(self.metrics, Mapping) or not isinstance(self.raw_metrics, Mapping):
            raise ValueError("observation metrics must be objects")
        object.__setattr__(self, "stability_reasons", tuple(str(item) for item in self.stability_reasons))
        object.__setattr__(self, "evidence_refs", tuple(dict(item) for item in self.evidence_refs))
        if self.schema != OBSERVATION_SCHEMA:
            raise ValueError("unsupported performance tuning observation schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "action_id": self.action_id,
            "mode": self.mode.to_dict(),
            "generation": self.generation,
            "metrics": _thaw(self.metrics),
            "raw_metrics": _thaw(self.raw_metrics),
            "stable": self.stable,
            "stability_reasons": list(self.stability_reasons),
            "checkpoint_ref": _thaw(self.checkpoint_ref),
            "evidence_refs": _thaw(self.evidence_refs),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Observation":
        raw_metrics = value.get("raw_metrics")
        if not isinstance(raw_metrics, Mapping):
            raw_metrics = {}
        metrics_value = value.get("metrics")
        if not isinstance(metrics_value, Mapping):
            # The legacy performance_sweep state stored normalized telemetry
            # at the observation top level.  Read it without rewriting the
            # canonical old state.
            metrics_value = {
                key: value[key]
                for key in (
                    "games", "moves", "selfplay_wall_time_sec", "games_per_hour",
                    "mean_game_length", "moves_per_sec", "mean_inference_batch",
                    "gpu_utilization_percent", "gpu_power_w", "cpu_utilization_percent",
                    "technical_games", "invalid_games", "stalls", "actual_execution", "timing",
                )
                if key in value
            }
        return cls(
            action_id=str(value["action_id"]),
            mode=Mode.from_dict(value["mode"]),  # type: ignore[arg-type]
            generation=int(value["generation"]),
            metrics=dict(metrics_value),  # type: ignore[arg-type]
            stable=bool(value.get("stable", False)),
            stability_reasons=tuple(value.get("stability_reasons", ())),  # type: ignore[arg-type]
            checkpoint_ref=value.get("checkpoint_ref"),  # type: ignore[arg-type]
            evidence_refs=tuple(value.get("evidence_refs", ())),  # type: ignore[arg-type]
            role=str(value.get("role", "measurement")),
            raw_metrics=dict(raw_metrics),
        )


@dataclass(frozen=True)
class Decision:
    kind: DecisionType
    action_id: str | None = None
    mode: Mode | None = None
    reason: str = ""
    failure_category: FailureCategory | None = None
    fallback_intent: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, DecisionType) else DecisionType(str(self.kind))
        object.__setattr__(self, "kind", kind)
        if self.action_id is not None:
            _component(self.action_id.replace(":", "_"), "decision action_id")
        if kind is DecisionType.RUN_MODE and self.mode is None:
            raise ValueError("RUN_MODE decision requires a mode")
        if kind is DecisionType.RETRY_BASELINE and self.fallback_intent is None:
            object.__setattr__(self, "fallback_intent", {"required": True})

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "mode": None if self.mode is None else self.mode.to_dict(),
            "reason": self.reason,
            "failure_category": None if self.failure_category is None else self.failure_category.value,
            "fallback_intent": _thaw(self.fallback_intent),
        }


@dataclass(frozen=True)
class SelectedProfile:
    profile_id: str
    topology: str
    mode: Mode
    execution_overrides: Mapping[str, object]
    scientific_config_fingerprint: str
    parent_checkpoint: Mapping[str, object]
    workers: int | None = None
    measurement_checkpoint_refs: tuple[Mapping[str, object], ...] = ()
    execution_code_commit: str | None = None
    device_characteristics: Mapping[str, object] = field(default_factory=dict)
    basis: Mapping[str, object] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    source: str = "measured_selection"
    schema_version: str = SELECTED_PROFILE_SCHEMA
    fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _component(self.profile_id, "profile_id"))
        object.__setattr__(self, "topology", _component(self.topology, "profile topology"))
        if not isinstance(self.mode, Mode):
            object.__setattr__(self, "mode", Mode.from_dict(self.mode))  # type: ignore[arg-type]
        overrides = validate_execution_overrides(self.execution_overrides)
        if overrides != self.mode.execution_overrides:
            raise ValueError("selected profile overrides do not match selected mode")
        object.__setattr__(self, "execution_overrides", overrides)
        if not isinstance(self.scientific_config_fingerprint, str) or not self.scientific_config_fingerprint:
            raise ValueError("selected profile requires scientific_config_fingerprint")
        if self.workers is not None:
            _positive_int(self.workers, "selected profile workers")
        if self.source not in {"measured_selection", "explicit_baseline_fallback"}:
            raise ValueError("selected profile source is unsupported")
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))
        object.__setattr__(self, "measurement_checkpoint_refs", tuple(dict(item) for item in self.measurement_checkpoint_refs))
        calculated = sha256_fingerprint(self.to_dict(include_fingerprint=False))
        if self.fingerprint and self.fingerprint != calculated:
            raise ValueError("selected profile fingerprint does not match its contents")
        object.__setattr__(self, "fingerprint", calculated)
        if self.schema_version != SELECTED_PROFILE_SCHEMA:
            raise ValueError("unsupported selected profile schema")

    @property
    def selected_mode(self) -> Mode:
        return self.mode

    @property
    def parent_start_checkpoint_ref(self) -> Mapping[str, object]:
        return self.parent_checkpoint

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "fingerprint": self.fingerprint if include_fingerprint else None,
            "topology": self.topology,
            "mode": self.mode.to_dict(),
            "selected_mode": self.mode.to_dict(),
            "execution_overrides": dict(self.execution_overrides),
            "execution_allowlist": dict(self.execution_overrides),
            "workers": self.workers,
            "scientific_config_fingerprint": self.scientific_config_fingerprint,
            "parent_checkpoint": dict(self.parent_checkpoint),
            "parent_start_checkpoint_ref": dict(self.parent_checkpoint),
            "measurement_checkpoint_refs": [dict(item) for item in self.measurement_checkpoint_refs],
            "execution_code_commit": self.execution_code_commit,
            "device_characteristics": _thaw(self.device_characteristics),
            "basis": _thaw(self.basis),
            "limitations": list(self.limitations),
            "source": self.source,
        }
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SelectedProfile":
        return cls(
            profile_id=str(value["profile_id"]),
            topology=str(value["topology"]),
            mode=Mode.from_dict(value.get("mode", value.get("selected_mode")), label="selected profile mode"),  # type: ignore[arg-type]
            execution_overrides=dict(value.get("execution_overrides", value.get("execution_allowlist", {}))),  # type: ignore[arg-type]
            scientific_config_fingerprint=str(value["scientific_config_fingerprint"]),
            parent_checkpoint=dict(value.get("parent_checkpoint", value.get("parent_start_checkpoint_ref"))),  # type: ignore[arg-type]
            workers=None if value.get("workers") is None else int(value["workers"]),
            measurement_checkpoint_refs=tuple(value.get("measurement_checkpoint_refs", ())),  # type: ignore[arg-type]
            execution_code_commit=None if value.get("execution_code_commit") is None else str(value["execution_code_commit"]),
            device_characteristics=dict(value.get("device_characteristics", {})),  # type: ignore[arg-type]
            basis=dict(value.get("basis", {})),  # type: ignore[arg-type]
            limitations=tuple(value.get("limitations", ())),  # type: ignore[arg-type]
            source=str(value.get("source", "measured_selection")),
            schema_version=str(value.get("schema_version", SELECTED_PROFILE_SCHEMA)),
            fingerprint=str(value.get("fingerprint", "")),
        )

    def validate_applicability(
        self,
        *,
        topology: str,
        scientific_config_fingerprint: str,
        execution_code_commit: str | None = None,
        device_characteristics: Mapping[str, object] | None = None,
        workers: int | None = None,
    ) -> None:
        if topology != self.topology:
            raise ValueError("selected profile topology does not match the run")
        if scientific_config_fingerprint != self.scientific_config_fingerprint:
            raise ValueError("selected profile scientific contract does not match the run")
        if self.workers is not None and workers is not None and self.workers != workers:
            raise ValueError("selected profile worker constraint does not match the run")
        if execution_code_commit is not None and self.execution_code_commit not in {None, execution_code_commit}:
            raise ValueError("selected profile execution code commit does not match the run")
        if device_characteristics is not None:
            for key, value in self.device_characteristics.items():
                if key in device_characteristics and device_characteristics[key] != value:
                    raise ValueError(f"selected profile device characteristic mismatch: {key}")


def execution_overrides_for(mode: Mode, *, workers: int, scientific_config: object | None = None) -> dict[str, int]:
    """Resolve the exact production mapping and assert its scientific fence."""

    _positive_int(workers, "plan workers")
    if mode.total_active_contexts > workers * mode.active_games_per_worker:
        raise ValueError("execution mode exceeds worker lane capacity")
    if scientific_config is not None:
        before = scientific_contract_fingerprint(scientific_config)
        # Profiles are passed as an override object, not merged into the
        # effective config.  Recompute the same immutable scientific view to
        # make that boundary explicit and testable.
        after = scientific_contract_fingerprint(scientific_config)
        if before != after:
            raise ValueError("execution profile changed the scientific contract")
    return validate_execution_overrides(mode.execution_overrides)


__all__ = [
    "Decision", "DecisionKind", "DecisionType", "EXECUTION_ALIAS_NAMES",
    "EXECUTION_OVERRIDE_ALLOWLIST", "FailureCategory", "MeasurementBudget",
    "Mode", "Observation", "PERFORMANCE_TUNING_SCHEMA", "PLAN_SCHEMA", "Plan",
    "SelectedProfile", "execution_overrides_for", "scientific_contract_fingerprint",
    "scientific_contract_payload", "validate_execution_overrides",
]
