"""Declarative production interface for Orchestrator V2."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

RUN_SPEC_SCHEMA = "gocube-orchestrator-v2-run-spec-v1"


class RunMode(str, Enum):
    CONTINUOUS = "continuous"
    ARENA = "arena"
    EVALUATION = "evaluation"
    EXPERIMENT = "experiment"
    CALIBRATION = "calibration"
    WORKFLOW = "workflow"
    SCENARIO = "scenario"


def _validate_search_payload(value: object, label: str = "search") -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    allowed = {
        "simulations",
        "mcts_simulations",
        "cpuct",
        "fpu",
        "watchdog",
        "technical_move_limit",
        "komi",
        "root_noise",
        "temperature",
        "fast_search",
        "resign",
        "deterministic_tie_break",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"{label} contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )


def _validate_arena_config_payload(value: object, label: str = "arena_config") -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    execution = {
        "games",
        "workers",
        "games_per_worker",
        "inference_batch_rows",
        "inference_batch_wait_ms",
        "inference_batch_cap",
        "inference_wait",
        "inference_wait_ms",
        "device",
        "strict_production",
        "monitoring_acceptance",
        "min_mean_inference_batch_rows",
        "min_effective_cpu_cores",
        "early_gate_enabled",
        "early_gate_min_forwards",
        "early_gate_min_wall_sec",
    }
    unknown = set(value) - execution - {
        "simulations", "mcts_simulations", "cpuct", "fpu", "watchdog",
        "technical_move_limit", "komi", "root_noise", "temperature",
        "fast_search", "resign", "deterministic_tie_break", "search",
    }
    if unknown:
        raise ValueError(
            f"{label} contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    if "search" in value:
        _validate_search_payload(value["search"], f"{label}.search")


def _validate_supervision_payload(value: object, label: str = "supervision") -> None:
    from .supervisor import SupervisorPolicy

    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    policy_keys = SupervisorPolicy.FIELD_NAMES
    for key, item in value.items():
        if key in {"default", "common", "generation", "arena", "evaluation", "calibration", "workflow"}:
            SupervisorPolicy.from_dict(item, label=f"{label}.{key}")
        elif key in policy_keys:
            # A flat policy is a useful shorthand for one action's default.
            continue
        else:
            raise ValueError(f"{label} contains unsupported field: {key}")
    flat = {key: value[key] for key in value if key in policy_keys}
    if flat:
        SupervisorPolicy.from_dict(flat, label=label)


@dataclass(frozen=True)
class RunSpecV2:
    mode: RunMode
    payload: Mapping[str, object]
    schema: str = RUN_SPEC_SCHEMA

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RunSpecV2":
        if not isinstance(value, Mapping):
            raise ValueError("Orchestrator V2 run-spec must be an object")
        schema = value.get("schema", RUN_SPEC_SCHEMA)
        if schema != RUN_SPEC_SCHEMA:
            raise ValueError(f"unsupported Orchestrator V2 run-spec schema: {schema!r}")
        raw_mode = value.get("mode")
        if not isinstance(raw_mode, str) or not raw_mode:
            raise ValueError("Orchestrator V2 run-spec requires mode")
        try:
            mode = RunMode(raw_mode)
        except ValueError as exc:
            raise ValueError(f"unsupported Orchestrator V2 run mode: {raw_mode!r}") from exc
        raw_payload = value.get(mode.value)
        if raw_payload is None and mode is RunMode.EVALUATION:
            raw_payload = value.get("arena")
        if raw_payload is None:
            raw_payload = {key: item for key, item in value.items() if key not in {"schema", "mode"}}
        if not isinstance(raw_payload, Mapping):
            raise ValueError(f"{mode.value} run-spec payload must be an object")
        payload = dict(raw_payload)
        if "supervision" in value and "supervision" not in payload:
            payload["supervision"] = value["supervision"]
        _validate_supervision_payload(payload.get("supervision"))
        for key in ("search", "evaluation_search", "arena_search"):
            if key in payload:
                _validate_search_payload(payload[key], f"{mode.value}.{key}")
        if mode in {RunMode.ARENA, RunMode.EVALUATION}:
            for key in ("arena_config", "config"):
                if key in payload:
                    _validate_arena_config_payload(payload[key], f"{mode.value}.{key}")
        return cls(mode=mode, payload=payload, schema=RUN_SPEC_SCHEMA)

    @property
    def supervision(self) -> Mapping[str, object]:
        raw = self.payload.get("supervision", {})
        return raw if isinstance(raw, Mapping) else {}

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "mode": self.mode.value, self.mode.value: dict(self.payload)}


def supervision_policy_for(
    payload: Mapping[str, object], action: str
):
    """Resolve common + action-specific supervision policy for one run."""
    from .supervisor import SupervisorPolicy

    raw = payload.get("supervision", {})
    if raw is None:
        return SupervisorPolicy()
    if not isinstance(raw, Mapping):
        raise ValueError("supervision must be an object")
    flat = {key: raw[key] for key in raw if key in SupervisorPolicy.FIELD_NAMES}
    common_raw = raw.get("default", raw.get("common", flat))
    common = SupervisorPolicy.from_dict(common_raw, label="supervision.default")
    action_raw = raw.get(action)
    if action_raw is None and action == "arena":
        action_raw = raw.get("evaluation")
    return SupervisorPolicy.from_dict(
        action_raw,
        base=common,
        label=f"supervision.{action}",
    )


__all__ = ["RUN_SPEC_SCHEMA", "RunMode", "RunSpecV2", "supervision_policy_for"]
