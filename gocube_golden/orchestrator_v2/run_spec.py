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
        return cls(mode=mode, payload=dict(raw_payload), schema=RUN_SPEC_SCHEMA)

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "mode": self.mode.value, self.mode.value: dict(self.payload)}


__all__ = ["RUN_SPEC_SCHEMA", "RunMode", "RunSpecV2"]
