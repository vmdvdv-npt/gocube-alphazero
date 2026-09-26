"""Structured operator facts shared by orchestrators and delivery services.

An :class:`OperatorEvent` is a durable description of a fact, not a Telegram
message.  In particular, ``event_id`` is derived from the logical identity of
the fact and never from a PID, wall-clock time, or formatted text.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
OPERATOR_EVENT_SCHEMA = "gocube-operator-event-v1"


class EventType(str, Enum):
    TRAINING_STARTED = "TRAINING_STARTED"
    GENERATION_STARTED = "GENERATION_STARTED"
    GENERATION_COMMITTED = "GENERATION_COMMITTED"
    ARENA_STARTED = "ARENA_STARTED"
    ARENA_COMPLETED = "ARENA_COMPLETED"
    ARENA_FAILED = "ARENA_FAILED"
    STOP_REQUESTED = "STOP_REQUESTED"
    RUN_STOPPED = "RUN_STOPPED"
    RUN_COMPLETED = "RUN_COMPLETED"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_STAGE_DECIDED = "EXPERIMENT_STAGE_DECIDED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    CALIBRATION_STARTED = "CALIBRATION_STARTED"
    CALIBRATION_DECIDED = "CALIBRATION_DECIDED"
    CALIBRATION_HANDOFF_COMPLETED = "CALIBRATION_HANDOFF_COMPLETED"
    TUNING_STARTED = "TUNING_STARTED"
    TUNING_MODE_REJECTED = "TUNING_MODE_REJECTED"
    TUNING_SELECTED = "TUNING_SELECTED"
    TUNING_FAILED = "TUNING_FAILED"
    RUN_FAILED = "RUN_FAILED"
    DELIVERY_DEGRADED = "DELIVERY_DEGRADED"


EVENT_TYPES = frozenset(item.value for item in EventType)
OWNER_TYPES = frozenset({"lineage", "evaluation", "experiment", "workflow"})
VALIDITIES = frozenset({"VALID", "TECHNICAL", "CRITICAL", "INVALID"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any, *, label: str = "value") -> Any:
    """Return a JSON-only copy and reject secrets/artifact objects early."""
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and "api.telegram.org/bot" in value.lower():
            raise ValueError(f"{label} must not contain a Telegram bot URL")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} object keys must be strings")
            lowered = key.lower()
            if lowered in {"token", "bot_token", "environ", "environment", "env", "tensor", "tensors", "replay_data", "checkpoint_contents", "stack_trace", "traceback"}:
                raise ValueError(f"{label} contains prohibited operator-event field: {key}")
            result[key] = _json_value(item, label=f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label=f"{label}[]") for item in value]
    raise ValueError(f"{label} must contain only JSON-compatible values")


def canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _event_type(value: str | EventType) -> str:
    result = value.value if isinstance(value, EventType) else str(value).upper()
    if result not in EVENT_TYPES:
        raise ValueError(f"unknown operator event type: {result}")
    return result


def event_id_for(event_type: str | EventType, identity: Mapping[str, Any]) -> str:
    """Build a stable ID from a complete logical identity.

    Callers should include every scientific identity component in ``identity``.
    ``recorded_at`` and formatted text are intentionally not accepted as part
    of this primitive's implicit input.
    """
    kind = _event_type(event_type)
    digest = hashlib.sha256(canonical_json({"event_type": kind, "identity": identity}).encode("utf-8")).hexdigest()
    return f"operator:{kind.lower()}:{digest}"


def evidence_refs(value: object) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("evidence_refs must be a list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each evidence_ref must be an object")
        result.append(_json_value(item, label="evidence_refs"))
    return tuple(result)


@dataclass(frozen=True)
class OperatorEvent:
    schema_version: int
    event_id: str
    event_type: str
    occurred_at: str
    recorded_at: str
    topology: str
    owner_type: str
    owner_id: str
    action_id: str
    launch_id: str | None = None
    attempt: int | None = None
    correlation_id: str | None = None
    payload: Mapping[str, Any] = None  # type: ignore[assignment]
    evidence_refs: tuple[dict[str, Any], ...] = ()
    producer_version: str = "unknown"
    execution_code_commit: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported operator event schema_version")
        object.__setattr__(self, "event_id", str(self.event_id))
        object.__setattr__(self, "event_type", _event_type(self.event_type))
        if not str(self.event_id).strip():
            raise ValueError("event_id is required")
        if not str(self.topology).strip():
            raise ValueError("topology is required")
        if self.owner_type not in OWNER_TYPES:
            raise ValueError(f"owner_type must be one of {sorted(OWNER_TYPES)}")
        for label in ("occurred_at", "recorded_at", "owner_id", "action_id", "producer_version"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label).strip():
                raise ValueError(f"{label} is required")
        if self.attempt is not None and (type(self.attempt) is not int or self.attempt < 1):
            raise ValueError("attempt must be a positive integer")
        object.__setattr__(self, "payload", _json_value(self.payload or {}, label="payload"))
        object.__setattr__(self, "evidence_refs", evidence_refs(self.evidence_refs))

    @classmethod
    def create(
        cls,
        event_type: str | EventType,
        *,
        topology: str,
        owner_type: str,
        owner_id: str,
        action_id: str,
        payload: Mapping[str, Any] | None = None,
        evidence_refs: object = None,
        occurred_at: str | None = None,
        recorded_at: str | None = None,
        launch_id: str | None = None,
        attempt: int | None = None,
        correlation_id: str | None = None,
        producer_version: str = "unknown",
        execution_code_commit: str | None = None,
        identity: Mapping[str, Any] | None = None,
    ) -> "OperatorEvent":
        kind = _event_type(event_type)
        logical_identity = identity or {
            "topology": topology,
            "owner_type": owner_type,
            "owner_id": owner_id,
            "action_id": action_id,
            "correlation_id": correlation_id,
            "payload": payload or {},
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            event_id=event_id_for(kind, logical_identity),
            event_type=kind,
            occurred_at=occurred_at or utc_now(),
            recorded_at=recorded_at or utc_now(),
            topology=topology,
            owner_type=owner_type,
            owner_id=owner_id,
            action_id=action_id,
            launch_id=launch_id,
            attempt=attempt,
            correlation_id=correlation_id,
            payload=payload or {},
            evidence_refs=evidence_refs,
            producer_version=producer_version,
            execution_code_commit=execution_code_commit,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(_json_value(self.payload))
        for key in ("legacy_message", "legacy_event", "formatted_text"):
            payload.pop(key, None)
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "recorded_at": self.recorded_at,
            "topology": self.topology,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "action_id": self.action_id,
            "launch_id": self.launch_id,
            "attempt": self.attempt,
            "correlation_id": self.correlation_id,
            "payload": payload,
            "evidence_refs": _json_value(self.evidence_refs),
            "producer_version": self.producer_version,
            "execution_code_commit": self.execution_code_commit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperatorEvent":
        required = {"schema_version", "event_id", "event_type", "occurred_at", "recorded_at", "topology", "owner_type", "owner_id", "action_id", "payload", "evidence_refs", "producer_version"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"operator event missing fields: {', '.join(missing)}")
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})

    def identity_dict(self) -> dict[str, Any]:
        """Fields compared when the same event ID is published twice."""
        # Wall-clock observation and process-attempt metadata are not
        # scientific content.  A retry reconstructed from the same durable
        # result must remain the same logical fact even if it is recorded by a
        # new process at a different time.
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "topology": self.topology,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "action_id": self.action_id,
            "correlation_id": self.correlation_id,
            "payload": _json_value(self.payload),
            "execution_code_commit": self.execution_code_commit,
        }


__all__ = [
    "EVENT_TYPES", "EventType", "OperatorEvent", "OPERATOR_EVENT_SCHEMA",
    "OWNER_TYPES", "SCHEMA_VERSION", "VALIDITIES", "canonical_json",
    "event_id_for", "utc_now",
]
