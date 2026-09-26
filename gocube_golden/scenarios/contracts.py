"""Small, JSON-stable envelopes between scenarios and execution services.

These contracts deliberately contain references to artifacts and results, not
artifact payloads.  A scenario can therefore persist intent and reconcile a
completed action without knowing how a process was launched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ..provenance import sha256_fingerprint


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    WAITING = "waiting"


class ScientificValidity(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    TECHNICAL = "TECHNICAL"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


def _safe(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[]") for item in value]
    raise ValueError(f"{label} contains an unsupported value")


@dataclass(frozen=True)
class ActionRequest:
    """Durable scenario intent for one concrete execution action."""

    owner: str
    scenario_id: str
    action_id: str
    action_type: str
    request_fingerprint: str
    request_ref: Mapping[str, object] | str | None = None
    dependency_refs: tuple[Mapping[str, object] | str, ...] = ()
    result_refs: tuple[Mapping[str, object] | str, ...] = ()
    supervisor_policy: Mapping[str, object] | None = None
    budget: Mapping[str, object] | None = None
    correlation_id: str | None = None
    schema_version: str = "gocube-scenario-action-request-v1"

    def __post_init__(self) -> None:
        for value, label in (
            (self.owner, "owner"),
            (self.scenario_id, "scenario_id"),
            (self.action_id, "action_id"),
            (self.action_type, "action_type"),
            (self.request_fingerprint, "request_fingerprint"),
        ):
            _safe(value, label)
        if self.schema_version != "gocube-scenario-action-request-v1":
            raise ValueError("unsupported ActionRequest schema")
        if self.request_ref is not None:
            if not isinstance(self.request_ref, (str, Mapping)):
                raise ValueError("request_ref must be a reference or object")
            if isinstance(self.request_ref, Mapping):
                object.__setattr__(self, "request_ref", _mapping(self.request_ref, "request_ref"))
        for label, refs in (("dependency_refs", self.dependency_refs), ("result_refs", self.result_refs)):
            normalized = tuple(refs)
            for ref in normalized:
                if not isinstance(ref, (str, Mapping)):
                    raise ValueError(f"{label} must contain references")
            object.__setattr__(self, label, normalized)
        for value, label in ((self.supervisor_policy, "supervisor_policy"), (self.budget, "budget")):
            if value is not None:
                object.__setattr__(self, label, _mapping(value, label))
        if self.correlation_id is not None:
            _safe(self.correlation_id, "correlation_id")

    @classmethod
    def from_payload(
        cls,
        *,
        owner: str,
        scenario_id: str,
        action_id: str,
        action_type: str,
        request: Mapping[str, object],
        **kwargs: object,
    ) -> "ActionRequest":
        normalized = _json_value(request, "request")
        return cls(
            owner=owner,
            scenario_id=scenario_id,
            action_id=action_id,
            action_type=action_type,
            request_fingerprint=sha256_fingerprint(normalized),
            request_ref=normalized,
            **kwargs,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "owner": self.owner,
            "scenario_id": self.scenario_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "request_ref": _json_value(self.request_ref, "request_ref"),
            "request_fingerprint": self.request_fingerprint,
            "dependency_refs": [_json_value(ref, "dependency_ref") for ref in self.dependency_refs],
            "result_refs": [_json_value(ref, "result_ref") for ref in self.result_refs],
            "supervisor_policy": _json_value(self.supervisor_policy, "supervisor_policy"),
            "budget": _json_value(self.budget, "budget"),
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ActionRequest":
        payload = _mapping(value, "action request")
        return cls(
            schema_version=str(payload.get("schema_version", "gocube-scenario-action-request-v1")),
            owner=str(payload["owner"]),
            scenario_id=str(payload["scenario_id"]),
            action_id=str(payload["action_id"]),
            action_type=str(payload["action_type"]),
            request_ref=payload.get("request_ref"),
            request_fingerprint=str(payload["request_fingerprint"]),
            dependency_refs=tuple(payload.get("dependency_refs", ())),  # type: ignore[arg-type]
            result_refs=tuple(payload.get("result_refs", ())),  # type: ignore[arg-type]
            supervisor_policy=payload.get("supervisor_policy"),  # type: ignore[arg-type]
            budget=payload.get("budget"),  # type: ignore[arg-type]
            correlation_id=None if payload.get("correlation_id") is None else str(payload["correlation_id"]),
        )


@dataclass(frozen=True)
class ActionOutcome:
    """Execution evidence returned to a scenario after one action."""

    action_id: str
    request_fingerprint: str
    execution_status: ExecutionStatus | str
    result_refs: tuple[Mapping[str, object] | str, ...] = ()
    commit_evidence: Mapping[str, object] | None = None
    scientific_validity: ScientificValidity | str = ScientificValidity.UNKNOWN
    execution_code_commit: str | None = None
    error_category: str | None = None
    error_reason: str | None = None
    reused: bool = False
    schema_version: str = "gocube-scenario-action-outcome-v1"

    def __post_init__(self) -> None:
        _safe(self.action_id, "action_id")
        _safe(self.request_fingerprint, "request_fingerprint")
        if self.schema_version != "gocube-scenario-action-outcome-v1":
            raise ValueError("unsupported ActionOutcome schema")
        object.__setattr__(self, "execution_status", ExecutionStatus(self.execution_status))
        object.__setattr__(self, "scientific_validity", ScientificValidity(self.scientific_validity))
        refs = tuple(self.result_refs)
        if any(not isinstance(ref, (str, Mapping)) for ref in refs):
            raise ValueError("result_refs must contain references")
        object.__setattr__(self, "result_refs", refs)
        if self.commit_evidence is not None:
            object.__setattr__(self, "commit_evidence", _mapping(self.commit_evidence, "commit_evidence"))
        if self.execution_code_commit is not None:
            _safe(self.execution_code_commit, "execution_code_commit")
        if self.execution_status is ExecutionStatus.COMPLETED:
            if not refs:
                raise ValueError("completed action requires result_refs")
            if self.scientific_validity is ScientificValidity.UNKNOWN:
                raise ValueError("completed action requires scientific validity")
            if self.commit_evidence is None:
                raise ValueError("completed action requires commit evidence")
        if self.execution_status is ExecutionStatus.FAILED and not (self.error_category or self.error_reason):
            raise ValueError("failed action requires an error category or reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "request_fingerprint": self.request_fingerprint,
            "execution_status": self.execution_status.value,
            "result_refs": [_json_value(ref, "result_ref") for ref in self.result_refs],
            "commit_evidence": _json_value(self.commit_evidence, "commit_evidence"),
            "scientific_validity": self.scientific_validity.value,
            "execution_code_commit": self.execution_code_commit,
            "error_category": self.error_category,
            "error_reason": self.error_reason,
            "reused": self.reused,
        }

    def validate_against(self, request: ActionRequest) -> None:
        """Fail closed when evidence belongs to another durable action."""
        if self.action_id != request.action_id:
            raise ValueError("action outcome id does not match the request")
        if self.request_fingerprint != request.request_fingerprint:
            raise ValueError("action outcome fingerprint does not match the request")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ActionOutcome":
        payload = _mapping(value, "action outcome")
        return cls(
            schema_version=str(payload.get("schema_version", "gocube-scenario-action-outcome-v1")),
            action_id=str(payload["action_id"]),
            request_fingerprint=str(payload["request_fingerprint"]),
            execution_status=str(payload["execution_status"]),
            result_refs=tuple(payload.get("result_refs", ())),  # type: ignore[arg-type]
            commit_evidence=payload.get("commit_evidence"),  # type: ignore[arg-type]
            scientific_validity=str(payload.get("scientific_validity", ScientificValidity.UNKNOWN.value)),
            execution_code_commit=None if payload.get("execution_code_commit") is None else str(payload["execution_code_commit"]),
            error_category=None if payload.get("error_category") is None else str(payload["error_category"]),
            error_reason=None if payload.get("error_reason") is None else str(payload["error_reason"]),
            reused=bool(payload.get("reused", False)),
        )


__all__ = ["ActionOutcome", "ActionRequest", "ExecutionStatus", "ScientificValidity"]
