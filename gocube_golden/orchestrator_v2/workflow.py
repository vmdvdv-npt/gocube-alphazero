"""Small durable DAG coordinator for Orchestrator V2.

The workflow layer composes existing actions.  It deliberately does not know
how Arena, calibration, experiments, or training work; action handlers own
those domains and retain their existing durable state machines.  This module
only persists step state, resolves simple output references, and implements a
small deterministic selection step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass, asdict
from datetime import datetime, timezone
import json
import inspect
from pathlib import Path
import re

from ..process_supervision import atomic_write_json
from ..provenance import sha256_fingerprint


WORKFLOW_SCHEMA = "gocube-orchestrator-v2-workflow-v1"
WORKFLOW_STATE_SCHEMA = "gocube-orchestrator-v2-workflow-state-v1"
WORKFLOW_ACTIONS = frozenset(
    {"arena", "calibration", "continuous_training", "experiment", "select", "stop"}
)
WORKFLOW_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"})
_COMPONENT_RE = re.compile(r"^[^/\\]+$")
_REF_RE = re.compile(r"^\$\{([^}]+)\}$")


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        raise ValueError(f"{label} must be one safe path component")
    return text


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    action: str
    config: Mapping[str, object] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    failure_policy: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _component(self.step_id, "workflow step_id"))
        action = str(self.action)
        if action not in WORKFLOW_ACTIONS:
            raise ValueError(f"unsupported workflow step action: {action!r}")
        object.__setattr__(self, "action", action)
        if not isinstance(self.config, Mapping):
            raise ValueError("workflow step config must be an object")
        dependencies = tuple(_component(item, "workflow dependency") for item in self.dependencies)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"workflow step {self.step_id} has duplicate dependencies")
        object.__setattr__(self, "dependencies", dependencies)
        if not isinstance(self.failure_policy, Mapping):
            raise ValueError("workflow failure_policy must be an object")
        unknown = set(self.failure_policy) - {"on_technical_failure", "max_retries"}
        if unknown:
            raise ValueError(
                f"workflow step {self.step_id} failure_policy contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        behavior = str(self.failure_policy.get("on_technical_failure", "stop"))
        if behavior not in {"stop", "retry", "continue"}:
            raise ValueError(f"workflow step {self.step_id} has unsupported failure behavior")
        retries = self.failure_policy.get("max_retries", 0)
        if type(retries) is not int or retries < 0:
            raise ValueError(f"workflow step {self.step_id} max_retries must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorkflowStep":
        raw = _mapping(value, "workflow step")
        allowed = {"step_id", "id", "action", "type", "config", "dependencies", "depends_on", "failure_policy", "policy"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "workflow step contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        dependencies = raw.get("dependencies", raw.get("depends_on", ()))
        if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, (list, tuple)):
            raise ValueError("workflow step dependencies must be a list")
        failure_policy = raw.get("failure_policy", raw.get("policy", {}))
        return cls(
            step_id=str(raw.get("step_id", raw.get("id", ""))),
            action=str(raw.get("action", raw.get("type", ""))),
            config=dict(_mapping(raw.get("config", {}), "workflow step config")),
            dependencies=tuple(str(item) for item in dependencies),
            failure_policy=dict(_mapping(failure_policy, "workflow failure_policy")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "config": dict(self.config),
            "dependencies": list(self.dependencies),
            "failure_policy": dict(self.failure_policy),
        }


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    topology: str
    steps: tuple[WorkflowStep, ...]
    schema: str = WORKFLOW_SCHEMA
    supervision: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _component(self.workflow_id, "workflow_id"))
        object.__setattr__(self, "topology", _component(self.topology, "workflow topology"))
        if self.schema != WORKFLOW_SCHEMA:
            raise ValueError("unsupported workflow schema")
        if not isinstance(self.supervision, Mapping):
            raise ValueError("workflow supervision must be an object")
        if not self.steps:
            raise ValueError("workflow requires at least one step")
        ids = [step.step_id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("workflow step_id values must be unique")
        known = set(ids)
        for step in self.steps:
            missing = set(step.dependencies) - known
            if missing:
                raise ValueError(f"workflow step {step.step_id} depends on unknown steps: {sorted(missing)}")
        # Kahn's algorithm gives a compact fail-fast cycle check.
        remaining = {step.step_id: set(step.dependencies) for step in self.steps}
        resolved: set[str] = set()
        while True:
            ready = {key for key, deps in remaining.items() if key not in resolved and deps <= resolved}
            if not ready:
                break
            resolved.update(ready)
        if len(resolved) != len(remaining):
            raise ValueError("workflow dependencies must form an acyclic graph")

    @property
    def fingerprint(self) -> str:
        return sha256_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorkflowSpec":
        raw = _mapping(value, "workflow")
        if isinstance(raw.get("workflow"), Mapping):
            raw = raw["workflow"]  # type: ignore[assignment]
        allowed = {"schema", "workflow_id", "id", "topology", "steps", "supervision"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "workflow contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        steps = raw.get("steps")
        if isinstance(steps, (str, bytes)) or not isinstance(steps, list):
            raise ValueError("workflow.steps must be a list")
        return cls(
            workflow_id=str(raw.get("workflow_id", raw.get("id", ""))),
            topology=str(raw.get("topology", "")),
            steps=tuple(WorkflowStep.from_dict(item) for item in steps),
            supervision=dict(_mapping(raw.get("supervision", {}), "workflow supervision")),
            schema=str(raw.get("schema", WORKFLOW_SCHEMA)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "workflow_id": self.workflow_id,
            "topology": self.topology,
            "supervision": dict(self.supervision),
            "steps": [step.to_dict() for step in self.steps],
        }


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return _jsonable(getattr(value, "to_dict")())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _path_value(value: object, path: str) -> object:
    current = value
    for part in path.split(".") if path else ():
        if isinstance(current, Mapping):
            if part not in current:
                raise WorkflowError(f"workflow output reference is missing field {part!r}")
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            raise WorkflowError(f"workflow output reference cannot descend through {part!r}")
    return current


class WorkflowError(RuntimeError):
    pass


class WorkflowRunner:
    """Execute a persisted workflow using injected existing-action handlers."""

    def __init__(
        self,
        spec: WorkflowSpec,
        *,
        root: str | Path,
        handlers: Mapping[str, Callable[..., object]] | None = None,
    ) -> None:
        self.spec = spec
        self.root = Path(root).resolve()
        self.handlers = dict(handlers or {})
        self.spec_path = self.root / "spec.json"
        self.state_path = self.root / "state.json"

    def _initial_state(self) -> dict[str, object]:
        now = _now()
        return {
            "schema": WORKFLOW_STATE_SCHEMA,
            "workflow_id": self.spec.workflow_id,
            "topology": self.spec.topology,
            "spec_fingerprint": self.spec.fingerprint,
            "state": "RUNNING",
            "created_at": now,
            "updated_at": now,
            "steps": {
                step.step_id: {
                    "step_id": step.step_id,
                    "action": step.action,
                    "status": "PENDING",
                    "attempts": 0,
                    "outputs": {},
                    "timestamps": {},
                }
                for step in self.spec.steps
            },
        }

    def _load_or_create(self) -> dict[str, object]:
        if not self.state_path.is_file():
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.spec_path, self.spec.to_dict())
            state = self._initial_state()
            self._persist(state)
            return state
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"cannot read workflow state: {self.state_path}") from exc
        if not isinstance(state, dict) or state.get("schema") != WORKFLOW_STATE_SCHEMA:
            raise WorkflowError("workflow state schema mismatch")
        if state.get("spec_fingerprint") != self.spec.fingerprint:
            raise WorkflowError("workflow spec changed after durable execution started")
        steps = state.get("steps")
        if not isinstance(steps, Mapping):
            raise WorkflowError("workflow state steps are malformed")
        # A process can die between RUNNING publication and the action's own
        # commit.  The existing action handler receives resume=True and is
        # responsible for reattaching to its SupervisorV2 identity.
        for step in self.spec.steps:
            record = steps.get(step.step_id)
            if not isinstance(record, Mapping):
                raise WorkflowError(f"workflow state is missing step {step.step_id}")
            if str(record.get("status")) not in WORKFLOW_STATUSES:
                raise WorkflowError(f"workflow step {step.step_id} has an invalid status")
        return dict(state)

    def _persist(self, state: dict[str, object]) -> None:
        state["updated_at"] = _now()
        atomic_write_json(self.state_path, state)

    @staticmethod
    def _resolve(value: object, outputs: Mapping[str, object]) -> object:
        if isinstance(value, Mapping):
            if set(value) == {"$ref"}:
                reference = value["$ref"]
                if not isinstance(reference, str):
                    raise WorkflowError("workflow $ref must be a string")
                return WorkflowRunner._resolve_reference(reference, outputs)
            return {str(key): WorkflowRunner._resolve(item, outputs) for key, item in value.items()}
        if isinstance(value, list):
            return [WorkflowRunner._resolve(item, outputs) for item in value]
        if isinstance(value, str):
            match = _REF_RE.fullmatch(value)
            if match:
                return WorkflowRunner._resolve_reference(match.group(1), outputs)
        return value

    @staticmethod
    def _resolve_reference(reference: str, outputs: Mapping[str, object]) -> object:
        parts = reference.split(".")
        if parts and parts[0] == "steps":
            parts = parts[1:]
        if len(parts) < 2 or parts[1] != "outputs":
            raise WorkflowError(
                "workflow references must use step_id.outputs[.field]"
            )
        step_id = parts[0]
        if step_id not in outputs:
            raise WorkflowError(f"workflow output reference names incomplete step {step_id!r}")
        return _path_value(outputs[step_id], ".".join(parts[2:]))

    @staticmethod
    def _select(config: Mapping[str, object], outputs: Mapping[str, object]) -> dict[str, object]:
        source_raw = config.get("source", config.get("candidates"))
        if source_raw is None:
            raise WorkflowError("select step requires source or candidates")
        source = WorkflowRunner._resolve(source_raw, outputs)
        score_path = str(config.get("score", config.get("metric", "score")))
        value_path = config.get("value", config.get("value_field"))
        direction = str(config.get("direction", "minimize"))
        if direction not in {"minimize", "maximize"}:
            raise WorkflowError("select direction must be minimize or maximize")
        if isinstance(source, Mapping):
            items = list(source.items())
            candidates = [(key, item) for key, item in items]
        elif isinstance(source, list):
            candidates = list(enumerate(source))
        else:
            raise WorkflowError("select source must resolve to an object or list")
        if not candidates:
            raise WorkflowError("select source is empty")
        scored: list[tuple[float, str, object, object]] = []
        for key, item in candidates:
            raw_score = item if isinstance(item, (int, float)) else _path_value(item, score_path)
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise WorkflowError(f"select score is not numeric for {key!r}") from exc
            selected_value = key if value_path is None else _path_value(item, str(value_path))
            scored.append((score, str(key), selected_value, item))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=direction == "maximize")
        score, key, selected, item = scored[0]
        return {
            "selected": _jsonable(selected),
            "selected_key": key,
            "selected_score": score,
            "selected_item": _jsonable(item),
            "rule": f"{direction}:{score_path}",
        }

    def _ready(self, step: WorkflowStep, state: Mapping[str, object]) -> bool:
        steps = state["steps"]
        assert isinstance(steps, Mapping)
        return all(str(steps[dependency].get("status")) in {"COMPLETED", "SKIPPED"} for dependency in step.dependencies)  # type: ignore[union-attr]

    def _all_outputs(self, state: Mapping[str, object]) -> dict[str, object]:
        steps = state["steps"]
        assert isinstance(steps, Mapping)
        return {
            str(step_id): dict(record.get("outputs", {}))
            for step_id, record in steps.items()
            if isinstance(record, Mapping) and isinstance(record.get("outputs", {}), Mapping)
        }

    def run(self) -> Mapping[str, object]:
        state = self._load_or_create()
        steps_state = state["steps"]
        assert isinstance(steps_state, dict)
        for step in self.spec.steps:
            record = steps_state[step.step_id]
            assert isinstance(record, dict)
            if record.get("status") == "COMPLETED" or record.get("status") == "SKIPPED":
                continue
            if record.get("status") == "FAILED":
                raise WorkflowError(f"workflow step {step.step_id} is durably failed")
            if not self._ready(step, state):
                continue
            outputs = self._all_outputs(state)
            resolved_config = self._resolve(step.config, outputs)
            if not isinstance(resolved_config, Mapping):
                raise WorkflowError(f"workflow step {step.step_id} config must resolve to an object")
            record["status"] = "RUNNING"
            record["attempts"] = int(record.get("attempts", 0)) + 1
            started_at = _now()
            record["started_at"] = started_at
            timestamps = record.setdefault("timestamps", {})
            if isinstance(timestamps, dict):
                timestamps["started_at"] = started_at
            self._persist(state)
            try:
                if step.action == "select":
                    result = self._select(resolved_config, outputs)
                elif step.action == "stop":
                    result = {"stopped": True, "reason": resolved_config.get("reason", "workflow stop")}
                else:
                    handler = self.handlers.get(step.action)
                    if handler is None:
                        raise WorkflowError(f"no handler registered for workflow action {step.action!r}")
                    parameters = inspect.signature(handler).parameters
                    accepts_keywords = (
                        "config" in parameters
                        or any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
                    )
                    if accepts_keywords:
                        result = handler(
                            config=dict(resolved_config),
                            step=step,
                            state=state,
                            resume=record["attempts"] > 1,
                        )
                    else:
                        # Small injected handlers in tests and integrations may
                        # use the simpler ``handler(config)`` form.
                        result = handler(dict(resolved_config))
                record["outputs"] = _jsonable(result)
                record["status"] = "COMPLETED"
                completed_at = _now()
                record["completed_at"] = completed_at
                timestamps = record.setdefault("timestamps", {})
                if isinstance(timestamps, dict):
                    timestamps["completed_at"] = completed_at
                record.pop("error", None)
                self._persist(state)
            except BaseException as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                error_at = _now()
                timestamps = record.setdefault("timestamps", {})
                if isinstance(timestamps, dict):
                    timestamps["error_at"] = error_at
                behavior = str(step.failure_policy.get("on_technical_failure", "stop"))
                max_retries = int(step.failure_policy.get("max_retries", 0))
                if behavior == "retry" and int(record["attempts"]) <= max_retries:
                    record["status"] = "PENDING"
                elif behavior == "continue":
                    record["status"] = "SKIPPED"
                else:
                    record["status"] = "FAILED"
                    state["state"] = "FAILED"
                    self._persist(state)
                    raise WorkflowError(f"workflow step {step.step_id} failed: {exc}") from exc
                self._persist(state)
                if record["status"] == "PENDING":
                    return self.run()
        incomplete = [
            step.step_id
            for step in self.spec.steps
            if str(steps_state[step.step_id].get("status")) not in {"COMPLETED", "SKIPPED"}
        ]
        if incomplete:
            raise WorkflowError(f"workflow has unresolved steps: {', '.join(incomplete)}")
        state["state"] = "COMPLETED"
        state["completed_at"] = state.get("completed_at", _now())
        self._persist(state)
        return state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "WORKFLOW_SCHEMA",
    "WORKFLOW_STATE_SCHEMA",
    "WorkflowError",
    "WorkflowRunner",
    "WorkflowSpec",
    "WorkflowStep",
]
