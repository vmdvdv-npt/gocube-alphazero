"""Durable, sequential DAG composition for Orchestrator V2.

This module owns only workflow order, dependencies, JSON state and references
between action results. Arena, calibration, experiment and training remain
owned by their existing action runners.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import fcntl
import inspect
import json
import math
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
WORKFLOW_STATES = frozenset({"RUNNING", "COMPLETED", "FAILED", "STOPPED"})
_COMPONENT_RE = re.compile(r"^[^/\\]+$")
_REF_RE = re.compile(r"^\$\{([^}]+)\}$")


class WorkflowError(RuntimeError):
    """A workflow cannot safely continue from its durable state."""


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        raise ValueError(f"{label} must be one safe path component")
    return text


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_fixed_search_options(value: Mapping[str, object], label: str) -> None:
    """Reject Arena options which the engine cannot vary."""
    fixed: dict[str, object] = {
        "root_noise": False,
        "temperature": 0.0,
        "fast_search": False,
        "resign": False,
        "deterministic_tie_break": True,
    }
    for key, expected in fixed.items():
        if key not in value:
            continue
        actual = value[key]
        if key == "temperature":
            try:
                valid = math.isfinite(float(actual)) and float(actual) == 0.0
            except (TypeError, ValueError):
                valid = False
        else:
            valid = type(actual) is bool and actual is expected
        if not valid:
            raise ValueError(
                f"{label}.{key} is not supported by the Arena engine; "
                f"the only supported value is {expected!r}"
            )


def _preflight_action_config(action: str, config: Mapping[str, object]) -> None:
    """Validate cheap shared action fields before any action starts."""
    if action not in {"arena", "calibration", "continuous_training", "experiment"}:
        return
    raw_arena = config.get("arena_config", config.get("config"))
    if raw_arena is None:
        return
    # The checked-in examples deliberately use placeholder strings for
    # artifact/config values.  Defer those shapes until reference
    # substitution; the concrete entrypoint will validate them before any
    # expensive action starts.
    if isinstance(raw_arena, str) and raw_arena.startswith("PLACEHOLDER"):
        return
    raw_arena = _mapping(raw_arena, "workflow arena config")
    allowed = {
        "games", "workers", "games_per_worker", "inference_batch_rows",
        "inference_batch_wait_ms", "inference_batch_cap", "inference_wait",
        "inference_wait_ms", "contexts", "device", "strict_production",
        "strict_performance", "monitoring_acceptance", "min_mean_inference_batch_rows",
        "min_effective_cpu_cores", "early_gate_enabled", "early_gate_min_forwards",
        "early_gate_min_wall_sec", "simulations", "mcts_simulations", "cpuct",
        "fpu", "watchdog", "technical_move_limit", "komi", "root_noise",
        "temperature", "fast_search", "resign", "deterministic_tie_break",
        "search", "evaluation",
    }
    unknown = set(raw_arena) - allowed
    if unknown:
        raise ValueError(
            "workflow arena config contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    aliases = {
        "inference_batch_cap": "inference_batch_rows",
        "inference_wait": "inference_batch_wait_ms",
        "inference_wait_ms": "inference_batch_wait_ms",
        "contexts": "games_per_worker",
        "mcts_simulations": "simulations",
        "technical_move_limit": "watchdog",
    }
    for source, target in aliases.items():
        if source in raw_arena and target in raw_arena:
            raise ValueError(f"workflow arena config specifies both {source} and {target}")
    if "search" in raw_arena and "evaluation" in raw_arena:
        raise ValueError("workflow arena config specifies both search and evaluation")
    nested = raw_arena.get("search", raw_arena.get("evaluation"))
    if nested is not None:
        nested_mapping = _mapping(nested, "workflow arena search")
        search_allowed = {
            "simulations", "mcts_simulations", "cpuct", "fpu", "watchdog",
            "technical_move_limit", "komi", "root_noise", "temperature",
            "fast_search", "resign", "deterministic_tie_break",
        }
        unknown = set(nested_mapping) - search_allowed
        if unknown:
            raise ValueError(
                "workflow Arena search contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        for key in nested_mapping:
            if key in raw_arena and key not in {"search", "evaluation"}:
                raise ValueError(
                    f"workflow Arena search field {key!r} conflicts with its top-level value"
                )
        _validate_fixed_search_options(nested_mapping, "workflow Arena search")
    _validate_fixed_search_options(raw_arena, "workflow Arena config")


def _references(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        if set(value) == {"$ref"} and isinstance(value.get("$ref"), str):
            found.append(str(value["$ref"]))
        else:
            for item in value.values():
                found.extend(_references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_references(item))
    elif isinstance(value, str):
        match = _REF_RE.fullmatch(value)
        if match:
            found.append(match.group(1))
    return found


def _ref_step(reference: str) -> str:
    parts = reference.split(".")
    if parts and parts[0] == "steps":
        parts = parts[1:]
    if len(parts) < 2 or parts[1] != "outputs":
        raise ValueError("workflow references must use step_id.outputs[.field]")
    return parts[0]


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
        _preflight_action_config(action, self.config)
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
        allowed = {
            "step_id", "id", "action", "type", "config", "dependencies", "depends_on",
            "failure_policy", "policy",
        }
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
                raise ValueError(
                    f"workflow step {step.step_id} depends on unknown steps: {sorted(missing)}"
                )
        # A reference is also a dependency. Requiring it in the graph makes
        # dynamic values validate before an action starts and prevents a step
        # from accidentally consuming a stale result from an unrelated branch.
        ancestors: dict[str, set[str]] = {step.step_id: set(step.dependencies) for step in self.steps}
        changed = True
        while changed:
            changed = False
            for step_id, deps in ancestors.items():
                expanded = set(deps)
                for dependency in deps:
                    expanded.update(ancestors[dependency])
                if expanded != deps:
                    ancestors[step_id] = expanded
                    changed = True
        for step in self.steps:
            for reference in _references(step.config):
                source = _ref_step(reference)
                if source not in known:
                    raise ValueError(
                        f"workflow step {step.step_id} references unknown step {source!r}"
                    )
                if source == step.step_id or source not in ancestors[step.step_id]:
                    raise ValueError(
                        f"workflow step {step.step_id} references {source!r} without declaring it as a dependency"
                    )
        remaining = {step.step_id: set(step.dependencies) for step in self.steps}
        resolved: set[str] = set()
        while True:
            ready = {
                key for key, deps in remaining.items()
                if key not in resolved and deps <= resolved
            }
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
        if isinstance(value, float) and not math.isfinite(value):
            raise WorkflowError("workflow action output contains a non-finite number")
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
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise WorkflowError(f"workflow output reference index {index} is out of range")
            current = current[index]
        else:
            raise WorkflowError(f"workflow output reference cannot descend through {part!r}")
    return current


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
        self.lock_path = self.root / "workflow.lock"

    @contextmanager
    def _exclusive_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkflowError(
                    f"workflow {self.spec.workflow_id!r} is already managed by another coordinator"
                ) from exc
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

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
            "events": [],
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
        if state.get("workflow_id") != self.spec.workflow_id or state.get("topology") != self.spec.topology:
            raise WorkflowError("workflow state identity does not match the supplied spec")
        if state.get("state") not in WORKFLOW_STATES:
            raise WorkflowError("workflow state has an invalid workflow status")
        steps = state.get("steps")
        if not isinstance(steps, Mapping):
            raise WorkflowError("workflow state steps are malformed")
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
            raise WorkflowError("workflow references must use step_id.outputs[.field]")
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
            candidates = list(source.items())
        elif isinstance(source, (list, tuple)):
            candidates = list(enumerate(source))
        else:
            raise WorkflowError("select source must resolve to an object or list")
        if not candidates:
            raise WorkflowError("select source is empty")
        scored: list[tuple[float, str, object, object]] = []
        for key, item in candidates:
            if isinstance(item, Mapping):
                validity = item.get("validity")
                if validity is not None and str(validity).upper() != "VALID":
                    raise WorkflowError(
                        f"select cannot use candidate {key!r}: result validity is {validity!r}"
                    )
                if item.get("valid") is False:
                    raise WorkflowError(f"select cannot use candidate {key!r}: result is invalid")
            raw_score = item if isinstance(item, (int, float)) else _path_value(item, score_path)
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise WorkflowError(f"select score is not numeric for {key!r}") from exc
            if not math.isfinite(score):
                raise WorkflowError(f"select score is not finite for {key!r}")
            selected_value = key if value_path is None else _path_value(item, str(value_path))
            scored.append((score, str(key), selected_value, item))
        scored.sort(key=lambda row: ((-row[0] if direction == "maximize" else row[0]), row[1]))
        score, key, selected, item = scored[0]
        result: dict[str, object] = {
            "selected": _jsonable(selected),
            "selected_value": _jsonable(selected),
            "selected_key": key,
            "selected_score": score,
            "selected_item": _jsonable(item),
            "rule": f"{direction}:{score_path}",
        }
        if isinstance(item, Mapping) and "effective_config" in item:
            result["selected_effective_config"] = _jsonable(item["effective_config"])
        return result

    def _ready(self, step: WorkflowStep, state: Mapping[str, object]) -> bool:
        steps = state["steps"]
        assert isinstance(steps, Mapping)
        return all(
            str(steps[dependency].get("status")) == "COMPLETED"  # type: ignore[union-attr]
            for dependency in step.dependencies
        )

    def _blocked_dependencies(self, step: WorkflowStep, state: Mapping[str, object]) -> tuple[str, ...]:
        steps = state["steps"]
        assert isinstance(steps, Mapping)
        return tuple(
            dependency
            for dependency in step.dependencies
            if str(steps[dependency].get("status")) in {"FAILED", "SKIPPED"}  # type: ignore[union-attr]
        )

    def _all_outputs(self, state: Mapping[str, object]) -> dict[str, object]:
        steps = state["steps"]
        assert isinstance(steps, Mapping)
        return {
            str(step_id): record["outputs"]
            for step_id, record in steps.items()
            if isinstance(record, Mapping) and "outputs" in record
        }

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, WorkflowError):
            return False
        return not isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError))

    @staticmethod
    def _is_durable_stop(exc: Exception) -> bool:
        text = str(exc).lower()
        return "supervisor-stop.json" in text or "durable supervisor stop" in text

    def run(self) -> Mapping[str, object]:
        with self._exclusive_lock():
            state = self._load_or_create()
            if state.get("state") in {"STOPPED", "COMPLETED"}:
                return state
            if state.get("state") == "FAILED":
                raise WorkflowError(str(state.get("failure", "workflow is durably failed")))
            state["state"] = "RUNNING"

            while True:
                steps_state = state["steps"]
                assert isinstance(steps_state, dict)
                progressed = False
                for step in self.spec.steps:
                    record = steps_state[step.step_id]
                    assert isinstance(record, dict)
                    status = str(record.get("status"))
                    if status in {"COMPLETED", "SKIPPED"}:
                        continue
                    if status == "FAILED":
                        state["state"] = "FAILED"
                        state["failure"] = f"workflow step {step.step_id} is durably failed"
                        self._persist(state)
                        raise WorkflowError(str(state["failure"]))
                    blocked = self._blocked_dependencies(step, state)
                    if blocked:
                        record["status"] = "SKIPPED"
                        record["skip_reason"] = (
                            "dependency did not produce a usable result: " + ", ".join(blocked)
                        )
                        self._persist(state)
                        progressed = True
                        continue
                    if not self._ready(step, state):
                        continue

                    outputs = self._all_outputs(state)
                    resolved_config = self._resolve(step.config, outputs)
                    if not isinstance(resolved_config, Mapping):
                        raise WorkflowError(
                            f"workflow step {step.step_id} config must resolve to an object"
                        )
                    record["status"] = "RUNNING"
                    record["attempts"] = int(record.get("attempts", 0)) + 1
                    record["resolved_config"] = _jsonable(resolved_config)
                    started_at = _now()
                    record["started_at"] = started_at
                    timestamps = record.setdefault("timestamps", {})
                    if isinstance(timestamps, dict):
                        timestamps["started_at"] = started_at
                    events = state.setdefault("events", [])
                    if isinstance(events, list):
                        events.append(
                            {
                                "at": started_at,
                                "step_id": step.step_id,
                                "action": step.action,
                                "attempt": record["attempts"],
                                "status": "RUNNING",
                                "config": _jsonable(resolved_config),
                            }
                        )
                    self._persist(state)
                    try:
                        if step.action == "select":
                            result = self._select(resolved_config, outputs)
                        elif step.action == "stop":
                            result = {
                                "stopped": True,
                                "reason": resolved_config.get("reason", "workflow stop"),
                            }
                        else:
                            handler = self.handlers.get(step.action)
                            if handler is None:
                                raise WorkflowError(
                                    f"no handler registered for workflow action {step.action!r}"
                                )
                            parameters = inspect.signature(handler).parameters
                            accepts_keywords = (
                                "config" in parameters
                                or any(
                                    item.kind is inspect.Parameter.VAR_KEYWORD
                                    for item in parameters.values()
                                )
                            )
                            if accepts_keywords:
                                result = handler(
                                    config=dict(resolved_config),
                                    step=step,
                                    state=state,
                                    resume=record["attempts"] > 1,
                                )
                            else:
                                result = handler(dict(resolved_config))
                        record["outputs"] = _jsonable(result)
                        record["status"] = "COMPLETED"
                        completed_at = _now()
                        record["completed_at"] = completed_at
                        timestamps = record.setdefault("timestamps", {})
                        if isinstance(timestamps, dict):
                            timestamps["completed_at"] = completed_at
                        record.pop("error", None)
                        events = state.setdefault("events", [])
                        if isinstance(events, list):
                            events.append(
                                {
                                    "at": completed_at,
                                    "step_id": step.step_id,
                                    "action": step.action,
                                    "attempt": record["attempts"],
                                    "status": "COMPLETED",
                                    "result": _jsonable(result),
                                }
                            )
                        self._persist(state)
                        progressed = True
                        if step.action == "stop":
                            state["state"] = "STOPPED"
                            state["stop_reason"] = str(result.get("reason", "workflow stop"))  # type: ignore[union-attr]
                            state["stopped_at"] = _now()
                            self._persist(state)
                            return state
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                        error_at = _now()
                        timestamps = record.setdefault("timestamps", {})
                        if isinstance(timestamps, dict):
                            timestamps["error_at"] = error_at
                        events = state.setdefault("events", [])
                        if isinstance(events, list):
                            events.append(
                                {
                                    "at": error_at,
                                    "step_id": step.step_id,
                                    "action": step.action,
                                    "attempt": record["attempts"],
                                    "status": "ERROR",
                                    "reason": str(exc),
                                }
                            )
                        if self._is_durable_stop(exc):
                            record["status"] = "FAILED"
                            record["stop_reason"] = str(exc)
                            state["state"] = "STOPPED"
                            state["stop_reason"] = str(exc)
                            state["stopped_at"] = error_at
                            self._persist(state)
                            return state
                        behavior = str(step.failure_policy.get("on_technical_failure", "stop"))
                        max_retries = int(step.failure_policy.get("max_retries", 0))
                        can_retry = (
                            behavior == "retry"
                            and self._retryable(exc)
                            and not self._is_durable_stop(exc)
                            and int(record["attempts"]) <= max_retries
                        )
                        if can_retry:
                            record["status"] = "PENDING"
                        elif behavior == "continue" and self._retryable(exc):
                            record["status"] = "SKIPPED"
                        else:
                            record["status"] = "FAILED"
                            state["state"] = "FAILED"
                            state["failure"] = f"workflow step {step.step_id} failed: {exc}"
                            self._persist(state)
                            raise WorkflowError(str(state["failure"])) from exc
                        self._persist(state)
                        progressed = True

                steps_state = state["steps"]
                assert isinstance(steps_state, Mapping)
                unresolved = [
                    step.step_id
                    for step in self.spec.steps
                    if str(steps_state[step.step_id].get("status"))
                    not in {"COMPLETED", "SKIPPED"}
                ]
                if not unresolved:
                    if any(
                        str(steps_state[step.step_id].get("status")) == "SKIPPED"
                        and step.action in {"select", "continuous_training", "experiment"}
                        for step in self.spec.steps
                    ):
                        state["state"] = "FAILED"
                        state["failure"] = "workflow skipped a required decision or action"
                        self._persist(state)
                        raise WorkflowError(str(state["failure"]))
                    state["state"] = "COMPLETED"
                    state["completed_at"] = state.get("completed_at", _now())
                    self._persist(state)
                    return state
                if not progressed:
                    raise WorkflowError(
                        "workflow has unresolved steps: " + ", ".join(unresolved)
                    )


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
