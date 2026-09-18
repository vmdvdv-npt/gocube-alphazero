"""Immutable persistence contracts for Orchestrator V2; no orchestration logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping

from ..provenance import canonical_json, sha256_fingerprint

CHECKPOINT_NODE_SCHEMA = "gocube-checkpoint-node-v2"
EFFECTIVE_CONFIG_SCHEMA = "gocube-effective-config-v2"
RUN_STATE_SCHEMA = "gocube-orchestrator-run-state-v2"
RUNTIME_AMENDMENT_SCHEMA = "gocube-runtime-amendment-v2"
EVALUATION_IDENTITY_SCHEMA = "gocube-arena-evaluation-identity-v2"
CONTRACT_VERSION = 2
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPONENT_RE = re.compile(r"^[^/\\]+$")


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        raise ValueError(f"{label} must be one safe path component")
    return text


def _sha(value: object, label: str) -> str:
    text = str(value)
    if not _SHA_RE.fullmatch(text):
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return text


def _path(value: object, label: str) -> str:
    text = str(value).strip()
    parsed = PurePosixPath(text)
    if not text or parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{label} must be a safe relative path")
    return parsed.as_posix()


def _freeze(value: Any, label: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{label} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} object keys must be strings")
            out[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{label}[]") for item in value)
    raise ValueError(f"{label} must contain only JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _map(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{key} must be an object")
    return item


def contract_fingerprint(payload: Mapping[str, object]) -> str:
    return sha256_fingerprint(dict(payload))


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "artifact path"))
        object.__setattr__(self, "sha256", _sha(self.sha256, "artifact sha256"))

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ArtifactRef":
        return cls(str(value["path"]), str(value["sha256"]))


@dataclass(frozen=True)
class CheckpointRef:
    topology: str
    lineage_id: str
    checkpoint_id: str
    generation: int
    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology", _component(self.topology, "topology"))
        object.__setattr__(self, "lineage_id", _component(self.lineage_id, "lineage_id"))
        object.__setattr__(self, "checkpoint_id", _component(self.checkpoint_id, "checkpoint_id"))
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("checkpoint generation must be a non-negative integer")
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "path", _path(self.path, "checkpoint path"))
        object.__setattr__(self, "sha256", _sha(self.sha256, "checkpoint sha256"))

    def to_dict(self) -> dict[str, object]:
        return {"topology": self.topology, "lineage_id": self.lineage_id,
                "checkpoint_id": self.checkpoint_id, "generation": self.generation,
                "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CheckpointRef":
        sha = value.get("sha256") or value.get("artifact_sha256")
        return cls(str(value["topology"]), str(value["lineage_id"]),
                   str(value["checkpoint_id"]), int(value["generation"]),
                   str(value["path"]), str(sha or ""))


@dataclass(frozen=True)
class EffectiveConfigRef:
    artifact: ArtifactRef
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", _sha(self.fingerprint, "effective config fingerprint"))

    def to_dict(self) -> dict[str, object]:
        return {"artifact": self.artifact.to_dict(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EffectiveConfigRef":
        return cls(ArtifactRef.from_dict(_map(value, "artifact")), str(value["fingerprint"]))


@dataclass(frozen=True)
class CheckpointNode:
    checkpoint: CheckpointRef
    genesis: bool
    parent: CheckpointRef | None
    fresh_replay: ArtifactRef | None
    effective_config: EffectiveConfigRef
    provenance: ArtifactRef
    schema: str = CHECKPOINT_NODE_SCHEMA
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema != CHECKPOINT_NODE_SCHEMA or self.version != CONTRACT_VERSION:
            raise ValueError("unsupported CheckpointNode schema")
        if self.genesis and self.parent is not None:
            raise ValueError("genesis checkpoint must not declare a parent")
        if not self.genesis:
            if self.parent is None:
                raise ValueError("non-genesis checkpoint requires exactly one immediate parent")
            if self.parent.topology != self.checkpoint.topology:
                raise ValueError("checkpoint parent must have compatible topology")
            if self.parent == self.checkpoint:
                raise ValueError("checkpoint cannot be its own parent")
            if self.fresh_replay is None:
                raise ValueError("non-genesis checkpoint requires its generation fresh replay artifact")

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "version": self.version,
                "checkpoint": self.checkpoint.to_dict(), "genesis": self.genesis,
                "parent": None if self.parent is None else self.parent.to_dict(),
                "fresh_replay": None if self.fresh_replay is None else self.fresh_replay.to_dict(),
                "effective_config": self.effective_config.to_dict(),
                "provenance": self.provenance.to_dict()}

    @property
    def fingerprint(self) -> str:
        return contract_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CheckpointNode":
        if value.get("schema") != CHECKPOINT_NODE_SCHEMA or value.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported CheckpointNode schema")
        if {"ancestors", "replay_references"}.intersection(value):
            raise ValueError("CheckpointNode forbids persisted ancestry/replay-chain fields")
        parent, replay = value.get("parent"), value.get("fresh_replay")
        return cls(CheckpointRef.from_dict(_map(value, "checkpoint")), bool(value["genesis"]),
                   CheckpointRef.from_dict(parent) if isinstance(parent, Mapping) else None,
                   ArtifactRef.from_dict(replay) if isinstance(replay, Mapping) else None,
                   EffectiveConfigRef.from_dict(_map(value, "effective_config")),
                   ArtifactRef.from_dict(_map(value, "provenance")))


@dataclass(frozen=True)
class EffectiveConfig:
    topology: str
    compatibility: Mapping[str, object]
    self_play: Mapping[str, object] = field(default_factory=dict)
    training: Mapping[str, object] = field(default_factory=dict)
    replay: Mapping[str, object] = field(default_factory=dict)
    execution: Mapping[str, object] = field(default_factory=dict)
    arena: Mapping[str, object] = field(default_factory=dict)
    supervision: Mapping[str, object] = field(default_factory=dict)
    extensions: Mapping[str, object] = field(default_factory=dict)
    schema: str = EFFECTIVE_CONFIG_SCHEMA
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema != EFFECTIVE_CONFIG_SCHEMA or self.version != CONTRACT_VERSION:
            raise ValueError("unsupported EffectiveConfig schema")
        object.__setattr__(self, "topology", _component(self.topology, "topology"))
        for name in ("compatibility", "self_play", "training", "replay", "execution", "arena", "supervision", "extensions"):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"effective config {name} must be an object")
            object.__setattr__(self, name, _freeze(raw, name))
        if not self.compatibility:
            raise ValueError("effective config requires compatibility identity")

    def to_dict(self) -> dict[str, object]:
        out = {"schema": self.schema, "version": self.version, "topology": self.topology}
        for name in ("compatibility", "self_play", "training", "replay", "execution", "arena", "supervision", "extensions"):
            out[name] = _thaw(getattr(self, name))
        return out

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return contract_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EffectiveConfig":
        if value.get("schema") != EFFECTIVE_CONFIG_SCHEMA or value.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported EffectiveConfig schema")
        return cls(str(value["topology"]), *(_map(value, name) for name in
                   ("compatibility", "self_play", "training", "replay", "execution", "arena", "supervision", "extensions")))


class ChangeClass(str, Enum):
    NEXT_GENERATION = "next_generation"
    NEXT_EXECUTION_UNIT = "next_execution_unit"
    INCOMPATIBLE_NEW_RUN = "incompatible_new_run"


@dataclass(frozen=True)
class ChangeabilityRule:
    path: str
    change_class: ChangeClass
    boundary_semantics: str


PARAMETER_CHANGEABILITY_V2: tuple[ChangeabilityRule, ...] = tuple(
    ChangeabilityRule(path, ChangeClass.NEXT_GENERATION, "first_not_started_generation")
    for path in (
        "self_play.mcts_simulations", "self_play.games_per_iteration", "self_play.cpuct", "self_play.fpu",
        "self_play.root_noise", "self_play.dirichlet_epsilon", "self_play.dirichlet_alpha",
        "self_play.temperature_schedule", "self_play.fast_search", "self_play.resign",
        "training.learning_rate", "training.optimizer_steps", "training.batch_size", "training.weight_decay",
        "training.warmup", "training.scheduler", "replay.window", "replay.cap",
        "arena.cadence", "arena.games", "arena.reference_gap",
    )
) + tuple(
    ChangeabilityRule(path, ChangeClass.NEXT_EXECUTION_UNIT, "next_child_start")
    for path in ("execution.workers", "execution.active_contexts", "execution.inference_batch_cap",
                 "execution.inference_wait_ms", "self_play.watchdog", "supervision.*")
) + (
    ChangeabilityRule("topology", ChangeClass.INCOMPATIBLE_NEW_RUN, "new_run_required"),
    ChangeabilityRule("compatibility.*", ChangeClass.INCOMPATIBLE_NEW_RUN, "new_run_required"),
)


@dataclass(frozen=True)
class ResolvedBoundary:
    kind: str
    generation: int | None = None
    execution_unit_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "generation":
            if self.generation is None or int(self.generation) < 0 or self.execution_unit_id is not None:
                raise ValueError("generation boundary requires only a non-negative generation")
            object.__setattr__(self, "generation", int(self.generation))
        elif self.kind == "execution_unit":
            if not self.execution_unit_id or self.generation is not None:
                raise ValueError("execution_unit boundary requires only execution_unit_id")
            object.__setattr__(self, "execution_unit_id", _component(self.execution_unit_id, "execution_unit_id"))
        else:
            raise ValueError("boundary kind must be generation or execution_unit")

    def to_dict(self) -> dict[str, object]:
        return ({"kind": self.kind, "generation": self.generation} if self.kind == "generation"
                else {"kind": self.kind, "execution_unit_id": self.execution_unit_id})

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ResolvedBoundary":
        return cls(str(value["kind"]), int(value["generation"]) if value.get("generation") is not None else None,
                   str(value["execution_unit_id"]) if value.get("execution_unit_id") is not None else None)


@dataclass(frozen=True)
class ParameterChange:
    path: str
    old_value: Any
    new_value: Any
    change_class: ChangeClass
    resolved_boundary: ResolvedBoundary

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith(".") or self.path.endswith("."):
            raise ValueError("parameter path must be a non-empty dotted path")
        if self.change_class is ChangeClass.INCOMPATIBLE_NEW_RUN:
            raise ValueError("incompatible/new-run parameters cannot be accepted as runtime amendments")
        object.__setattr__(self, "old_value", _freeze(self.old_value, f"{self.path}.old_value"))
        object.__setattr__(self, "new_value", _freeze(self.new_value, f"{self.path}.new_value"))
        expected = "generation" if self.change_class is ChangeClass.NEXT_GENERATION else "execution_unit"
        if self.resolved_boundary.kind != expected:
            raise ValueError(f"{self.change_class.value} change requires {expected} boundary")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "old_value": _thaw(self.old_value), "new_value": _thaw(self.new_value),
                "change_class": self.change_class.value, "resolved_boundary": self.resolved_boundary.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ParameterChange":
        return cls(str(value["path"]), value.get("old_value"), value.get("new_value"),
                   ChangeClass(str(value["change_class"])), ResolvedBoundary.from_dict(_map(value, "resolved_boundary")))


@dataclass(frozen=True)
class RuntimeAmendment:
    amendment_id: str
    run_id: str
    accepted_at: str
    requested_changes: tuple[ParameterChange, ...]
    base_effective_config: EffectiveConfigRef
    resulting_effective_config: EffectiveConfigRef
    schema: str = RUNTIME_AMENDMENT_SCHEMA
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_AMENDMENT_SCHEMA or self.version != CONTRACT_VERSION:
            raise ValueError("unsupported RuntimeAmendment schema")
        object.__setattr__(self, "amendment_id", _component(self.amendment_id, "amendment_id"))
        object.__setattr__(self, "run_id", _component(self.run_id, "run_id"))
        object.__setattr__(self, "requested_changes", tuple(self.requested_changes))
        if not self.accepted_at or not self.requested_changes:
            raise ValueError("accepted_at and at least one requested change are required")
        if self.base_effective_config.fingerprint == self.resulting_effective_config.fingerprint:
            raise ValueError("runtime amendment must produce a different effective config")

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "version": self.version, "amendment_id": self.amendment_id,
                "run_id": self.run_id, "accepted_at": self.accepted_at,
                "requested_changes": [c.to_dict() for c in self.requested_changes],
                "base_effective_config": self.base_effective_config.to_dict(),
                "resulting_effective_config": self.resulting_effective_config.to_dict()}

    @property
    def fingerprint(self) -> str:
        return contract_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuntimeAmendment":
        if value.get("schema") != RUNTIME_AMENDMENT_SCHEMA or value.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported RuntimeAmendment schema")
        changes = value.get("requested_changes")
        if not isinstance(changes, list):
            raise ValueError("requested_changes must be a list")
        return cls(str(value["amendment_id"]), str(value["run_id"]), str(value["accepted_at"]),
                   tuple(ParameterChange.from_dict(c) for c in changes if isinstance(c, Mapping)),
                   EffectiveConfigRef.from_dict(_map(value, "base_effective_config")),
                   EffectiveConfigRef.from_dict(_map(value, "resulting_effective_config")))


class RunMode(str, Enum):
    CONTINUOUS = "continuous"
    EXPERIMENT = "experiment"


class BusinessState(str, Enum):
    READY = "READY"
    RUNNING_GENERATION = "RUNNING_GENERATION"
    GENERATION_COMMITTED = "GENERATION_COMMITTED"
    RUNNING_ARENA = "RUNNING_ARENA"
    STOPPED = "STOPPED"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


class ExecutionKind(str, Enum):
    GENERATION = "generation"
    ARENA = "arena"


@dataclass(frozen=True)
class ActiveExecution:
    kind: ExecutionKind
    unit_id: str
    attempt: int
    generation: int | None = None
    evaluation_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _component(self.unit_id, "execution unit id"))
        if isinstance(self.attempt, bool) or int(self.attempt) < 1:
            raise ValueError("active execution attempt must be >= 1")
        object.__setattr__(self, "attempt", int(self.attempt))
        if self.kind is ExecutionKind.GENERATION:
            if self.generation is None or int(self.generation) < 0 or self.evaluation_fingerprint is not None:
                raise ValueError("generation execution requires generation only")
            object.__setattr__(self, "generation", int(self.generation))
        elif self.generation is not None or self.evaluation_fingerprint is None:
            raise ValueError("Arena execution requires evaluation_fingerprint only")
        else:
            object.__setattr__(self, "evaluation_fingerprint", _sha(self.evaluation_fingerprint, "evaluation fingerprint"))

    def to_dict(self) -> dict[str, object]:
        base = {"kind": self.kind.value, "unit_id": self.unit_id, "attempt": self.attempt}
        base["generation" if self.kind is ExecutionKind.GENERATION else "evaluation_fingerprint"] = (
            self.generation if self.kind is ExecutionKind.GENERATION else self.evaluation_fingerprint)
        return base

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ActiveExecution":
        return cls(ExecutionKind(str(value["kind"])), str(value["unit_id"]), int(value["attempt"]),
                   int(value["generation"]) if value.get("generation") is not None else None,
                   str(value["evaluation_fingerprint"]) if value.get("evaluation_fingerprint") is not None else None)


@dataclass(frozen=True)
class RunState:
    run_id: str
    mode: RunMode
    state: BusinessState
    lineage_id: str | None
    base_config: EffectiveConfigRef
    created_at: str
    updated_at: str
    last_committed_checkpoint: CheckpointRef | None = None
    generation_commit: ArtifactRef | None = None
    active_execution: ActiveExecution | None = None
    retry_attempt: int = 0
    soft_stop_requested: bool = False
    pending_required_step: str | None = None
    queued_transition: Mapping[str, object] | None = None
    applied_amendments: tuple[ArtifactRef, ...] = ()
    schema: str = RUN_STATE_SCHEMA
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema != RUN_STATE_SCHEMA or self.version != CONTRACT_VERSION:
            raise ValueError("unsupported RunState schema")
        object.__setattr__(self, "run_id", _component(self.run_id, "run_id"))
        if self.lineage_id is not None:
            object.__setattr__(self, "lineage_id", _component(self.lineage_id, "lineage_id"))
        if not self.created_at or not self.updated_at:
            raise ValueError("run state timestamps are required")
        if isinstance(self.retry_attempt, bool) or int(self.retry_attempt) < 0:
            raise ValueError("retry_attempt must be non-negative")
        object.__setattr__(self, "retry_attempt", int(self.retry_attempt))
        object.__setattr__(self, "applied_amendments", tuple(self.applied_amendments))
        if self.queued_transition is not None:
            object.__setattr__(self, "queued_transition", _freeze(self.queued_transition, "queued_transition"))
        if (self.last_committed_checkpoint is None) != (self.generation_commit is None):
            raise ValueError("last committed checkpoint and authoritative commit reference must be recorded together")
        if self.state is BusinessState.GENERATION_COMMITTED and self.generation_commit is None:
            raise ValueError("GENERATION_COMMITTED state requires an external generation commit reference")
        if self.state is BusinessState.RUNNING_GENERATION:
            if self.active_execution is None or self.active_execution.kind is not ExecutionKind.GENERATION:
                raise ValueError("RUNNING_GENERATION requires active generation execution")
        elif self.state is BusinessState.RUNNING_ARENA:
            if self.active_execution is None or self.active_execution.kind is not ExecutionKind.ARENA:
                raise ValueError("RUNNING_ARENA requires active Arena execution")
        elif self.active_execution is not None:
            raise ValueError("non-running business state must not carry active execution")

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "version": self.version, "run_id": self.run_id, "mode": self.mode.value,
                "lineage_id": self.lineage_id, "state": self.state.value,
                "last_committed_checkpoint": None if self.last_committed_checkpoint is None else self.last_committed_checkpoint.to_dict(),
                "generation_commit": None if self.generation_commit is None else self.generation_commit.to_dict(),
                "active_execution": None if self.active_execution is None else self.active_execution.to_dict(),
                "retry_attempt": self.retry_attempt, "soft_stop_requested": self.soft_stop_requested,
                "pending_required_step": self.pending_required_step,
                "queued_transition": None if self.queued_transition is None else _thaw(self.queued_transition),
                "base_config": self.base_config.to_dict(),
                "applied_amendments": [a.to_dict() for a in self.applied_amendments],
                "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RunState":
        if value.get("schema") != RUN_STATE_SCHEMA or value.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported RunState schema")
        cp, commit, active = value.get("last_committed_checkpoint"), value.get("generation_commit"), value.get("active_execution")
        amendments = value.get("applied_amendments", [])
        if not isinstance(amendments, list):
            raise ValueError("applied_amendments must be a list")
        return cls(str(value["run_id"]), RunMode(str(value["mode"])), BusinessState(str(value["state"])),
                   str(value["lineage_id"]) if value.get("lineage_id") is not None else None,
                   EffectiveConfigRef.from_dict(_map(value, "base_config")), str(value["created_at"]), str(value["updated_at"]),
                   CheckpointRef.from_dict(cp) if isinstance(cp, Mapping) else None,
                   ArtifactRef.from_dict(commit) if isinstance(commit, Mapping) else None,
                   ActiveExecution.from_dict(active) if isinstance(active, Mapping) else None,
                   int(value.get("retry_attempt", 0)), bool(value.get("soft_stop_requested", False)),
                   str(value["pending_required_step"]) if value.get("pending_required_step") is not None else None,
                   value.get("queued_transition") if isinstance(value.get("queued_transition"), Mapping) else None,
                   tuple(ArtifactRef.from_dict(a) for a in amendments if isinstance(a, Mapping)))


@dataclass(frozen=True)
class StartsetRef:
    id: str
    artifact: ArtifactRef
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _component(self.id, "startset id"))
        object.__setattr__(self, "fingerprint", _sha(self.fingerprint, "startset fingerprint"))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "artifact": self.artifact.to_dict(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StartsetRef":
        return cls(str(value["id"]), ArtifactRef.from_dict(_map(value, "artifact")), str(value["fingerprint"]))


@dataclass(frozen=True)
class EvaluationIdentity:
    candidate: CheckpointRef
    reference: CheckpointRef
    games: int
    master_seed: int
    startset: StartsetRef
    scientific_contract: Mapping[str, object]
    execution_contract: Mapping[str, object]
    workload: Mapping[str, object] = field(default_factory=dict)
    schema: str = EVALUATION_IDENTITY_SCHEMA
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema != EVALUATION_IDENTITY_SCHEMA or self.version != CONTRACT_VERSION:
            raise ValueError("unsupported EvaluationIdentity schema")
        if self.candidate.topology != self.reference.topology:
            raise ValueError("Arena checkpoints must share topology")
        if isinstance(self.games, bool) or int(self.games) <= 0:
            raise ValueError("Arena games must be positive")
        object.__setattr__(self, "games", int(self.games))
        if isinstance(self.master_seed, bool):
            raise ValueError("master_seed must be an integer")
        object.__setattr__(self, "master_seed", int(self.master_seed))
        for name in ("scientific_contract", "execution_contract", "workload"):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be an object")
            object.__setattr__(self, name, _freeze(raw, name))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "version": self.version, "candidate": self.candidate.to_dict(),
                "reference": self.reference.to_dict(), "games": self.games, "master_seed": self.master_seed,
                "startset": self.startset.to_dict(), "scientific_contract": _thaw(self.scientific_contract),
                "execution_contract": _thaw(self.execution_contract), "workload": _thaw(self.workload)}

    @property
    def fingerprint(self) -> str:
        return contract_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EvaluationIdentity":
        if value.get("schema") != EVALUATION_IDENTITY_SCHEMA or value.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported EvaluationIdentity schema")
        return cls(CheckpointRef.from_dict(_map(value, "candidate")), CheckpointRef.from_dict(_map(value, "reference")),
                   int(value["games"]), int(value["master_seed"]), StartsetRef.from_dict(_map(value, "startset")),
                   _map(value, "scientific_contract"), _map(value, "execution_contract"), _map(value, "workload"))


class EvaluationValidity(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    TECHNICAL = "TECHNICAL"
    CRITICAL = "CRITICAL"


def reusable_scientific_result(validity: EvaluationValidity | str) -> bool:
    return EvaluationValidity(validity) is EvaluationValidity.VALID


__all__ = [
    "CHECKPOINT_NODE_SCHEMA", "EFFECTIVE_CONFIG_SCHEMA", "RUN_STATE_SCHEMA", "RUNTIME_AMENDMENT_SCHEMA",
    "EVALUATION_IDENTITY_SCHEMA", "CONTRACT_VERSION", "ArtifactRef", "CheckpointRef", "EffectiveConfigRef",
    "CheckpointNode", "EffectiveConfig", "ChangeClass", "ChangeabilityRule", "PARAMETER_CHANGEABILITY_V2",
    "ResolvedBoundary", "ParameterChange", "RuntimeAmendment", "RunMode", "BusinessState", "ExecutionKind",
    "ActiveExecution", "RunState", "StartsetRef", "EvaluationIdentity", "EvaluationValidity",
    "contract_fingerprint", "reusable_scientific_result",
]
