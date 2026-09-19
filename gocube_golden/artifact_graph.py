"""Low-level immutable checkpoint and artifact graph contracts.

These contracts are shared by production topology paths and orchestration.
They intentionally contain no orchestration policy or replay-selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping

from .provenance import canonical_json, sha256_fingerprint


CHECKPOINT_NODE_SCHEMA = "gocube-checkpoint-node-v2"
EFFECTIVE_CONFIG_SCHEMA = "gocube-effective-config-v2"
CONTRACT_VERSION = 2
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPONENT_RE = re.compile(r"^[^/\\]+$")


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required(value: Mapping[str, object], key: str) -> object:
    if key not in value:
        raise ValueError(f"{key} is required")
    return value[key]


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _component(value: object, label: str) -> str:
    text = _string(value, label).strip()
    if not text or text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        raise ValueError(f"{label} must be one safe path component")
    return text


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if not _SHA_RE.fullmatch(text):
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return text


def _path(value: object, label: str) -> str:
    text = _string(value, label).strip()
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
    return _object(_required(value, key), key)


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
        payload = _object(value, "artifact")
        return cls(_string(_required(payload, "path"), "artifact path"),
                   _string(_required(payload, "sha256"), "artifact sha256"))


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
        generation = _integer(self.generation, "checkpoint generation")
        if generation < 0:
            raise ValueError("checkpoint generation must be a non-negative integer")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "path", _path(self.path, "checkpoint path"))
        object.__setattr__(self, "sha256", _sha(self.sha256, "checkpoint sha256"))

    def to_dict(self) -> dict[str, object]:
        return {"topology": self.topology, "lineage_id": self.lineage_id,
                "checkpoint_id": self.checkpoint_id, "generation": self.generation,
                "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CheckpointRef":
        payload = _object(value, "checkpoint")
        if "sha256" in payload:
            sha = payload["sha256"]
        elif "artifact_sha256" in payload:
            sha = payload["artifact_sha256"]
        else:
            raise ValueError("checkpoint sha256 is required")
        return cls(_string(_required(payload, "topology"), "topology"),
                   _string(_required(payload, "lineage_id"), "lineage_id"),
                   _string(_required(payload, "checkpoint_id"), "checkpoint_id"),
                   _integer(_required(payload, "generation"), "checkpoint generation"),
                   _string(_required(payload, "path"), "checkpoint path"),
                   _string(sha, "checkpoint sha256"))


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
        payload = _object(value, "effective config reference")
        return cls(ArtifactRef.from_dict(_map(payload, "artifact")),
                   _string(_required(payload, "fingerprint"), "effective config fingerprint"))


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
        _boolean(self.genesis, "genesis")
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
        payload = _object(value, "checkpoint node")
        if payload.get("schema") != CHECKPOINT_NODE_SCHEMA or payload.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported CheckpointNode schema")
        if {"ancestors", "replay_references"}.intersection(payload):
            raise ValueError("CheckpointNode forbids persisted ancestry/replay-chain fields")
        parent, replay = payload.get("parent"), payload.get("fresh_replay")
        return cls(CheckpointRef.from_dict(_map(payload, "checkpoint")),
                   _boolean(_required(payload, "genesis"), "genesis"),
                   None if parent is None else CheckpointRef.from_dict(_object(parent, "parent")),
                   None if replay is None else ArtifactRef.from_dict(_object(replay, "fresh_replay")),
                   EffectiveConfigRef.from_dict(_map(payload, "effective_config")),
                   ArtifactRef.from_dict(_map(payload, "provenance")))


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
        payload = _object(value, "effective config")
        if payload.get("schema") != EFFECTIVE_CONFIG_SCHEMA or payload.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported EffectiveConfig schema")
        return cls(_string(_required(payload, "topology"), "topology"), *(_map(payload, name) for name in
                   ("compatibility", "self_play", "training", "replay", "execution", "arena", "supervision", "extensions")))


__all__ = [
    "CHECKPOINT_NODE_SCHEMA",
    "EFFECTIVE_CONFIG_SCHEMA",
    "CONTRACT_VERSION",
    "ArtifactRef",
    "CheckpointRef",
    "EffectiveConfigRef",
    "CheckpointNode",
    "EffectiveConfig",
    "contract_fingerprint",
]
