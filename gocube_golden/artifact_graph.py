"""Low-level immutable checkpoint and artifact graph contracts.

These contracts are shared by production topology paths and orchestration.
They intentionally contain no orchestration policy or replay-selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping

from .artifact_catalog import ArtifactCatalog, sha256_file
from .process_supervision import atomic_write_text
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


def _graph_node_path(root: Path, checkpoint: CheckpointRef) -> Path:
    return root / "metadata" / "checkpoints" / f"{checkpoint.checkpoint_id}.json"


def _owned_path(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes lineage root: {relative}") from exc
    return path


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    content = canonical_json(dict(payload)) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite immutable graph artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content)


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return payload


def _artifact_identity(path: Path, root: Path, *, sha256: str | None = None) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256 or sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def publish_checkpoint_graph(
    *,
    root: str | Path,
    parent: CheckpointRef,
    checkpoint: CheckpointRef,
    fresh_replay: ArtifactRef,
    effective_config: EffectiveConfigRef,
    generation_commit: ArtifactRef,
    artifact_identities: Mapping[str, Mapping[str, object]] | None = None,
    checkpoint_reload_verified: bool = True,
) -> CheckpointNode:
    """Publish graph evidence before the generation completion fence.

    All inputs are already immutable identities.  The completion marker is
    deliberately referenced but not required to exist yet; its atomic rename
    is performed by the caller only after this function returns successfully.
    """
    lineage_root = Path(root).resolve()
    if checkpoint.generation != parent.generation + 1:
        raise ValueError("checkpoint is not an immediate child of parent")
    if checkpoint.lineage_id != str(_read_object(lineage_root / "manifest.json", "lineage manifest").get("lineage_id")):
        raise ValueError("checkpoint lineage does not match manifest")
    for ref, label in (
        (checkpoint, "checkpoint"),
        (fresh_replay, "fresh replay"),
        (effective_config.artifact, "effective config"),
    ):
        path = _owned_path(lineage_root, ref.path, label)
        if not path.is_file() or sha256_file(path) != ref.sha256:
            raise ValueError(f"{label} identity is not ready for commit: {path}")

    provenance_path = lineage_root / "metadata" / "provenance-v2" / f"M{checkpoint.generation}.json"
    node_path = _graph_node_path(lineage_root, checkpoint)
    artifact_payload = {
        name: dict(identity)
        for name, identity in (artifact_identities or {}).items()
    }
    provenance_payload: dict[str, object] = {
        "schema": "gocube-orchestrator-v2-production-provenance-v2",
        "version": 2,
        "checkpoint": checkpoint.to_dict(),
        "immediate_parent": parent.to_dict(),
        "fresh_replay": fresh_replay.to_dict(),
        "generation_commit": generation_commit.to_dict(),
        "effective_config": effective_config.to_dict(),
        "checkpoint_reload_verified": bool(checkpoint_reload_verified),
        "artifact_identities": artifact_payload,
    }
    _write_immutable_json(provenance_path, provenance_payload)
    provenance = ArtifactRef(
        provenance_path.relative_to(lineage_root).as_posix(),
        sha256_file(provenance_path),
    )
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=False,
        parent=parent,
        fresh_replay=fresh_replay,
        effective_config=effective_config,
        provenance=provenance,
    )
    _write_immutable_json(node_path, node.to_dict())

    manifest_path = lineage_root / "manifest.json"
    manifest = _read_object(manifest_path, "lineage manifest")
    if manifest.get("lineage_id") != checkpoint.lineage_id or manifest.get("topology") != checkpoint.topology:
        raise ValueError("lineage manifest owner does not match checkpoint")
    # ``manifest.parent_checkpoint`` identifies the immutable parent at the
    # lineage boundary.  After the first generation, the immediate parent is
    # the preceding same-lineage CheckpointNode and is intentionally not
    # promoted into that lineage-root field.
    if parent.lineage_id != checkpoint.lineage_id and manifest.get("parent_checkpoint") != parent.to_dict():
        raise ValueError("lineage manifest parent changed during graph publication")
    hashes = manifest.get("checkpoint_hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("lineage manifest checkpoint_hashes is malformed")
    updated = dict(manifest)
    updated_hashes = dict(hashes)
    updated_hashes[checkpoint.path] = checkpoint.sha256
    updated["checkpoint_hashes"] = updated_hashes
    commits = updated.get("generation_commits")
    if not isinstance(commits, Mapping):
        commits = {}
    updated_commits = dict(commits)
    updated_commits[str(checkpoint.generation)] = generation_commit.to_dict()
    updated["generation_commits"] = updated_commits
    updated["effective_config"] = effective_config.to_dict()
    atomic_write_text(manifest_path, canonical_json(updated) + "\n")

    catalog_path = lineage_root / "runtime" / "artifact-catalog.json"
    catalog = (
        ArtifactCatalog.load(catalog_path, root=lineage_root)
        if catalog_path.is_file()
        else ArtifactCatalog.initialize(
            catalog_path,
            lineage_id=checkpoint.lineage_id,
            root=lineage_root,
        )
    )
    catalog_items = [
        identity
        for name, identity in artifact_payload.items()
        if name != "completion_marker"
    ]
    catalog_items.extend(
        (
            _artifact_identity(provenance_path, lineage_root),
            _artifact_identity(node_path, lineage_root),
        )
    )
    catalog.register_generation(checkpoint.generation, catalog_items)
    return node


def validate_generation_commit(
    *,
    root: str | Path,
    lineage_id: str,
    generation: int,
) -> CheckpointNode:
    """Validate the evidence behind a completion marker, fail-closed."""
    lineage_root = Path(root).resolve()
    marker_path = lineage_root / f"generation-{generation:02d}.complete.json"
    marker = _read_object(marker_path, "generation completion marker")
    if int(marker.get("generation", -1)) != int(generation):
        raise ValueError("completion marker generation mismatch")
    if str(marker.get("lineage_id", marker.get("run_id", lineage_id))) != lineage_id:
        raise ValueError("completion marker lineage mismatch")
    required_marker_fields = (
        "checkpoint_sha256",
        "fresh_replay_sha256",
        "rolling_replay_sha256",
        "checkpoint_metadata_sha256",
        "training_metrics_sha256",
        "summary_sha256",
    )
    if any(not str(marker.get(field, "")).startswith("sha256:") for field in required_marker_fields):
        raise ValueError("completion marker lacks mandatory artifact identities")

    node_path = lineage_root / "metadata" / "checkpoints" / f"M{generation}.json"
    node_payload = _read_object(node_path, "CheckpointNode")
    node = CheckpointNode.from_dict(node_payload)
    if node.checkpoint.lineage_id != lineage_id or node.checkpoint.generation != generation:
        raise ValueError("CheckpointNode does not belong to committed generation")
    checkpoint_path = _owned_path(lineage_root, node.checkpoint.path, "checkpoint")
    fresh_path = None if node.fresh_replay is None else _owned_path(lineage_root, node.fresh_replay.path, "fresh replay")
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != node.checkpoint.sha256:
        raise ValueError("CheckpointNode checkpoint identity is not physically valid")
    if fresh_path is None or not fresh_path.is_file() or sha256_file(fresh_path) != node.fresh_replay.sha256:
        raise ValueError("CheckpointNode fresh replay identity is not physically valid")
    if str(marker["checkpoint_sha256"]) != node.checkpoint.sha256 or str(marker["fresh_replay_sha256"]) != node.fresh_replay.sha256:
        raise ValueError("completion marker disagrees with CheckpointNode")
    marker_artifacts = {
        "rolling_replay_sha256": lineage_root / "replay" / f"rolling-after-{generation:02d}.jsonl",
        "checkpoint_metadata_sha256": checkpoint_path.with_suffix(".metadata.json"),
        "training_metrics_sha256": lineage_root / "training" / f"iter-{generation:02d}.json",
        "summary_sha256": lineage_root / f"iter-{generation:02d}-summary.json",
    }
    for field, path in marker_artifacts.items():
        if not path.is_file() or sha256_file(path) != str(marker[field]):
            raise ValueError(f"completion marker artifact identity is invalid: {field}")

    provenance_path = _owned_path(lineage_root, node.provenance.path, "provenance")
    if not provenance_path.is_file() or sha256_file(provenance_path) != node.provenance.sha256:
        raise ValueError("CheckpointNode provenance is missing or has the wrong identity")
    provenance = _read_object(provenance_path, "generation provenance")
    if provenance.get("checkpoint") != node.checkpoint.to_dict():
        raise ValueError("generation provenance checkpoint mismatch")
    if provenance.get("immediate_parent") != (None if node.parent is None else node.parent.to_dict()):
        raise ValueError("generation provenance parent mismatch")
    if provenance.get("fresh_replay") != node.fresh_replay.to_dict():
        raise ValueError("generation provenance replay mismatch")
    if provenance.get("effective_config") != node.effective_config.to_dict():
        raise ValueError("generation provenance config mismatch")
    commit_ref = ArtifactRef.from_dict(provenance.get("generation_commit", {}))
    if commit_ref.path != marker_path.relative_to(lineage_root).as_posix() or commit_ref.sha256 != sha256_file(marker_path):
        raise ValueError("generation provenance commit marker mismatch")
    if provenance.get("checkpoint_reload_verified") is not True:
        raise ValueError("checkpoint reload was not verified before commit")

    config_path = _owned_path(lineage_root, node.effective_config.artifact.path, "effective config")
    if not config_path.is_file() or sha256_file(config_path) != node.effective_config.artifact.sha256:
        raise ValueError("effective config evidence is missing or invalid")
    manifest = _read_object(lineage_root / "manifest.json", "lineage manifest")
    hashes = manifest.get("checkpoint_hashes")
    if manifest.get("lineage_id") != lineage_id or not isinstance(hashes, Mapping):
        raise ValueError("lineage manifest is not valid commit evidence")
    if hashes.get(node.checkpoint.path) != node.checkpoint.sha256:
        raise ValueError("lineage manifest does not contain the committed checkpoint identity")
    catalog_path = lineage_root / "runtime" / "artifact-catalog.json"
    if catalog_path.is_file():
        catalog = ArtifactCatalog.load(catalog_path, root=lineage_root)
        if str(catalog.entries.get(node.checkpoint.path, {}).get("sha256", "")) != node.checkpoint.sha256:
            raise ValueError("artifact catalog does not contain the committed checkpoint")
    return node


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
    "publish_checkpoint_graph",
    "validate_generation_commit",
]
