"""Lazy, fail-closed resolution of the Orchestrator V2 artifact graph.

``CheckpointNode`` records contain one immediate parent and one fresh replay
artifact.  This module is the runtime owner of traversing those links.  It
deliberately does not know about legacy metadata, model loading, replay
contents, or any topology-specific orchestration.

The canonical V2 node locator is a storage detail chosen by Stage 1:
``metadata/checkpoints/<checkpoint_id>.json`` relative to the owning lineage.
It is intentionally separate from the legacy ``checkpoints/M*.metadata.json``
files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Union

from .. import run_storage
from ..artifact_catalog import ArtifactCatalog, sha256_file
from .contracts import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)


NODE_DIRECTORY = Path("metadata") / "checkpoints"


class ArtifactResolutionError(ValueError):
    """Base error for a failed bounded V2 resolution."""


class ArtifactIntegrityError(ArtifactResolutionError):
    """A physical artifact or immutable content identity is invalid."""


class GraphIntegrityError(ArtifactResolutionError):
    """A persisted V2 graph edge or traversal is invalid."""


@dataclass(frozen=True)
class ResolvedArtifact:
    """An explicitly referenced immutable artifact opened inside its owner."""

    ref: ArtifactRef
    path: Path
    owner_root: Path
    owner_topology: str
    owner_lineage_id: str
    owner_status: str | None = None
    # Durable catalog/checkpoint evidence, when the owner committed it. The
    # resolver has already verified ``ref.sha256`` against the physical file;
    # consumers may use this evidence without hashing the artifact again.
    identity: Mapping[str, object] | None = None

    @property
    def artifact(self) -> ArtifactRef:
        """Compatibility alias for callers that name the logical value artifact."""
        return self.ref

    @property
    def sha256(self) -> str:
        return self.ref.sha256

    @property
    def topology(self) -> str:
        return self.owner_topology

    @property
    def lineage_id(self) -> str:
        return self.owner_lineage_id


@dataclass(frozen=True)
class ResolvedEffectiveConfig:
    """Parsed effective config plus the verified immutable artifact that owns it."""

    ref: EffectiveConfigRef
    artifact: ResolvedArtifact
    config: EffectiveConfig

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint

    @property
    def topology(self) -> str:
        return self.config.topology

    @property
    def compatibility(self) -> Mapping[str, object]:
        return self.config.compatibility

    @property
    def path(self) -> Path:
        return self.artifact.path

    def to_dict(self) -> dict[str, object]:
        return self.config.to_dict()


@dataclass(frozen=True)
class ResolvedCheckpointNode:
    """A parsed V2 node with all eagerly required immutable references verified."""

    node: CheckpointNode
    checkpoint: run_storage.ResolvedCheckpoint
    effective_config: ResolvedEffectiveConfig
    provenance: ResolvedArtifact
    owner_root: Path
    owner_status: str

    @property
    def ref(self) -> CheckpointRef:
        return self.node.checkpoint

    @property
    def checkpoint_ref(self) -> CheckpointRef:
        return self.node.checkpoint

    @property
    def resolved_checkpoint(self) -> run_storage.ResolvedCheckpoint:
        return self.checkpoint

    @property
    def path(self) -> Path:
        return self.checkpoint.path

    @property
    def topology(self) -> str:
        return self.node.checkpoint.topology

    @property
    def lineage_id(self) -> str:
        return self.node.checkpoint.lineage_id

    @property
    def checkpoint_id(self) -> str:
        return self.node.checkpoint.checkpoint_id

    @property
    def generation(self) -> int:
        return self.node.checkpoint.generation


@dataclass(frozen=True)
class _ArtifactOwner:
    root: Path
    topology: str
    lineage_id: str
    status: str | None


Owner = Union[ResolvedCheckpointNode, run_storage.ResolvedCheckpoint, CheckpointRef, _ArtifactOwner]


def checkpoint_node_path(owner_root: str | Path, checkpoint: CheckpointRef) -> Path:
    """Return the one canonical persisted-node path for ``checkpoint``."""
    return Path(owner_root).resolve() / NODE_DIRECTORY / f"{checkpoint.checkpoint_id}.json"


def _checkpoint_identity(ref: CheckpointRef) -> tuple[str, str, str, int, str, str]:
    """Return the complete immutable checkpoint identity used for cycle checks."""
    return (
        ref.topology,
        ref.lineage_id,
        ref.checkpoint_id,
        ref.generation,
        ref.path,
        ref.sha256,
    )


class ArtifactResolver:
    """Resolve V2 checkpoint nodes, parent chains, and fresh replay windows.

    Resolution is lazy and bounded by the requested traversal.  ``runs_root``
    is injectable so tests can use tiny synthetic lineages without touching
    production storage.
    """

    def __init__(self, runs_root: str | Path | None = None) -> None:
        self.runs_root = (
            Path(runs_root).resolve()
            if runs_root is not None
            else run_storage.RUNS_ROOT.resolve()
        )

    def checkpoint(self, ref: CheckpointRef | Mapping[str, object]) -> ResolvedCheckpointNode:
        """Open one full checkpoint reference and its canonical V2 node."""
        checkpoint_ref = self._checkpoint_ref(ref)
        physical = self._resolve_checkpoint(checkpoint_ref)
        owner = self._owner_for_physical(checkpoint_ref, physical)

        node_path = checkpoint_node_path(owner.root, checkpoint_ref).resolve()
        self._require_confined(node_path, owner.root, label="checkpoint node")
        try:
            payload = json.loads(node_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactResolutionError(
                f"Cannot read canonical CheckpointNode for "
                f"{self._checkpoint_label(checkpoint_ref)}: {node_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ArtifactResolutionError(f"CheckpointNode is not an object: {node_path}")
        try:
            node = CheckpointNode.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise ArtifactResolutionError(f"Invalid CheckpointNode {node_path}: {exc}") from exc

        self._validate_checkpoint_identity(checkpoint_ref, physical, node, node_path)
        resolved_config = self._resolve_effective_config(node.effective_config, owner, checkpoint_ref)
        provenance = self.open_artifact(node.provenance, owner=physical)
        return ResolvedCheckpointNode(
            node=node,
            checkpoint=physical,
            effective_config=resolved_config,
            provenance=provenance,
            owner_root=owner.root,
            owner_status=owner.status or physical.owner_status,
        )

    def parent(self, checkpoint: ResolvedCheckpointNode) -> ResolvedCheckpointNode | None:
        """Resolve exactly the persisted immediate parent, if any."""
        self._require_resolved_node(checkpoint)
        if checkpoint.node.genesis:
            return None
        parent_ref = checkpoint.node.parent
        if parent_ref is None:  # Defensive; strict parsing already rejects this.
            raise GraphIntegrityError(
                f"Non-genesis checkpoint {self._checkpoint_label(checkpoint.ref)} has no parent"
            )
        return self._resolve_parent(checkpoint, seen=set())

    def ancestor(self, checkpoint: ResolvedCheckpointNode, n: int) -> ResolvedCheckpointNode:
        """Return the checkpoint ``n`` immediate-parent edges behind ``checkpoint``."""
        distance = self._count(n, label="ancestor distance")
        self._require_resolved_node(checkpoint)
        current = checkpoint
        seen: set[tuple[str, str, str, int, str, str]] = set()
        for step in range(distance + 1):
            self._record_visit(current, seen)
            if step == distance:
                return current
            self._raise_if_next_is_cycle(current, seen)
            next_checkpoint = self._resolve_parent(current, seen=seen)
            if next_checkpoint is None:
                raise GraphIntegrityError(
                    f"Ancestry ended before distance {distance} from "
                    f"{self._checkpoint_label(checkpoint.ref)} at step {step}"
                )
            current = next_checkpoint
        raise AssertionError("ancestor traversal did not return")

    def ancestors(
        self,
        checkpoint: ResolvedCheckpointNode,
        count: int,
    ) -> tuple[ResolvedCheckpointNode, ...]:
        """Return nearest-to-oldest immediate ancestors, excluding ``checkpoint``."""
        requested = self._count(count, label="ancestor count")
        self._require_resolved_node(checkpoint)
        if requested == 0:
            return ()

        current = checkpoint
        seen: set[tuple[str, str, str, int, str, str]] = set()
        result: list[ResolvedCheckpointNode] = []
        for step in range(requested):
            self._record_visit(current, seen)
            self._raise_if_next_is_cycle(current, seen)
            next_checkpoint = self._resolve_parent(current, seen=seen)
            if next_checkpoint is None:
                raise GraphIntegrityError(
                    f"Requested {requested} ancestors from "
                    f"{self._checkpoint_label(checkpoint.ref)}, but ancestry ended at {step}"
                )
            result.append(next_checkpoint)
            current = next_checkpoint
        self._record_visit(current, seen)
        return tuple(result)

    def replay_window(
        self,
        parent: ResolvedCheckpointNode,
        count: int,
    ) -> tuple[ResolvedArtifact, ...]:
        """Open ``count`` fresh replays from ``parent``, ordered oldest-to-newest."""
        return self._fresh_replay_window(parent, count)

    def _fresh_replay_window(
        self,
        parent: ResolvedCheckpointNode,
        count: int,
    ) -> tuple[ResolvedArtifact, ...]:
        """Open graph-owned fresh replay artifacts, oldest-to-newest."""
        requested = self._count(count, label="replay window count")
        self._require_resolved_node(parent)
        if requested == 0:
            return ()

        current = parent
        seen: set[tuple[str, str, str, int, str, str]] = set()
        newest_to_oldest: list[ResolvedArtifact] = []
        for step in range(requested):
            self._record_visit(current, seen)
            replay_ref = current.node.fresh_replay
            if replay_ref is None:
                raise GraphIntegrityError(
                    f"Checkpoint {self._checkpoint_label(current.ref)} has no fresh replay "
                    f"for replay-window step {step}"
                )
            newest_to_oldest.append(self.open_artifact(replay_ref, owner=current))
            if step == requested - 1:
                break
            self._raise_if_next_is_cycle(current, seen)
            next_checkpoint = self._resolve_parent(current, seen=seen)
            if next_checkpoint is None:
                raise GraphIntegrityError(
                    f"Requested replay window of {requested} from "
                    f"{self._checkpoint_label(parent.ref)}, but ancestry ended at {step + 1}"
                )
            current = next_checkpoint
        return tuple(reversed(newest_to_oldest))

    def open_artifact(
        self,
        ref: ArtifactRef | Mapping[str, object],
        *,
        owner: Owner,
    ) -> ResolvedArtifact:
        """Open one owner-relative immutable artifact and verify its SHA-256."""
        artifact_ref = self._artifact_ref(ref)
        owner_context = self._owner_context(owner)
        path = (owner_context.root / artifact_ref.path).resolve()
        self._require_confined(path, owner_context.root, label="artifact")
        exists = path.is_file()
        try:
            actual = sha256_file(path) if exists else None
        except OSError as exc:
            raise ArtifactIntegrityError(
                f"Cannot read immutable artifact for hashing: path={path}, owner={owner_context.root}"
            ) from exc
        if actual != artifact_ref.sha256:
            raise ArtifactIntegrityError(
                "Artifact integrity check failed: "
                f"owner={owner_context.topology}/{owner_context.lineage_id}, "
                f"path={path}, exists={str(exists).lower()}, "
                f"expected SHA={artifact_ref.sha256}, actual SHA={actual or '<missing>'}"
            )
        identity = self._durable_artifact_identity(
            artifact_ref,
            path,
            owner=owner,
            owner_context=owner_context,
        )
        return ResolvedArtifact(
            ref=artifact_ref,
            path=path,
            owner_root=owner_context.root,
            owner_topology=owner_context.topology,
            owner_lineage_id=owner_context.lineage_id,
            owner_status=owner_context.status,
            identity=identity,
        )

    def _durable_artifact_identity(
        self,
        artifact_ref: ArtifactRef,
        path: Path,
        *,
        owner: Owner,
        owner_context: _ArtifactOwner,
    ) -> Mapping[str, object] | None:
        """Attach existing catalog/metadata evidence to an already-open file."""
        relative = artifact_ref.path
        identity: dict[str, object] = {
            "path": relative,
            "sha256": artifact_ref.sha256,
            "immutable_verified": True,
        }
        catalog_path = owner_context.root / "runtime" / "artifact-catalog.json"
        if catalog_path.is_file():
            try:
                catalog = ArtifactCatalog.load(catalog_path, root=owner_context.root)
                catalog_entry = catalog.entries.get(relative)
            except ValueError as exc:
                raise ArtifactIntegrityError(
                    f"Cannot load artifact catalog for resolved artifact: {catalog_path}"
                ) from exc
            if isinstance(catalog_entry, Mapping):
                if str(catalog_entry.get("sha256", "")) != artifact_ref.sha256:
                    raise ArtifactIntegrityError(
                        f"Artifact catalog SHA disagrees with resolved artifact: {relative}"
                    )
                if int(catalog_entry.get("size_bytes", -1)) != path.stat().st_size:
                    raise ArtifactIntegrityError(
                        f"Artifact catalog size disagrees with resolved artifact: {relative}"
                    )
                identity.update(dict(catalog_entry))
                identity["validation_schema"] = catalog.payload.get("validation_schema")
        # A size is useful for the downstream evidence check even when the
        # catalog is absent, but validation_schema remains absent so restore
        # correctly falls back to the existing full-validation path.  Replay
        # composition evidence is interpreted by the production training
        # path, not by this generic graph resolver.
        identity.setdefault("size_bytes", path.stat().st_size)
        return identity

    def _resolve_effective_config(
        self,
        ref: EffectiveConfigRef,
        owner: _ArtifactOwner,
        checkpoint_ref: CheckpointRef,
    ) -> ResolvedEffectiveConfig:
        artifact = self.open_artifact(ref.artifact, owner=owner)
        try:
            payload = json.loads(artifact.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactResolutionError(
                f"Cannot read EffectiveConfig for {self._checkpoint_label(checkpoint_ref)}: "
                f"{artifact.path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ArtifactResolutionError(f"EffectiveConfig is not an object: {artifact.path}")
        try:
            config = EffectiveConfig.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise ArtifactResolutionError(f"Invalid EffectiveConfig {artifact.path}: {exc}") from exc
        if config.fingerprint != ref.fingerprint:
            raise ArtifactIntegrityError(
                "EffectiveConfig fingerprint mismatch: "
                f"checkpoint={self._checkpoint_label(checkpoint_ref)}, path={artifact.path}, "
                f"expected fingerprint={ref.fingerprint}, actual fingerprint={config.fingerprint}"
            )
        if config.topology != checkpoint_ref.topology:
            raise GraphIntegrityError(
                "EffectiveConfig topology mismatch: "
                f"checkpoint={self._checkpoint_label(checkpoint_ref)} topology={checkpoint_ref.topology}, "
                f"config={config.topology}"
            )
        compatibility_topology = config.compatibility.get("topology")
        if compatibility_topology is not None and compatibility_topology != checkpoint_ref.topology:
            raise GraphIntegrityError(
                "EffectiveConfig compatibility topology mismatch: "
                f"checkpoint={self._checkpoint_label(checkpoint_ref)} topology={checkpoint_ref.topology}, "
                f"compatibility.topology={compatibility_topology}"
            )
        return ResolvedEffectiveConfig(ref=ref, artifact=artifact, config=config)

    def _resolve_parent(
        self,
        checkpoint: ResolvedCheckpointNode,
        *,
        seen: set[tuple[str, str, str, int, str, str]],
    ) -> ResolvedCheckpointNode:
        parent_ref = checkpoint.node.parent
        if parent_ref is None:
            if checkpoint.node.genesis:
                raise GraphIntegrityError(
                    f"ancestry ended at genesis checkpoint {self._checkpoint_label(checkpoint.ref)}"
                )
            raise GraphIntegrityError(
                f"Non-genesis checkpoint {self._checkpoint_label(checkpoint.ref)} has no parent"
            )
        parent = self.checkpoint(parent_ref)
        if parent.node.parent is not None and _checkpoint_identity(parent.node.parent) in seen:
            raise GraphIntegrityError(
                "Checkpoint parent cycle detected: "
                f"{parent.ref.to_dict()} points to an already visited "
                f"{parent.node.parent.to_dict()}"
            )
        expected_generation = parent.node.checkpoint.generation + 1
        if checkpoint.node.checkpoint.generation != expected_generation:
            raise GraphIntegrityError(
                "Immediate-parent generation mismatch: "
                f"child={self._checkpoint_label(checkpoint.ref)} "
                f"generation={checkpoint.node.checkpoint.generation}, "
                f"parent={self._checkpoint_label(parent.ref)} "
                f"generation={parent.node.checkpoint.generation}; "
                f"expected child generation={expected_generation}"
            )
        return parent

    def _resolve_checkpoint(self, ref: CheckpointRef) -> run_storage.ResolvedCheckpoint:
        try:
            physical = run_storage.resolve_checkpoint(
                ref.to_dict(),
                topology=ref.topology,
                runs_root=self.runs_root,
            )
        except run_storage.CheckpointResolutionError as exc:
            raise ArtifactIntegrityError(
                f"Cannot open checkpoint {self._checkpoint_label(ref)}: {exc}"
            ) from exc
        expected = _checkpoint_identity(ref)
        actual = (
            physical.topology,
            physical.lineage_id,
            physical.checkpoint_id,
            physical.generation,
            physical.path,
            physical.sha256,
        )
        owner = self._owner_root_for_path(ref, physical.path)
        expected_path = (owner.root / ref.path).resolve()
        if actual[:4] != expected[:4] or actual[5] != expected[5] or physical.path != expected_path:
            raise ArtifactIntegrityError(
                "Resolved checkpoint identity mismatch: "
                f"expected={ref.to_dict()}, actual={{'topology': {physical.topology!r}, "
                f"'lineage_id': {physical.lineage_id!r}, 'checkpoint_id': {physical.checkpoint_id!r}, "
                f"'generation': {physical.generation!r}, 'path': {str(physical.path)!r}, "
                f"'sha256': {physical.sha256!r}}}"
            )
        return physical

    def _owner_for_physical(
        self,
        ref: CheckpointRef,
        physical: run_storage.ResolvedCheckpoint,
    ) -> _ArtifactOwner:
        return self._owner_root_for_path(ref, physical.path, status=physical.owner_status)

    def _owner_root_for_path(
        self,
        ref: CheckpointRef,
        path: Path,
        *,
        status: str | None = None,
    ) -> _ArtifactOwner:
        matches: list[Path] = []
        for namespace in (run_storage.ACTIVE, run_storage.ARCHIVE):
            candidate = (self.runs_root / ref.topology / namespace / ref.lineage_id).resolve()
            try:
                path.resolve().relative_to(candidate)
            except ValueError:
                continue
            if candidate.is_dir():
                matches.append(candidate)
        if len(matches) != 1:
            raise ArtifactResolutionError(
                f"Cannot identify unique owner lineage for checkpoint "
                f"{self._checkpoint_label(ref)} at {path}: matches={matches}"
            )
        return _ArtifactOwner(matches[0], ref.topology, ref.lineage_id, status)

    def _owner_context(self, owner: Owner) -> _ArtifactOwner:
        if isinstance(owner, _ArtifactOwner):
            return owner
        if isinstance(owner, ResolvedCheckpointNode):
            return _ArtifactOwner(
                owner.owner_root,
                owner.topology,
                owner.lineage_id,
                owner.owner_status,
            )
        if isinstance(owner, run_storage.ResolvedCheckpoint):
            ref = CheckpointRef(
                owner.topology,
                owner.lineage_id,
                owner.checkpoint_id,
                owner.generation if owner.generation is not None else 0,
                self._relative_checkpoint_path(owner),
                owner.sha256,
            )
            return self._owner_root_for_path(ref, owner.path, status=owner.owner_status)
        if isinstance(owner, CheckpointRef):
            physical = self._resolve_checkpoint(owner)
            return self._owner_for_physical(owner, physical)
        raise TypeError(
            "artifact owner must be ResolvedCheckpointNode, ResolvedCheckpoint, or CheckpointRef"
        )

    def _relative_checkpoint_path(self, physical: run_storage.ResolvedCheckpoint) -> str:
        owner = self._owner_root_for_path(
            CheckpointRef(
                physical.topology,
                physical.lineage_id,
                physical.checkpoint_id,
                physical.generation if physical.generation is not None else 0,
                "checkpoints/placeholder.pt",
                physical.sha256,
            ),
            physical.path,
            status=physical.owner_status,
        )
        return physical.path.relative_to(owner.root).as_posix()

    @staticmethod
    def _require_confined(path: Path, root: Path, *, label: str) -> None:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"{label} path escapes owner lineage: path={path}, owner={root.resolve()}"
            ) from exc

    @staticmethod
    def _checkpoint_ref(ref: CheckpointRef | Mapping[str, object]) -> CheckpointRef:
        if isinstance(ref, CheckpointRef):
            return ref
        if isinstance(ref, Mapping):
            try:
                return CheckpointRef.from_dict(ref)
            except (TypeError, ValueError) as exc:
                raise ArtifactResolutionError(f"Invalid checkpoint reference: {exc}") from exc
        raise TypeError("checkpoint reference must be CheckpointRef or a mapping")

    @staticmethod
    def _artifact_ref(ref: ArtifactRef | Mapping[str, object]) -> ArtifactRef:
        if isinstance(ref, ArtifactRef):
            return ref
        if isinstance(ref, Mapping):
            try:
                return ArtifactRef.from_dict(ref)
            except (TypeError, ValueError) as exc:
                raise ArtifactResolutionError(f"Invalid artifact reference: {exc}") from exc
        raise TypeError("artifact reference must be ArtifactRef or a mapping")

    @staticmethod
    def _validate_checkpoint_identity(
        ref: CheckpointRef,
        physical: run_storage.ResolvedCheckpoint,
        node: CheckpointNode,
        node_path: Path,
    ) -> None:
        if node.checkpoint != ref:
            raise GraphIntegrityError(
                "CheckpointNode checkpoint identity mismatch: "
                f"requested={ref.to_dict()}, node={node.checkpoint.to_dict()}, node_path={node_path}"
            )
        if physical.topology != node.checkpoint.topology:
            raise GraphIntegrityError(f"CheckpointNode topology mismatch at {node_path}")
        if physical.lineage_id != node.checkpoint.lineage_id:
            raise GraphIntegrityError(f"CheckpointNode lineage mismatch at {node_path}")
        if physical.checkpoint_id != node.checkpoint.checkpoint_id:
            raise GraphIntegrityError(f"CheckpointNode checkpoint id mismatch at {node_path}")
        if physical.generation != node.checkpoint.generation:
            raise GraphIntegrityError(f"CheckpointNode generation mismatch at {node_path}")
        if physical.sha256 != node.checkpoint.sha256:
            raise GraphIntegrityError(f"CheckpointNode checkpoint SHA mismatch at {node_path}")

    @staticmethod
    def _require_resolved_node(checkpoint: ResolvedCheckpointNode) -> None:
        if not isinstance(checkpoint, ResolvedCheckpointNode):
            raise TypeError("checkpoint must be a ResolvedCheckpointNode")

    @staticmethod
    def _count(value: int, *, label: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    @staticmethod
    def _record_visit(
        checkpoint: ResolvedCheckpointNode,
        seen: set[tuple[str, str, str, int, str, str]],
    ) -> None:
        identity = _checkpoint_identity(checkpoint.ref)
        if identity in seen:
            raise GraphIntegrityError(
                f"Checkpoint parent cycle detected at {checkpoint.ref.to_dict()}"
            )
        seen.add(identity)

    @staticmethod
    def _raise_if_next_is_cycle(
        checkpoint: ResolvedCheckpointNode,
        seen: set[tuple[str, str, str, int, str, str]],
    ) -> None:
        parent_ref = checkpoint.node.parent
        if parent_ref is not None and _checkpoint_identity(parent_ref) in seen:
            raise GraphIntegrityError(
                "Checkpoint parent cycle detected: "
                f"{checkpoint.ref.to_dict()} points to an already visited "
                f"{parent_ref.to_dict()}"
            )

    @staticmethod
    def _checkpoint_label(ref: CheckpointRef) -> str:
        return f"{ref.topology}/{ref.lineage_id}/{ref.checkpoint_id}"


__all__ = [
    "NODE_DIRECTORY",
    "ArtifactResolutionError",
    "ArtifactIntegrityError",
    "GraphIntegrityError",
    "ResolvedArtifact",
    "ResolvedEffectiveConfig",
    "ResolvedCheckpointNode",
    "checkpoint_node_path",
    "ArtifactResolver",
]
