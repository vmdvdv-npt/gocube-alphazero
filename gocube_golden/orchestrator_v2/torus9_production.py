"""Concrete V2 bridge to the existing Torus9 production driver.

The driver remains the owner of self-play, training, checkpoint bytes, replay
bytes, and the atomic generation commit.  The small V2 publication tail in
this module records the already-produced checkpoint graph edge; it does not
discover ancestry or reconstruct replay inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
from pathlib import Path

from .. import run_storage
from ..artifact_catalog import sha256_file
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json, capture_code_identity
from .artifact_resolver import (
    ResolvedArtifact,
    ResolvedCheckpointNode,
    ResolvedEffectiveConfig,
    checkpoint_node_path,
)
from .contracts import ArtifactRef, CheckpointNode, CheckpointRef, EffectiveConfigRef
from .generation_runner import GenerationExecutionResult, ResolvedGenerationInput


def _default_driver(resolved_input: ResolvedGenerationInput) -> Mapping[str, object]:
    # Keep the scientific driver lazy: importing the V2 contracts remains
    # cheap for resolver/tests and does not import torch until execution.
    from tools.torus9_run_driver import run_generation_v2

    return run_generation_v2(resolved_input)


def _relative_artifact(root: Path, value: object, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Production driver result is missing {label} path")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Production driver {label} escapes output lineage") from exc
    if not path.is_file():
        raise ValueError(f"Production driver {label} is missing: {relative}")
    return relative, path


def _artifact_identity(
    *,
    root: Path,
    payload: Mapping[str, object],
    key: str,
    fallback_path: str | None = None,
) -> ArtifactRef:
    raw = payload.get(key)
    declared: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
    raw_path = declared.get("path", fallback_path)
    relative, path = _relative_artifact(root, raw_path, key)
    declared_sha = declared.get("sha256") or declared.get("artifact_sha256")
    sha256 = str(declared_sha or sha256_file(path))
    if sha256 != sha256_file(path):
        raise ValueError(f"Production driver {key} SHA-256 mismatch")
    return ArtifactRef(path=relative, sha256=sha256)


class V2CheckpointPublisher:
    """Publish canonical V2 metadata after the existing generation commit."""

    def publish(
        self,
        resolved_input: ResolvedGenerationInput,
        execution: GenerationExecutionResult,
    ) -> None:
        parent = resolved_input.parent_checkpoint
        checkpoint = execution.checkpoint
        fresh_replay = execution.fresh_replay
        commit_artifact = execution.commit_artifact
        if checkpoint is None or fresh_replay is None or commit_artifact is None:
            raise ValueError("Committed V2 generation lacks canonical artifact identities")

        output = resolved_input.output_lineage
        if checkpoint.topology != output.topology or checkpoint.lineage_id != output.lineage_id:
            raise ValueError("Committed checkpoint does not belong to the output lineage")
        if checkpoint.generation != parent.generation + 1:
            raise ValueError("Committed checkpoint is not an immediate child of the resolved parent")

        config_ref = resolved_input.effective_config.ref
        config_path = (output.root / config_ref.artifact.path).resolve()
        self._require_owned_file(output.root, config_path, "effective config")
        if sha256_file(config_path) != config_ref.artifact.sha256:
            raise ValueError("Effective config artifact SHA-256 mismatch during publication")
        self._require_owned_file(output.root, (output.root / checkpoint.path).resolve(), "checkpoint")
        self._require_owned_file(output.root, (output.root / fresh_replay.path).resolve(), "fresh replay")
        self._require_owned_file(output.root, (output.root / commit_artifact.path).resolve(), "commit artifact")

        provenance_path = output.root / "metadata" / "provenance-v2" / f"M{checkpoint.generation}.json"
        provenance_payload = {
            "schema": "gocube-orchestrator-v2-production-provenance-v1",
            "version": 1,
            "checkpoint": checkpoint.to_dict(),
            "immediate_parent": parent.ref.to_dict(),
            "fresh_replay": fresh_replay.to_dict(),
            "generation_commit": commit_artifact.to_dict(),
            "effective_config": config_ref.to_dict(),
        }
        self._publish_json(provenance_path, provenance_payload)
        provenance = ArtifactRef(
            provenance_path.relative_to(output.root).as_posix(),
            sha256_file(provenance_path),
        )
        node = CheckpointNode(
            checkpoint=checkpoint,
            genesis=False,
            parent=parent.ref,
            fresh_replay=fresh_replay,
            effective_config=config_ref,
            provenance=provenance,
        )
        node_path = checkpoint_node_path(output.root, checkpoint)
        if node_path.is_file():
            try:
                existing = CheckpointNode.from_dict(self._read_json(node_path))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Existing V2 checkpoint node is invalid: {node_path}") from exc
            if existing != node:
                raise ValueError(f"Existing V2 checkpoint node conflicts: {node_path}")
        else:
            self._publish_json(node_path, node.to_dict())

        manifest_path = output.root / "manifest.json"
        manifest = self._read_json(manifest_path)
        if manifest.get("parent_checkpoint") != parent.ref.to_dict():
            raise ValueError("Lineage manifest parent changed during V2 publication")
        hashes = manifest.get("checkpoint_hashes")
        if not isinstance(hashes, Mapping):
            raise ValueError("Lineage manifest checkpoint_hashes is malformed")
        updated = dict(manifest)
        updated_hashes = dict(hashes)
        updated_hashes[checkpoint.path] = checkpoint.sha256
        updated["checkpoint_hashes"] = updated_hashes
        updated.setdefault("effective_config", config_ref.to_dict())
        if updated != manifest:
            self._write_json(manifest_path, updated)

    @staticmethod
    def _require_owned_file(root: Path, path: Path, label: str) -> None:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"{label} escapes output lineage") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Committed {label} is missing: {path}")

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON object required: {path}")
        return payload

    @staticmethod
    def _publish_json(path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = canonical_json(dict(payload)) + "\n"
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return
        if path.is_file():
            raise ValueError(f"Refusing to overwrite immutable V2 artifact: {path}")
        atomic_write_text(path, content)

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        """Atomically write mutable coordination metadata such as manifest.json."""
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, canonical_json(dict(payload)) + "\n")


class Torus9ProductionLineage:
    """Prepare one active arm lineage without copying the external parent."""

    def __init__(self, runs_root: str | Path, *, repo_root: str | Path | None = None) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )

    def prepare(
        self,
        *,
        topology: str,
        lineage_id: str,
        parent: ResolvedCheckpointNode,
        effective_config: object,
        experiment_id: str,
        arm_id: str,
    ) -> tuple[Path, ResolvedEffectiveConfig]:
        config = getattr(effective_config, "config", effective_config)
        if not hasattr(config, "to_dict") or not hasattr(config, "fingerprint"):
            raise TypeError("production lineage requires an EffectiveConfig")
        root = (self.runs_root / topology / run_storage.ACTIVE / lineage_id).resolve()
        code = capture_code_identity(self.repo_root)
        manifest = {
            "schema": "gocube-orchestrator-v2-production-lineage-v1",
            "lineage_id": lineage_id,
            "topology": topology,
            "status": "ACTIVE",
            "parent_checkpoint": parent.ref.to_dict(),
            "git_commit": code.git_commit_sha,
            "config_fingerprint": config.fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_hashes": {},
            "experiment": {"id": experiment_id, "arm": arm_id},
        }
        if root.is_dir():
            existing = V2CheckpointPublisher._read_json(root / "manifest.json")
            for key in ("lineage_id", "topology", "status", "parent_checkpoint", "config_fingerprint"):
                if existing.get(key) != manifest[key]:
                    raise ValueError(f"Production arm lineage {key} changed: {root}")
            if existing.get("git_commit") != code.git_commit_sha:
                raise ValueError("Production arm lineage code pin changed during resume")
            manifest = existing
        else:
            run_storage.ensure_lineage_layout(
                root,
                manifest=manifest,
                extra_directories=("metadata", "runtime", "replay", "selfplay", "training"),
            )

        config_path = root / "metadata" / "effective-config-v2" / f"{config.fingerprint}.json"
        content = canonical_json(config.to_dict()) + "\n"
        if config_path.is_file() and config_path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Effective config artifact changed: {config_path}")
        if not config_path.is_file():
            atomic_write_text(config_path, content)
        config_ref = EffectiveConfigRef(
            ArtifactRef(config_path.relative_to(root).as_posix(), sha256_file(config_path)),
            config.fingerprint,
        )
        updated_manifest = dict(manifest)
        if updated_manifest.get("effective_config") != config_ref.to_dict():
            updated_manifest["effective_config"] = config_ref.to_dict()
            V2CheckpointPublisher._write_json(root / "manifest.json", updated_manifest)
        resolved_artifact = ResolvedArtifact(
            ref=config_ref.artifact,
            path=config_path,
            owner_root=root,
            owner_topology=topology,
            owner_lineage_id=lineage_id,
            owner_status="ACTIVE",
            identity={
                "path": config_ref.artifact.path,
                "sha256": config_ref.artifact.sha256,
                "immutable_verified": True,
                "size_bytes": config_path.stat().st_size,
            },
        )
        return root, ResolvedEffectiveConfig(
            ref=config_ref,
            artifact=resolved_artifact,
            config=config,
        )


class Torus9ProductionGenerationPath:
    """Use the existing Torus9 production generation driver as a V2 path."""

    def __init__(
        self,
        driver: Callable[[ResolvedGenerationInput], Mapping[str, object] | GenerationExecutionResult]
        | None = None,
        *,
        publisher: V2CheckpointPublisher | None = None,
    ) -> None:
        self._driver = driver or _default_driver
        self._publisher = publisher

    def run_generation(
        self, resolved_input: ResolvedGenerationInput
    ) -> GenerationExecutionResult:
        if resolved_input.output_lineage.topology != "torus9":
            raise ValueError("Torus9 production path requires topology=torus9")
        produced = self._driver(resolved_input)
        if isinstance(produced, GenerationExecutionResult):
            if produced.committed and self._publisher is not None:
                if produced.fresh_replay is None:
                    raise ValueError("Committed production result is missing fresh replay identity")
                self._publisher.publish(resolved_input, produced)
            return produced
        if not isinstance(produced, Mapping):
            raise TypeError("Torus9 production driver returned an invalid result")

        generation = int(resolved_input.generation)
        committed = (
            str(produced.get("status", "")).upper() == "COMPLETED"
            and bool(produced.get("checkpoint_reload_verified", True))
        )
        if not committed:
            return GenerationExecutionResult(generation=generation, committed=False)

        root = resolved_input.output_lineage.root
        checkpoint_artifact = _artifact_identity(
            root=root,
            payload=produced,
            key="checkpoint",
            fallback_path=f"checkpoints/M{generation}.pt",
        )
        commit_artifact = _artifact_identity(
            root=root,
            payload=produced,
            key="commit_artifact",
            fallback_path=f"generation-{generation:02d}.complete.json",
        )
        fresh_replay = None
        if self._publisher is not None:
            fresh_replay = _artifact_identity(root=root, payload=produced, key="fresh_replay")
        checkpoint_id = f"M{generation}"
        checkpoint_payload = produced.get("checkpoint")
        if isinstance(checkpoint_payload, Mapping) and checkpoint_payload.get("checkpoint_id"):
            checkpoint_id = str(checkpoint_payload["checkpoint_id"])
        execution = GenerationExecutionResult(
            generation=generation,
            committed=True,
            checkpoint=CheckpointRef(
                topology=resolved_input.output_lineage.topology,
                lineage_id=resolved_input.output_lineage.lineage_id,
                checkpoint_id=checkpoint_id,
                generation=generation,
                path=checkpoint_artifact.path,
                sha256=checkpoint_artifact.sha256,
            ),
            commit_artifact=commit_artifact,
            fresh_replay=fresh_replay,
        )
        if self._publisher is not None:
            self._publisher.publish(resolved_input, execution)
        return execution


__all__ = [
    "Torus9ProductionGenerationPath",
    "Torus9ProductionLineage",
    "V2CheckpointPublisher",
]
