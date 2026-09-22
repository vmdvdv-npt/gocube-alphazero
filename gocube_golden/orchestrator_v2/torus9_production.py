"""Concrete V2 bridge to the existing Torus9 production driver."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
from pathlib import Path

from .. import run_storage
from ..artifact_catalog import ArtifactCatalog, sha256_file
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json, capture_code_identity
from .artifact_resolver import ResolvedArtifact, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import ArtifactRef, CheckpointRef, EffectiveConfigRef
from .generation_runner import GenerationExecutionResult, ResolvedGenerationInput
from .version import ORCHESTRATOR_ENTRYPOINT, ORCHESTRATOR_VERSION


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


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, canonical_json(dict(payload)) + "\n")


def _advance_lineage_code_pin(
    manifest: Mapping[str, object],
    *,
    git_commit: str,
    git_tree: str,
    working_tree_clean: bool,
    allow_code_rollover: bool,
) -> dict[str, object]:
    """Record a clean application-code rollover while preserving lineage origin."""
    updated = dict(manifest)
    previous = str(updated.get("git_commit", ""))
    if previous == git_commit:
        return updated
    if not allow_code_rollover:
        raise ValueError(
            "Production arm lineage code pin changed during resume; "
            "set allow_code_rollover=True"
        )
    if not working_tree_clean:
        raise ValueError("Production arm lineage resume requires a clean working tree")
    updated["lineage_initial_git_commit"] = str(
        updated.get("lineage_initial_git_commit") or previous
    )
    updated["git_commit"] = git_commit
    updated["current_git_tree"] = git_tree
    return updated


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
        allow_code_rollover: bool = False,
    ) -> tuple[Path, ResolvedEffectiveConfig]:
        if type(allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        config = getattr(effective_config, "config", effective_config)
        if not hasattr(config, "to_dict") or not hasattr(config, "fingerprint"):
            raise TypeError("production lineage requires an EffectiveConfig")
        root = (self.runs_root / topology / run_storage.ACTIVE / lineage_id).resolve()
        code = capture_code_identity(self.repo_root)
        manifest = {
            "schema": "gocube-orchestrator-v2-production-lineage-v1",
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "orchestrator_entrypoint": ORCHESTRATOR_ENTRYPOINT,
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
            existing = _read_json(root / "manifest.json")
            for key in ("lineage_id", "topology", "status", "parent_checkpoint", "config_fingerprint"):
                if existing.get(key) != manifest[key]:
                    raise ValueError(f"Production arm lineage {key} changed: {root}")
            existing_version = existing.get("orchestrator_version")
            if existing_version not in (None, ORCHESTRATOR_VERSION):
                raise ValueError(
                    f"Production arm lineage uses unsupported orchestrator version: {existing_version!r}"
                )
            manifest = _advance_lineage_code_pin(
                existing,
                git_commit=code.git_commit_sha,
                git_tree=code.git_tree_sha,
                working_tree_clean=code.working_tree_clean,
                allow_code_rollover=allow_code_rollover,
            )
            manifest.setdefault("orchestrator_version", ORCHESTRATOR_VERSION)
            manifest.setdefault("orchestrator_entrypoint", ORCHESTRATOR_ENTRYPOINT)
            if manifest != existing:
                _write_json(root / "manifest.json", manifest)
        else:
            run_storage.ensure_lineage_layout(
                root,
                manifest=manifest,
                extra_directories=("metadata", "runtime", "replay", "selfplay", "training"),
            )
        catalog_path = root / "runtime" / "artifact-catalog.json"
        if not catalog_path.is_file():
            ArtifactCatalog.initialize(catalog_path, lineage_id=lineage_id, root=root)

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
            _write_json(root / "manifest.json", updated_manifest)
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
    ) -> None:
        self._driver = driver or _default_driver

    def run_generation(
        self, resolved_input: ResolvedGenerationInput
    ) -> GenerationExecutionResult:
        if resolved_input.output_lineage.topology != "torus9":
            raise ValueError("Torus9 production path requires topology=torus9")
        produced = self._driver(resolved_input)
        if isinstance(produced, GenerationExecutionResult):
            if produced.committed and produced.fresh_replay is None:
                raise ValueError("Committed production result is missing fresh replay identity")
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
        return execution


__all__ = [
    "Torus9ProductionGenerationPath",
    "Torus9ProductionLineage",
]
