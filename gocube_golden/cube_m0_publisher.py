"""Canonical Cube-family genesis checkpoint publisher.

The Stage-6 ``create_cube_m0_state`` helper is intentionally scientific and
in-memory.  This module is the small production boundary that turns that
state into one immutable M0 lineage without routing genesis through a normal
training generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Mapping

from training_engine import CheckpointContext, sequence_fingerprint, value_fingerprint

from .artifact_catalog import ArtifactCatalog, sha256_file
from .artifact_graph import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from .cube_checkpoint_v2 import sidecar_path
from .cube_game_contract_v2 import validate_cube_size
from .cube_training_contract_v2 import CubeTrainingConfig
from .cube_training_v2 import create_cube_m0_state, load_cube_checkpoint
from .process_supervision import atomic_write_text
from .provenance import CodeIdentity, canonical_json, capture_code_identity, file_sha256
from .run_storage import ARCHIVE, ACTIVE, active_lineage_dir, archived_lineage_dir, ensure_lineage_layout


def _safe_component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _required(mapping: Mapping[str, object], names: tuple[str, ...], label: str) -> object:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise ValueError(f"{label} must be explicit in the effective config")


def _effective_config(value: EffectiveConfig | Mapping[str, object]) -> EffectiveConfig:
    if isinstance(value, EffectiveConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("effective_config must be an EffectiveConfig or mapping")
    payload = dict(value)
    if "schema" not in payload:
        topology = str(payload.get("topology", ""))
        payload = {
            "schema": "gocube-effective-config-v2",
            "version": 2,
            "topology": topology,
            "compatibility": payload.get("compatibility", {"topology": topology}),
            "self_play": payload.get("self_play", {}),
            "training": payload.get("training", {}),
            "replay": payload.get("replay", {}),
            "execution": payload.get("execution", {}),
            "arena": payload.get("arena", {}),
            "supervision": payload.get("supervision", {}),
            "extensions": payload.get("extensions", {}),
        }
    return EffectiveConfig.from_dict(payload)


def _training_config(config: EffectiveConfig) -> CubeTrainingConfig:
    training = config.training
    replay = config.replay
    cap = _required(replay, ("cap",), "Cube replay cap")
    return CubeTrainingConfig(
        learning_rate=float(_required(training, ("learning_rate",), "Cube learning rate")),
        batch_size=int(_required(training, ("batch_size",), "Cube batch size")),
        optimizer_steps=int(
            _required(
                training,
                ("optimizer_steps", "optimizer_steps_per_iteration"),
                "Cube optimizer steps",
            )
        ),
        replay_generations=int(
            _required(replay, ("generations", "window"), "Cube replay generations")
        ),
        replay_cap=None if cap is None else int(cap),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def _identity(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    content = canonical_json(dict(payload)) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite immutable M0 artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content)


@dataclass(frozen=True)
class CubeM0Publication:
    """Published identities returned by :func:`publish_cube_m0`."""

    root: Path
    lineage_id: str
    topology: str
    size: int
    seed: int
    checkpoint: CheckpointRef
    checkpoint_metadata: ArtifactRef
    rolling_replay: ArtifactRef
    effective_config: EffectiveConfigRef
    node: CheckpointNode
    code_identity: CodeIdentity

    @property
    def checkpoint_sha256(self) -> str:
        return self.checkpoint.sha256

    def checkpoint_reference(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "lineage_id": self.lineage_id,
            "checkpoint_id": "M0",
            "label": "M0",
            "generation": 0,
            "path": str((self.root / self.checkpoint.path).resolve()),
            "metadata_path": str(
                (self.root / self.checkpoint_metadata.path).resolve()
            ),
            "sha256": self.checkpoint.sha256,
            "artifact_sha256": self.checkpoint.sha256,
            "model_hash": self._model_hash(),
            "size": self.size,
        }

    def _model_hash(self) -> str:
        import json

        payload = json.loads(
            (self.root / self.checkpoint_metadata.path).read_text(encoding="utf-8")
        )
        return str(payload["model_hash"])

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "lineage_id": self.lineage_id,
            "root": str(self.root),
            "seed": self.seed,
            "checkpoint": self.checkpoint.to_dict(),
            "checkpoint_metadata": self.checkpoint_metadata.to_dict(),
            "rolling_replay": self.rolling_replay.to_dict(),
            "effective_config": self.effective_config.to_dict(),
            "checkpoint_node": self.node.to_dict(),
            "code_identity": {
                "git_commit_sha": self.code_identity.git_commit_sha,
                "git_tree_sha": self.code_identity.git_tree_sha,
                "working_tree_clean": self.code_identity.working_tree_clean,
            },
        }


def publish_cube_m0(
    *,
    size: int,
    lineage_id: str,
    effective_config: EffectiveConfig | Mapping[str, object],
    seed: int,
    runs_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    code_identity: CodeIdentity | None = None,
    require_clean_code: bool = True,
) -> CubeM0Publication:
    """Publish one immutable genesis M0 for Cube2..Cube7.

    The function refuses an existing active or archived lineage id and writes
    M0 directly with the normal Cube checkpoint writer.  No ``TrainingEngine``
    generation-0 transaction is used and no parent checkpoint is accepted.
    """

    size = validate_cube_size(size)
    lineage = _safe_component(lineage_id, "lineage id")
    if type(seed) is not int or seed <= 0:
        raise ValueError("Cube M0 seed must be a positive explicit integer")
    config = _effective_config(effective_config)
    topology = f"cube{size}"
    if config.topology != topology:
        raise ValueError("Cube M0 effective config topology does not match size")
    training_config = _training_config(config)

    final_root = active_lineage_dir(topology, lineage).resolve()
    archive_root = archived_lineage_dir(topology, lineage).resolve()
    if final_root.exists() or archive_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Cube M0 lineage id: {topology}/{lineage}"
        )

    identity = code_identity or capture_code_identity(repo_root)
    identity.validate(require_canonical=require_clean_code)

    root_base = (
        Path(runs_root).resolve()
        if runs_root is not None
        else final_root.parents[2].resolve()
    )
    # ``active_lineage_dir`` is rooted at the module default.  The publisher
    # accepts an injected root for temporary qualification/tests while keeping
    # the same topology/active/lineage layout.
    final_root = root_base / topology / ACTIVE / lineage
    archive_root = root_base / topology / ARCHIVE / lineage
    if final_root.exists() or archive_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Cube M0 lineage id: {topology}/{lineage}"
        )
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{lineage}.m0-publishing-", dir=final_root.parent)
    )
    staging_root = staging_parent / lineage

    manifest: dict[str, object] = {
        "schema": "gocube-orchestrator-v2-production-lineage-v1",
        "lineage_id": lineage,
        "topology": topology,
        "status": "ACTIVE",
        "parent_checkpoint": None,
        "genesis": True,
        "git_commit": identity.git_commit_sha,
        "git_tree": identity.git_tree_sha,
        "config_fingerprint": config.fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_hashes": {},
        "m0_seed": int(seed),
        "m0_checkpoint": "M0",
    }

    try:
        ensure_lineage_layout(
            staging_root,
            manifest=manifest,
            extra_directories=("metadata", "runtime", "replay", "selfplay", "training"),
            include_data=False,
        )
        config_path = (
            staging_root
            / "metadata"
            / "effective-config-v2"
            / f"{config.fingerprint}.json"
        )
        _write_immutable_json(config_path, config.to_dict())
        config_ref = EffectiveConfigRef(
            ArtifactRef(
                config_path.relative_to(staging_root).as_posix(),
                sha256_file(config_path),
            ),
            config.fingerprint,
        )

        rolling_path = staging_root / "replay" / "rolling-after-00.jsonl"
        rolling_path.parent.mkdir(parents=True, exist_ok=True)
        rolling_path.write_bytes(b"")

        adapter, state = create_cube_m0_state(
            size=size,
            config=training_config,
            seed=seed,
            device="cpu",
        )
        context = CheckpointContext(
            run_id=lineage,
            label="M0",
            parent_label=None,
            generation=0,
            training_seed=seed,
            fresh_positions=0,
            replay_positions=0,
            replay_generations=(),
            replay_fingerprint=sequence_fingerprint(()),
            sampled_row_ids_fingerprint=value_fingerprint(()),
            completed_games=0,
            parent_checkpoint_identity=None,
            code_identity=identity,
            device="cpu",
        )
        metadata = adapter.prepare_checkpoint(state, context, {})
        checkpoint_path = staging_root / "checkpoints" / "M0.pt"
        saved_metadata = adapter.save_checkpoint(checkpoint_path, state, metadata)
        # This is the publisher's strict reload gate. It also verifies the
        # empty rolling replay identity and the fresh Adam optimizer state.
        adapter.verify_checkpoint(checkpoint_path, state, saved_metadata)
        load_cube_checkpoint(
            checkpoint_path,
            config=training_config,
            replay_path=rolling_path,
            map_location="cpu",
            expected_size=size,
        )

        checkpoint_ref = CheckpointRef(
            topology=topology,
            lineage_id=lineage,
            checkpoint_id="M0",
            generation=0,
            path="checkpoints/M0.pt",
            sha256=str(saved_metadata["checkpoint_sha256"]),
        )
        checkpoint_metadata_ref = ArtifactRef(
            "checkpoints/M0.metadata.json",
            sha256_file(sidecar_path(checkpoint_path)),
        )
        rolling_ref = ArtifactRef(
            "replay/rolling-after-00.jsonl",
            sha256_file(rolling_path),
        )
        artifact_identities = {
            "checkpoint": _identity(checkpoint_path, staging_root),
            "checkpoint_metadata": _identity(sidecar_path(checkpoint_path), staging_root),
            "rolling_replay": _identity(rolling_path, staging_root),
            "effective_config": _identity(config_path, staging_root),
        }
        provenance_path = staging_root / "metadata" / "provenance-v2" / "M0.json"
        provenance = {
            "schema": "gocube-orchestrator-v2-production-provenance-v2",
            "version": 2,
            "genesis": True,
            "checkpoint": checkpoint_ref.to_dict(),
            "immediate_parent": None,
            "fresh_replay": None,
            "generation_commit": None,
            "effective_config": config_ref.to_dict(),
            "initialization_seed": int(seed),
            "master_seed": int(seed),
            "checkpoint_reload_verified": True,
            "artifact_identities": artifact_identities,
            "code_identity": {
                "git_commit_sha": identity.git_commit_sha,
                "git_tree_sha": identity.git_tree_sha,
                "working_tree_clean": identity.working_tree_clean,
            },
        }
        _write_immutable_json(provenance_path, provenance)
        provenance_ref = ArtifactRef(
            "metadata/provenance-v2/M0.json",
            sha256_file(provenance_path),
        )
        node = CheckpointNode(
            checkpoint=checkpoint_ref,
            genesis=True,
            parent=None,
            fresh_replay=None,
            effective_config=config_ref,
            provenance=provenance_ref,
        )
        node_path = staging_root / "metadata" / "checkpoints" / "M0.json"
        _write_immutable_json(node_path, node.to_dict())

        catalog_path = staging_root / "runtime" / "artifact-catalog.json"
        catalog = ArtifactCatalog.initialize(
            catalog_path,
            lineage_id=lineage,
            root=staging_root,
        )
        catalog.register_generation(
            0,
            [
                *artifact_identities.values(),
                _identity(provenance_path, staging_root),
                _identity(node_path, staging_root),
            ],
            transaction={"kind": "genesis", "seed": int(seed)},
        )

        manifest["checkpoint_hashes"] = {checkpoint_ref.path: checkpoint_ref.sha256}
        manifest["effective_config"] = config_ref.to_dict()
        manifest["artifact_catalog_fingerprint"] = catalog.fingerprint
        atomic_write_text(staging_root / "manifest.json", canonical_json(manifest) + "\n")

        if final_root.exists() or archive_root.exists():
            raise FileExistsError(
                f"Refusing to publish Cube M0 over a concurrently-created lineage: {lineage}"
            )
        os.rename(staging_root, final_root)
        staging_parent.rmdir()
        staging_root = final_root

        # Reopen through the canonical resolver after publication.  The
        # returned node is the identity production Orchestrator will consume.
        from .artifact_resolver import ArtifactResolver

        resolved = ArtifactResolver(root_base).checkpoint(checkpoint_ref)
        if not resolved.node.genesis or resolved.node.parent is not None or resolved.node.fresh_replay is not None:
            raise RuntimeError("Published Cube M0 is not a canonical genesis node")
        return CubeM0Publication(
            root=final_root,
            lineage_id=lineage,
            topology=topology,
            size=size,
            seed=seed,
            checkpoint=checkpoint_ref,
            checkpoint_metadata=checkpoint_metadata_ref,
            rolling_replay=rolling_ref,
            effective_config=config_ref,
            node=resolved.node,
            code_identity=identity,
        )
    except Exception:
        if staging_root != final_root and staging_parent.exists():
            import shutil

            shutil.rmtree(staging_parent)
        raise


class CubeM0Publisher:
    """Reusable family-level publisher facade."""

    def __init__(self, *, runs_root: str | Path | None = None, repo_root: str | Path | None = None) -> None:
        self.runs_root = None if runs_root is None else Path(runs_root)
        self.repo_root = None if repo_root is None else Path(repo_root)

    def publish(
        self,
        *,
        size: int,
        lineage_id: str,
        effective_config: EffectiveConfig | Mapping[str, object],
        seed: int,
        code_identity: CodeIdentity | None = None,
        require_clean_code: bool = True,
    ) -> CubeM0Publication:
        return publish_cube_m0(
            size=size,
            lineage_id=lineage_id,
            effective_config=effective_config,
            seed=seed,
            runs_root=self.runs_root,
            repo_root=self.repo_root,
            code_identity=code_identity,
            require_clean_code=require_clean_code,
        )


__all__ = ["CubeM0Publisher", "CubeM0Publication", "publish_cube_m0"]
