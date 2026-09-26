"""Training bootstrap for the 5-channel M137-derived Torus9 line.

This module deliberately does not choose a new komi.  It prepares the
``new_komi`` lineage from the immutable M137 checkpoint, migrates every Adam
state that has an unambiguous 6->5 mapping, resets only the folded bias
moments, and starts with an empty replay history.

The parent checkpoint is referenced by identity; it is never copied into the
new lineage.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import torch

from .artifact_catalog import ArtifactCatalog, sha256_file
from .artifact_graph import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from .process_supervision import atomic_write_text
from .neural import model_hash
from .provenance import canonical_json, file_sha256
from .run_storage import ACTIVE, ensure_lineage_layout
from .torus9_m137_5ch import (
    M137_FIVE_CHANNEL_CHANNELS,
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    M137_FIVE_CHANNEL_FORMULA,
    Torus9M137FiveChannelGraphNet,
    convert_m137_model,
    load_canonical_m137,
)
from .torus9_monolith import TORUS9_TOPOLOGY_FINGERPRINT, Torus9CurrentGraphNet
from .torus9_contract import TORUS9_POINT_COUNT


NEW_KOMI_LINEAGE_ID = "new_komi"
NEW_KOMI_SOURCE_LINEAGE = "torus9-m125-continuous-v2-gen6-20260922-v1"
NEW_KOMI_SOURCE_CHECKPOINT = "M137"
NEW_KOMI_SOURCE_CHECKPOINT_SHA256 = (
    "71cfc78dab3fe217b3c435a765790efe6f6c4fa42a7d21479f3fd909adf341fe"
)
NEW_KOMI_CONVERTED_MODEL_HASH = (
    "sha256:f4fc0e173ed9cea1a6274bb613a95cb6ba428461f87eb60313deb390261927bc"
)
NEW_KOMI_OPTIMIZER_CONVERSION = (
    "adam-preserve-exact-crop-input-reset-folded-bias-moments-v1"
)
NEW_KOMI_REPLAY_POLICY = "fresh-only-no-parent-history"


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one immutable JSON graph artifact, refusing content drift."""

    content = canonical_json(dict(payload)) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite immutable bootstrap graph artifact: {path}")
        return
    atomic_write_text(path, content)


def _bootstrap_effective_config(metadata: Mapping[str, object]) -> EffectiveConfig:
    architecture = metadata.get("architecture_config")
    if not isinstance(architecture, Mapping):
        raise ValueError("M137 5CH metadata has no architecture_config")
    return EffectiveConfig(
        topology="torus9",
        compatibility={
            "topology": "torus9",
            "topology_id": architecture.get("topology_id"),
            "topology_fingerprint": architecture.get("topology_fingerprint"),
            "architecture_id": metadata.get("architecture_id"),
            "input_channels": metadata.get("observation_shape", [None])[0],
            "observation_shape": metadata.get("observation_shape"),
            "model_hash": metadata.get("converted_model_hash"),
            "source_checkpoint": metadata.get("source_checkpoint"),
            "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
        },
        self_play={"komi": 0.5},
        training={"training_ready": False},
        replay={"policy": NEW_KOMI_REPLAY_POLICY},
        execution={},
        arena={"model_hash": metadata.get("converted_model_hash")},
        extensions={
            "kind": "m137-6ch-to-5ch-inference-bootstrap",
            "conversion_formula": metadata.get("conversion_formula"),
            "optimizer_conversion": metadata.get("optimizer_conversion"),
        },
    )


def publish_new_komi_bootstrap_graph(root: str | Path) -> dict[str, object]:
    """Publish the canonical V2 genesis reference for the existing 5CH model.

    The bootstrap checkpoint is already the lineage-owned converted artifact.
    This function only adds immutable graph/config/provenance references and
    catalog evidence; it never copies or rewrites model weights.
    """

    lineage_root = Path(root).resolve()
    manifest_path = lineage_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("new_komi manifest must be an object")
    if manifest.get("lineage_id") != NEW_KOMI_LINEAGE_ID or manifest.get("topology") != "torus9":
        raise ValueError("bootstrap graph owner is not the canonical new_komi Torus9 lineage")
    if manifest.get("status") != "ACTIVE":
        raise ValueError("cannot publish a bootstrap graph for an inactive lineage")

    bootstrap = manifest.get("bootstrap_checkpoint")
    parent = manifest.get("parent_checkpoint")
    if not isinstance(bootstrap, Mapping) or not isinstance(parent, Mapping):
        raise ValueError("new_komi manifest lacks bootstrap/source checkpoint evidence")
    checkpoint_rel = str(bootstrap.get("path", ""))
    checkpoint_path = (lineage_root / checkpoint_rel).resolve()
    if checkpoint_rel != "checkpoints/M137-5CH-bootstrap.pt" or not checkpoint_path.is_file():
        raise ValueError("canonical M137 5CH bootstrap checkpoint is missing")
    if str(parent.get("checkpoint_id")) != "M137" or str(parent.get("sha256")) != (
        "sha256:" + NEW_KOMI_SOURCE_CHECKPOINT_SHA256
    ):
        raise ValueError("new_komi source parent is not the canonical M137 identity")

    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("M137 5CH checkpoint metadata must be an object")
    architecture = metadata.get("architecture_config")
    if (
        metadata.get("architecture_id") != M137_FIVE_CHANNEL_ARCHITECTURE_ID
        or metadata.get("converted_model_hash") != NEW_KOMI_CONVERTED_MODEL_HASH
        or metadata.get("source_checkpoint") != "M137"
        or metadata.get("source_checkpoint_sha256")
        != "sha256:" + NEW_KOMI_SOURCE_CHECKPOINT_SHA256
        or metadata.get("observation_shape") != [5, TORUS9_POINT_COUNT]
        or not isinstance(architecture, Mapping)
        or architecture.get("input_channels") != 5
        or architecture.get("blocks") != 8
        or architecture.get("hidden") != 80
        or architecture.get("topology_fingerprint") != TORUS9_TOPOLOGY_FINGERPRINT
    ):
        raise ValueError("M137 5CH checkpoint metadata does not match the canonical identity")
    artifact_sha = file_sha256(checkpoint_path)
    declared_sha = str(bootstrap.get("sha256", ""))
    if declared_sha != artifact_sha:
        raise ValueError("new_komi manifest/bootstrap checkpoint SHA mismatch")

    checkpoint = CheckpointRef(
        topology="torus9",
        lineage_id=NEW_KOMI_LINEAGE_ID,
        checkpoint_id="M137-5CH-bootstrap",
        generation=137,
        path=checkpoint_rel,
        sha256=artifact_sha,
    )
    effective = _bootstrap_effective_config(metadata)
    config_rel = f"metadata/effective-config-v2/{effective.fingerprint}.json"
    config_path = lineage_root / config_rel
    _write_immutable_json(config_path, effective.to_dict())
    config_ref = EffectiveConfigRef(
        artifact=ArtifactRef(config_rel, sha256_file(config_path)),
        fingerprint=effective.fingerprint,
    )

    provenance_rel = "metadata/provenance-v2/M137-5CH-bootstrap.json"
    provenance_path = lineage_root / provenance_rel
    provenance_payload = {
        "schema": "gocube-orchestrator-v2-m137-5ch-bootstrap-provenance-v1",
        "version": 1,
        "genesis": True,
        "checkpoint": checkpoint.to_dict(),
        "source_checkpoint": dict(parent),
        "checkpoint_metadata": {
            "path": metadata_path.relative_to(lineage_root).as_posix(),
            "sha256": file_sha256(metadata_path),
        },
        "converted_model_hash": NEW_KOMI_CONVERTED_MODEL_HASH,
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "observation_channels": list(M137_FIVE_CHANNEL_CHANNELS),
        "observation_shape": [5, TORUS9_POINT_COUNT],
        "conversion_formula": metadata.get("conversion_formula"),
        "optimizer_conversion": metadata.get("optimizer_conversion"),
        "effective_config": config_ref.to_dict(),
        "weights_copied": False,
    }
    _write_immutable_json(provenance_path, provenance_payload)
    provenance_ref = ArtifactRef(provenance_rel, sha256_file(provenance_path))
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=True,
        parent=None,
        fresh_replay=None,
        effective_config=config_ref,
        provenance=provenance_ref,
    )
    node_rel = "metadata/checkpoints/M137-5CH-bootstrap.json"
    node_path = lineage_root / node_rel
    _write_immutable_json(node_path, node.to_dict())

    catalog_path = lineage_root / "runtime" / "artifact-catalog.json"
    catalog = ArtifactCatalog.load(catalog_path, root=lineage_root)
    catalog.register_generation(
        137,
        (
            {
                "path": checkpoint_rel,
                "sha256": artifact_sha,
                "size_bytes": checkpoint_path.stat().st_size,
                "kind": "checkpoint",
                "model_hash": NEW_KOMI_CONVERTED_MODEL_HASH,
            },
            {
                "path": config_rel,
                "sha256": config_ref.artifact.sha256,
                "size_bytes": config_path.stat().st_size,
                "kind": "effective_config",
            },
            {
                "path": provenance_rel,
                "sha256": provenance_ref.sha256,
                "size_bytes": provenance_path.stat().st_size,
                "kind": "provenance",
            },
            {
                "path": node_rel,
                "sha256": file_sha256(node_path),
                "size_bytes": node_path.stat().st_size,
                "kind": "checkpoint_node",
            },
        ),
    )

    updated_manifest = dict(manifest)
    checkpoint_hashes = dict(updated_manifest.get("checkpoint_hashes", {}))
    checkpoint_hashes[checkpoint_rel] = artifact_sha
    updated_manifest["checkpoint_hashes"] = checkpoint_hashes
    updated_manifest["effective_config"] = config_ref.to_dict()
    updated_manifest["v2_bootstrap_checkpoint"] = {
        "checkpoint": checkpoint.to_dict(),
        "node": {"path": node_rel, "sha256": file_sha256(node_path)},
        "provenance": provenance_ref.to_dict(),
        "effective_config": config_ref.to_dict(),
        "converted_model_hash": NEW_KOMI_CONVERTED_MODEL_HASH,
    }
    atomic_write_text(
        manifest_path,
        json.dumps(updated_manifest, indent=2, sort_keys=True) + "\n",
    )
    return updated_manifest["v2_bootstrap_checkpoint"]  # type: ignore[return-value]


def _sha_body(value: object) -> str:
    text = str(value).strip().lower()
    return text.removeprefix("sha256:")


def _clone_state_value(value: object) -> object:
    return value.detach().clone() if torch.is_tensor(value) else deepcopy(value)


def _adam_step_value(state: Mapping[str, object]) -> int:
    raw = state.get("step")
    if raw is None:
        return 0
    if torch.is_tensor(raw):
        return int(raw.item())
    return int(raw)


def migrate_m137_adam(
    *,
    source_model: Torus9CurrentGraphNet,
    target_model: Torus9M137FiveChannelGraphNet,
    source_optimizer_state: Mapping[str, object],
) -> tuple[torch.optim.Adam, dict[str, object]]:
    """Migrate M137 Adam state onto the folded 5-channel model.

    Exact mappings:
    - every unchanged parameter: state copied bit-for-bit;
    - ``input_projection.weight``: tensor moments are cropped from 6 to 5
      columns, matching the model-weight conversion.

    Non-exact mapping:
    - ``input_projection.bias`` now represents ``b + 0.5*W6``.  There is no
      exact Adam second-moment transform for the merged coordinate, so its
      first/second moments are zeroed.  The global Adam step is retained so
      optimizer continuity remains valid for the rest of the model.
    """
    source_names = [name for name, _ in source_model.named_parameters()]
    target_names = [name for name, _ in target_model.named_parameters()]
    if source_names != target_names:
        raise ValueError("M137 6->5 Adam migration requires identical parameter names/order")

    raw_groups = source_optimizer_state.get("param_groups")
    raw_state = source_optimizer_state.get("state")
    if not isinstance(raw_groups, list) or len(raw_groups) != 1:
        raise ValueError("M137 Adam migration requires exactly one optimizer parameter group")
    if not isinstance(raw_state, Mapping):
        raise ValueError("M137 checkpoint has no Adam state mapping")

    source_group = dict(raw_groups[0])
    source_param_ids = list(source_group.get("params", ()))
    if len(source_param_ids) != len(source_names):
        raise ValueError("M137 Adam parameter order disagrees with model parameter order")

    betas_value = source_group.get("betas", (0.9, 0.999))
    if not isinstance(betas_value, (tuple, list)) or len(betas_value) != 2:
        raise ValueError("M137 Adam betas are malformed")
    target_optimizer = torch.optim.Adam(
        target_model.parameters(),
        lr=float(source_group.get("lr", 1e-4)),
        betas=(float(betas_value[0]), float(betas_value[1])),
        eps=float(source_group.get("eps", 1e-8)),
        weight_decay=float(source_group.get("weight_decay", 0.0)),
        amsgrad=bool(source_group.get("amsgrad", False)),
    )
    target_state_dict = target_optimizer.state_dict()
    target_group = dict(target_state_dict["param_groups"][0])
    target_param_ids = list(target_group["params"])

    source_parameters = dict(source_model.named_parameters())
    target_parameters = dict(target_model.named_parameters())
    migrated_state: dict[object, dict[str, object]] = {}
    observed_steps: set[int] = set()
    exact_parameters: list[str] = []
    cropped_parameters: list[str] = []
    reset_parameters: list[str] = []

    for name, source_id, target_id in zip(source_names, source_param_ids, target_param_ids):
        source_parameter = source_parameters[name]
        target_parameter = target_parameters[name]
        source_parameter_state = raw_state.get(source_id)
        if not isinstance(source_parameter_state, Mapping):
            raise ValueError(f"M137 Adam state missing parameter {name}")
        source_step = _adam_step_value(source_parameter_state)
        if source_step:
            observed_steps.add(source_step)
        converted: dict[str, object] = {}

        if name == "input_projection.weight":
            if tuple(source_parameter.shape) != (target_parameter.shape[0], target_parameter.shape[1] + 1):
                raise ValueError("M137 input-projection Adam source/target shape mismatch")
            for key, value in source_parameter_state.items():
                if key == "step":
                    converted[key] = _clone_state_value(value)
                elif torch.is_tensor(value) and tuple(value.shape) == tuple(source_parameter.shape):
                    converted[key] = value[:, : target_parameter.shape[1]].detach().clone()
                elif torch.is_tensor(value) and value.ndim == 0:
                    converted[key] = value.detach().clone()
                else:
                    converted[key] = deepcopy(value)
            cropped_parameters.append(name)
        elif name == "input_projection.bias":
            if tuple(source_parameter.shape) != tuple(target_parameter.shape):
                raise ValueError("M137 folded-bias Adam source/target shape mismatch")
            for key, value in source_parameter_state.items():
                if key == "step":
                    converted[key] = _clone_state_value(value)
                elif torch.is_tensor(value) and tuple(value.shape) == tuple(source_parameter.shape):
                    converted[key] = torch.zeros_like(target_parameter.detach(), device=value.device)
                elif torch.is_tensor(value) and value.ndim == 0:
                    converted[key] = value.detach().clone()
                else:
                    converted[key] = deepcopy(value)
            reset_parameters.append(name)
        else:
            if tuple(source_parameter.shape) != tuple(target_parameter.shape):
                raise ValueError(f"M137 Adam state shape changed unexpectedly for {name}")
            for key, value in source_parameter_state.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and tuple(value.shape) != tuple(target_parameter.shape)
                ):
                    raise ValueError(f"M137 Adam tensor state shape mismatch for {name}.{key}")
                converted[key] = _clone_state_value(value)
            exact_parameters.append(name)

        migrated_state[target_id] = converted

    if len(observed_steps) > 1:
        raise ValueError(f"M137 Adam state has discontinuous steps: {sorted(observed_steps)}")
    adam_step = next(iter(observed_steps), 0)

    migrated_group = deepcopy(source_group)
    migrated_group["params"] = target_param_ids
    target_optimizer.load_state_dict(
        {"state": migrated_state, "param_groups": [migrated_group]}
    )

    report = {
        "schema": "torus9-m137-5ch-adam-migration-v1",
        "conversion": NEW_KOMI_OPTIMIZER_CONVERSION,
        "adam_step": adam_step,
        "exact_parameter_states": exact_parameters,
        "cropped_parameter_states": cropped_parameters,
        "reset_parameter_moments": reset_parameters,
        "reset_policy": (
            "input_projection.bias moments zeroed; global Adam step preserved"
        ),
        "source_parameter_count": len(source_names),
        "target_parameter_count": len(target_names),
    }
    return target_optimizer, report


def build_training_ready_checkpoint(
    *,
    source_checkpoint: str | Path,
    destination: str | Path,
    converter_git_commit: str,
) -> dict[str, object]:
    """Create the lineage-owned 5CH optimizer-ready bootstrap checkpoint."""
    source_path = Path(source_checkpoint).resolve()
    actual_sha = file_sha256(source_path)
    if _sha_body(actual_sha) != NEW_KOMI_SOURCE_CHECKPOINT_SHA256:
        raise ValueError(
            "new_komi requires canonical M137 checkpoint SHA-256 "
            f"{NEW_KOMI_SOURCE_CHECKPOINT_SHA256}; got {actual_sha}"
        )

    source_model, source_metadata = load_canonical_m137(source_path, device="cpu")
    raw = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("Canonical M137 checkpoint payload must be an object")
    source_optimizer_state = raw.get("optimizer_state_dict")
    if not isinstance(source_optimizer_state, Mapping):
        raise ValueError("Canonical M137 checkpoint has no resumable Adam state")

    target_model = convert_m137_model(source_model)
    target_optimizer, migration = migrate_m137_adam(
        source_model=source_model,
        target_model=target_model,
        source_optimizer_state=source_optimizer_state,
    )
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "checkpoint_schema_version": 1,
        "checkpoint_label": "M137-5CH-bootstrap",
        "lineage_id": NEW_KOMI_LINEAGE_ID,
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "architecture_config": {
            **target_model.architecture_config,
            "training_ready": False,
        },
        "observation_shape": [5, TORUS9_POINT_COUNT],
        "source_lineage": NEW_KOMI_SOURCE_LINEAGE,
        "source_checkpoint": NEW_KOMI_SOURCE_CHECKPOINT,
        "source_checkpoint_sha256": actual_sha,
        "source_model_hash": source_metadata.get("model_hash"),
        "converted_model_hash": model_hash(target_model),
        "converter_git_commit": str(converter_git_commit),
        "conversion_formula": M137_FIVE_CHANNEL_FORMULA,
        "optimizer_conversion": NEW_KOMI_OPTIMIZER_CONVERSION,
        "optimizer_migration": migration,
        "optimizer_migration_ready": True,
        "training_ready": False,
        "replay_bootstrap": NEW_KOMI_REPLAY_POLICY,
        "replay_references": [],
        "selected_komi": None,
        "source_training_komi": 0.5,
        "status": "BLOCKED_PENDING_KOMI_CALIBRATION",
    }
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "metadata": metadata,
            "model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": target_optimizer.state_dict(),
        },
        destination_path,
    )
    metadata["artifact_sha256"] = file_sha256(destination_path)
    destination_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def create_new_komi_lineage(
    *,
    source_checkpoint: str | Path,
    runs_root: str | Path,
    converter_git_commit: str,
) -> tuple[Path, dict[str, object]]:
    """Create ``runs/torus9/active/new_komi`` with no inherited replay."""
    source_path = Path(source_checkpoint).resolve()
    root = Path(runs_root).resolve() / "torus9" / ACTIVE / NEW_KOMI_LINEAGE_ID
    if root.exists():
        raise FileExistsError(f"new_komi lineage already exists: {root}")

    source_sha = file_sha256(source_path)
    manifest: dict[str, object] = {
        "schema": "gocube-new-komi-lineage-v1",
        "lineage_id": NEW_KOMI_LINEAGE_ID,
        "topology": "torus9",
        "status": "ACTIVE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": str(converter_git_commit),
        "config_fingerprint": "bootstrap-pending-komi-calibration",
        "checkpoint_hashes": {},
        "parent_checkpoint": {
            "lineage_id": NEW_KOMI_SOURCE_LINEAGE,
            "checkpoint_id": NEW_KOMI_SOURCE_CHECKPOINT,
            "generation": 137,
            "path": str(source_path),
            "sha256": source_sha,
        },
        "bootstrap": {
            "architecture": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
            "optimizer_conversion": NEW_KOMI_OPTIMIZER_CONVERSION,
            "replay_policy": NEW_KOMI_REPLAY_POLICY,
            "selected_komi": None,
            "training_status": "BLOCKED_PENDING_KOMI_CALIBRATION",
        },
        "replay_references": [],
    }
    ensure_lineage_layout(
        root,
        manifest=manifest,
        extra_directories=("metadata", "runtime", "replay", "selfplay", "training"),
        include_data=True,
    )
    catalog_path = root / "runtime" / "artifact-catalog.json"
    ArtifactCatalog.initialize(catalog_path, lineage_id=NEW_KOMI_LINEAGE_ID, root=root)

    checkpoint_path = root / "checkpoints" / "M137-5CH-bootstrap.pt"
    try:
        metadata = build_training_ready_checkpoint(
            source_checkpoint=source_path,
            destination=checkpoint_path,
            converter_git_commit=converter_git_commit,
        )
        manifest["bootstrap_checkpoint"] = {
            "path": checkpoint_path.relative_to(root).as_posix(),
            "sha256": metadata["artifact_sha256"],
            "model_hash": metadata["converted_model_hash"],
        }
        manifest["checkpoint_hashes"] = {
            "M137-5CH-bootstrap": metadata["artifact_sha256"]
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_new_komi_bootstrap_graph(root)
    except Exception:
        manifest["status"] = "DISCARDED"
        manifest["discard_reason"] = "bootstrap failed before training-ready checkpoint publication"
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    return root, metadata


__all__ = [
    "NEW_KOMI_LINEAGE_ID",
    "NEW_KOMI_OPTIMIZER_CONVERSION",
    "NEW_KOMI_REPLAY_POLICY",
    "NEW_KOMI_CONVERTED_MODEL_HASH",
    "NEW_KOMI_SOURCE_CHECKPOINT",
    "NEW_KOMI_SOURCE_CHECKPOINT_SHA256",
    "NEW_KOMI_SOURCE_LINEAGE",
    "build_training_ready_checkpoint",
    "create_new_komi_lineage",
    "migrate_m137_adam",
    "publish_new_komi_bootstrap_graph",
]
