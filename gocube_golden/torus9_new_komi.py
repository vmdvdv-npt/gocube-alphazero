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

from .artifact_catalog import ArtifactCatalog
from .neural import model_hash
from .provenance import file_sha256
from .run_storage import ACTIVE, ensure_lineage_layout
from .torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    M137_FIVE_CHANNEL_FORMULA,
    Torus9M137FiveChannelGraphNet,
    convert_m137_model,
    load_canonical_m137,
)
from .torus9_monolith import Torus9CurrentGraphNet
from .torus9_contract import TORUS9_POINT_COUNT


NEW_KOMI_LINEAGE_ID = "new_komi"
NEW_KOMI_SOURCE_LINEAGE = "torus9-m125-continuous-v2-gen6-20260922-v1"
NEW_KOMI_SOURCE_CHECKPOINT = "M137"
NEW_KOMI_SOURCE_CHECKPOINT_SHA256 = (
    "71cfc78dab3fe217b3c435a765790efe6f6c4fa42a7d21479f3fd909adf341fe"
)
NEW_KOMI_OPTIMIZER_CONVERSION = (
    "adam-preserve-exact-crop-input-reset-folded-bias-moments-v1"
)
NEW_KOMI_REPLAY_POLICY = "fresh-only-no-parent-history"


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
    "NEW_KOMI_SOURCE_CHECKPOINT",
    "NEW_KOMI_SOURCE_CHECKPOINT_SHA256",
    "NEW_KOMI_SOURCE_LINEAGE",
    "build_training_ready_checkpoint",
    "create_new_komi_lineage",
    "migrate_m137_adam",
]
