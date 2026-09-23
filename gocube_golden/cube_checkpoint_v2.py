"""Strict Cube V2 training checkpoint serialization and compatibility checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from .cube_network_v2 import (
    ARCHITECTURE_FINGERPRINT,
    ARCHITECTURE_ID,
    CubeGraphNetV2,
    build_cube_model_from_metadata,
    cube_graphnet_v2_model_hash,
    validate_cube_model_metadata,
)
from .cube_selfplay_contract import (
    CUBE_SELFPLAY_SEMANTICS_FINGERPRINT,
    CUBE_TARGET_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
)
from .cube_training_contract_v2 import CHECKPOINT_SCHEMA, OPTIMIZER_FAMILY, REPLAY_SCHEMA, TRAINING_CONTRACT_ID


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sidecar_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".metadata.json")


def _cpu_clone(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return deepcopy(value)


def validate_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    expected_size: int,
    game_fingerprint: str,
    observation_fingerprint: str,
    training_semantics_fingerprint: str,
    concrete_training_config_fingerprint: str,
    effective_learning_rate: float,
) -> None:
    if metadata.get("checkpoint_schema") != CHECKPOINT_SCHEMA or metadata.get("checkpoint_schema_version") != 2:
        raise ValueError("Cube checkpoint schema is incompatible")
    if int(metadata.get("size", -1)) != int(expected_size):
        raise ValueError("Cube checkpoint topology size mismatch")
    if metadata.get("game_fingerprint") != game_fingerprint:
        raise ValueError("Cube checkpoint game fingerprint mismatch")
    if metadata.get("observation_fingerprint") != observation_fingerprint:
        raise ValueError("Cube checkpoint observation fingerprint mismatch")
    if metadata.get("architecture_id") != ARCHITECTURE_ID or metadata.get("architecture_fingerprint") != ARCHITECTURE_FINGERPRINT:
        raise ValueError("Cube checkpoint architecture fingerprint mismatch")
    model_metadata = metadata.get("model_metadata")
    if not isinstance(model_metadata, Mapping):
        raise ValueError("Cube checkpoint model metadata is missing")
    validate_cube_model_metadata(model_metadata)
    if int(model_metadata["size"]) != int(expected_size):
        raise ValueError("Cube checkpoint model topology mismatch")
    if metadata.get("target_contract_id") != CUBE_TARGET_CONTRACT_ID or metadata.get("target_contract_fingerprint") != CUBE_TARGET_FINGERPRINT:
        raise ValueError("Cube checkpoint target contract mismatch")
    if metadata.get("training_contract_id") != TRAINING_CONTRACT_ID or metadata.get("training_semantics_fingerprint") != training_semantics_fingerprint:
        raise ValueError("Cube checkpoint training contract mismatch")
    if metadata.get("concrete_training_config_fingerprint") != concrete_training_config_fingerprint:
        raise ValueError("Cube checkpoint concrete training config mismatch")
    if metadata.get("optimizer_type") != OPTIMIZER_FAMILY:
        raise ValueError("Cube checkpoint optimizer type mismatch")
    if metadata.get("replay_schema") != REPLAY_SCHEMA:
        raise ValueError("Cube checkpoint replay schema mismatch")
    if metadata.get("selfplay_semantics_fingerprint") != CUBE_SELFPLAY_SEMANTICS_FINGERPRINT:
        raise ValueError("Cube checkpoint self-play semantics mismatch")
    try:
        lr = float(metadata["effective_learning_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cube checkpoint learning rate is invalid") from exc
    if not math.isfinite(lr) or lr <= 0.0 or not math.isclose(lr, float(effective_learning_rate), rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Cube checkpoint learning rate/config mismatch")


def save_checkpoint(
    path: str | Path,
    *,
    model: CubeGraphNetV2,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, object],
    runtime_state: Mapping[str, object],
    optimizer_updates: int,
    samples_consumed: int,
) -> dict[str, object]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "metadata": deepcopy(dict(metadata)),
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "optimizer_state_dict": _cpu_clone(optimizer.state_dict()),
        "runtime_state": deepcopy(dict(runtime_state)),
        "optimizer_updates": int(optimizer_updates),
        "samples_consumed": int(samples_consumed),
        "generation": int(metadata["generation"]),
    }
    torch.save(payload, target)
    enriched = dict(metadata)
    enriched["checkpoint_sha256"] = file_sha256(target)
    sidecar_path(target).write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return enriched


def load_payload(
    path: str | Path,
    *,
    map_location: str | torch.device,
    expected_size: int,
    game_fingerprint: str,
    observation_fingerprint: str,
    training_semantics_fingerprint: str,
    concrete_training_config_fingerprint: str,
    effective_learning_rate: float,
) -> tuple[dict[str, object], dict[str, object]]:
    target = Path(path)
    metadata_path = sidecar_path(target)
    if not metadata_path.is_file():
        raise ValueError("Cube checkpoint metadata sidecar is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Cube checkpoint metadata sidecar is invalid")
    validate_checkpoint_metadata(
        metadata,
        expected_size=expected_size,
        game_fingerprint=game_fingerprint,
        observation_fingerprint=observation_fingerprint,
        training_semantics_fingerprint=training_semantics_fingerprint,
        concrete_training_config_fingerprint=concrete_training_config_fingerprint,
        effective_learning_rate=effective_learning_rate,
    )
    if metadata.get("checkpoint_sha256") != file_sha256(target):
        raise ValueError("Cube checkpoint SHA-256 mismatch")
    payload = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Cube checkpoint payload schema mismatch")
    payload_metadata = payload.get("metadata")
    if not isinstance(payload_metadata, Mapping):
        raise ValueError("Cube checkpoint payload metadata is missing")
    semantic_sidecar = {key: value for key, value in metadata.items() if key != "checkpoint_sha256"}
    if dict(payload_metadata) != semantic_sidecar:
        raise ValueError("Cube checkpoint payload/sidecar metadata mismatch")
    for key in ("model_state_dict", "optimizer_state_dict", "runtime_state"):
        if key not in payload:
            raise ValueError(f"Cube checkpoint payload is missing {key}")
    return payload, metadata


def restore_model_optimizer(
    payload: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    map_location: str | torch.device,
) -> tuple[CubeGraphNetV2, torch.optim.Adam]:
    model_metadata = metadata["model_metadata"]
    if not isinstance(model_metadata, Mapping):
        raise ValueError("Cube checkpoint model metadata is missing")
    model = build_cube_model_from_metadata(model_metadata)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(map_location)
    if cube_graphnet_v2_model_hash(model) != metadata.get("model_hash"):
        raise ValueError("Cube checkpoint strict reload model hash mismatch")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(metadata["effective_learning_rate"]), weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, optimizer


__all__ = [
    "file_sha256",
    "load_payload",
    "restore_model_optimizer",
    "save_checkpoint",
    "sidecar_path",
    "validate_checkpoint_metadata",
]
