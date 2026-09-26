"""Strict Cube V2 training checkpoint serialization and compatibility checks."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CubeConfigTransition:
    """Auditable result of an explicit run-owned config transition."""

    schema: str
    source_checkpoint: Mapping[str, object]
    source_config_fingerprint: str
    target_config_fingerprint: str
    changed_fields: Mapping[str, Mapping[str, object]]
    optimizer_state_preserved: bool
    source_replay_fingerprint: str
    source_replay_count: int
    target_replay_fingerprint: str
    target_replay_count: int
    evicted_generation_count: int
    evicted_position_count: int
    target_replay_generations: int
    target_replay_cap: int | None
    effective_generation: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_checkpoint": dict(self.source_checkpoint),
            "source_config_fingerprint": self.source_config_fingerprint,
            "target_config_fingerprint": self.target_config_fingerprint,
            "changed_fields": {
                str(key): dict(value) for key, value in self.changed_fields.items()
            },
            "optimizer_state_preserved": bool(self.optimizer_state_preserved),
            "source_replay_fingerprint": self.source_replay_fingerprint,
            "source_replay_count": int(self.source_replay_count),
            "target_replay_fingerprint": self.target_replay_fingerprint,
            "target_replay_count": int(self.target_replay_count),
            "replay_evictions": {
                "generation_count": int(self.evicted_generation_count),
                "position_count": int(self.evicted_position_count),
            },
            "target_replay_generations": int(self.target_replay_generations),
            "target_replay_cap": self.target_replay_cap,
            "effective_generation": int(self.effective_generation),
        }


def validate_immutable_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    expected_size: int,
    game_fingerprint: str,
    observation_fingerprint: str,
    training_semantics_fingerprint: str,
) -> None:
    """Validate compatibility that an explicit config transition may not change."""

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
    if metadata.get("optimizer_type") != OPTIMIZER_FAMILY:
        raise ValueError("Cube checkpoint optimizer type mismatch")
    if metadata.get("weight_decay", 0.0) != 0.0:
        raise ValueError("Cube checkpoint weight-decay semantics mismatch")
    if metadata.get("replay_schema") != REPLAY_SCHEMA:
        raise ValueError("Cube checkpoint replay schema mismatch")
    if metadata.get("selfplay_semantics_fingerprint") != CUBE_SELFPLAY_SEMANTICS_FINGERPRINT:
        raise ValueError("Cube checkpoint self-play semantics mismatch")
    for key in ("model_hash", "replay_fingerprint"):
        if not isinstance(metadata.get(key), str) or not str(metadata[key]):
            raise ValueError(f"Cube checkpoint is missing {key}")


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
    validate_immutable_checkpoint_metadata(
        metadata,
        expected_size=expected_size,
        game_fingerprint=game_fingerprint,
        observation_fingerprint=observation_fingerprint,
        training_semantics_fingerprint=training_semantics_fingerprint,
    )
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
    _validate_adam_state(
        optimizer,
        updates=int(payload.get("optimizer_updates", 0)),
    )
    return model, optimizer


def _validate_adam_state(optimizer: torch.optim.Adam, *, updates: int) -> None:
    """Reject partial, malformed, non-finite, or non-Adam state after restore."""

    if updates < 0:
        raise ValueError("Cube checkpoint optimizer update counter is invalid")

    trainable_parameters = tuple(
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
        if parameter.requires_grad
    )
    if updates > 0:
        missing = sum(
            parameter not in optimizer.state for parameter in trainable_parameters
        )
        if missing:
            raise ValueError(
                "Cube checkpoint Adam state is missing for trainable parameter(s)"
            )

    populated = [bool(value) for value in optimizer.state.values()]
    if updates > 0 and not populated:
        raise ValueError("Cube checkpoint Adam state is missing")
    if populated and not all(populated):
        raise ValueError("Cube checkpoint Adam state is partially populated")
    for parameter, state in optimizer.state.items():
        if not state:
            continue
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not required.issubset(state):
            raise ValueError("Cube checkpoint Adam moments are incomplete")
        step = state["step"]
        try:
            if torch.is_tensor(step):
                if step.numel() != 1:
                    raise ValueError
                step_value = float(step.detach().cpu().item())
            else:
                step_value = float(step)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Cube checkpoint Adam step is invalid") from exc
        if not math.isfinite(step_value) or step_value < 0:
            raise ValueError("Cube checkpoint Adam step is invalid")
        if step_value != float(updates):
            raise ValueError(
                "Cube checkpoint Adam step does not match optimizer update counter"
            )
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Cube checkpoint Adam {name} shape is invalid")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Cube checkpoint Adam {name} contains NaN/Inf")


__all__ = [
    "CubeConfigTransition",
    "file_sha256",
    "load_payload",
    "restore_model_optimizer",
    "save_checkpoint",
    "sidecar_path",
    "validate_checkpoint_metadata",
    "validate_immutable_checkpoint_metadata",
]
