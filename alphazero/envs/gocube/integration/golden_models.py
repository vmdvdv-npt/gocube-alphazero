"""Golden checkpoint loading for the GoCube integration boundary.

This module has no dependency on the legacy AlphaZero wrapper or player
stack.  It accepts only the two currently registered Golden profiles and
returns a small playable-model object carrying the exact evaluator and
metadata needed by the Golden generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from .catalog import CheckpointCatalog, CheckpointDescriptor
from .errors import CheckpointLoadFailed, CheckpointMetadataInvalid


GOLDEN_BACKEND_KIND = "golden"
GOLDEN_CHECKPOINT_FORMAT = "golden_pt"
DEVICE_CHOICES = ("auto", "cpu", "cuda")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def resolve_device(device: str) -> str:
    if device not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported device {device!r}; expected one of {DEVICE_CHOICES}")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _metadata_value(metadata: Mapping[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def _require(metadata: Mapping[str, object], key: str) -> object:
    if key not in metadata:
        raise CheckpointMetadataInvalid(f"Golden checkpoint metadata is missing {key}")
    return metadata[key]


def _read_sidecar(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointMetadataInvalid(f"Golden checkpoint metadata sidecar is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointMetadataInvalid(f"Invalid Golden checkpoint metadata sidecar {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointMetadataInvalid("Golden checkpoint metadata sidecar must be an object")
    return value


def _load_payload(path: Path) -> Mapping[str, object]:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except FileNotFoundError as exc:
        raise CheckpointLoadFailed(f"Golden checkpoint file is missing: {path}") from exc
    except Exception as exc:
        raise CheckpointLoadFailed(f"Cannot read Golden checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointMetadataInvalid("Golden checkpoint payload must be an object")
    return payload


def _compare_embedded_metadata(sidecar: Mapping[str, object], payload: Mapping[str, object]) -> dict[str, object]:
    embedded = payload.get("metadata")
    if embedded is None:
        raise CheckpointMetadataInvalid("Golden checkpoint payload is missing embedded metadata")
    if not isinstance(embedded, Mapping):
        raise CheckpointMetadataInvalid("Golden checkpoint embedded metadata must be an object")
    # Cube M0 was published with an artifact-only sidecar enrichment.  It is
    # safe to allow sidecar keys absent from the embedded object, but every
    # embedded value must agree exactly with the sidecar.
    for key, value in embedded.items():
        if key not in sidecar or sidecar[key] != value:
            raise CheckpointMetadataInvalid(
                f"Golden checkpoint sidecar disagrees with embedded metadata for {key}"
            )
    return dict(embedded)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CheckpointMetadataInvalid(f"Golden checkpoint {field} is malformed")
    return value


def _validate_torus9_metadata(metadata: Mapping[str, object]) -> None:
    from gocube_golden.torus9_contract import (
        TORUS9_ACTION_COUNT,
        TORUS9_CURRENT_ARCHITECTURE_ID,
        TORUS9_CURRENT_PROFILE_FINGERPRINT,
        TORUS9_CURRENT_PROFILE_ID,
        TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        TORUS9_CURRENT_TARGET_FINGERPRINT,
        TORUS9_KOMI,
        TORUS9_OBSERVATION_FINGERPRINT,
        TORUS9_OBSERVATION_SCHEMA_ID,
        TORUS9_OBSERVATION_SCHEMA_VERSION,
        TORUS9_POINT_COUNT,
        TORUS9_RULES_FINGERPRINT,
        TORUS9_TARGET_CONTRACT_ID,
        current_torus9_selfplay_contract_fingerprint,
    )
    from gocube_golden.torus9 import TORUS9_TOPOLOGY_ID, TORUS9_TOPOLOGY_FINGERPRINT

    exact = {
        "checkpoint_schema_version": 1,
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": TORUS9_CURRENT_PROFILE_FINGERPRINT,
        "architecture_id": TORUS9_CURRENT_ARCHITECTURE_ID,
        "topology_id": TORUS9_TOPOLOGY_ID,
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "board_size": [9, 9],
        "point_id_order_identity": "row-major-yx:point_id=y*width+x",
        "komi": TORUS9_KOMI,
        "observation_schema_id": TORUS9_OBSERVATION_SCHEMA_ID,
        "observation_schema_version": TORUS9_OBSERVATION_SCHEMA_VERSION,
        "observation_fingerprint": TORUS9_OBSERVATION_FINGERPRINT,
        "observation_shape": [6, TORUS9_POINT_COUNT],
        "target_contract_id": TORUS9_TARGET_CONTRACT_ID,
        "target_contract_version": 1,
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "rules_profile_id": "graph-area-v1",
        "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
        "network_heads_and_shapes": {
            "policy": [TORUS9_ACTION_COUNT],
            "value": [3],
            "ownership": [TORUS9_POINT_COUNT, 3],
            "score": [1],
        },
        "auxiliary_heads": True,
        "selfplay_contract_id": TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        "selfplay_contract_fingerprint": current_torus9_selfplay_contract_fingerprint(),
    }
    for key, expected in exact.items():
        actual = _require(metadata, key)
        if actual != expected:
            raise CheckpointMetadataInvalid(
                f"Golden Torus9 metadata mismatch for {key}: saved={actual!r}, expected={expected!r}"
            )
    if _require(metadata, "model_parameter_count") <= 0:
        raise CheckpointMetadataInvalid("Golden Torus9 model parameter count must be positive")
    _validate_hash(_require(metadata, "model_hash"), "model_hash")


def _validate_cube_metadata(metadata: Mapping[str, object]) -> None:
    from gocube_golden.cube_contract import CUBE_PROFILE_ID, load_profile
    from gocube_golden.cube_neural import (
        CUBE_ACTION_COUNT,
        CUBE_OBSERVATION_FINGERPRINT,
        CUBE_OBSERVATION_SCHEMA_ID,
    )
    from gocube_golden.cube_topology import (
        CUBE4_GEOMETRY_FINGERPRINT,
        CUBE4_TOPOLOGY_FINGERPRINT,
        CUBE4_TOPOLOGY_ID,
    )
    from gocube_golden.cube_training import CUBE_TARGET_CONTRACT_ID, CUBE_TARGET_FINGERPRINT

    profile = load_profile()
    profile_fingerprint = profile.get("profile_fingerprint")
    exact = {
        "checkpoint_schema_version": 1,
        "architecture_id": "GoldenCubeGraphNetV1",
        "topology_id": CUBE4_TOPOLOGY_ID,
        "topology_fingerprint": CUBE4_TOPOLOGY_FINGERPRINT,
        "geometry_fingerprint": CUBE4_GEOMETRY_FINGERPRINT,
        "point_count": 96,
        "action_count": CUBE_ACTION_COUNT,
        "point_ordering_fingerprint": profile["topology"]["point_ordering_fingerprint"],
        "rules_id": "graph-area-v1",
        "rules_fingerprint": profile["rules"]["fingerprint"],
        "komi": 0.5,
        "observation_schema_id": CUBE_OBSERVATION_SCHEMA_ID,
        "observation_fingerprint": CUBE_OBSERVATION_FINGERPRINT,
        "target_contract_id": CUBE_TARGET_CONTRACT_ID,
        "target_fingerprint": CUBE_TARGET_FINGERPRINT,
        "network_heads_and_shapes": {"policy": [CUBE_ACTION_COUNT], "value": [3]},
        "training_profile_id": CUBE_PROFILE_ID,
        "training_profile_fingerprint": profile_fingerprint,
    }
    for key, expected in exact.items():
        actual = _require(metadata, key)
        if actual != expected:
            raise CheckpointMetadataInvalid(
                f"Golden Cube metadata mismatch for {key}: saved={actual!r}, expected={expected!r}"
            )
    # Support the common normalized spelling as an additional consistency
    # check, but never infer an architecture from a filename.
    if "profile_id" in metadata and metadata["profile_id"] != CUBE_PROFILE_ID:
        raise CheckpointMetadataInvalid("Golden Cube profile_id does not match the current profile")
    if "profile_fingerprint" in metadata and metadata["profile_fingerprint"] != profile_fingerprint:
        raise CheckpointMetadataInvalid("Golden Cube profile_fingerprint does not match the current profile")
    _validate_hash(_require(metadata, "model_hash"), "model_hash")


@dataclass
class GoldenPlayableModel:
    """The narrow model boundary consumed by ``GoldenGameGenerator``."""

    descriptor: CheckpointDescriptor
    network: object
    evaluator: object
    metadata: Mapping[str, object]
    device: str

    def evaluate(self, state, legal_context=None):
        evaluate = getattr(self.evaluator, "evaluate_prepared", None)
        if legal_context is not None and callable(evaluate):
            return evaluate(state, legal_context)
        return self.evaluator.evaluate(state)


class GoldenCheckpointLoader:
    """Load supported Golden ``.pt`` artifacts with strict validation."""

    def __init__(self, catalog: CheckpointCatalog, *, device: str = "cpu", cache=None):
        self.catalog = catalog
        self.device = resolve_device(device)
        self.cache = cache

    def descriptor(self, checkpoint_id: str) -> CheckpointDescriptor:
        descriptor = self.catalog.get(checkpoint_id)
        if descriptor is None:
            raise CheckpointMetadataInvalid(f"Unknown Golden checkpoint: {checkpoint_id}")
        return descriptor

    def load(self, checkpoint_id: str):
        descriptor = self.descriptor(checkpoint_id)
        key = (descriptor.checkpoint_id, self.device, GOLDEN_BACKEND_KIND)
        if self.cache is not None:
            return descriptor, self.cache.get_or_load(key, lambda: self._load_uncached(descriptor))
        return descriptor, self._load_uncached(descriptor)

    def _load_uncached(self, descriptor: CheckpointDescriptor) -> GoldenPlayableModel:
        if descriptor.metadata_error:
            raise CheckpointMetadataInvalid(descriptor.metadata_error)
        path = Path(descriptor.path)
        metadata_path = Path(descriptor.metadata_path or f"{path.with_suffix('')}.metadata.json")
        sidecar = _read_sidecar(metadata_path)
        if descriptor.profile_id is not None and _metadata_value(sidecar, "profile_id", "training_profile_id") != descriptor.profile_id:
            raise CheckpointMetadataInvalid("Golden checkpoint profile differs from catalog descriptor")

        profile_id = str(_metadata_value(sidecar, "profile_id", "training_profile_id", default=""))
        if profile_id == "gocube-torus9-golden-v3":
            _validate_torus9_metadata(sidecar)
        elif profile_id == "gocube-cube4-golden-training-v1":
            _validate_cube_metadata(sidecar)
        else:
            raise CheckpointMetadataInvalid(f"Unsupported Golden profile: {profile_id!r}")

        payload = _load_payload(path)
        embedded = _compare_embedded_metadata(sidecar, payload)
        # Validate the object actually embedded in the .pt as well.  Sidecar
        # agreement alone must not make a stale/tampered payload playable.
        if profile_id == "gocube-torus9-golden-v3":
            _validate_torus9_metadata(embedded)
        else:
            _validate_cube_metadata(embedded)
        if "artifact_sha256" in sidecar:
            expected_artifact = _validate_hash(sidecar["artifact_sha256"], "artifact_sha256")
            if _file_sha256(path) != expected_artifact:
                raise CheckpointMetadataInvalid("Golden checkpoint artifact hash mismatch")

        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, Mapping):
            raise CheckpointLoadFailed("Golden checkpoint is missing model_state_dict")

        try:
            if profile_id == "gocube-torus9-golden-v3":
                from gocube_golden.torus9 import (
                    Torus9CurrentGraphNet,
                    Torus9NeuralEvaluator,
                    torus9_load_checkpoint,
                    torus9_model_from_metadata,
                )
                network = torus9_model_from_metadata(sidecar)
                loaded = torus9_load_checkpoint(
                    path,
                    model=network,
                    optimizer=None,
                    expected={
                        "model_hash": sidecar["model_hash"],
                        "profile_id": sidecar["profile_id"],
                        "profile_fingerprint": sidecar["profile_fingerprint"],
                        "target_fingerprint": sidecar["target_fingerprint"],
                    },
                    device=self.device,
                )
                if not isinstance(network, Torus9CurrentGraphNet):
                    raise CheckpointMetadataInvalid("Golden Torus9 loader resolved a non-current network")
                evaluator = Torus9NeuralEvaluator(network, device=self.device)
            else:
                from gocube_golden.cube_neural import GoldenCubeGraphNetV1, GoldenCubeNeuralEvaluator, cube_model_hash

                network = GoldenCubeGraphNetV1()
                if sidecar.get("architecture_config") != network.architecture_config:
                    raise CheckpointMetadataInvalid("Golden Cube architecture metadata does not match the network")
                network.load_state_dict(state_dict, strict=True)
                network.to(self.device)
                network.eval()
                actual_hash = cube_model_hash(network)
                if actual_hash != sidecar["model_hash"]:
                    raise CheckpointMetadataInvalid("Golden Cube model hash mismatch")
                loaded = dict(sidecar)
                evaluator = GoldenCubeNeuralEvaluator(network, device=self.device)
        except (CheckpointMetadataInvalid, CheckpointLoadFailed):
            raise
        except Exception as exc:
            raise CheckpointLoadFailed(f"Failed to load Golden checkpoint {descriptor.checkpoint_id}: {exc}") from exc

        # The Torus utility performs the model-hash check after strict loading;
        # repeat the identity check here so both Golden families expose one
        # uniform loader guarantee.
        if loaded.get("model_hash") != sidecar.get("model_hash"):
            raise CheckpointMetadataInvalid("Golden model hash mismatch")
        evaluator.checkpoint_path = str(path)
        evaluator.checkpoint_metadata = dict(sidecar)
        return GoldenPlayableModel(
            descriptor=descriptor,
            network=network,
            evaluator=evaluator,
            metadata=dict(sidecar),
            device=self.device,
        )


__all__ = [
    "DEVICE_CHOICES",
    "GOLDEN_BACKEND_KIND",
    "GOLDEN_CHECKPOINT_FORMAT",
    "GoldenPlayableModel",
    "GoldenCheckpointLoader",
    "resolve_device",
]
