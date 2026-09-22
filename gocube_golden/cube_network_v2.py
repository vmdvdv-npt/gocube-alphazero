"""Metadata-driven CubeGraphNetV2 for the Stage-2/3 Cube family.

Stage 4 intentionally stops at model construction, forward/backward semantics,
and model-only test bundles. It does not connect this network to production
self-play, replay, training, Arena, or run storage.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cube_family import (
    CROSS_FACE_SEAM,
    SAME_FACE,
    CubeFamilyTopology,
    cube_family_topology,
)
from .cube_game_contract_v2 import SUPPORTED_SIZES, validate_cube_size
from .cube_observation_v2 import (
    CHANNEL_COUNT,
    SCHEMA_FINGERPRINT as OBSERVATION_SCHEMA_FINGERPRINT,
    SCHEMA_ID as OBSERVATION_SCHEMA_ID,
    concrete_observation_identity,
)

ARCHITECTURE_SCHEMA_VERSION = 2
ARCHITECTURE_ID = "gocube-cube-graphnet-v2"
FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"
HIDDEN = 112
BLOCKS = 10
GLOBAL_CONTEXT_AFTER_BLOCKS = (3, 6, 9)
ARCHITECTURE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "gocube" / "cube_network_v2.json"
)
MODEL_METADATA_SCHEMA_VERSION = 1

_ARCHITECTURE_PAYLOAD: dict[str, object] = {
    "architecture_schema_version": ARCHITECTURE_SCHEMA_VERSION,
    "architecture_id": ARCHITECTURE_ID,
    "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
    "hidden": HIDDEN,
    "blocks": BLOCKS,
    "input": {
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "channels": CHANNEL_COUNT,
        "layout": "[B,channels,points]",
        "projection": "shared-linear:30->112",
    },
    "trunk": {
        "normalization": "pre-norm-layernorm-hidden-per-point",
        "activation": "relu",
        "aggregation": "mean-over-four-game-neighbors",
        "relation_types": [SAME_FACE, CROSS_FACE_SEAM],
        "relation_specific_transforms": True,
        "residual_style": "graph-message-output-projection-add",
        "operation_order": "block->global(if 3,6,9)->corner(if <10)",
    },
    "corner_context": {
        "enabled": True,
        "after_blocks": list(range(1, BLOCKS)),
        "physical_corner_count": 8,
        "incident_points_per_corner": 3,
        "pooling": "arithmetic-mean-unordered-three-points",
        "transform": "layernorm-linear-relu-linear",
        "injection": "shared-residual-add-identical-context-to-three-points",
    },
    "global_context": {
        "enabled": True,
        "after_blocks": list(GLOBAL_CONTEXT_AFTER_BLOCKS),
        "pooling": "arithmetic-mean-all-points",
        "transform": "layernorm-linear-relu-linear",
        "injection": "broadcast-residual-add",
    },
    "final": {
        "normalization": "layernorm-hidden-per-point",
        "activation": "relu",
    },
    "heads": {
        "policy": {
            "point": "shared-linear:112->1",
            "pass": "mean-final-nodes->linear:112->1",
            "output": "logits",
            "shape": "[B,P+1]",
            "pass_index": "P",
        },
        "wdl": {
            "pooling": "arithmetic-mean-final-nodes",
            "transform": "linear:112->112,relu,linear:112->3",
            "classes": ["WIN", "DRAW", "LOSS"],
            "perspective": "current-side-to-move",
            "output": "logits",
            "shape": "[B,3]",
        },
        "ownership": {
            "point": "shared-linear:112->3",
            "classes": ["OWN", "OPPONENT", "NEUTRAL"],
            "perspective": "current-side-to-move",
            "output": "logits",
            "shape": "[B,P,3]",
        },
        "score": {
            "pooling": "arithmetic-mean-final-nodes",
            "transform": "linear:112->112,relu,linear:112->1",
            "output": "raw-scalar",
            "shape": "[B,1]",
        },
    },
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


ARCHITECTURE_FINGERPRINT = _fingerprint(_ARCHITECTURE_PAYLOAD)


def cube_network_architecture_contract() -> dict[str, object]:
    contract = copy.deepcopy(_ARCHITECTURE_PAYLOAD)
    contract["architecture_fingerprint"] = ARCHITECTURE_FINGERPRINT
    return contract


def validate_cube_network_architecture_contract(
    contract: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(contract, Mapping):
        raise ValueError("Cube network architecture contract must be a mapping")
    candidate = copy.deepcopy(dict(contract))
    supplied_fingerprint = candidate.pop("architecture_fingerprint", None)
    if candidate != _ARCHITECTURE_PAYLOAD:
        raise ValueError("Cube network architecture contract does not match Stage-4 semantics")
    if supplied_fingerprint != ARCHITECTURE_FINGERPRINT:
        raise ValueError("Cube network architecture fingerprint mismatch")
    return contract


def load_cube_network_architecture_contract(
    path: Path | str = ARCHITECTURE_PATH,
) -> Mapping[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_cube_network_architecture_contract(payload)


class CubeRelationMessageLayer(nn.Module):
    """Shared relation-aware messages over the four Stage-2 game neighbors."""

    def __init__(self, hidden: int, topology: CubeFamilyTopology) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("Cube relation layer hidden size must be positive")
        if any(len(row) != 4 for row in topology.adjacency):
            raise ValueError("CubeGraphNetV2 requires degree-4 Stage-2 adjacency")
        self.register_buffer(
            "neighbors",
            torch.as_tensor(topology.adjacency, dtype=torch.long),
            persistent=False,
        )
        seam_flags = [
            [relation == CROSS_FACE_SEAM for relation in row]
            for row in topology.relation_types
        ]
        self.register_buffer(
            "seam_flags",
            torch.as_tensor(seam_flags, dtype=torch.bool),
            persistent=False,
        )
        self.self_transform = nn.Linear(hidden, hidden)
        self.same_face_transform = nn.Linear(hidden, hidden)
        self.cross_face_seam_transform = nn.Linear(hidden, hidden)

    def forward(self, nodes: Tensor) -> Tensor:
        neighbor_nodes = nodes[:, self.neighbors, :]
        same_messages = self.same_face_transform(neighbor_nodes)
        seam_messages = self.cross_face_seam_transform(neighbor_nodes)
        messages = torch.where(
            self.seam_flags[None, :, :, None], seam_messages, same_messages
        )
        return self.self_transform(nodes) + messages.mean(dim=2)


class CubeGraphResidualBlock(nn.Module):
    def __init__(self, hidden: int, topology: CubeFamilyTopology) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.message = CubeRelationMessageLayer(hidden, topology)
        self.output_projection = nn.Linear(hidden, hidden)

    def forward(self, nodes: Tensor) -> Tensor:
        normalized = self.norm(nodes)
        message = self.message(normalized)
        return nodes + self.output_projection(F.relu(message))


class CornerContext(nn.Module):
    """Shared unordered physical-corner context; it does not alter game topology."""

    def __init__(self, hidden: int, topology: CubeFamilyTopology) -> None:
        super().__init__()
        if len(topology.physical_corners) != 8 or any(
            len(set(corner)) != 3 for corner in topology.physical_corners
        ):
            raise ValueError("CubeGraphNetV2 requires eight unordered three-point corners")
        self.register_buffer(
            "corner_groups",
            torch.as_tensor(topology.physical_corners, dtype=torch.long),
            persistent=False,
        )
        self.transform = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def pooled_context(self, nodes: Tensor) -> Tensor:
        return nodes[:, self.corner_groups, :].mean(dim=2)

    def forward(self, nodes: Tensor) -> Tensor:
        context = self.transform(self.pooled_context(nodes))
        result = nodes.clone()
        result[:, self.corner_groups, :] = (
            result[:, self.corner_groups, :] + context[:, :, None, :]
        )
        return result


class GlobalContext(nn.Module):
    """Permutation-invariant global mean context broadcast to every point."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, nodes: Tensor) -> Tensor:
        context = self.transform(nodes.mean(dim=1))
        return nodes + context[:, None, :]


@dataclass(frozen=True)
class CubeNetworkOutput:
    policy_logits: Tensor
    wdl_logits: Tensor
    ownership_logits: Tensor
    score: Tensor


@dataclass(frozen=True)
class CubeInferenceOutput:
    policy_logits: Tensor
    wdl_logits: Tensor


class CubeGraphNetV2(nn.Module):
    """Stage-4 Cube-family architecture shared across cube2..cube7."""

    architecture_id = ARCHITECTURE_ID
    architecture_schema_version = ARCHITECTURE_SCHEMA_VERSION
    architecture_fingerprint = ARCHITECTURE_FINGERPRINT

    def __init__(self, *, topology: CubeFamilyTopology) -> None:
        super().__init__()
        if not isinstance(topology, CubeFamilyTopology):
            raise ValueError("CubeGraphNetV2 requires CubeFamilyTopology")
        validate_cube_size(topology.size)
        if topology.point_count != 6 * topology.size * topology.size:
            raise ValueError("CubeGraphNetV2 topology point count mismatch")

        self.size = topology.size
        self.point_count = topology.point_count
        self.action_count = topology.action_count
        self.topology_id = topology.topology_id
        self.game_graph_fingerprint = topology.game_graph_fingerprint
        self.geometry_fingerprint = topology.geometry_fingerprint
        self.hidden = HIDDEN
        self.blocks_count = BLOCKS
        self.model_metadata: dict[str, object] | None = None

        self.input_projection = nn.Linear(CHANNEL_COUNT, HIDDEN)
        self.blocks = nn.ModuleList(
            CubeGraphResidualBlock(HIDDEN, topology) for _ in range(BLOCKS)
        )
        self.corner_contexts = nn.ModuleList(
            CornerContext(HIDDEN, topology) for _ in range(BLOCKS - 1)
        )
        self.global_contexts = nn.ModuleDict(
            {str(index): GlobalContext(HIDDEN) for index in GLOBAL_CONTEXT_AFTER_BLOCKS}
        )
        self.final_norm = nn.LayerNorm(HIDDEN)

        self.point_policy_head = nn.Linear(HIDDEN, 1)
        self.pass_head = nn.Linear(HIDDEN, 1)
        self.wdl_head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 3),
        )
        self.ownership_head = nn.Linear(HIDDEN, 3)
        self.score_head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def _validate_observation(self, observation: Tensor) -> None:
        if not isinstance(observation, Tensor):
            raise ValueError("CubeGraphNetV2 observation must be a torch.Tensor")
        if observation.ndim != 3:
            raise ValueError("CubeGraphNetV2 expects batched [B,30,P] observations")
        if observation.shape[0] <= 0:
            raise ValueError("CubeGraphNetV2 batch must be non-empty")
        if observation.shape[1] != CHANNEL_COUNT:
            raise ValueError(
                f"CubeGraphNetV2 expects {CHANNEL_COUNT} observation channels"
            )
        if observation.shape[2] != self.point_count:
            raise ValueError(
                f"CubeGraphNetV2 topology expects P={self.point_count} points"
            )
        if observation.dtype != torch.float32:
            raise ValueError("CubeGraphNetV2 observation must have dtype float32")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("CubeGraphNetV2 observation contains NaN or Inf")

    def encode(self, observation: Tensor) -> Tensor:
        self._validate_observation(observation)
        nodes = self.input_projection(observation.transpose(1, 2))
        for block_number, block in enumerate(self.blocks, start=1):
            nodes = block(nodes)
            key = str(block_number)
            if key in self.global_contexts:
                nodes = self.global_contexts[key](nodes)
            if block_number < BLOCKS:
                nodes = self.corner_contexts[block_number - 1](nodes)
        return F.relu(self.final_norm(nodes))

    def _policy_and_wdl(self, features: Tensor) -> CubeInferenceOutput:
        global_features = features.mean(dim=1)
        point_logits = self.point_policy_head(features).squeeze(-1)
        pass_logit = self.pass_head(global_features)
        return CubeInferenceOutput(
            policy_logits=torch.cat((point_logits, pass_logit), dim=1),
            wdl_logits=self.wdl_head(global_features),
        )

    def infer_policy_wdl(self, observation: Tensor) -> CubeInferenceOutput:
        return self._policy_and_wdl(self.encode(observation))

    def forward(self, observation: Tensor) -> CubeNetworkOutput:
        features = self.encode(observation)
        inference = self._policy_and_wdl(features)
        global_features = features.mean(dim=1)
        return CubeNetworkOutput(
            policy_logits=inference.policy_logits,
            wdl_logits=inference.wdl_logits,
            ownership_logits=self.ownership_head(features),
            score=self.score_head(global_features),
        )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@lru_cache(maxsize=len(SUPPORTED_SIZES))
def _parameter_count_for_size(size: int) -> int:
    topology = cube_family_topology(validate_cube_size(size))
    return trainable_parameter_count(CubeGraphNetV2(topology=topology))


def _metadata_without_contract_fingerprint(size: int) -> dict[str, object]:
    topology = cube_family_topology(validate_cube_size(size))
    observation = concrete_observation_identity(topology)
    architecture = cube_network_architecture_contract()
    return {
        "metadata_schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "architecture_id": ARCHITECTURE_ID,
        "architecture_schema_version": ARCHITECTURE_SCHEMA_VERSION,
        "architecture_fingerprint": ARCHITECTURE_FINGERPRINT,
        "size": topology.size,
        "point_count": topology.point_count,
        "action_count": topology.action_count,
        "hidden": HIDDEN,
        "blocks": BLOCKS,
        "input_channels": CHANNEL_COUNT,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "observation_schema_fingerprint": OBSERVATION_SCHEMA_FINGERPRINT,
        "concrete_observation_fingerprint": observation[
            "concrete_observation_fingerprint"
        ],
        "topology_id": topology.topology_id,
        "game_graph_fingerprint": topology.game_graph_fingerprint,
        "geometry_fingerprint": topology.geometry_fingerprint,
        "normalization": {
            "trunk": architecture["trunk"]["normalization"],
            "final": architecture["final"]["normalization"],
        },
        "activation": architecture["trunk"]["activation"],
        "aggregation": architecture["trunk"]["aggregation"],
        "relation_types": list(architecture["trunk"]["relation_types"]),
        "corner_context": copy.deepcopy(architecture["corner_context"]),
        "global_context": copy.deepcopy(architecture["global_context"]),
        "heads": copy.deepcopy(architecture["heads"]),
        "parameter_count": _parameter_count_for_size(topology.size),
    }


def cube_model_metadata(size: int) -> dict[str, object]:
    metadata = _metadata_without_contract_fingerprint(validate_cube_size(size))
    metadata["model_contract_fingerprint"] = _fingerprint(metadata)
    return metadata


def _validate_metadata_header(metadata: Mapping[str, object]) -> int:
    if not isinstance(metadata, Mapping):
        raise ValueError("Cube model metadata must be a mapping")
    if metadata.get("architecture_id") == "GoldenCubeGraphNetV1":
        raise ValueError("Historical GoldenCubeGraphNetV1 metadata is incompatible")
    if metadata.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Unknown Cube architecture id")
    if metadata.get("architecture_schema_version") != ARCHITECTURE_SCHEMA_VERSION:
        raise ValueError("Cube architecture schema version mismatch")
    if metadata.get("architecture_fingerprint") != ARCHITECTURE_FINGERPRINT:
        raise ValueError("Cube architecture fingerprint mismatch")
    if metadata.get("metadata_schema_version") != MODEL_METADATA_SCHEMA_VERSION:
        raise ValueError("Cube model metadata schema version mismatch")
    try:
        size = validate_cube_size(metadata.get("size"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cube model metadata has invalid size") from exc
    return size


def validate_cube_model_metadata(
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    size = _validate_metadata_header(metadata)
    expected = cube_model_metadata(size)
    candidate = dict(metadata)

    supplied_contract_fingerprint = candidate.get("model_contract_fingerprint")
    candidate_without_contract = dict(candidate)
    candidate_without_contract.pop("model_contract_fingerprint", None)
    if supplied_contract_fingerprint != _fingerprint(candidate_without_contract):
        raise ValueError("Cube concrete model contract fingerprint mismatch")

    if candidate != expected:
        keys = sorted(set(candidate) | set(expected))
        mismatches = [key for key in keys if candidate.get(key) != expected.get(key)]
        detail = ", ".join(mismatches[:6])
        raise ValueError(f"Cube model metadata is incompatible: {detail}")
    return metadata


def build_cube_model_from_metadata(
    metadata: Mapping[str, object],
) -> CubeGraphNetV2:
    validate_cube_model_metadata(metadata)
    size = int(metadata["size"])
    topology = cube_family_topology(size)
    model = CubeGraphNetV2(topology=topology)
    actual_count = trainable_parameter_count(model)
    if actual_count != metadata["parameter_count"]:
        raise ValueError("Cube model parameter count mismatch")
    model.model_metadata = copy.deepcopy(dict(metadata))
    return model


def cube_graphnet_v2_model_hash(model: CubeGraphNetV2) -> str:
    if not isinstance(model, CubeGraphNetV2) or model.model_metadata is None:
        raise ValueError("CubeGraphNetV2 model hash requires validated concrete metadata")
    validate_cube_model_metadata(model.model_metadata)
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(model.model_metadata))
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def save_cube_model_bundle(
    model: CubeGraphNetV2,
    directory: Path | str,
) -> dict[str, Path]:
    if not isinstance(model, CubeGraphNetV2) or model.model_metadata is None:
        raise ValueError("Model-only bundle requires a metadata-built CubeGraphNetV2")
    validate_cube_model_metadata(model.model_metadata)
    bundle = Path(directory)
    bundle.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle / "metadata.json"
    weights_path = bundle / "state_dict.pt"
    hash_path = bundle / "model.sha256"

    metadata_path.write_text(
        json.dumps(model.model_metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(model.state_dict(), weights_path)
    hash_path.write_text(cube_graphnet_v2_model_hash(model) + "\n", encoding="utf-8")
    return {
        "metadata": metadata_path,
        "state_dict": weights_path,
        "model_hash": hash_path,
    }


def load_cube_model_bundle(
    directory: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> CubeGraphNetV2:
    bundle = Path(directory)
    metadata_path = bundle / "metadata.json"
    weights_path = bundle / "state_dict.pt"
    hash_path = bundle / "model.sha256"

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model = build_cube_model_from_metadata(metadata)
    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(map_location)

    if hash_path.exists():
        expected_hash = hash_path.read_text(encoding="utf-8").strip()
        if cube_graphnet_v2_model_hash(model) != expected_hash:
            raise ValueError("Cube model-only bundle hash mismatch")
    return model


__all__ = [
    "ARCHITECTURE_FINGERPRINT",
    "ARCHITECTURE_ID",
    "ARCHITECTURE_PATH",
    "ARCHITECTURE_SCHEMA_VERSION",
    "BLOCKS",
    "CubeGraphNetV2",
    "CubeGraphResidualBlock",
    "CubeInferenceOutput",
    "CubeNetworkOutput",
    "CubeRelationMessageLayer",
    "CornerContext",
    "GLOBAL_CONTEXT_AFTER_BLOCKS",
    "GlobalContext",
    "HIDDEN",
    "MODEL_METADATA_SCHEMA_VERSION",
    "build_cube_model_from_metadata",
    "cube_graphnet_v2_model_hash",
    "cube_model_metadata",
    "cube_network_architecture_contract",
    "load_cube_model_bundle",
    "load_cube_network_architecture_contract",
    "save_cube_model_bundle",
    "trainable_parameter_count",
    "validate_cube_model_metadata",
    "validate_cube_network_architecture_contract",
]
