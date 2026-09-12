"""Geometry-aware neural representation for the canonical Golden Cube."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from typing import Mapping, Sequence

torch = importlib.import_module("torch")
Tensor = torch.Tensor
nn = torch.nn
F = importlib.import_module("torch.nn.functional")

from .rules import legal_actions
from .search import Evaluation
from .state import BLACK, PASS, WHITE, GoldenState
from .cube_topology import (
    CROSS_FACE_SEAM,
    CUBE4_TOPOLOGY,
    FACE_CORNER,
    FACE_EDGE,
    FACE_INTERIOR,
    GEOMETRY_SCHEMA_ID,
    SAME_FACE,
    CubeGoldenTopology,
)

CUBE_POINT_COUNT = 96
CUBE_ACTION_COUNT = 97
CUBE_PASS_INDEX = 96
CUBE_OBSERVATION_SCHEMA_ID = "gocube-cube4-golden-observation-v1"
CUBE_OBSERVATION_SCHEMA_VERSION = 1
CUBE_OBSERVATION_LAYOUT = "[channels,points]"
CUBE_OBSERVATION_CHANNELS = (
    "own_stones",
    "opponent_stones",
    "side_to_move_color",
    "previous_pass",
    "legal_point_mask",
    "komi",
    "is_face_interior",
    "is_face_edge",
    "is_face_corner",
    "corner_distance_0",
    "corner_distance_1",
    "corner_distance_2",
    "corner_distance_3_plus",
    "has_cross_face_neighbor",
    "num_cross_face_neighbors",
)
CUBE_OBSERVATION_CHANNEL_COUNT = len(CUBE_OBSERVATION_CHANNELS)
CUBE_VALUE_HEAD_SEMANTICS = "side-to-move:[WIN,DRAW,LOSS]"


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _observation_payload(topology: CubeGoldenTopology) -> dict[str, object]:
    return {
        "schema_id": CUBE_OBSERVATION_SCHEMA_ID,
        "schema_version": CUBE_OBSERVATION_SCHEMA_VERSION,
        "layout": CUBE_OBSERVATION_LAYOUT,
        "dynamic_channels": list(CUBE_OBSERVATION_CHANNELS[:6]),
        "static_node_channels": list(CUBE_OBSERVATION_CHANNELS[6:]),
        "static_geometry_schema_id": GEOMETRY_SCHEMA_ID,
        "topology_fingerprint": topology.fingerprint,
        "geometry_fingerprint": topology.geometry_fingerprint,
        "corner_distance_buckets": ["0", "1", "2", "3_plus"],
        "relation_schema": [SAME_FACE, CROSS_FACE_SEAM],
        "physical_corner_group_size": 3,
        "physical_corner_count": 8,
        "point_ordering": list(topology.point_ids),
        "action_ordering": list(range(CUBE_ACTION_COUNT)),
    }


CUBE_OBSERVATION_FINGERPRINT = _fingerprint(_observation_payload(CUBE4_TOPOLOGY))


@dataclass(frozen=True)
class CubeObservation:
    tensor: Tensor
    action_mask: tuple[bool, ...]
    state_key: tuple[object, ...]

    def __post_init__(self) -> None:
        if tuple(self.tensor.shape) != (CUBE_OBSERVATION_CHANNEL_COUNT, CUBE_POINT_COUNT):
            raise ValueError("Cube observation tensor must have shape [15,96]")
        if self.tensor.dtype != torch.float32:
            raise ValueError("Cube observation tensor must be float32")
        if len(self.action_mask) != CUBE_ACTION_COUNT:
            raise ValueError("Cube action mask must have length 97")
        if not bool(torch.isfinite(self.tensor).all()):
            raise ValueError("Cube observation contains NaN or Inf")


def build_cube_action_mask(state: GoldenState) -> tuple[bool, ...]:
    if state.is_terminal:
        raise ValueError("Terminal Golden Cube states must never be passed to the NN")
    if state.topology.fingerprint != CUBE4_TOPOLOGY.fingerprint:
        raise ValueError("Cube observation requires canonical Golden Cube topology")
    legal = set(legal_actions(state))
    return tuple(point in legal for point in range(CUBE_POINT_COUNT)) + (PASS in legal,)


def build_cube_observation_bundle(
    state: GoldenState,
    *,
    topology: CubeGoldenTopology = CUBE4_TOPOLOGY,
) -> CubeObservation:
    if state.is_terminal:
        raise ValueError("Terminal Golden Cube states must never be passed to the NN")
    if topology.fingerprint != CUBE4_TOPOLOGY.fingerprint or state.topology.fingerprint != topology.fingerprint:
        raise ValueError("Cube observation requires canonical Golden Cube topology")
    mask = build_cube_action_mask(state)
    own = int(state.side_to_move)
    other = int(WHITE if state.side_to_move == BLACK else BLACK)
    values = torch.zeros((CUBE_OBSERVATION_CHANNEL_COUNT, CUBE_POINT_COUNT), dtype=torch.float32)
    for point, stone in enumerate(state.stones):
        values[0, point] = float(int(stone) == own)
        values[1, point] = float(int(stone) == other)
    values[2].fill_(1.0 if state.side_to_move == BLACK else -1.0)
    values[3].fill_(1.0 if state.consecutive_passes == 1 else 0.0)
    values[4] = torch.tensor(mask[:CUBE_POINT_COUNT], dtype=torch.float32)
    values[5].fill_(0.5)
    for point in range(CUBE_POINT_COUNT):
        geometry = topology.geometry(point)
        class_channel = {
            FACE_INTERIOR: 6,
            FACE_EDGE: 7,
            FACE_CORNER: 8,
        }[geometry.geometry_class]
        values[class_channel, point] = 1.0
        bucket_channel = {
            "corner_distance_0": 9,
            "corner_distance_1": 10,
            "corner_distance_2": 11,
            "corner_distance_3_plus": 12,
        }[geometry.corner_distance_bucket]
        values[bucket_channel, point] = 1.0
        values[13, point] = float(geometry.has_cross_face_neighbor)
        values[14, point] = float(geometry.num_cross_face_neighbors) / 4.0
    return CubeObservation(values, mask, state.state_key)


def build_cube_observation(state: GoldenState) -> Tensor:
    return build_cube_observation_bundle(state).tensor


class CubeRelationMessageLayer(nn.Module):
    """Relation-aware messages over actual game edges only."""

    def __init__(self, hidden: int, topology: CubeGoldenTopology) -> None:
        super().__init__()
        self.register_buffer("neighbors", torch.as_tensor(topology.adjacency, dtype=torch.long))
        seam_flags = [
            [relation == CROSS_FACE_SEAM for relation in row]
            for row in topology.relation_types
        ]
        self.register_buffer("seam_flags", torch.as_tensor(seam_flags, dtype=torch.bool))
        self.self_linear = nn.Linear(hidden, hidden)
        self.same_face_linear = nn.Linear(hidden, hidden)
        self.seam_linear = nn.Linear(hidden, hidden)

    def forward(self, nodes: Tensor) -> Tensor:
        neighbor_nodes = nodes[:, self.neighbors, :]
        same_messages = self.same_face_linear(neighbor_nodes)
        seam_messages = self.seam_linear(neighbor_nodes)
        messages = torch.where(self.seam_flags[None, :, :, None], seam_messages, same_messages)
        return self.self_linear(nodes) + messages.mean(dim=2)


class CubeGraphResidualBlock(nn.Module):
    def __init__(self, hidden: int, topology: CubeGoldenTopology) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.message = CubeRelationMessageLayer(hidden, topology)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, nodes: Tensor) -> Tensor:
        return nodes + self.output(F.relu(self.message(self.norm(nodes))))


class CornerContextBlock(nn.Module):
    """Aggregate each physical corner's three face-corner cells and return it."""

    def __init__(self, hidden: int, topology: CubeGoldenTopology) -> None:
        super().__init__()
        self.register_buffer(
            "corner_groups",
            torch.as_tensor(topology.physical_corners, dtype=torch.long),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.injection = nn.Linear(hidden, hidden)

    def forward(self, nodes: Tensor) -> Tensor:
        grouped = nodes[:, self.corner_groups, :]
        context = self.context_projection(grouped.mean(dim=2))
        injected = self.injection(context)
        result = nodes.clone()
        result[:, self.corner_groups, :] = result[:, self.corner_groups, :] + injected[:, :, None, :]
        return result


class GoldenCubeGraphNetV1(nn.Module):
    """Cube-aware residual graph network with relation and corner context."""

    architecture_id = "GoldenCubeGraphNetV1"

    def __init__(
        self,
        *,
        topology: CubeGoldenTopology = CUBE4_TOPOLOGY,
        hidden: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        if topology.fingerprint != CUBE4_TOPOLOGY.fingerprint:
            raise ValueError("GoldenCubeGraphNetV1 is fixed to canonical Cube 4x4x6")
        if hidden <= 0 or blocks <= 0:
            raise ValueError("GoldenCubeGraphNetV1 hidden and blocks must be positive")
        self.topology_id = topology.topology_id
        self.topology_fingerprint = topology.fingerprint
        self.geometry_fingerprint = topology.geometry_fingerprint
        self.observation_fingerprint = CUBE_OBSERVATION_FINGERPRINT
        self.hidden = int(hidden)
        self.blocks_count = int(blocks)
        self.input_projection = nn.Linear(CUBE_OBSERVATION_CHANNEL_COUNT, hidden)
        self.blocks = nn.ModuleList(
            CubeGraphResidualBlock(hidden, topology) for _ in range(blocks)
        )
        self.corner_context_blocks = nn.ModuleList(
            CornerContextBlock(hidden, topology) for _ in range(max(0, blocks - 1))
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.point_policy = nn.Linear(hidden, 1)
        self.pass_policy = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    @property
    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture_id": self.architecture_id,
            "topology_id": self.topology_id,
            "topology_fingerprint": self.topology_fingerprint,
            "geometry_fingerprint": self.geometry_fingerprint,
            "observation_schema_id": CUBE_OBSERVATION_SCHEMA_ID,
            "observation_fingerprint": self.observation_fingerprint,
            "input_channels": CUBE_OBSERVATION_CHANNEL_COUNT,
            "point_count": CUBE_POINT_COUNT,
            "hidden": self.hidden,
            "blocks": self.blocks_count,
            "relation_types": [SAME_FACE, CROSS_FACE_SEAM],
            "corner_context": {
                "enabled": True,
                "physical_corner_count": 8,
                "incident_points_per_corner": 3,
            },
            "heads": {"policy": [CUBE_ACTION_COUNT], "value": [3]},
        }

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        if tuple(observation.shape[1:]) != (CUBE_OBSERVATION_CHANNEL_COUNT, CUBE_POINT_COUNT):
            raise ValueError("GoldenCubeGraphNetV1 expects [batch,15,96]")
        nodes = self.input_projection(observation.transpose(1, 2))
        for index, block in enumerate(self.blocks):
            nodes = block(nodes)
            if index < len(self.corner_context_blocks):
                nodes = self.corner_context_blocks[index](nodes)
        nodes = F.relu(self.output_norm(nodes))
        policy_logits = torch.cat(
            (self.point_policy(nodes).squeeze(-1), self.pass_policy(nodes.mean(dim=1))),
            dim=1,
        )
        value_logits = self.value_head(nodes.mean(dim=1))
        return policy_logits, value_logits


def cube_model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    config = getattr(model, "architecture_config", {"class": type(model).__qualname__})
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, parameter in sorted(model.state_dict().items()):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def cube_count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class GoldenCubeNeuralEvaluator:
    def __init__(self, model: nn.Module, *, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path: str | None = None
        self.checkpoint_metadata: Mapping[str, object] | None = None
        self.nn_evaluations = 0

    def evaluate(self, state: GoldenState) -> Evaluation:
        if state.is_terminal:
            raise RuntimeError("GoldenCubeNeuralEvaluator must never evaluate a terminal state")
        observation = build_cube_observation(state).to(self.device)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            wdl = torch.softmax(value_logits[0], dim=0)
        self.nn_evaluations += 1
        policy_values = tuple(float(value) for value in policy.detach().cpu())
        wdl_values = tuple(float(value) for value in wdl.detach().cpu())
        if len(policy_values) != CUBE_ACTION_COUNT or len(wdl_values) != 3:
            raise RuntimeError("Golden Cube neural head shape drift")
        if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
            raise RuntimeError("Golden Cube neural evaluator produced non-finite output")
        if not math.isclose(sum(policy_values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError("Golden Cube policy probabilities are not normalized")
        if not math.isclose(sum(wdl_values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError("Golden Cube WDL probabilities are not normalized")
        return Evaluation(policy=policy_values, wdl=wdl_values)


class SelfPlayCubeRootNoiseEvaluator:
    def __init__(
        self,
        evaluator: GoldenCubeNeuralEvaluator,
        root_state: GoldenState,
        *,
        seed: int,
        epsilon: float = 0.25,
        alpha: float = 0.30,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0 or alpha <= 0.0:
            raise ValueError("Invalid Cube self-play Dirichlet parameters")
        self.evaluator = evaluator
        self.root_state_key = root_state.state_key
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))

    def evaluate(self, state: GoldenState) -> Evaluation:
        base = self.evaluator.evaluate(state)
        if state.state_key != self.root_state_key:
            return base
        legal = legal_actions(state)
        if not legal:
            raise RuntimeError("Cube self-play root has no legal actions")
        base_policy = [float(value) for value in base.policy]
        if len(base_policy) != CUBE_ACTION_COUNT:
            raise RuntimeError("Cube root-noise evaluator received the wrong policy shape")
        legal_indices = [CUBE_PASS_INDEX if action == PASS else int(action) for action in legal]
        prior = torch.tensor([base_policy[index] for index in legal_indices], dtype=torch.float64)
        prior = prior / prior.sum() if float(prior.sum()) > 0.0 else torch.full_like(prior, 1.0 / len(legal))
        noise = torch._standard_gamma(
            torch.full((len(legal),), self.alpha, dtype=torch.float64),
            generator=self._generator,
        )
        noise = noise / noise.sum()
        mixed = (1.0 - self.epsilon) * prior + self.epsilon * noise
        output = list(base_policy)
        for index, value in zip(legal_indices, mixed.tolist()):
            output[index] = float(value)
        return Evaluation(policy=tuple(output), wdl=base.wdl)
