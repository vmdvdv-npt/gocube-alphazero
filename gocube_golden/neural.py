"""The small, semantic boundary between Golden rules and PyTorch.

This module deliberately does not import anything from the legacy AlphaZero
training tree.  The network consumes the canonical Golden observation and
returns raw policy/value logits; search owns legality masking and WDL utility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .search import Evaluation
from .state import BLACK, PASS, WHITE, GoldenState
from .rules import legal_actions
from .topology import TORUS_5X5


OBSERVATION_SCHEMA_ID = "gocube-torus-golden-observation-v1"
OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_LAYOUT = "[channels,points]"
OBSERVATION_CHANNELS = (
    "own_stones",
    "opponent_stones",
    "side_to_move_color",
    "previous_pass",
    "legal_point_mask",
    "komi",
)
ACTION_COUNT = 26
PASS_INDEX = 25
VALUE_HEAD_SEMANTICS = "side-to-move:[WIN,DRAW,LOSS]"


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# This is deliberately the frozen Golden v2 observation identity referenced by
# Stage 3.  The Stage-3 profile adds the canonical tensor layout explicitly,
# but does not mint a semantically different observation identity.
OBSERVATION_FINGERPRINT = "sha256:d6e3aecc89f7df84f6e758423da4e3fe9269abeca644070db0be9261b30c6361"


@dataclass(frozen=True)
class GoldenObservation:
    """A checked observation plus the complete action mask used by search."""

    tensor: Tensor
    action_mask: tuple[bool, ...]
    state_key: tuple[object, ...]

    def __post_init__(self) -> None:
        if tuple(self.tensor.shape) != (6, 25):
            raise ValueError("Golden observation tensor must have shape [6,25]")
        if self.tensor.dtype != torch.float32:
            raise ValueError("Golden observation tensor must be float32")
        if len(self.action_mask) != ACTION_COUNT:
            raise ValueError("Golden action mask must have length 26")
        if not bool(torch.isfinite(self.tensor).all()):
            raise ValueError("Golden observation contains NaN or Inf")


def build_action_mask(state: GoldenState) -> tuple[bool, ...]:
    """Return the 25-point legality mask plus PASS at index 25."""
    if state.is_terminal:
        raise ValueError("Terminal Golden states must never be passed to the NN")
    legal = set(legal_actions(state))
    mask = tuple(point in legal for point in range(state.topology.point_count))
    if state.topology.point_count != 25:
        raise ValueError("Stage-3 observation is fixed to the 25-point Golden Torus")
    return mask + (PASS in legal,)


def build_observation_bundle(state: GoldenState) -> GoldenObservation:
    """Build the canonical [6,25] float32 observation for a live state."""
    if state.is_terminal:
        raise ValueError("Terminal Golden states must never be passed to the NN")
    if state.topology.fingerprint != TORUS_5X5.fingerprint:
        raise ValueError("Stage-3 observation requires canonical Golden Torus 5x5")
    mask = build_action_mask(state)
    own = int(state.side_to_move)
    other = int(WHITE if state.side_to_move == BLACK else BLACK)
    values = torch.zeros((6, 25), dtype=torch.float32)
    for point, stone in enumerate(state.stones):
        values[0, point] = float(int(stone) == own)
        values[1, point] = float(int(stone) == other)
    values[2].fill_(1.0 if state.side_to_move == BLACK else -1.0)
    values[3].fill_(1.0 if state.consecutive_passes == 1 else 0.0)
    values[4] = torch.tensor(mask[:25], dtype=torch.float32)
    values[5].fill_(0.5)
    return GoldenObservation(values, mask, state.state_key)


def build_observation(state: GoldenState) -> Tensor:
    """Convenience API returning only the canonical tensor."""
    return build_observation_bundle(state).tensor


class GoldenGraphMessageLayer(nn.Module):
    def __init__(self, hidden: int, neighbors: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.register_buffer("neighbors", torch.as_tensor(neighbors, dtype=torch.long))
        self.self_linear = nn.Linear(hidden, hidden)
        self.neighbor_linear = nn.Linear(hidden, hidden)

    def forward(self, nodes: Tensor) -> Tensor:
        # nodes: [batch, points, hidden]; adjacency is created once at init.
        gathered = nodes[:, self.neighbors, :]
        neighbor_mean = gathered.mean(dim=2)
        return self.self_linear(nodes) + self.neighbor_linear(neighbor_mean)


class GoldenGraphResidualBlock(nn.Module):
    def __init__(self, hidden: int, neighbors: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.message = GoldenGraphMessageLayer(hidden, neighbors)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, nodes: Tensor) -> Tensor:
        return nodes + self.output(F.relu(self.message(self.norm(nodes))))


class GoldenGraphNetV1(nn.Module):
    """Minimal graph-aware policy + side-to-move WDL network."""

    architecture_id = "GoldenGraphNetV1"

    def __init__(
        self,
        *,
        topology=TORUS_5X5,
        hidden: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        if topology.fingerprint != TORUS_5X5.fingerprint:
            raise ValueError("GoldenGraphNetV1 is fixed to the canonical Golden Torus")
        if hidden <= 0 or blocks <= 0:
            raise ValueError("GoldenGraphNetV1 hidden and blocks must be positive")
        self.topology_id = topology.topology_id
        self.topology_fingerprint = topology.fingerprint
        self.hidden = int(hidden)
        self.blocks_count = int(blocks)
        neighbors = topology.adjacency
        self.input_projection = nn.Linear(6, hidden)
        self.blocks = nn.ModuleList(
            GoldenGraphResidualBlock(hidden, neighbors) for _ in range(blocks)
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
            "input_channels": 6,
            "point_count": 25,
            "hidden": self.hidden,
            "blocks": self.blocks_count,
            "heads": {"policy": [26], "value": [3]},
        }

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        if tuple(observation.shape[1:]) != (6, 25):
            raise ValueError("GoldenGraphNetV1 expects [batch,6,25]")
        nodes = observation.transpose(1, 2)
        nodes = self.input_projection(nodes)
        for block in self.blocks:
            nodes = block(nodes)
        nodes = F.relu(self.output_norm(nodes))
        policy_logits = torch.cat(
            (self.point_policy(nodes).squeeze(-1), self.pass_policy(nodes.mean(dim=1))),
            dim=1,
        )
        value_logits = self.value_head(nodes.mean(dim=1))
        return policy_logits, value_logits


def model_hash(model: nn.Module) -> str:
    """Hash architecture identity and canonical parameter bytes, not a label."""
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


class GoldenNeuralEvaluator:
    """GoldenState -> finite policy probabilities and [WIN,DRAW,LOSS]."""

    def __init__(self, model: nn.Module, *, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        # These names make provenance/audit explicit without importing legacy code.
        self.checkpoint_path: str | None = None
        self.checkpoint_metadata: Mapping[str, object] | None = None
        self.nn_evaluations = 0

    def evaluate(self, state: GoldenState) -> Evaluation:
        if state.is_terminal:
            raise RuntimeError("GoldenNeuralEvaluator must never evaluate a terminal state")
        observation = build_observation(state).to(self.device)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            wdl = torch.softmax(value_logits[0], dim=0)
        self.nn_evaluations += 1
        policy_values = tuple(float(value) for value in policy.detach().cpu())
        wdl_values = tuple(float(value) for value in wdl.detach().cpu())
        if len(policy_values) != ACTION_COUNT or len(wdl_values) != 3:
            raise RuntimeError("Golden neural head shape drift")
        if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
            raise RuntimeError("Golden neural evaluator produced non-finite output")
        if not math.isclose(sum(policy_values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError("Golden policy probabilities are not normalized")
        if not math.isclose(sum(wdl_values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError("Golden WDL probabilities are not normalized")
        return Evaluation(policy=policy_values, wdl=wdl_values)


class SelfPlayRootNoiseEvaluator:
    """Add root-only Dirichlet exploration while leaving Arena PUCT unchanged."""

    def __init__(
        self,
        evaluator: GoldenNeuralEvaluator,
        root_state: GoldenState,
        *,
        seed: int,
        epsilon: float = 0.25,
        alpha: float = 0.30,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0 or alpha <= 0.0:
            raise ValueError("Invalid self-play Dirichlet parameters")
        self.evaluator = evaluator
        self.root_state_key = root_state.state_key
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def evaluate(self, state: GoldenState) -> Evaluation:
        base = self.evaluator.evaluate(state)
        if state.state_key != self.root_state_key:
            return base
        legal = legal_actions(state)
        if not legal:
            raise RuntimeError("Self-play root has no legal actions")
        base_policy = [float(value) for value in base.policy]
        if len(base_policy) != ACTION_COUNT:
            raise RuntimeError("Root-noise evaluator received the wrong policy shape")
        legal_indices = [PASS_INDEX if action == PASS else int(action) for action in legal]
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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
