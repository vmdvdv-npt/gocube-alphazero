"""Isolated Torus9 GraphNet 8x120 experiment, independent of production GoCube.

The layer/heads are structurally analogous to the GoCube Torus9 GraphNet;
experimental checkpoints use the isolated diagnostics adapter, never the production loader.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

BOARD = 9
POINTS = BOARD * BOARD
ACTIONS = POINTS + 1


def torus_neighbors() -> torch.Tensor:
    rows = []
    for row in range(BOARD):
        for col in range(BOARD):
            rows.append([
                ((row - 1) % BOARD) * BOARD + col,
                ((row + 1) % BOARD) * BOARD + col,
                row * BOARD + (col - 1) % BOARD,
                row * BOARD + (col + 1) % BOARD,
            ])
    return torch.tensor(rows, dtype=torch.long)


class GraphMessageLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("neighbors", torus_neighbors())
        self.self_linear = nn.Linear(width, width)
        self.neighbor_linear = nn.Linear(width, width)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        neigh = nodes[:, self.neighbors, :].mean(dim=2)
        return self.self_linear(nodes) + self.neighbor_linear(neigh)


class GraphResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.message = GraphMessageLayer(width)
        self.output = nn.Linear(width, width)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        return nodes + self.output(F.relu(self.message(self.norm(nodes))))


class TorusNet(nn.Module):
    """Input [batch,C,81]; policy [batch,82], WDL [batch,3]."""

    def __init__(self, channels: int = 5, width: int = 120, blocks: int = 8):
        super().__init__()
        if channels not in (5, 6):
            raise ValueError("Experiment supports exactly 5 or 6 input channels")
        if width <= 0 or blocks <= 0:
            raise ValueError("width/blocks must be positive")
        self.channels = channels
        self.width = width
        self.blocks_count = blocks
        self.input_projection = nn.Linear(channels, width)
        self.blocks = nn.ModuleList(GraphResidualBlock(width) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(width)
        self.point_policy = nn.Linear(width, 1)
        self.pass_policy = nn.Linear(width, 1)
        self.value_head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))
        self.ownership_head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))
        self.score_head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        zero_all_weights(self)

    @property
    def architecture(self):
        return {
            "architecture_id": "experimental-torus9-graphnet8x120-direct-weights-v0",
            "board": [BOARD, BOARD],
            "topology": "torus",
            "input_channels": self.channels,
            "width": self.width,
            "blocks": self.blocks_count,
            "policy_actions": ACTIONS,
            "value_classes": 3,
            "parameter_count": sum(x.numel() for x in self.parameters()),
        }

    def forward(self, observation: torch.Tensor):
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        if observation.shape[1:] != (self.channels, POINTS):
            raise ValueError(f"Expected [batch,{self.channels},{POINTS}], got {tuple(observation.shape)}")
        nodes = self.input_projection(observation.transpose(1, 2))
        for block in self.blocks:
            nodes = block(nodes)
        nodes = F.relu(self.output_norm(nodes))
        pooled = nodes.mean(dim=1)
        policy = torch.cat([self.point_policy(nodes).squeeze(-1), self.pass_policy(pooled)], dim=1)
        return policy, self.value_head(pooled)

    def forward_auxiliary(self, observation: torch.Tensor):
        """Auxiliary output contract is intentionally separate from policy/WDL."""
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        nodes = self.input_projection(observation.transpose(1, 2))
        for block in self.blocks:
            nodes = block(nodes)
        nodes = F.relu(self.output_norm(nodes))
        pooled = nodes.mean(dim=1)
        policy = torch.cat([self.point_policy(nodes).squeeze(-1), self.pass_policy(pooled)], dim=1)
        return policy, self.value_head(pooled), self.ownership_head(nodes), self.score_head(pooled).squeeze(-1)


def zero_all_weights(model: nn.Module):
    """Reset ALL learned parameters; adjacency buffers are topology data, not weights."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


def write_demo_weights(model: TorusNet):
    """Tiny proof-of-mechanism: deterministic handcrafted adjacency prior.

    It is *not* a full Go policy. In particular it cannot recognize groups,
    liberties, ko or life/death. Astra should replace this procedure.
    """
    if model.channels < 5 or model.width < 10 or model.blocks_count < 1:
        raise ValueError("Demo requires >=5 channels, >=10 width, >=1 block")
    zero_all_weights(model)
    with torch.no_grad():
        proj = model.input_projection
        # Layout from latest 5CH Torus contract: own, opponent, side color,
        # last move was pass, legal-point mask; optional 6th channel ignored.
        proj.weight[0, 0] = 1.0  # own occupancy
        proj.weight[1, 1] = 1.0  # opponent occupancy
        proj.weight[2, 4] = 1.0  # legal
        proj.bias[3] = 1.0      # empty
        proj.weight[3, 0] = -1.0
        proj.weight[3, 1] = -1.0
        proj.weight[4, 3] = 1.0  # previous pass
        # To keep the point-level result translation-equivariant, all other
        # transformations are shared across 81 vertices.
        for block in model.blocks:
            block.norm.weight.fill_(1.0)
        model.output_norm.weight.fill_(1.0)
        block = model.blocks[0]
        block.message.neighbor_linear.weight[6, 0] = 1.0
        block.message.neighbor_linear.weight[7, 1] = 1.0
        block.message.neighbor_linear.weight[8, 3] = 1.0
        block.output.weight[6, 6] = 1.0
        block.output.weight[7, 7] = 1.0
        block.output.weight[8, 8] = 1.0
        model.point_policy.weight[0, 2] = 0.4
        model.point_policy.weight[0, 6] = 0.7
        model.point_policy.weight[0, 7] = 0.3
        model.point_policy.weight[0, 8] = 0.2
        model.pass_policy.bias.fill_(-2.0)
        # WDL/ownership/score heads remain zero until designed analytically.


def mask_illegal(policy: torch.Tensor, legal_actions: torch.Tensor):
    """A simple interface contract; real legal masks come from GoCube referee.

    legal_actions must have shape [batch,82] and includes PASS legality.
    """
    if legal_actions.shape != policy.shape or legal_actions.dtype != torch.bool:
        raise ValueError("legal_actions must be boolean [batch,82]")
    if not bool(legal_actions.any(dim=1).all()):
        raise ValueError("No legal action")
    return policy.masked_fill(~legal_actions, -torch.inf)
