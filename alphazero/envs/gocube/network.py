from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMessageLayer(nn.Module):
    """One topology-aware message-passing layer over the logical Go graph."""

    def __init__(self, hidden_size: int, neighbors):
        super().__init__()
        neighbor_index = torch.as_tensor(neighbors, dtype=torch.long)
        if neighbor_index.ndim != 2:
            raise ValueError("neighbors must be point_count x degree")
        self.register_buffer("neighbors", neighbor_index)
        self.self_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.neighbor_linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        # nodes: batch x points x hidden
        neighbor_nodes = nodes[:, self.neighbors, :]
        neighbor_mean = neighbor_nodes.mean(dim=2)
        return self.self_linear(nodes) + self.neighbor_linear(neighbor_mean)


class GraphResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, neighbors):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.conv1 = GraphMessageLayer(hidden_size, neighbors)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.conv2 = GraphMessageLayer(hidden_size, neighbors)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        residual = nodes
        nodes = self.conv1(F.relu(self.norm1(nodes)))
        nodes = self.conv2(F.relu(self.norm2(nodes)))
        return nodes + residual


def _mlp(input_size: int, hidden_sizes, output_size: int) -> nn.Sequential:
    sizes = [input_size, *hidden_sizes, output_size]
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GraphNet(nn.Module):
    """Policy/value network whose local receptive field is Topology.neighbors().

    The stock framework ResNet assumes planar 3x3 adjacency and would therefore
    be wrong at Torus wraps and Cube seams. This network uses the exact logical
    four-neighbor graph supplied by the configured game class.
    """

    def __init__(self, game_cls, args):
        super().__init__()
        channels, point_count, singleton = game_cls.observation_size()
        if singleton != 1:
            raise ValueError("GoCube graph observations require singleton H dimension")
        self.channels = channels
        self.point_count = point_count
        self.action_size = game_cls.action_size()

        neighbors = game_cls.graph_neighbors()
        if len(neighbors) != point_count:
            raise ValueError("Graph neighbor table does not match observation point count")

        hidden_size = int(args.num_channels)
        depth = int(args.depth)
        self.input_linear = nn.Linear(channels, hidden_size)
        self.blocks = nn.ModuleList(
            GraphResidualBlock(hidden_size, neighbors) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

        self.point_policy = nn.Linear(hidden_size, 1)
        self.pass_policy = nn.Linear(hidden_size, 1)
        self.value_head = _mlp(
            hidden_size,
            list(args.value_dense_layers),
            game_cls.num_players() + game_cls.has_draw(),
        )

    def forward(self, observation: torch.Tensor):
        # Framework input: batch x channels x points x 1.
        observation = observation.view(-1, self.channels, self.point_count)
        nodes = observation.transpose(1, 2)
        nodes = self.input_linear(nodes)
        for block in self.blocks:
            nodes = block(nodes)
        nodes = F.relu(self.output_norm(nodes))

        point_logits = self.point_policy(nodes).squeeze(-1)
        pooled = nodes.mean(dim=1)
        pass_logit = self.pass_policy(pooled)
        policy_logits = torch.cat((point_logits, pass_logit), dim=1)
        if policy_logits.shape[1] != self.action_size:
            raise RuntimeError("Graph policy head action size mismatch")

        value_logits = self.value_head(pooled)
        return F.log_softmax(policy_logits, dim=1), F.log_softmax(value_logits, dim=1)
