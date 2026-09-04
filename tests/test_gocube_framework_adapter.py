from types import SimpleNamespace

import numpy as np
import torch

from alphazero.envs.gocube import (
    Cube4ChineseGame,
    Torus9ChineseGame,
    game_class,
)
from alphazero.envs.gocube.network import GraphMessageLayer, GraphNet


def test_configured_game_classes_expose_exact_action_spaces():
    assert Torus9ChineseGame.action_size() == 82
    assert Torus9ChineseGame.pass_action() == 81
    assert Torus9ChineseGame.action_for_point_id("4,3") == 31
    assert Torus9ChineseGame.point_id_for_action(31) == "4,3"
    assert Torus9ChineseGame.point_id_for_action(81) is None

    assert Cube4ChineseGame.action_size() == 6 * 4 * 4 + 1
    assert Cube4ChineseGame.pass_action() == 96
    assert Cube4ChineseGame.action_for_point_id("back:0:0") == 16
    assert Cube4ChineseGame.action_for_point_id("bottom:3:3") == 95


def test_game_class_registry_matches_supported_training_configs():
    assert game_class("torus", 9) is Torus9ChineseGame
    assert game_class("cube", 4) is Cube4ChineseGame


def test_game_adapter_starts_with_framework_compatible_state():
    game = Torus9ChineseGame()

    assert game.player == 0
    assert game.turns == 0
    assert not game.win_state().any()
    assert game.valid_moves().shape == (82,)
    assert game.valid_moves().sum() == 82
    assert game.observation().shape == (8, 81, 1)
    assert game.observation().dtype == np.float32
    assert np.all(game.observation()[4] == 1.0)


def test_play_action_synchronizes_player_turns_board_and_observation():
    game = Torus9ChineseGame()
    action = game.action_for_point_id("0,0")

    game.play_action(action)

    assert game.player == 1
    assert game.turns == 1
    assert game.semantic_state.board[action] == 1
    observation = game.observation()
    assert observation[0, action, 0] == 1.0
    assert observation[1, action, 0] == 0.0
    assert np.all(observation[4] == -1.0)
    assert np.all(observation[5] == 0.0)


def test_previous_board_and_pass_context_are_present_in_observation():
    game = Torus9ChineseGame()
    point = game.action_for_point_id("0,0")
    game.play_action(point)
    game.play_action(game.pass_action())

    observation = game.observation()
    assert game.player == 0
    assert game.turns == 2
    assert observation[2, point, 0] == 1.0
    assert observation[3, point, 0] == 0.0
    assert np.all(observation[4] == 1.0)
    assert np.all(observation[5] == 1.0)


def test_clone_is_independent_and_preserves_semantic_identity():
    original = Torus9ChineseGame()
    clone = original.clone()

    assert clone == original
    clone.play_action(clone.action_for_point_id("1,1"))

    assert clone != original
    assert original.turns == 0
    assert clone.turns == 1
    assert original.semantic_state.board[original.action_for_point_id("1,1")] == 0


def test_second_pass_is_a_real_terminal_node_with_value_vector():
    game = Torus9ChineseGame()

    game.play_action(game.pass_action())
    assert not game.win_state().any()
    game.play_action(game.pass_action())

    assert game.terminal_adjudication is not None
    assert game.terminal_adjudication.winner == "white"
    assert np.array_equal(game.win_state(), np.array([0, 1, 0], dtype=np.uint8))
    assert not game.valid_moves().any()


def graph_args():
    return SimpleNamespace(
        num_channels=16,
        depth=2,
        value_dense_layers=[16],
    )


def test_graph_network_policy_and_value_shapes_for_torus_and_cube():
    for game_cls in (Torus9ChineseGame, Cube4ChineseGame):
        network = GraphNet(game_cls, graph_args())
        observations = torch.from_numpy(
            np.stack((game_cls().observation(), game_cls().observation()))
        )

        log_policy, log_value = network(observations)

        assert log_policy.shape == (2, game_cls.action_size())
        assert log_value.shape == (2, 3)
        assert torch.allclose(torch.exp(log_policy).sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.allclose(torch.exp(log_value).sum(dim=1), torch.ones(2), atol=1e-6)


def test_graph_message_layer_crosses_torus_wrap_using_logical_neighbors():
    game_cls = Torus9ChineseGame
    topology = game_cls.logical_topology()
    layer = GraphMessageLayer(1, game_cls.graph_neighbors())
    with torch.no_grad():
        layer.self_linear.weight.zero_()
        layer.neighbor_linear.weight.fill_(1.0)

    nodes = torch.zeros((1, topology.point_count, 1))
    nodes[0, topology.point_index("8,0"), 0] = 1.0
    output = layer(nodes)

    assert output[0, topology.point_index("0,0"), 0].item() == 0.25


def test_graph_message_layer_crosses_cube_seam_using_logical_neighbors():
    game_cls = Cube4ChineseGame
    topology = game_cls.logical_topology()
    layer = GraphMessageLayer(1, game_cls.graph_neighbors())
    with torch.no_grad():
        layer.self_linear.weight.zero_()
        layer.neighbor_linear.weight.fill_(1.0)

    nodes = torch.zeros((1, topology.point_count, 1))
    seam_neighbor = topology.neighbor_ids("front:0:0")[0]
    nodes[0, topology.point_index(seam_neighbor), 0] = 1.0
    output = layer(nodes)

    assert output[0, topology.point_index("front:0:0"), 0].item() == 0.25
