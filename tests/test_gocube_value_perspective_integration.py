from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyximport
import pytest
import torch

pyximport.install(setup_args={"include_dirs": np.get_include()})

from alphazero.MCTS import MCTS
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube import (
    NO_RESULT,
    SCORED,
    V3Terminal,
    Cube4JapaneseGame,
    build_v3_training_targets,
    initial_v3_state,
)
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.katago_v3 import apply_v3_action
from alphazero.envs.gocube.diversified_game import diversified_structural_pinned_game_class


GAME = diversified_structural_pinned_game_class(Cube4JapaneseGame)


def _search_args():
    game_cls, args = build_katago_training_args(parse_args([]))
    assert game_cls is GAME
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    args.gocube_win_loss_utility_factor = 1.0
    args.gocube_static_score_utility_factor = 0.0
    args.gocube_dynamic_score_utility_factor = 0.0
    args.add_root_noise = False
    args.add_root_temp = False
    return game_cls, args


def _terminal_fixture(outcome, point_count):
    if outcome == "no_result":
        return V3Terminal(NO_RESULT, None, None, None, "cycle")

    score = SimpleNamespace(
        black=0.0,
        white=0.0,
        winner=outcome,
    )
    ownership = np.zeros((point_count, 3), dtype=np.float32)
    ownership[:, 2] = 1.0
    return V3Terminal(
        SCORED,
        score,
        ownership,
        np.ones(point_count, dtype=np.float32),
    )


class _FixedSearchNetwork(torch.nn.Module):
    def __init__(self, action_size, point_count, value):
        super().__init__()
        self.log_policy = torch.full((1, action_size), -np.log(action_size), dtype=torch.float32)
        self.log_value = torch.log(torch.as_tensor(value, dtype=torch.float32)).view(1, 3)
        ownership = torch.zeros(point_count, 3, dtype=torch.float32)
        ownership[:, 2] = 1.0
        self.log_ownership = torch.log(ownership).view(1, point_count, 3)
        self.score = torch.zeros((1, 1), dtype=torch.float32)

    def forward(self, _batch):
        return self.log_policy, self.log_value, self.log_ownership, self.score


def _wrapper(game_cls, value):
    wrapper = object.__new__(NNetWrapper)
    wrapper.args = SimpleNamespace(cuda=False)
    wrapper.nnet = _FixedSearchNetwork(
        game_cls.action_size(),
        game_cls.logical_topology().point_count,
        value,
    )
    return wrapper


def _nonterminal_game(game_cls, side_to_move):
    state = replace(initial_v3_state(game_cls.logical_topology()), current_player=side_to_move)
    return game_cls(state)


def _expected_utility(outcome):
    return {
        "black": -1.0,
        "white": 1.0,
        "draw": 0.0,
        "no_result": 0.0,
    }[outcome]


def _value_target(outcome, side_to_move, topology):
    terminal = _terminal_fixture(outcome, topology.point_count)
    return build_v3_training_targets(terminal, side_to_move, topology).value_target


@pytest.mark.parametrize("side_to_move", [0, 1], ids=["black-to-move", "white-to-move"])
@pytest.mark.parametrize("outcome", ["black", "white", "draw", "no_result"])
def test_v3_training_target_network_mcts_preserves_absolute_white_utility(side_to_move, outcome):
    game_cls, args = _search_args()
    topology = game_cls.logical_topology()
    target = _value_target(outcome, side_to_move, topology)
    game = _nonterminal_game(game_cls, side_to_move)
    network = _wrapper(game_cls, target)

    network_output = network.predict_for_search(game.observation())
    np.testing.assert_allclose(network_output.value, target)

    mcts = MCTS(args)
    mcts.search(game, network, 1, False, False)

    assert np.isclose(mcts._root.q, _expected_utility(outcome))


def _after_passes(game_cls, count, starting_player):
    topology = game_cls.logical_topology()
    state = replace(initial_v3_state(topology), current_player=starting_player)
    for _ in range(count):
        state = apply_v3_action(state, topology.pass_action, topology)
    return state


def test_exact_terminal_white_result_bypasses_relative_value_conversion():
    game_cls, args = _search_args()
    state = _after_passes(game_cls, 6, starting_player=1)
    game = game_cls(state)

    assert game.player == 1
    np.testing.assert_array_equal(game.win_state(), np.array([0, 1, 0], dtype=np.uint8))

    wrong_neural_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    network = _wrapper(game_cls, wrong_neural_value)
    mcts = MCTS(args)
    mcts.search(game, network, 1, False, False)

    assert np.isclose(mcts._root.q, 1.0)
