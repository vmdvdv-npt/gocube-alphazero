import numpy as np
import pytest

from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.search_contract import (
    ownership_root_ending_bonus_points,
    ownership_root_move_useful,
)


def _neutral_ownership(game):
    ownership = np.zeros((game.logical_topology().point_count, 3), dtype=np.float32)
    ownership[:, 2] = 1.0
    return ownership


def test_neutral_dame_like_point_remains_useful_and_unpenalized():
    game = Cube4JapaneseGame()
    ownership = _neutral_ownership(game)
    action = 0

    assert ownership_root_move_useful(game, action, ownership) is True
    assert ownership_root_ending_bonus_points(game, action, ownership, 0.5) == pytest.approx(0.0)


def test_filling_strong_own_territory_is_discouraged_without_changing_legality():
    game = Cube4JapaneseGame()
    ownership = _neutral_ownership(game)
    action = 0
    ownership[action] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert game.valid_moves()[action] == 1
    assert ownership_root_move_useful(game, action, ownership) is False
    assert ownership_root_ending_bonus_points(game, action, ownership, 0.5) == pytest.approx(-0.5)


def test_deep_opponent_territory_is_not_forced_unless_near_strong_own_ownership():
    game = Cube4JapaneseGame()
    ownership = _neutral_ownership(game)
    action = 0
    ownership[action] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert ownership_root_move_useful(game, action, ownership) is False

    neighbor = game.logical_topology().neighbor_indices(action)[0]
    ownership[neighbor] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert ownership_root_move_useful(game, action, ownership) is True


def test_pass_ending_adjustment_is_score_point_utility_input_only():
    game = Cube4JapaneseGame()
    ownership = _neutral_ownership(game)
    pass_action = game.pass_action()

    assert game.valid_moves()[pass_action] == 1
    assert ownership_root_move_useful(game, pass_action, ownership) is False
    assert ownership_root_ending_bonus_points(game, pass_action, ownership, 0.5) == pytest.approx(-1.0 / 3.0)
