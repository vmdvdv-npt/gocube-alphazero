import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyximport
import pytest

pyximport.install(setup_args={"include_dirs": np.get_include()})

from alphazero.MCTS import MCTS
from alphazero.envs.connect4.connect4 import Game as Connect4Game
from alphazero.envs.gocube import (
    Cube4JapaneseGame,
    Cube3JapaneseGame,
    Cube3JapaneseV2Game,
    initial_v3_state,
    Torus9JapaneseGame,
)
from alphazero.envs.gocube.training_common import GoCubeCoach
from alphazero.search_contract import SearchOutput


def _mcts_args(mode):
    return SimpleNamespace(
        root_noise_frac=0.0,
        root_policy_temp=1.0,
        min_discount=1.0,
        fpu_reduction=0.0,
        cpuct=1.0,
        _num_players=3,
        search_utility_mode=mode,
    )


def _v3_game(game_cls, side_to_move):
    state = replace(
        initial_v3_state(game_cls.logical_topology()),
        current_player=int(side_to_move),
    )
    return game_cls(state)


def _policy(game_cls):
    return np.full(
        game_cls.action_size(),
        1.0 / game_cls.action_size(),
        dtype=np.float32,
    )


class _LegacyNetwork:
    def __init__(self, game_cls, value):
        self.game_cls = game_cls
        self.value = np.asarray(value, dtype=np.float32)
        self.calls = 0

    def __call__(self, _observation):
        self.calls += 1
        return _policy(self.game_cls), self.value


class _PinnedNetwork:
    def __init__(self, game_cls, value):
        self.game_cls = game_cls
        self.value = np.asarray(value, dtype=np.float32)
        self.calls = 0

    def predict_for_search(self, _observation):
        self.calls += 1
        point_count = self.game_cls.logical_topology().point_count
        return SearchOutput(
            policy=_policy(self.game_cls),
            value=self.value,
            score=np.array([0.0], dtype=np.float32),
            ownership=np.tile(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                (point_count, 1),
            ),
        )


class _UnprotectedV3ProbeGame(Cube3JapaneseGame):
    """V3 rules with the marker disabled only to observe legacy process_results."""

    GOCUBE_V3 = False


@pytest.mark.parametrize(
    ("side_to_move", "outcome", "value", "legacy_slot", "pinned_utility"),
    (
        (0, "win", [1.0, 0.0, 0.0], 1.0, -1.0),
        (0, "loss", [0.0, 1.0, 0.0], 0.0, 1.0),
        (1, "win", [1.0, 0.0, 0.0], 0.0, 1.0),
        (1, "loss", [0.0, 1.0, 0.0], 1.0, -1.0),
    ),
    ids=(
        "black-to-move-win",
        "black-to-move-loss",
        "white-to-move-win",
        "white-to-move-loss",
    ),
)
def test_v3_value_target_reproduction_compares_legacy_process_results_and_pinned_search(
    side_to_move, outcome, value, legacy_slot, pinned_utility
):
    """The legacy slot lookup is the exact white-to-move semantic mismatch."""

    # Run the actual legacy MCTS.process_results() with only the boundary
    # marker disabled, so the new invariant does not hide the observed bug.
    legacy_game = _v3_game(_UnprotectedV3ProbeGame, side_to_move)
    legacy_mcts = MCTS(_mcts_args("legacy"))
    legacy_leaf = legacy_mcts.find_leaf(legacy_game)
    legacy_mcts.process_results(
        legacy_leaf,
        np.asarray(value, dtype=np.float32),
        _policy(legacy_game),
        False,
        False,
    )
    assert float(legacy_mcts._root.v) == legacy_slot

    # The real V3 game is accepted only by the pinned process_search_results()
    # path, which converts the relative response to absolute Black/White.
    pinned_game = _v3_game(Cube3JapaneseGame, side_to_move)
    pinned_network = _PinnedNetwork(Cube3JapaneseGame, value)
    pinned_mcts = MCTS(_mcts_args("katago-pinned-f6bc4b19"))
    pinned_mcts.search(pinned_game, pinned_network, 1, False, False)
    assert pinned_network.calls == 1
    assert float(pinned_mcts._root.q) == pinned_utility


def test_gocube_v3_legacy_search_fails_before_neural_inference():
    game = _v3_game(Cube3JapaneseGame, 0)
    network = _LegacyNetwork(Cube3JapaneseGame, [1.0, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="GoCube V3 requires the pinned KataGo search contract"):
        MCTS(_mcts_args("legacy")).search(game, network, 1, False, False)

    assert network.calls == 0


def test_gocube_coach_rejects_legacy_search_before_selfplay_setup():
    with pytest.raises(RuntimeError, match="GoCube V3 requires the pinned KataGo search contract"):
        GoCubeCoach(
            Cube3JapaneseGame,
            object(),
            _mcts_args("legacy"),
        )


def test_raw_legacy_search_also_fails_closed_for_v3():
    with pytest.raises(RuntimeError, match="GoCube V3 requires the pinned KataGo search contract"):
        MCTS(_mcts_args("katago-pinned-f6bc4b19")).raw_search(
            _v3_game(Cube3JapaneseGame, 1), 1, False, False
        )


@pytest.mark.parametrize("game_cls", (Cube3JapaneseGame, Cube4JapaneseGame, Torus9JapaneseGame))
def test_all_ci_gocube_v3_topologies_use_the_pinned_search_contract(game_cls):
    game = _v3_game(game_cls, 0)
    network = _PinnedNetwork(game_cls, [1.0, 0.0, 0.0])

    mcts = MCTS(_mcts_args("katago-pinned-f6bc4b19"))
    mcts.search(game, network, 1, False, False)

    assert game_cls.GOCUBE_V3 is True
    assert network.calls == 1
    assert int(mcts._root.n) == 1
    assert float(mcts._root.q) == -1.0


def test_v3_legacy_value_update_method_fails_even_on_pinned_mcts():
    game = _v3_game(Cube3JapaneseGame, 0)
    mcts = MCTS(_mcts_args("katago-pinned-f6bc4b19"))
    leaf = mcts.find_leaf(game)

    with pytest.raises(RuntimeError, match="GoCube V3 requires the pinned KataGo search contract"):
        mcts.process_results(
            leaf,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            _policy(Cube3JapaneseGame),
            False,
            False,
        )


def test_historical_v2_go_cube_can_still_use_legacy_mcts_surface():
    game = Cube3JapaneseV2Game()
    mcts = MCTS(_mcts_args("legacy"))
    leaf = mcts.find_leaf(game)
    mcts.process_results(
        leaf,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        _policy(Cube3JapaneseV2Game),
        False,
        False,
    )
    assert float(mcts._root.v) == 1.0


def test_non_gocube_legacy_mcts_behavior_is_unchanged():
    game = Connect4Game()
    mcts = MCTS(_mcts_args("legacy"))
    leaf = mcts.find_leaf(game)
    mcts.process_results(
        leaf,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        _policy(Connect4Game),
        False,
        False,
    )
    assert int(mcts._root.n) == 1


def test_legacy_training_module_is_retired_fail_closed():
    result = subprocess.run(
        [sys.executable, "-m", "alphazero.envs.gocube.train"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "legacy entrypoint is retired" in output
    assert "katago_train" in output
