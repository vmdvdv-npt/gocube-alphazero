from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.utils import const_temp_scaling, dotdict


def _bare_agent(*, is_arena, prob_fast, arena_sims=100, include_arena_sims=True):
    agent = SelfPlayAgent.__new__(SelfPlayAgent)
    args = dotdict({
        "numMCTSSims": 100,
        "numFastSims": 20,
        "numWarmupSims": 5,
        "probFastSim": prob_fast,
    })
    if include_arena_sims:
        args.arenaMCTSSims = arena_sims
    agent.args = args
    agent._is_arena = is_arena
    agent._is_warmup = False
    agent.fast = False
    return agent


@pytest.mark.parametrize("prob_fast", [0.0, 0.25, 0.75, 1.0])
def test_arena_never_uses_fast_search(prob_fast):
    agent = _bare_agent(is_arena=True, prob_fast=prob_fast, arena_sims=73)
    for _ in range(16):
        assert agent._select_search_sims() == 73
        assert agent.fast is False


def test_arena_budget_falls_back_to_regular_budget_for_legacy_args():
    agent = _bare_agent(
        is_arena=True, prob_fast=1.0, include_arena_sims=False,
    )
    assert agent._select_search_sims() == 100
    assert agent.fast is False


def test_self_play_prob_zero_uses_regular_search():
    agent = _bare_agent(is_arena=False, prob_fast=0.0)
    assert agent._select_search_sims() == 100
    assert agent.fast is False


def test_self_play_prob_one_uses_fast_search():
    agent = _bare_agent(is_arena=False, prob_fast=1.0)
    assert agent._select_search_sims() == 20
    assert agent.fast is True


class _FakePause:
    @staticmethod
    def is_set():
        return False


class _FakeMCTS:
    def __init__(self):
        self.flags = None

    def process_results(self, _game, _value, _policy, add_root_noise, add_root_temp):
        self.flags = (add_root_noise, add_root_temp)


def test_arena_process_results_forces_root_noise_and_temp_off():
    agent = SelfPlayAgent.__new__(SelfPlayAgent)
    agent._is_arena = True
    agent._is_warmup = True
    agent.batch_size = 1
    agent.batch_indices = [0]
    agent.games = [SimpleNamespace(player=0)]
    agent.pause_event = _FakePause()
    agent.args = SimpleNamespace(add_root_noise=True, add_root_temp=True)
    agent.value_tensor = torch.zeros((1, 3), dtype=torch.float32)
    agent.policy_tensor = torch.zeros((1, 2), dtype=torch.float32)
    mcts = _FakeMCTS()
    agent.mcts = [(mcts, _FakeMCTS())]

    agent.processBatch()

    assert mcts.flags == (False, False)


class _DummyGame:
    @staticmethod
    def num_players():
        return 2


class _DummyPlayer:
    def __init__(self, args):
        self.args = args
        self.temp = args.startTemp

    @staticmethod
    def supports_process():
        return False


def test_arena_normalizes_budget_and_exploration_without_mutating_training_args():
    training_args = dotdict({
        "numMCTSSims": 100,
        "arenaMCTSSims": 61,
        "probFastSim": 0.75,
        "add_root_noise": True,
        "add_root_temp": True,
        "startTemp": 1.0,
        "arenaTemp": 0.25,
    })
    players = [_DummyPlayer(training_args), _DummyPlayer(training_args)]

    arena = Arena(players, _DummyGame, use_batched_mcts=False, args=training_args)

    assert arena.args.numMCTSSims == 61
    assert arena.args.arenaMCTSSims == 61
    assert arena.args.probFastSim == 0.0
    assert arena.args.add_root_noise is False
    assert arena.args.add_root_temp is False
    assert arena.args.startTemp == 0.25
    assert arena.args.arenaTemp == 0.25
    assert all(player.args is arena.args for player in players)
    assert all(player.temp == 0.25 for player in players)

    assert training_args.numMCTSSims == 100
    assert training_args.probFastSim == 0.75
    assert training_args.add_root_noise is True
    assert training_args.add_root_temp is True
    assert training_args.startTemp == 1.0


def test_deterministic_arena_policy_is_one_hot_at_zero_temperature():
    visits = np.array([1.0, 7.0, 3.0], dtype=np.float32)
    max_index = int(np.argmax(visits))
    policy = np.zeros_like(visits)
    policy[max_index] = 1.0
    assert np.random.choice(len(policy), p=policy) == max_index


class _UniformTechnicalNet:
    def __init__(self, game_cls):
        self.action_size = game_cls.action_size()
        self.value_size = game_cls.num_players() + game_cls.has_draw()
        self.calls = 0

    def __call__(self, _observation):
        self.calls += 1
        policy = np.full(self.action_size, 1.0 / self.action_size, dtype=np.float32)
        value = np.zeros(self.value_size, dtype=np.float32)
        return policy, value


def _functional_arena_args():
    return dotdict({
        "numMCTSSims": 1,
        "arenaMCTSSims": 2,
        "numFastSims": 1,
        "probFastSim": 1.0,
        "add_root_noise": True,
        "add_root_temp": True,
        "startTemp": 1.0,
        "arenaTemp": 0.0,
        "root_noise_frac": 0.1,
        "root_policy_temp": 1.1,
        "min_discount": 1.0,
        "fpu_reduction": 0.2,
        "cpuct": 1.25,
        "_num_players": Cube4JapaneseGame.num_players() + Cube4JapaneseGame.has_draw(),
        "use_draws_for_winrate": True,
        "temp_scaling_fn": const_temp_scaling,
    })


def test_gocube_arena_functional_smoke_with_technical_mcts_models():
    np.random.seed(7)
    training_args = _functional_arena_args()
    candidate_net = _UniformTechnicalNet(Cube4JapaneseGame)
    baseline_net = _UniformTechnicalNet(Cube4JapaneseGame)
    players = [
        MCTSPlayer(candidate_net, Cube4JapaneseGame, training_args),
        MCTSPlayer(baseline_net, Cube4JapaneseGame, training_args),
    ]

    arena = Arena(players, Cube4JapaneseGame, use_batched_mcts=False, args=training_args)
    wins, draws, winrates = arena.play_games(2, shuffle_players=True)

    assert arena.args.numMCTSSims == 2
    assert arena.args.arenaMCTSSims == 2
    assert arena.args.probFastSim == 0.0
    assert arena.args.add_root_noise is False
    assert arena.args.add_root_temp is False
    assert arena.args.arenaTemp == 0.0
    assert arena.games_played == 2
    assert sum(wins) + draws + arena.no_results == 2
    assert len(winrates) == 2
    assert candidate_net.calls > 0
    assert baseline_net.calls > 0


class _OneTurnGame:
    def __init__(self):
        self.player = 0
        self.turns = 0

    @staticmethod
    def num_players():
        return 2

    def play_action(self, _action):
        self.turns = 1

    def win_state(self):
        if self.turns:
            return np.array([True, False, False], dtype=np.bool_)
        return np.array([False, False, False], dtype=np.bool_)


class _RecordingPlayer:
    def __init__(self, label, first_movers):
        self.label = label
        self.first_movers = first_movers
        self.args = None

    def __call__(self, _state):
        self.first_movers.append(self.label)
        return 0

    @staticmethod
    def reset():
        return None

    @staticmethod
    def update(_state, _action):
        return None

    @staticmethod
    def supports_process():
        return False


def test_two_player_non_batched_arena_balances_colors_for_even_game_count():
    first_movers = []
    args = dotdict({
        "numMCTSSims": 100,
        "arenaMCTSSims": 100,
        "probFastSim": 0.75,
        "add_root_noise": True,
        "add_root_temp": True,
        "startTemp": 1.0,
        "arenaTemp": 0.0,
        "use_draws_for_winrate": True,
    })
    players = [
        _RecordingPlayer(0, first_movers),
        _RecordingPlayer(1, first_movers),
    ]

    arena = Arena(players, _OneTurnGame, use_batched_mcts=False, args=args)
    arena.play_games(4, shuffle_players=True)

    assert first_movers.count(0) == 2
    assert first_movers.count(1) == 2
