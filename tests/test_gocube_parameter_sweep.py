from types import SimpleNamespace

import pytest
import torch

import alphazero.envs.gocube.pinned_selfplay as pinned_selfplay_module
from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.MCTS import MCTS
from alphazero.envs.gocube.exploration_contract import KATAGO_PINNED_EXPLORATION_DEFAULTS
from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import (
    KataGoSearchCoach,
    build_katago_training_args,
    checkpoint_arg_overrides,
    parse_args,
)
from alphazero.envs.gocube.pinned_selfplay import PinnedSelfPlayAgent
from alphazero.envs.gocube.production_training import build_replay_training_plan
from alphazero.search_contract import SearchOutput
from alphazero.utils import const_temp_scaling


def test_parameter_sweep_defaults_do_not_change_production_contract():
    cli = parse_args([])
    game_cls, args = build_hardened_training_args(cli)
    defaults = KATAGO_PINNED_EXPLORATION_DEFAULTS

    assert args.numMCTSSims == 50
    assert args.numFastSims == 20
    assert args.gocube_komi == 0.5
    assert args.gocube_chosen_move_temperature_halflife == defaults[
        "chosen_move_temperature_halflife"
    ] == 19.0
    assert args.gocube_root_dirichlet_noise_weight == defaults[
        "root_dirichlet_noise_weight"
    ] == 0.25
    assert args.root_noise_frac == 0.25
    assert args.gocube_root_dirichlet_noise_total_concentration == 10.83
    assert args.gocube_replay_window_iters is None
    assert args.gocube_arena_batched is False
    assert args.arenaBatched is False
    assert checkpoint_arg_overrides(cli, args) == {}
    assert game_cls.logical_topology().point_count == 96


def test_temperature_halflife_override_reaches_pinned_selfplay(monkeypatch):
    cli = parse_args(["--chosen-move-temperature-halflife", "11"])
    _, args = build_hardened_training_args(cli)
    captured = []

    class StopAfterTemperature(Exception):
        pass

    def capture_temperature(
        _turn_number,
        _point_count,
        *,
        early_temperature,
        temperature,
        halflife,
    ):
        captured.append((early_temperature, temperature, halflife))
        raise StopAfterTemperature

    monkeypatch.setattr(
        pinned_selfplay_module,
        "chosen_move_temperature",
        capture_temperature,
    )

    agent = object.__new__(PinnedSelfPlayAgent)
    agent.score_aware = True
    agent._is_arena = False
    agent._is_warmup = False
    agent.args = args
    agent.batch_size = 1
    agent.temps = [1.0]
    agent.games = [SimpleNamespace(turns=3)]
    agent.game_cls = SimpleNamespace(
        logical_topology=lambda: SimpleNamespace(point_count=96)
    )

    with pytest.raises(StopAfterTemperature):
        agent.playMoves()

    assert captured == [(0.75, 0.15, 11.0)]
    assert checkpoint_arg_overrides(cli, args) == {
        "gocube_chosen_move_temperature_halflife": 11.0,
    }


def test_dirichlet_weight_override_reaches_used_root_noise_arg():
    cli = parse_args(["--root-dirichlet-noise-weight", "0.4"])
    _, args = build_katago_training_args(cli)
    args._num_players = 3
    mcts = MCTS(args)

    assert args.gocube_root_dirichlet_noise_weight == pytest.approx(0.4)
    assert args.root_noise_frac == pytest.approx(0.4)
    assert mcts.root_noise_frac == pytest.approx(0.4)
    assert args.gocube_root_dirichlet_noise_total_concentration == pytest.approx(10.83)
    overrides = checkpoint_arg_overrides(cli, args)
    assert set(overrides) == {
        "gocube_root_dirichlet_noise_weight",
        "root_noise_frac",
    }
    assert overrides["gocube_root_dirichlet_noise_weight"] == pytest.approx(0.4)
    assert overrides["root_noise_frac"] == pytest.approx(0.4)


def _replay_coach(explicit_window):
    coach = object.__new__(KataGoSearchCoach)
    coach.args = SimpleNamespace(
        gocube_replay_window_iters=explicit_window,
        minTrainHistoryWindow=4,
        trainHistoryIncrementIters=2,
        maxTrainHistoryWindow=20,
    )
    return coach


def test_explicit_replay_window_selects_exact_last_iterations_and_none_keeps_old_schedule():
    explicit = _replay_coach(3)
    assert list(explicit._replay_iterations(10)) == [8, 9, 10]
    assert list(explicit._replay_iterations(2)) == [1, 2]

    production = _replay_coach(None)
    assert list(production._replay_iterations(10)) == list(range(3, 11))


def test_replay_window_does_not_change_sample_based_training_budget():
    small = build_replay_training_plan(
        new_selfplay_samples=500,
        replay_window_samples=1_000,
        train_samples_per_new_sample=1.25,
        batch_size=256,
    )
    large = build_replay_training_plan(
        new_selfplay_samples=500,
        replay_window_samples=100_000,
        train_samples_per_new_sample=1.25,
        batch_size=256,
    )
    assert small.planned_training_samples == large.planned_training_samples == 625
    assert small.planned_optimizer_steps == large.planned_optimizer_steps


class _TechnicalFourHeadNet:
    def __init__(self, game_cls):
        self.action_size = game_cls.action_size()
        self.value_size = game_cls.num_players() + game_cls.has_draw()
        self.point_count = game_cls.logical_topology().point_count
        self.batch_calls = 0

    def process_for_search(self, batch):
        self.batch_calls += 1
        rows = int(batch.shape[0])
        return SearchOutput(
            policy=torch.full((rows, self.action_size), 1.0 / self.action_size),
            value=torch.full((rows, self.value_size), 1.0 / self.value_size),
            score=torch.zeros((rows, 1)),
            ownership=torch.full((rows, self.point_count, 3), 1.0 / 3.0),
        )


def test_batched_arena_smoke_completes_with_four_heads_and_keeps_exploration_off():
    cli = parse_args([
        "--size", "2",
        "--workers", "2",
        "--sims", "1",
        "--arena-sims", "1",
        "--games-per-iteration", "2",
        "--arena-batched",
        "--smoke",
    ])
    game_cls, args = build_hardened_training_args(cli)
    args.cuda = False
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    args.arena_batch_size = 4
    args.numMCTSSims = 1
    args.arenaMCTSSims = 1
    args.probFastSim = 1.0
    args.add_root_noise = True
    args.add_root_temp = True
    args.startTemp = 1.0
    args.arenaTemp = 0.0
    args.temp_scaling_fn = const_temp_scaling
    args.use_draws_for_winrate = True

    net_a = _TechnicalFourHeadNet(game_cls)
    net_b = _TechnicalFourHeadNet(game_cls)
    players = [
        MCTSPlayer(net_a, game_cls=game_cls, args=args),
        MCTSPlayer(net_b, game_cls=game_cls, args=args),
    ]
    arena = Arena(players, game_cls, use_batched_mcts=True, args=args)
    wins, draws, winrates = arena.play_games(2, shuffle_players=True)

    assert arena.games_played == 2
    assert sum(wins) + draws + arena.no_results == 2
    assert len(winrates) == 2
    assert net_a.batch_calls > 0
    assert net_b.batch_calls > 0
    assert arena.args.arena_batch_size == 1
    assert arena.args.numMCTSSims == 1
    assert arena.args.arenaMCTSSims == 1
    assert arena.args.probFastSim == 0.0
    assert arena.args.add_root_noise is False
    assert arena.args.add_root_temp is False
    assert arena.args.startTemp == 0.0
    assert arena.args.arenaTemp == 0.0
    color_split = arena.player_color_results(0)
    assert color_split["black"]["games"] + color_split["white"]["games"] == 2
