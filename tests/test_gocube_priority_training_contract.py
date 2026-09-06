from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from alphazero.envs.gocube.exploration_contract import (
    KATAGO_PINNED_EXPLORATION_DEFAULTS,
    retrospectively_reduce_root_visits,
    root_policy_temperature,
    shaped_dirichlet_alpha_distribution,
)
from alphazero.envs.gocube.katago_train import KataGoSearchCoach
from alphazero.envs.gocube.production_training import (
    anchor_checkpoint_iteration,
    arena_regression_signals,
    build_replay_training_plan,
    summarize_arena_outcomes,
)
from alphazero.envs.gocube.train import GoCubeCoach
from alphazero.utils import dotdict


def test_training_budget_depends_on_new_samples_not_iteration_or_replay_size():
    small_history = build_replay_training_plan(
        new_selfplay_samples=1234,
        replay_window_samples=1234,
        train_samples_per_new_sample=1.5,
        batch_size=256,
    )
    large_history = build_replay_training_plan(
        new_selfplay_samples=1234,
        replay_window_samples=98_765,
        train_samples_per_new_sample=1.5,
        batch_size=256,
    )
    assert small_history.planned_training_samples == 1851
    assert large_history.planned_training_samples == 1851
    assert small_history.planned_optimizer_steps == large_history.planned_optimizer_steps == 8
    assert small_history.planned_passes_over_replay_window != large_history.planned_passes_over_replay_window


def test_replay_sampler_consumes_exact_planned_samples_independent_of_window_size():
    plan = build_replay_training_plan(
        new_selfplay_samples=37,
        replay_window_samples=500,
        train_samples_per_new_sample=1.5,
        batch_size=16,
    )
    assert plan.planned_training_samples == 56
    assert plan.planned_optimizer_steps == 4

    consumed = []
    for replay_size in (100, 500):
        dataset = TensorDataset(torch.arange(replay_size))
        sampler = RandomSampler(dataset, replacement=True, num_samples=plan.planned_training_samples)
        loader = DataLoader(dataset, batch_size=16, sampler=sampler)
        batch_sizes = [int(batch[0].size(0)) for batch in loader]
        consumed.append(sum(batch_sizes))
        assert len(batch_sizes) == plan.planned_optimizer_steps
        assert batch_sizes[-1] == 8
    assert consumed == [plan.planned_training_samples, plan.planned_training_samples]


def test_training_budget_fractional_ratio_is_monotonic_and_zero_is_explicit():
    assert build_replay_training_plan(
        new_selfplay_samples=1,
        replay_window_samples=100,
        train_samples_per_new_sample=0.25,
        batch_size=256,
    ).planned_training_samples == 1
    assert build_replay_training_plan(
        new_selfplay_samples=10,
        replay_window_samples=100,
        train_samples_per_new_sample=0.0,
        batch_size=256,
    ).planned_optimizer_steps == 0


def test_katago_early_root_temperature_uses_logical_point_count_scaling():
    defaults = KATAGO_PINNED_EXPLORATION_DEFAULTS
    assert root_policy_temperature(
        0,
        361,
        early_temperature=defaults["root_policy_temperature_early"],
        temperature=defaults["root_policy_temperature"],
        halflife=defaults["root_policy_temperature_halflife"],
    ) == pytest.approx(1.25)
    assert root_policy_temperature(
        19,
        361,
        early_temperature=defaults["root_policy_temperature_early"],
        temperature=defaults["root_policy_temperature"],
        halflife=defaults["root_policy_temperature_halflife"],
    ) == pytest.approx(1.175)


def test_shaped_dirichlet_alpha_is_half_uniform_half_policy_shaped():
    alpha = shaped_dirichlet_alpha_distribution(np.array([0.8, 0.15, 0.05, 1e-6]))
    assert np.sum(alpha) == pytest.approx(1.0)
    assert np.all(alpha > 0.0)
    assert alpha[0] == pytest.approx(alpha[1]) == pytest.approx(alpha[2])
    assert alpha[0] > alpha[3]
    uniform = shaped_dirichlet_alpha_distribution(np.full(4, 0.25))
    assert np.allclose(uniform, np.full(4, 0.25))


def test_retrospective_target_reduction_removes_exploration_overspend():
    raw = np.array([20, 10, 0], dtype=np.int32)
    policy = np.array([0.9, 0.1, 0.0], dtype=np.float64)
    utility = np.zeros(3, dtype=np.float64)
    corrected = retrospectively_reduce_root_visits(
        raw,
        policy,
        utility,
        root_player=1,
        explore_scaling=1.1 * np.sqrt(30.01),
        legal_mask=np.array([True, True, True]),
    )
    assert corrected[0] == 20
    assert 0 <= corrected[1] < raw[1]
    assert corrected[2] == 0
    assert corrected.sum() < raw.sum()


def test_periodic_anchor_is_strictly_older_than_current_checkpoint():
    assert anchor_checkpoint_iteration(1, 10) == 0
    assert anchor_checkpoint_iteration(10, 10) == 0
    assert anchor_checkpoint_iteration(11, 10) == 10
    assert anchor_checkpoint_iteration(20, 10) == 10


def test_new_iteration_clears_prior_arena_telemetry(monkeypatch):
    monkeypatch.setattr(
        GoCubeCoach,
        "_reset_selfplay_telemetry",
        lambda self: {},
    )
    coach = object.__new__(KataGoSearchCoach)
    coach._arena_telemetry = {"current_checkpoint": {"iteration": 4}}
    coach._iteration_telemetry = {}

    telemetry = coach._reset_selfplay_telemetry()

    assert telemetry.keys() >= {
        "exploration_telemetry_positions",
        "exploration_raw_visits",
        "exploration_forced_visits",
        "exploration_target_visits",
    }
    assert coach._arena_telemetry is None


def test_arena_summary_preserves_color_split_and_scored_win_rate():
    summary = summarize_arena_outcomes([
        ("black", "win"),
        ("white", "loss"),
        ("black", "draw"),
        ("white", "no_result"),
    ])
    assert summary["games"] == 4
    assert summary["scored_games"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["draws"] == 1
    assert summary["no_results"] == 1
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["by_color"]["black"]["games"] == 2
    assert summary["by_color"]["white"]["games"] == 2


def test_checkpoint_arena_enforces_deterministic_contract_and_alternates_colors(monkeypatch):
    captured = {"player_args": [], "arena_args": None, "orders": []}

    class FakeGame:
        @staticmethod
        def num_players():
            return 2

    class FakePlayer:
        def __init__(self, _net, _game_cls, args):
            captured["player_args"].append(args.copy())

    class FakeArena:
        def __init__(self, _players, _game_cls, use_batched_mcts, args):
            assert use_batched_mcts is False
            captured["arena_args"] = args.copy()

        def play_game(self, _verbose, order):
            captured["orders"].append(list(order))
            # Black always wins. Because current/opponent alternate colors this
            # produces an exactly balanced result for four games.
            return SimpleNamespace(terminal_kind="scored"), np.array([1, 0, 0], dtype=np.uint8)

    monkeypatch.setattr("alphazero.envs.gocube.katago_train.MCTSPlayer", FakePlayer)
    monkeypatch.setattr("alphazero.envs.gocube.katago_train.Arena", FakeArena)
    monkeypatch.setattr(KataGoSearchCoach, "_load_model", lambda self, model, iteration: None)

    coach = object.__new__(KataGoSearchCoach)
    coach.game_cls = FakeGame
    coach.train_net = object()
    coach.self_play_net = object()
    coach.stop_train = SimpleNamespace(is_set=lambda: False)
    coach.args = dotdict({
        "run_name": "arena-test",
        "checkpoint": "checkpoint",
        "gocube_arena_games_per_opponent": 4,
        "gocube_arena_seed": 12345,
        "arenaMCTSSims": 17,
        "numMCTSSims": 99,
        "probFastSim": 0.75,
        "add_root_noise": True,
        "add_root_temp": True,
        "startTemp": 1.0,
        "arenaTemp": 0.25,
        "gocube_rules_fingerprint": "rules",
        "gocube_komi": 0.5,
    })

    summary = coach._run_checkpoint_arena(5, 3)
    for args in captured["player_args"] + [captured["arena_args"]]:
        assert args.numMCTSSims == 17
        assert args.arenaMCTSSims == 17
        assert args.probFastSim == 0.0
        assert args.add_root_noise is False
        assert args.add_root_temp is False
        assert args.startTemp == 0.0
        assert args.arenaTemp == 0.0
    assert captured["orders"] == [[0, 1], [1, 0], [0, 1], [1, 0]]
    assert summary["wins"] == 2
    assert summary["losses"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["evaluation_contract"]["deterministic"] is True
    assert summary["evaluation_contract"]["fast_search"] is False
    assert summary["evaluation_contract"]["root_noise"] is False
    assert summary["evaluation_contract"]["komi"] == 0.5


def test_arena_regression_signals_are_observational_not_a_gate():
    signals = arena_regression_signals(0.42, [0.53, 0.48, 0.47], material_threshold=0.45)
    assert signals["below_even_against_previous"] is True
    assert signals["material_regression"] is True
    assert signals["multi_checkpoint_regression_signal"] is True
    assert "gate" not in signals
