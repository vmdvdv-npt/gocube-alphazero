from __future__ import annotations

import numpy as np
from torch import multiprocessing as mp

import alphazero.envs.gocube.pinned_selfplay as pinned_selfplay_module
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.pinned_game import PinnedCube4JapaneseGame
from alphazero.envs.gocube.pinned_selfplay import FORK_PRELUDE, POLICY_INIT_PRELUDE, PinnedSelfPlayAgent
from alphazero.envs.gocube.selfplay_semantics import (
    KATAGO_PINNED_SELFPLAY_DEFAULTS,
    sample_early_fork_depth,
    sample_plain_fork_kind,
    sample_policy_init_moves,
)
from alphazero.utils import dotdict


def test_pinned_diversification_defaults_match_selfplay8b20():
    defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
    assert defaults["early_fork_game_probability"] == 0.04
    assert defaults["early_fork_expected_move_prop"] == 0.025
    assert defaults["fork_game_probability"] == 0.01
    assert defaults["fork_game_min_choices"] == 3
    assert defaults["early_fork_game_max_choices"] == 12
    assert defaults["fork_game_max_choices"] == 36
    assert defaults["init_games_with_policy"] is True
    assert defaults["policy_init_area_prop"] == 0.04
    assert defaults["policy_init_gamma_shape"] == 1.0
    assert defaults["policy_init_temperature"] == 1.0

    _, args = build_katago_training_args(parse_args([]))
    assert args.numMCTSSims == 50
    assert args.numFastSims == 20
    assert args.gocube_komi == 0.5


def test_fork_sampler_matches_pinned_expected_frequencies():
    rng = np.random.RandomState(1234)
    trials = 100_000
    counts = {"early": 0, "ordinary": 0, None: 0}
    for _ in range(trials):
        counts[sample_plain_fork_kind(rng, 0.04, 0.01)] += 1

    early_fraction = counts["early"] / trials
    ordinary_fraction = counts["ordinary"] / trials
    assert abs(early_fraction - 0.04) < 0.002
    assert abs(ordinary_fraction - 0.0096) < 0.001


def test_early_fork_and_policy_init_depth_distributions_scale_with_logical_area():
    rng = np.random.RandomState(7)
    point_count = PinnedCube4JapaneseGame.logical_topology().point_count
    fork_depths = [sample_early_fork_depth(rng, point_count, 0.025) for _ in range(50_000)]
    init_moves = [sample_policy_init_moves(rng, point_count, 0.04, 1.0) for _ in range(50_000)]

    fork_mean = 1.0 / (np.exp(1.0 / (point_count * 0.025)) - 1.0)
    init_mean = 1.0 / (np.exp(1.0 / (point_count * 0.04)) - 1.0)
    assert abs(np.mean(fork_depths) - fork_mean) < 0.08
    assert abs(np.mean(init_moves) - init_mean) < 0.10


def _finish_by_passes(game):
    for _ in range(6):
        if game.win_state().any():
            break
        game.play_action(game.pass_action())
    assert game.win_state().any()


def test_finished_games_seed_and_consume_early_and_ordinary_forks():
    pool = PinnedCube4JapaneseGame._plain_fork_pool()
    pool.clear()

    early_source = PinnedCube4JapaneseGame()
    early_source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.0,
        early_fork_game_prob=1.0,
        fork_game_prob=0.0,
    )
    _finish_by_passes(early_source)
    assert pool and pool[-1][2] == "early"

    early_target = PinnedCube4JapaneseGame()
    early_target.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.0,
    )
    early = early_target.maybe_start_plain_fork()
    assert early is not None and early["mode"] == "early"
    assert early_target.pinned_selfplay_config()["started_from_plain_fork"] is True

    ordinary_source = PinnedCube4JapaneseGame()
    ordinary_source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.0,
        early_fork_game_prob=0.0,
        fork_game_prob=1.0,
    )
    _finish_by_passes(ordinary_source)
    assert pool and pool[-1][2] == "ordinary"

    ordinary_target = PinnedCube4JapaneseGame()
    ordinary_target.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.0,
    )
    ordinary = ordinary_target.maybe_start_plain_fork()
    assert ordinary is not None and ordinary["mode"] == "ordinary"


def _minimal_agent():
    agent = PinnedSelfPlayAgent.__new__(PinnedSelfPlayAgent)
    agent._is_arena = False
    agent._is_warmup = False
    agent.score_aware = True
    agent.game_cls = PinnedCube4JapaneseGame
    agent.games = [PinnedCube4JapaneseGame()]
    agent.histories = [[]]
    agent.temps = [1.0]
    agent.mcts = [object()]
    agent.next_reset = [0]
    agent.root_policy_cache = [None]
    agent.cleanup_training_phase = [None]
    agent.cleanup_training_moves_left = [0]
    agent.cleanup_training_prelude_total = [0]
    agent.cleanup_training_metadata = [None]
    agent.cleanup_training_prob = 0.0
    agent.args = dotdict({
        "startTemp": 1.0,
        "gocube_pass_alive_auto_end_probability": 0.98,
        "gocube_root_prune_useless_moves": True,
        "gocube_seki_fork_hack_probability": 0.0,
        "gocube_early_fork_game_probability": 0.04,
        "gocube_early_fork_expected_move_prop": 0.025,
        "gocube_fork_game_probability": 0.01,
        "gocube_fork_game_min_choices": 3,
        "gocube_early_fork_game_max_choices": 12,
        "gocube_fork_game_max_choices": 36,
        "gocube_init_games_with_policy": True,
        "gocube_policy_init_area_prop": 0.04,
        "gocube_policy_init_gamma_shape": 1.0,
        "gocube_policy_init_temperature": 1.0,
    })
    agent.telemetry = {
        key: mp.Value('q', 0)
        for key in (
            "normal_starts", "early_forks", "ordinary_forks",
            "policy_initialized_starts", "fork_depth_sum", "fork_depth_count",
        )
    }
    agent._get_mcts = lambda: object()
    return agent


def test_agent_schedules_policy_init_as_setup_not_training_target(monkeypatch):
    PinnedCube4JapaneseGame._plain_fork_pool().clear()
    agent = _minimal_agent()
    monkeypatch.setattr(pinned_selfplay_module, "sample_policy_init_moves", lambda *args: 3)
    agent._sample_cleanup_training_plan(0)

    assert agent.cleanup_training_phase[0] == POLICY_INIT_PRELUDE
    assert agent.cleanup_training_moves_left[0] == 3
    assert agent.telemetry["normal_starts"].value == 1
    assert agent.telemetry["policy_initialized_starts"].value == 1

    game = agent.games[0]
    game.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.0,
    )
    game.play_action(0)
    state_after_setup = game.semantic_state
    agent.histories[0] = [("must-not-survive", None)]
    agent.cleanup_training_phase[0] = POLICY_INIT_PRELUDE
    assert agent._start_cleanup_training(0) is True
    assert agent.histories[0] == []
    assert agent.games[0].semantic_state == state_after_setup


def test_agent_emits_both_plain_fork_start_types_without_training_the_experimental_move():
    pool = PinnedCube4JapaneseGame._plain_fork_pool()
    pool.clear()
    seed_game = PinnedCube4JapaneseGame()
    seed_game.play_action(0)
    seed = (seed_game.semantic_state, seed_game._pinned_move_history, "early", 1)
    pool.append(seed)

    agent = _minimal_agent()
    agent._sample_cleanup_training_plan(0)
    assert agent.cleanup_training_phase[0] == FORK_PRELUDE
    assert agent.cleanup_training_moves_left[0] == 1
    assert agent.telemetry["early_forks"].value == 1
    assert agent.telemetry["fork_depth_sum"].value == 1

    pool.append((seed_game.semantic_state, seed_game._pinned_move_history, "ordinary", 1))
    agent.games[0] = PinnedCube4JapaneseGame()
    agent.cleanup_training_phase[0] = None
    agent._sample_cleanup_training_plan(0)
    assert agent.cleanup_training_phase[0] == FORK_PRELUDE
    assert agent.telemetry["ordinary_forks"].value == 1
