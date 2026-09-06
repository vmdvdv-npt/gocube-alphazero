from dataclasses import replace
import math

import numpy as np
import pytest

from alphazero.envs.gocube.diversified_game import (
    DiversifiedPinnedCube4JapaneseGame,
    sample_fork_depth,
    sample_plain_fork_kind,
)
from alphazero.envs.gocube.diversified_selfplay import (
    KATAGO_PINNED_DIVERSIFICATION_DEFAULTS,
    DiversifiedPinnedSelfPlayAgent,
    _SETUP_EARLY_FORK,
    _SETUP_ORDINARY_FORK,
    _SETUP_POLICY_INIT,
    sample_policy_init_moves,
)
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.katago_v3 import SCORED


def test_pinned_diversification_defaults_match_selfplay8b20():
    defaults = KATAGO_PINNED_DIVERSIFICATION_DEFAULTS
    assert defaults["early_fork_game_prob"] == 0.04
    assert defaults["early_fork_game_expected_move_prop"] == 0.025
    assert defaults["fork_game_prob"] == 0.01
    assert defaults["fork_game_min_choices"] == 3
    assert defaults["early_fork_game_max_choices"] == 12
    assert defaults["fork_game_max_choices"] == 36
    assert defaults["init_games_with_policy"] is True
    assert defaults["policy_init_area_prop"] == 0.04
    assert defaults["policy_init_gamma_shape"] == 1.0
    assert defaults["policy_init_temperature"] == 1.0

    _, args = build_katago_training_args(parse_args([]))
    assert args.gocube_early_fork_game_prob == 0.04
    assert args.gocube_early_fork_game_expected_move_prop == 0.025
    assert args.gocube_fork_game_prob == 0.01
    assert args.gocube_fork_game_min_choices == 3
    assert args.gocube_early_fork_game_max_choices == 12
    assert args.gocube_fork_game_max_choices == 36
    assert args.gocube_policy_init_area_prop == 0.04
    assert args.gocube_policy_init_gamma_shape == 1.0
    assert args.gocube_policy_init_temperature == 1.0


def test_fork_sampler_realizes_expected_early_and_conditional_ordinary_rates():
    rng = np.random.RandomState(12345)
    draws = [
        sample_plain_fork_kind(rng, early_prob=0.04, ordinary_prob=0.01)
        for _ in range(100_000)
    ]
    early = draws.count("early_fork") / len(draws)
    ordinary = draws.count("ordinary_fork") / len(draws)
    assert early == pytest.approx(0.04, abs=0.002)
    # KataGo samples ordinary only when early did not fire: (1-.04)*.01=.0096.
    assert ordinary == pytest.approx(0.0096, abs=0.0015)


def test_early_fork_depth_uses_logical_point_count_not_planar_dimensions():
    rng = np.random.RandomState(7)
    depths = [
        sample_fork_depth(
            rng,
            kind="early_fork",
            move_count=10_000,
            point_count=96,
            early_expected_move_prop=0.025,
        )
        for _ in range(30_000)
    ]
    continuous_mean = 96 * 0.025
    expected_floor_mean = 1.0 / (math.exp(1.0 / continuous_mean) - 1.0)
    assert np.mean(depths) == pytest.approx(expected_floor_mean, abs=0.12)


def test_policy_init_sampler_generates_both_blank_and_nonblank_starts():
    rng = np.random.RandomState(99)
    samples = [
        sample_policy_init_moves(rng, point_count=96, area_prop=0.04, gamma_shape=1.0)
        for _ in range(20_000)
    ]
    assert min(samples) == 0
    assert max(samples) > 0
    assert sum(value > 0 for value in samples) > 10_000


def test_finished_game_seeds_both_plain_fork_types_when_forced():
    pool = DiversifiedPinnedCube4JapaneseGame._plain_fork_pool()
    for early_prob, ordinary_prob, expected_mode in (
        (1.0, 0.0, "early_fork"),
        (0.0, 1.0, "ordinary_fork"),
    ):
        pool.clear()
        source = DiversifiedPinnedCube4JapaneseGame()
        source.configure_pinned_selfplay(
            auto_end_pass_alive=False,
            root_prune_useless_moves=True,
            seki_fork_hack_prob=0.0,
        )
        source.configure_diversification(
            early_fork_prob=early_prob,
            ordinary_fork_prob=ordinary_prob,
            early_expected_move_prop=0.025,
        )
        source.play_action(0)
        source._state = replace(source.semantic_state, phase=SCORED, terminal_kind=SCORED)
        source._maybe_store_plain_fork()
        assert len(pool) == 1
        assert pool[0][0] == expected_mode
        assert pool[0][1].terminal_kind is None


def test_plain_fork_restore_preserves_v3_history_ko_and_pass_state():
    source = DiversifiedPinnedCube4JapaneseGame()
    source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.0,
    )
    source.play_action(0)
    source.play_action(source.pass_action())
    candidate_state = source.semantic_state
    candidate_history = source._pinned_move_history

    pool = DiversifiedPinnedCube4JapaneseGame._plain_fork_pool()
    pool.clear()
    pool.append(("early_fork", candidate_state, candidate_history, 2))

    target = DiversifiedPinnedCube4JapaneseGame()
    fork = target.maybe_start_plain_fork()
    assert fork == {"mode": "early_fork", "fork_depth": 2}
    assert target.semantic_state == candidate_state
    assert target.semantic_state.phase_history == candidate_state.phase_history
    assert target.semantic_state.black_pass_states == candidate_state.black_pass_states
    assert target.semantic_state.white_pass_states == candidate_state.white_pass_states
    assert target.semantic_state.ko_recap_blocked == candidate_state.ko_recap_blocked
    assert target._pinned_move_history == candidate_history


def test_all_setup_modes_are_seen_as_non_training_preludes():
    agent = DiversifiedPinnedSelfPlayAgent.__new__(DiversifiedPinnedSelfPlayAgent)
    for phase in (_SETUP_POLICY_INIT, _SETUP_EARLY_FORK, _SETUP_ORDINARY_FORK):
        agent.cleanup_training_phase = [phase]
        assert agent._cleanup_training_active(0) is True


def test_50_regular_and_20_fast_sims_remain_unchanged():
    _, args = build_katago_training_args(parse_args([]))
    assert args.numMCTSSims == 50
    assert args.numFastSims == 20
