import math

import numpy as np

from alphazero.MCTS import MCTS
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.search_contract import (
    KATAGO_PINNED_SEARCH_UTILITY_MODE,
    KATAGO_REFERENCE_COMMIT,
    KATAGO_SEARCH_DEFAULTS,
    SearchOutput,
    combined_white_utility,
    recent_score_center,
    score_value,
    white_owner_map,
    white_win_loss_value,
)


def _cube4_args():
    cli = parse_args([])
    game_cls, args = build_katago_training_args(cli)
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    return game_cls, args


def test_pinned_reference_and_defaults_are_explicit():
    assert KATAGO_REFERENCE_COMMIT == "f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
    assert KATAGO_SEARCH_DEFAULTS["win_loss_utility_factor"] == 1.0
    assert KATAGO_SEARCH_DEFAULTS["dynamic_score_utility_factor"] == 0.30
    assert KATAGO_SEARCH_DEFAULTS["dynamic_score_center_zero_weight"] == 0.25
    assert KATAGO_SEARCH_DEFAULTS["dynamic_score_center_scale"] == 0.50
    assert KATAGO_SEARCH_DEFAULTS["cpuct_exploration"] == 1.10
    assert KATAGO_SEARCH_DEFAULTS["fpu_reduction_max"] == 0.20
    assert KATAGO_SEARCH_DEFAULTS["root_fpu_reduction_max"] == 0.0
    assert KATAGO_SEARCH_DEFAULTS["root_ending_bonus_points"] == 0.50


def test_dynamic_score_center_caps_relative_to_expected_score_not_zero():
    # Raw center would be 75. KataGo caps the movement relative to the
    # expected score: sqrt(100)*0.5 = 5, therefore 95 rather than 5.
    center = recent_score_center(
        100.0,
        zero_weight=0.25,
        center_scale=0.50,
        point_count=100,
    )
    assert center == 95.0

    assert recent_score_center(
        4.0,
        zero_weight=0.25,
        center_scale=0.50,
        point_count=100,
    ) == 3.0


def test_score_value_matches_pinned_zero_stdev_formula():
    expected = (2.0 / math.pi) * math.atan(3.0 / (2.0 * math.sqrt(100.0)))
    assert score_value(3.0, 0.0, 2.0, 100) == expected


def test_white_result_utility_uses_white_win_minus_black_win():
    value = np.array([0.20, 0.70, 0.10], dtype=np.float32)
    assert np.isclose(white_win_loss_value(value), 0.50)
    utility = combined_white_utility(
        value,
        None,
        recent_center=0.0,
        point_count=96,
        win_loss_factor=1.0,
        static_score_factor=0.0,
        dynamic_score_factor=0.30,
        dynamic_score_scale=0.50,
    )
    assert np.isclose(utility, 0.50)


def test_ownership_conversion_is_only_signed_white_minus_black():
    ownership = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.1, 0.8],
    ])
    assert np.allclose(white_owner_map(ownership), [-0.7, 0.6, 0.0])


def test_cube4_from_scratch_defaults_pin_requested_experiment_dimensions():
    game_cls, args = _cube4_args()

    assert game_cls.topology_kind() == "cube"
    assert game_cls.board_size() == 4
    assert game_cls.logical_topology().point_count == 4 * 4 * 6
    assert args.workers == 16
    assert args.numMCTSSims == 50
    assert args.arenaMCTSSims == 50
    assert args.probFastSim == 0.25
    assert args.search_utility_mode == KATAGO_PINNED_SEARCH_UTILITY_MODE
    assert args.cpuct == 1.10
    assert args.fpu_reduction == 0.20
    assert args.gocube_root_fpu_reduction == 0.0
    assert args.numWarmupIters == 0

    # The abandoned branch used local recovery thresholds. The clean port has
    # no such knobs in the new experiment contract.
    assert not hasattr(args, "gocube_score_improvement_threshold_points")
    assert not hasattr(args, "gocube_win_probability_tolerance")


def test_pinned_mcts_consumes_all_heads_and_searches_cube4():
    game_cls, args = _cube4_args()
    game = game_cls()
    point_count = game.logical_topology().point_count
    action_size = game.action_size()

    class StubNet:
        def predict_for_search(self, observation):
            assert observation.shape == game.observation_size()
            return SearchOutput(
                policy=np.full(action_size, 1.0 / action_size, dtype=np.float32),
                value=np.array([0.5, 0.5, 0.0], dtype=np.float32),
                score=np.array([0.0], dtype=np.float32),
                ownership=np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    (point_count, 1),
                ),
            )

    mcts = MCTS(args)
    mcts.search(game, StubNet(), 4, False, False)
    counts = np.asarray(mcts.raw_counts(game))

    assert counts.shape == (action_size,)
    assert counts.sum() == 3  # first playout expands/evaluates the root itself
    assert np.isclose(np.asarray(mcts.probs(game, 1.0)).sum(), 1.0)
