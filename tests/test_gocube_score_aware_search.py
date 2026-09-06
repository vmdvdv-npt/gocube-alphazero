from types import SimpleNamespace

import numpy as np
import pytest

from alphazero.MCTS import MCTS
from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.search_contract import (
    GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE,
    SearchOutput,
    equivalent_win_probability,
    exact_player_score_points,
    player_score_points,
    score_value,
)


def _args():
    return SimpleNamespace(
        root_noise_frac=0.25,
        root_policy_temp=1.0,
        min_discount=1.0,
        fpu_reduction=0.0,
        cpuct=1.25,
        _num_players=3,
        search_utility_mode=GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE,
        gocube_win_loss_utility_factor=1.0,
        gocube_static_score_utility_factor=0.0,
        gocube_dynamic_score_utility_factor=0.4,
        gocube_dynamic_score_center_zero_weight=0.25,
        gocube_dynamic_score_center_scale=0.5,
        gocube_root_ending_bonus_points=0.5,
        gocube_score_improvement_threshold_points=1.0,
        gocube_win_probability_tolerance=0.005,
        gocube_fill_dame_before_pass=True,
        gocube_conservative_pass=True,
    )


class _SecondPassNet:
    def __init__(self, *, pass_is_good=False):
        self.pass_is_good = pass_is_good

    def predict_for_search(self, observation):
        point_count = observation.shape[1]
        action_size = point_count + 1
        root_after_one_pass = bool(np.max(observation[5, :, 0]) > 0.5) and not (
            bool(np.max(observation[8, :, 0]) > 0.5)
            or bool(np.max(observation[9, :, 0]) > 0.5)
        )
        if root_after_one_pass:
            policy = np.full(action_size, 0.001 / point_count, dtype=np.float32)
            policy[-1] = 0.999
        else:
            # The fixture is testing the root choice, not fast repeated passing
            # in descendants. Keep later play on-board so exact terminal results
            # do not manufacture a win-value difference that the NN did not
            # predict at the root children.
            policy = np.full(action_size, 1.0 / point_count, dtype=np.float32)
            policy[-1] = 0.0
            policy[:-1] /= np.sum(policy[:-1])

        value = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        ownership = np.full((point_count, 3), 1.0 / 3.0, dtype=np.float32)

        cleanup1 = bool(np.max(observation[8, :, 0]) > 0.5)
        white_stones = int(np.sum(observation[1, :, 0]))
        if self.pass_is_good:
            black_minus_white = -12.0 if cleanup1 else (8.0 if white_stones else -12.0)
        else:
            black_minus_white = 22.32 if cleanup1 else (5.59 if white_stones else 22.32)
        score = np.array([black_minus_white / point_count], dtype=np.float32)
        return SearchOutput(policy=policy, value=value, score=score, ownership=ownership)


def _after_black_pass():
    game = Cube4JapaneseGame()
    game.play_action(game.pass_action())
    assert game.player == 1
    assert game.semantic_state.phase == "main"
    assert game.semantic_state.consecutive_passes == 1
    return game


def test_score_sign_inverts_between_black_and_white():
    points = Cube4JapaneseGame.logical_topology().point_count
    normalized = 11.5 / points
    assert player_score_points(normalized, 0, points) == pytest.approx(11.5)
    assert player_score_points(normalized, 1, points) == pytest.approx(-11.5)


def test_score_value_matches_pinned_katago_smooth_transform():
    points = Cube4JapaneseGame.logical_topology().point_count
    expected = (2.0 / np.pi) * np.arctan(7.0 / (2.0 * np.sqrt(points)))
    assert score_value(7.0, 0.0, 2.0, points) == pytest.approx(expected)


def test_framework_result_width_keeps_draw_as_half_win():
    value = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    assert equivalent_win_probability(value, 0, 3) == pytest.approx(0.5)
    assert equivalent_win_probability(value, 1, 3) == pytest.approx(0.5)


@pytest.mark.parametrize("sims", [20, 50])
def test_score_aware_search_rejects_score_dominated_second_pass(sims):
    game = _after_black_pass()
    mcts = MCTS(_args())
    mcts.search(game, _SecondPassNet(pass_is_good=False), sims, False, False)

    diagnostic = mcts.pass_diagnostic(game)
    assert diagnostic["best_nonpass_score_gain"] >= 1.0, diagnostic
    assert diagnostic["best_nonpass_win_delta"] >= -0.005, diagnostic
    assert diagnostic["score_dominated_pass"] is True, diagnostic
    assert diagnostic["pass_suppressed"] is True, diagnostic
    assert mcts.best_action(game) != game.pass_action(), diagnostic
    assert np.asarray(mcts.counts(game))[game.pass_action()] == 0, diagnostic


@pytest.mark.parametrize("sims", [20, 50])
def test_pass_remains_selectable_when_nonpass_does_not_improve_score(sims):
    game = _after_black_pass()
    mcts = MCTS(_args())
    mcts.search(game, _SecondPassNet(pass_is_good=True), sims, False, False)

    diagnostic = mcts.pass_diagnostic(game)
    assert diagnostic["score_dominated_pass"] is False, diagnostic
    assert diagnostic["pass_suppressed"] is False, diagnostic
    assert mcts.best_action(game) == game.pass_action(), diagnostic


def test_terminal_search_uses_exact_formal_score_not_nn_score():
    game = Cube4JapaneseGame()
    pass_action = game.pass_action()
    for _ in range(6):
        if game.terminal_kind is not None:
            break
        game.play_action(pass_action)
    assert game.terminal_kind == "scored"
    assert game.terminal_adjudication.score is not None

    mcts = MCTS(_args())
    leaf = mcts.find_leaf(game)
    bogus_policy = np.full(game.action_size(), 1.0 / game.action_size(), dtype=np.float32)
    bogus_value = np.array([0.99, 0.01, 0.0], dtype=np.float32)
    bogus_score = np.array([1.0], dtype=np.float32)
    ownership = np.full((game.logical_topology().point_count, 3), 1.0 / 3.0, dtype=np.float32)
    mcts.process_search_results(leaf, bogus_value, bogus_policy, bogus_score, ownership, False, False)

    expected = exact_player_score_points(game, game.player)
    assert mcts._root.score_v == pytest.approx(expected)


def test_legacy_mcts_does_not_require_score_or_ownership_heads():
    class LegacyNet:
        def __call__(self, _observation):
            policy = np.full(Cube4JapaneseGame.action_size(), 1.0 / Cube4JapaneseGame.action_size(), dtype=np.float32)
            value = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            return policy, value

    args = _args()
    args.search_utility_mode = "legacy"
    game = Cube4JapaneseGame()
    mcts = MCTS(args)
    mcts.search(game, LegacyNet(), 3, False, False)
    assert int(np.sum(np.asarray(mcts.raw_counts(game)))) > 0
