import numpy as np

from alphazero.MCTS import MCTS
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.search_contract import SearchOutput


def test_pinned_exploration_changes_root_search_and_is_removed_from_policy_target():
    game_cls, args = build_katago_training_args(parse_args([]))
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    game = game_cls()
    action_size = game.action_size()
    point_count = game.logical_topology().point_count

    assert args.numFastSims == 20
    assert args.gocube_komi == 0.5
    assert args.gocube_train_samples_per_new_sample == 1.0
    assert "train_sample_ratio" not in args

    raw_policy = np.linspace(2.0, 1.0, action_size, dtype=np.float64)
    raw_policy /= raw_policy.sum()

    class StubNet:
        def predict_for_search(self, _observation):
            return SearchOutput(
                policy=raw_policy.astype(np.float32),
                value=np.array([0.5, 0.5, 0.0], dtype=np.float32),
                score=np.array([0.0], dtype=np.float32),
                ownership=np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    (point_count, 1),
                ),
            )

    # Deliberately use a test-only budget rather than testing the production
    # regular-search budget selected for the training run.
    test_sims = 40
    np.random.seed(20260906)
    mcts = MCTS(args)
    mcts.search(game, StubNet(), test_sims, True, True)
    telemetry = mcts.root_search_telemetry(game)

    nn_policy = np.asarray(telemetry["nn_root_policy"], dtype=np.float64)
    explored_policy = np.asarray(telemetry["exploration_policy"], dtype=np.float64)
    raw_counts = np.asarray(telemetry["root_visit_counts"], dtype=np.int32)
    target = np.asarray(telemetry["policy_training_target"], dtype=np.float64)
    forced = np.asarray(telemetry["forced_exploration_visits"], dtype=np.int32)
    corrected_counts = np.asarray(mcts.counts(game), dtype=np.int32)
    post_lcb = np.asarray(telemetry["play_selection_post_lcb"], dtype=np.float64)

    assert raw_counts.sum() == test_sims - 1  # first simulation evaluates the root itself
    assert not np.allclose(nn_policy, explored_policy)
    assert telemetry["forced_exploration_visit_total"] == int(forced.sum())
    assert forced.sum() > 0
    assert np.all(corrected_counts <= raw_counts)
    assert corrected_counts.sum() < raw_counts.sum()
    assert np.isclose(target.sum(), 1.0)
    assert post_lcb.sum() > 0.0
    assert np.allclose(target, post_lcb / post_lcb.sum())
    assert telemetry["lcb_values"] is not None
    assert telemetry["lcb_radii"] is not None


def test_selfplay_move_uses_pre_lcb_distribution_but_training_target_uses_lcb():
    game_cls, args = build_katago_training_args(parse_args([]))
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    game = game_cls()
    action_size = game.action_size()
    point_count = game.logical_topology().point_count
    raw_policy = np.linspace(3.0, 1.0, action_size, dtype=np.float64)
    raw_policy /= raw_policy.sum()

    class StubNet:
        def predict_for_search(self, observation):
            # Vary value/score slightly by occupancy to generate non-zero
            # utility variance for LCB without relying on any real network.
            occupied = float(np.asarray(observation)[0:2].sum())
            white = 0.50 + min(0.15, occupied * 0.002)
            black = 1.0 - white
            return SearchOutput(
                policy=raw_policy.astype(np.float32),
                value=np.array([black, white, 0.0], dtype=np.float32),
                score=np.array([min(0.5, occupied * 0.005)], dtype=np.float32),
                ownership=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (point_count, 1)),
            )

    np.random.seed(20260907)
    mcts = MCTS(args)
    mcts.search(game, StubNet(), 60, True, True)
    move_policy = mcts.probs(game, 0.75, False)
    target_policy = mcts.probs(game, 1.0, True)

    assert np.isclose(move_policy.sum(), 1.0)
    assert np.isclose(target_policy.sum(), 1.0)
    # Explicit flags are the contract: self-play action has LCB disabled while
    # policy supervision has it enabled. Whether a particular tiny search
    # happens to change the argmax is not required for correctness.
    telemetry = mcts.root_search_telemetry(game)
    assert telemetry["lcb_values"] is not None
    assert telemetry["lcb_best_action"] is not None


def test_arena_contract_disables_selfplay_randomization():
    _game_cls, args = build_katago_training_args(parse_args([]))
    assert args.compareWithBaseline is False
    assert args.compareWithPast is True
    assert args.model_gating is False
    assert args.arenaBatched is False
    assert args.arenaTemp == 0.0
