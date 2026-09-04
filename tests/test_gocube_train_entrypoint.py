from argparse import Namespace

import pytest

from alphazero.envs.gocube.train import build_training_args


def cli_args(**overrides):
    values = {
        "topology": "torus",
        "size": 9,
        "workers": 2,
        "sims": 20,
        "games_per_iteration": 8,
        "iterations": 3,
        "train_batch_size": 1024,
        "fast_game_prob": 0.75,
        "no_arena": False,
        "smoke": False,
        "run_name": "gocube-train-test",
    }
    values.update(overrides)
    return Namespace(**values)


def test_iterations_and_process_batch_size_are_forwarded():
    _, args = build_training_args(cli_args())

    assert args.numIters == 3
    assert args.gamesPerIteration == 8
    assert args.process_batch_size == 4
    assert args.train_batch_size == 1024
    assert args.compareWithBaseline is True
    assert args.compareWithPast is True
    assert args.model_gating is True
    assert args.autoTrainSteps is True
    assert args.train_steps_per_iteration == 64
    assert args.probFastSim == 0.75


def test_smoke_mode_is_one_iteration_without_arena_comparisons():
    _, args = build_training_args(cli_args(iterations=50, smoke=True))

    assert args.numIters == 1
    assert args.process_batch_size == 4
    assert args.compareWithBaseline is False
    assert args.compareWithPast is False
    assert args.model_gating is False
    assert args.autoTrainSteps is False
    assert args.train_steps_per_iteration == 1
    assert args.probFastSim == 0.0


def test_no_arena_uses_latest_train_net_and_forwards_short_run_controls():
    _, args = build_training_args(
        cli_args(
            no_arena=True,
            train_batch_size=256,
            fast_game_prob=0.0,
            iterations=5,
            games_per_iteration=16,
        )
    )

    assert args.numIters == 5
    assert args.gamesPerIteration == 16
    assert args.process_batch_size == 8
    assert args.train_batch_size == 256
    assert args.compareWithBaseline is False
    assert args.compareWithPast is False
    assert args.model_gating is False
    assert args.autoTrainSteps is True
    assert args.probFastSim == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workers", 0, "workers must be at least 1"),
        ("games_per_iteration", 0, "games-per-iteration must be at least 1"),
        ("iterations", 0, "iterations must be at least 1"),
        ("train_batch_size", 0, "train-batch-size must be at least 1"),
        ("fast_game_prob", -0.1, "fast-game-prob must be between 0 and 1"),
        ("fast_game_prob", 1.1, "fast-game-prob must be between 0 and 1"),
    ],
)
def test_invalid_training_counts_fail_fast(field, value, message):
    with pytest.raises(ValueError, match=message):
        build_training_args(cli_args(**{field: value}))
