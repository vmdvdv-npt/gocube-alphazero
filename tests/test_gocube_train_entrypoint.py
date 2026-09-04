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
    assert args.compareWithBaseline is True
    assert args.compareWithPast is True


def test_smoke_mode_is_one_iteration_without_arena_comparisons():
    _, args = build_training_args(cli_args(iterations=50, smoke=True))

    assert args.numIters == 1
    assert args.process_batch_size == 4
    assert args.compareWithBaseline is False
    assert args.compareWithPast is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workers", 0, "workers must be at least 1"),
        ("games_per_iteration", 0, "games-per-iteration must be at least 1"),
        ("iterations", 0, "iterations must be at least 1"),
    ],
)
def test_invalid_training_counts_fail_fast(field, value, message):
    with pytest.raises(ValueError, match=message):
        build_training_args(cli_args(**{field: value}))
