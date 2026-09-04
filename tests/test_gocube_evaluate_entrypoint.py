from argparse import Namespace

import pytest

from alphazero.envs.gocube import Cube4ChineseGame
from alphazero.envs.gocube.evaluate import prepare_arena_args, validate_cli
from alphazero.utils import dotdict


def test_prepare_arena_args_disables_exploration_noise_and_temperature():
    saved = dotdict({
        "numMCTSSims": 100,
        "_num_players": None,
        "add_root_noise": True,
        "add_root_temp": True,
        "startTemp": 1.0,
        "arenaTemp": 0.25,
    })

    args = prepare_arena_args(saved, Cube4ChineseGame, sims=20)

    assert args.numMCTSSims == 20
    assert args._num_players == 3
    assert args.add_root_noise is False
    assert args.add_root_temp is False
    assert args.startTemp == 0.0
    assert args.arenaTemp == 0.0
    # Do not mutate checkpoint args.
    assert saved.add_root_noise is True
    assert saved.startTemp == 1.0


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"candidate": -1}, "checkpoint iterations must be non-negative"),
        ({"baseline": -1}, "checkpoint iterations must be non-negative"),
        ({"games": 1}, "games must be a positive even number of at least 2"),
        ({"games": 3}, "games must be a positive even number of at least 2"),
        ({"sims": 0}, "sims must be at least 1"),
    ],
)
def test_invalid_evaluation_counts_fail_fast(overrides, message):
    values = {"candidate": 5, "baseline": 0, "games": 32, "sims": 20}
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_cli(Namespace(**values))
