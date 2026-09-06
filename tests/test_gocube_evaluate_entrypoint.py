from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import pytest

from alphazero.envs.gocube import Cube4ChineseGame
from alphazero.envs.gocube import evaluation as evaluation_module
from alphazero.envs.gocube.evaluate import prepare_arena_args, validate_cli
from alphazero.utils import dotdict


def test_prepare_arena_args_disables_exploration_and_enables_batching():
    saved = dotdict({
        "numMCTSSims": 100,
        "probFastSim": 0.75,
        "_num_players": None,
        "add_root_noise": True,
        "add_root_temp": True,
        "startTemp": 1.0,
        "arenaTemp": 0.25,
        "workers": 16,
        "arena_batch_size": 64,
    })

    args = prepare_arena_args(saved, Cube4ChineseGame, sims=20)

    assert args.numMCTSSims == 20
    assert args.arenaMCTSSims == 20
    assert args.probFastSim == 0.0
    assert args._num_players == 3
    assert args.add_root_noise is False
    assert args.add_root_temp is False
    assert args.startTemp == 0.0
    assert args.arenaTemp == 0.0
    assert args.arenaBatched is True
    assert args.arena_batch_size == 16
    assert args.workers == 1
    # Do not mutate checkpoint args.
    assert saved.numMCTSSims == 100
    assert saved.probFastSim == 0.75
    assert saved.add_root_noise is True
    assert saved.startTemp == 1.0
    assert saved.workers == 16
    assert saved.arena_batch_size == 64


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"candidate": -1}, "checkpoint iterations must be non-negative"),
        ({"baseline": -1}, "checkpoint iterations must be non-negative"),
        ({"games": 1}, "games must be a positive even number of at least 2"),
        ({"games": 3}, "games must be a positive even number of at least 2"),
        ({"sims": 0}, "sims must be at least 1"),
        ({"arena_batch_size": 1}, "arena-batch-size must be at least 2"),
        ({"arena_workers": 0}, "arena-workers must be at least 1"),
    ],
)
def test_invalid_evaluation_counts_fail_fast(overrides, message):
    values = {
        "candidate": 5,
        "baseline": 0,
        "games": 32,
        "sims": 20,
        "arena_batch_size": 16,
        "arena_workers": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_cli(Namespace(**values))


def test_balanced_batched_match_runs_two_fixed_color_halves(monkeypatch):
    calls = []

    class FakePlayer:
        def __init__(self, net, game_cls=None, args=None):
            self.net = net
            self.game_cls = game_cls
            self.args = args

    class FakeArena:
        def __init__(self, players, game_cls, use_batched_mcts, args):
            self.players = players
            self.game_cls = game_cls
            self.use_batched_mcts = use_batched_mcts
            self.args = args
            self.no_results = 1

        def play_games(self, games, verbose=False, shuffle_players=True):
            calls.append({
                "players": [player.net for player in self.players],
                "batched": self.use_batched_mcts,
                "batch_size": self.args.arena_batch_size,
                "workers": self.args.workers,
                "games": games,
                "shuffle_players": shuffle_players,
            })
            if self.players[0].net == "candidate":
                return [3, 1], 0, [0.75, 0.25]
            return [2, 1], 1, [0.625, 0.375]

    monkeypatch.setattr(evaluation_module, "MCTSPlayer", FakePlayer)
    monkeypatch.setattr(evaluation_module, "Arena", FakeArena)

    args = dotdict({
        "arena_batch_size": 16,
        "workers": 1,
    })
    result = evaluation_module.play_balanced_batched_match(
        "candidate",
        "reference",
        Cube4ChineseGame,
        args,
        8,
    )

    assert len(calls) == 2
    assert calls[0] == {
        "players": ["candidate", "reference"],
        "batched": True,
        "batch_size": 4,
        "workers": 1,
        "games": 4,
        "shuffle_players": False,
    }
    assert calls[1] == {
        "players": ["reference", "candidate"],
        "batched": True,
        "batch_size": 4,
        "workers": 1,
        "games": 4,
        "shuffle_players": False,
    }
    assert result["candidate_black_games"] == 4
    assert result["candidate_white_games"] == 4
    assert result["wins"] == [4, 3]
    assert result["draws"] == 1
    assert result["no_results"] == 2


def test_checkpoint_evaluator_supports_direct_script_execution():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/evaluate_gocube_checkpoints.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--arena-batch-size" in result.stdout
