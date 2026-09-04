import argparse
import os

import pyximport

pyximport.install()

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import game_class
from alphazero.utils import get_iter_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate two GoCube AlphaZero checkpoints without training"
    )
    parser.add_argument("--topology", choices=("torus", "cube"), default="cube")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--candidate", type=int, default=5)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=20)
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    return parser.parse_args()


def prepare_arena_args(saved_args, game_cls, sims):
    args = saved_args.copy()
    args.numMCTSSims = sims
    args._num_players = game_cls.num_players() + game_cls.has_draw()

    # Evaluation should measure model strength, not exploration randomness.
    args.add_root_noise = False
    args.add_root_temp = False
    args.startTemp = 0.0
    args.arenaTemp = 0.0
    return args


def load_checkpoint(game_cls, folder, iteration):
    return NNetWrapper.from_checkpoint(
        game_cls,
        folder=folder,
        filename=get_iter_file(iteration),
    )


def main():
    cli = parse_args()
    if cli.candidate < 0 or cli.baseline < 0:
        raise ValueError("checkpoint iterations must be non-negative")
    if cli.games < 2 or cli.games % 2:
        raise ValueError("games must be a positive even number of at least 2")
    if cli.sims < 1:
        raise ValueError("sims must be at least 1")

    game_cls = game_class(cli.topology, cli.size)
    folder = os.path.join(cli.checkpoint_dir, cli.run_name)

    candidate = load_checkpoint(game_cls, folder, cli.candidate)
    baseline = load_checkpoint(game_cls, folder, cli.baseline)
    args = prepare_arena_args(candidate.args, game_cls, cli.sims)

    # Non-batched Arena alternates the player order every game, giving an exact
    # colour/seat balance for an even number of games.
    players = [
        MCTSPlayer(candidate, game_cls, args),
        MCTSPlayer(baseline, game_cls, args),
    ]
    arena = Arena(players, game_cls, use_batched_mcts=False, args=args)
    wins, draws, winrates = arena.play_games(cli.games, shuffle_players=True)

    print()
    print(
        f"candidate iteration {cli.candidate}: {wins[0]} wins "
        f"({winrates[0]:.3f} score rate)"
    )
    print(
        f"baseline iteration {cli.baseline}: {wins[1]} wins "
        f"({winrates[1]:.3f} score rate)"
    )
    print(f"draws: {draws}")


if __name__ == "__main__":
    main()
