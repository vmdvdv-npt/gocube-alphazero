import argparse
import math

import pyximport

pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import game_class


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaZero on a GoCube topology")
    parser.add_argument("--topology", choices=("torus", "cube"), default="torus")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one bounded iteration and skip baseline/past arena comparisons",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_training_args(cli):
    if cli.workers < 1:
        raise ValueError("workers must be at least 1")
    if cli.games_per_iteration < 1:
        raise ValueError("games-per-iteration must be at least 1")
    if cli.iterations < 1:
        raise ValueError("iterations must be at least 1")

    game_cls = game_class(cli.topology, cli.size)
    run_name = cli.run_name or f"gocube-{cli.topology}-{cli.size}-chinese75"

    # process_batch_size is per worker. Keep the total number of concurrently
    # simulated games close to gamesPerIteration instead of inheriting the
    # generic framework default of 256 games *per worker*.
    process_batch_size = max(1, math.ceil(cli.games_per_iteration / cli.workers))
    iterations = 1 if cli.smoke else cli.iterations

    args = get_args(
        run_name=run_name,
        workers=cli.workers,
        gamesPerIteration=cli.games_per_iteration,
        numIters=iterations,
        numMCTSSims=cli.sims,
        process_batch_size=process_batch_size,
        compareWithBaseline=not cli.smoke,
        compareWithPast=not cli.smoke,
        nnet_type="graph",
        # No augmentation is allowed until a topology/action permutation is
        # explicitly proven for Torus/Cube Compatibility V1.
        symmetricSamples=False,
        num_channels=64,
        depth=6,
        value_dense_layers=[128, 64],
        policy_dense_layers=[128],  # retained in checkpoint args; GraphNet does not use it
    )
    return game_cls, args


def main():
    cli = parse_args()
    game_cls, args = build_training_args(cli)
    network = NNetWrapper(game_cls, args)
    coach = Coach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
