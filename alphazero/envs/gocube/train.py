import argparse

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
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_training_args(cli):
    game_cls = game_class(cli.topology, cli.size)
    run_name = cli.run_name or f"gocube-{cli.topology}-{cli.size}-chinese75"
    args = get_args(
        run_name=run_name,
        workers=cli.workers,
        gamesPerIteration=cli.games_per_iteration,
        numMCTSSims=cli.sims,
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
