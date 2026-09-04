import argparse
import math

import pyximport

pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import game_class


class GoCubeCoach(Coach):
    """Coach variant that tracks the live self-play model without gating.

    Base Coach uses ``self_play_iter == 0`` as a reason to keep workers in
    random warmup mode. When model gating is disabled, self-play inference
    correctly uses ``train_net`` directly, but ``self_play_iter`` otherwise
    never advances. Recording every newly saved train checkpoint here makes
    iteration 2 leave warmup and keeps the logged self-play model version
    aligned with the network actually used.
    """

    def _save_model(self, model, iteration):
        super()._save_model(model, iteration)
        if hasattr(self, "args") and not self.args.model_gating:
            self.self_play_iter = iteration


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaZero on a GoCube topology")
    parser.add_argument("--topology", choices=("torus", "cube"), default="torus")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--fast-game-prob", type=float, default=0.75)
    parser.add_argument(
        "--no-arena",
        action="store_true",
        help=(
            "Skip baseline/past arena comparisons and disable model gating so "
            "self-play immediately uses the latest trained network"
        ),
    )
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
    if cli.train_batch_size < 1:
        raise ValueError("train-batch-size must be at least 1")
    if not 0.0 <= cli.fast_game_prob <= 1.0:
        raise ValueError("fast-game-prob must be between 0 and 1")

    game_cls = game_class(cli.topology, cli.size)
    run_name = cli.run_name or f"gocube-{cli.topology}-{cli.size}-chinese75"

    # process_batch_size is per worker. Keep the total number of concurrently
    # simulated games close to gamesPerIteration instead of inheriting the
    # generic framework default of 256 games *per worker*.
    process_batch_size = max(1, math.ceil(cli.games_per_iteration / cli.workers))
    iterations = 1 if cli.smoke else cli.iterations
    arena_enabled = not (cli.smoke or cli.no_arena)

    args = get_args(
        run_name=run_name,
        workers=cli.workers,
        gamesPerIteration=cli.games_per_iteration,
        numIters=iterations,
        numMCTSSims=cli.sims,
        process_batch_size=process_batch_size,
        train_batch_size=cli.train_batch_size,
        compareWithBaseline=arena_enabled,
        compareWithPast=arena_enabled,
        # Gating only advances via compareToPast(). If arena comparisons are
        # disabled, gating must also be disabled; GoCubeCoach then advances the
        # live self-play model version whenever the train checkpoint is saved.
        model_gating=arena_enabled,
        # A smoke run must actually exercise the optimizer once, even when the
        # sample count is below the configured train batch size.
        autoTrainSteps=not cli.smoke,
        train_steps_per_iteration=1 if cli.smoke else 64,
        # Fast games intentionally do not retain histories. Disable them in a
        # smoke run so the single iteration is guaranteed to produce samples.
        probFastSim=0.0 if cli.smoke else cli.fast_game_prob,
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
    coach = GoCubeCoach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
