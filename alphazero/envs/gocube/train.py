import argparse
import math
from time import time

import pyximport

pyximport.install()

from alphazero.Coach import Coach, TrainState, _set_state, get_args
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import game_class
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.inference_batching import collect_ready_worker_ids, process_coalesced_inference
from alphazero.pytorch_classification.utils import Bar, AverageMeter


class GoCubeCoach(Coach):
    """Coach variant for GoCube self-play model tracking and GPU batching."""

    def _save_model(self, model, iteration):
        """Track the live self-play model when gating is disabled.

        Base Coach uses ``self_play_iter == 0`` as a reason to keep workers in
        random warmup mode. When model gating is disabled, self-play inference
        uses ``train_net`` directly, so recording each newly saved checkpoint
        makes iteration 2 leave warmup and keeps the logged self-play version
        aligned with the network actually used.
        """
        super()._save_model(model, iteration)
        if hasattr(self, "args") and not self.args.model_gating:
            self.self_play_iter = iteration

    @_set_state(TrainState.SELF_PLAY)
    def processSelfPlayBatches(self, iteration):
        """Coalesce ready workers into larger neural-network inference calls.

        The upstream Coach processes each ready worker independently. For
        GoCube that can mean CUDA batches of only a few positions even when
        many workers are waiting. A short coalescing window combines those
        requests, runs one GraphNet call, splits the outputs, and then releases
        all participating workers.
        """
        sample_time = AverageMeter()
        inference_batch_size = AverageMeter()
        bar = Bar("Generating Samples", max=self.args.gamesPerIteration)
        end = time()

        n = 0
        while self.completed.value != self.args.workers:
            if self.stop_train.is_set() and not self.stop_agents.is_set():
                self.stop_agents.set()

            worker_ids = collect_ready_worker_ids(
                self.ready_queue,
                self.args.workers,
                self.args.inference_batch_wait_ms,
            )
            if worker_ids:
                nnet = self.self_play_net if self.args.model_gating else self.train_net
                rows = process_coalesced_inference(
                    nnet,
                    worker_ids,
                    self.input_tensors,
                    self.policy_tensors,
                    self.value_tensors,
                    self.batch_ready,
                )
                inference_batch_size.update(rows)

            size = self.games_played.value
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()

            bar.suffix = (
                f"({size}/{self.args.gamesPerIteration}) "
                f"Sample Time: {sample_time.avg:.3f}s | "
                f"Infer Batch: {inference_batch_size.avg:.1f} | "
                f"Total: {bar.elapsed_td} | ETA: {bar.eta_td:}"
            )
            bar.goto(size)
            self.sample_time = sample_time.avg
            self.iter_time = bar.elapsed_td
            self.eta = bar.eta_td

        if not self.stop_agents.is_set():
            self.stop_agents.set()
        bar.update()
        bar.finish()
        self.writer.add_scalar("loss/sample_time", sample_time.avg, iteration)
        self.writer.add_scalar(
            "performance/inference_batch_size",
            inference_batch_size.avg,
            iteration,
        )
        print()


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
        "--inference-batch-wait-ms",
        type=float,
        default=1.0,
        help=(
            "Maximum time after the first ready worker to wait for more workers "
            "and combine them into one neural-network inference batch"
        ),
    )
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
    if cli.inference_batch_wait_ms < 0:
        raise ValueError("inference-batch-wait-ms must be non-negative")

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
        inference_batch_wait_ms=cli.inference_batch_wait_ms,
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
        # Immutable integration metadata is also persisted inside new checkpoint args.
        gocube_topology=game_cls.topology_kind(),
        gocube_size=game_cls.board_size(),
        gocube_rule_set=game_cls.RULESET,
        gocube_komi=float(game_cls.KOMI),
        gocube_terminal_adjudicator=game_cls.TERMINAL_ADJUDICATOR_ID,
    )
    return game_cls, args


def main():
    cli = parse_args()
    game_cls, args = build_training_args(cli)
    ensure_training_manifest(args.checkpoint, args.run_name, game_cls)
    network = NNetWrapper(game_cls, args)
    coach = GoCubeCoach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
