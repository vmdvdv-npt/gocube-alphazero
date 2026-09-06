from __future__ import annotations

import argparse
import os
from time import time

import torch
from torch import multiprocessing as mp

from alphazero.Coach import TrainState, _set_state
from alphazero.NNetWrapper import NNetWrapper
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.selfplay_semantics import KATAGO_CLEANUP_TRAINING_DEFAULTS
from alphazero.envs.gocube.train import GoCubeCoach, build_training_args, print_training_configuration
from alphazero.inference_batching import collect_ready_worker_ids, process_coalesced_inference
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.search_contract import (
    KATAGO_PINNED_SEARCH_UTILITY_MODE,
    KATAGO_REFERENCE_COMMIT,
    KATAGO_SEARCH_CONTRACT,
    KATAGO_SEARCH_DEFAULTS,
)


DEFAULT_RUN_NAME = "gocube-cube-4-katago-pinned-s50-fast025-20260906"


class KataGoSearchCoach(GoCubeCoach):
    """GoCube coach with one-forward policy/value/score/ownership inference."""

    def __init__(self, game_cls, nnet, args):
        super().__init__(game_cls, nnet, args)
        self.score_tensors = []
        self.ownership_tensors = []

    @_set_state(TrainState.INIT_AGENTS)
    def generateSelfPlayAgents(self):
        telemetry = self._reset_selfplay_telemetry()
        self._iteration_record_context = self._build_iteration_record_context(self.model_iter)
        self.stop_agents = mp.Event()
        self.ready_queue = mp.Queue()
        point_count = self.game_cls.logical_topology().point_count

        for i in range(self.args.workers):
            self.input_tensors.append(torch.zeros(
                [self.args.process_batch_size, *self.game_cls.observation_size()]
            ))
            self.input_tensors[i].share_memory_()

            self.policy_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game_cls.action_size()]
            ))
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game_cls.num_players() + 1]
            ))
            self.value_tensors[i].share_memory_()

            self.score_tensors.append(torch.zeros(
                [self.args.process_batch_size, 1]
            ))
            self.score_tensors[i].share_memory_()

            self.ownership_tensors.append(torch.zeros(
                [self.args.process_batch_size, point_count, 3]
            ))
            self.ownership_tensors[i].share_memory_()

            self.batch_ready.append(mp.Event())

            if self.args.cuda:
                self.input_tensors[i].pin_memory()
                self.policy_tensors[i].pin_memory()
                self.value_tensors[i].pin_memory()
                self.score_tensors[i].pin_memory()
                self.ownership_tensors[i].pin_memory()

            self.agents.append(
                SelfPlayAgent(
                    i,
                    self.game_cls,
                    self.ready_queue,
                    self.batch_ready[i],
                    self.input_tensors[i],
                    self.policy_tensors[i],
                    self.value_tensors[i],
                    self.file_queue,
                    self.result_queue,
                    self.completed,
                    self.games_played,
                    self.stop_agents,
                    self.pause_train,
                    self.args,
                    _is_warmup=self.warmup,
                    telemetry=telemetry,
                    score_tensor=self.score_tensors[i],
                    ownership_tensor=self.ownership_tensors[i],
                )
            )
            self.agents[i].daemon = True
            self.agents[i].start()

    @_set_state(TrainState.SELF_PLAY)
    def processSelfPlayBatches(self, iteration):
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
                    score_tensors=self.score_tensors,
                    ownership_tensors=self.ownership_tensors,
                )
                inference_batch_size.update(rows)

            size = self.games_played.value
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()

            bar.suffix = (
                f"({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | "
                f"Infer Batch: {inference_batch_size.avg:.1f} | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}"
            )
            bar.goto(size)
            self.sample_time = sample_time.avg
            self.iter_time = bar.elapsed_td
            self.eta = bar.eta_td

        if not self.stop_agents.is_set():
            self.stop_agents.set()
        bar.update()
        bar.finish()
        self.writer.add_scalar("performance/sample_time", sample_time.avg, iteration)
        self.writer.add_scalar("performance/inference_batch_size", inference_batch_size.avg, iteration)
        self._snapshot_selfplay_telemetry(iteration)
        print()


def parse_args(argv=None):
    cleanup_defaults = KATAGO_CLEANUP_TRAINING_DEFAULTS
    parser = argparse.ArgumentParser(
        description="Train GoCube from scratch with search semantics ported from pinned KataGo"
    )
    parser.add_argument("--topology", choices=("torus", "cube"), default="cube")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--arena-sims", type=int, default=50)
    parser.add_argument("--games-per-iteration", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--train-steps-per-iteration", type=int, default=None)
    parser.add_argument("--fast-game-prob", type=float, default=0.25)
    parser.add_argument("--endgame-sample-weight", type=int, default=1)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=1.0)
    parser.add_argument(
        "--cleanup-training-prob",
        type=float,
        default=cleanup_defaults["probability"],
        help="KataGo-style probability of rebasing a self-play game into cleanup/encore training.",
    )
    parser.add_argument(
        "--cleanup-training-prelude-area-prop",
        type=float,
        default=cleanup_defaults["prelude_area_prop"],
        help="Mean pure-policy prelude length as a fraction of logical board area.",
    )
    parser.add_argument("--no-arena", action="store_true")
    parser.add_argument("--model-gating", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--allow-existing-run",
        action="store_true",
        help="Allow resuming an existing namespace. Omit this for a guaranteed from-scratch run.",
    )
    return parser.parse_args(argv)


def build_katago_training_args(cli):
    game_cls, args = build_training_args(cli)
    defaults = KATAGO_SEARCH_DEFAULTS
    cleanup_defaults = KATAGO_CLEANUP_TRAINING_DEFAULTS

    args.search_utility_mode = KATAGO_PINNED_SEARCH_UTILITY_MODE
    # This is intentionally metadata only here. The checkpoint already saves the
    # complete args object, while the separate branch prevents accidental mixing
    # with the old score-aware experiment.
    args.gocube_katago_search_contract = KATAGO_SEARCH_CONTRACT
    args.gocube_katago_search_reference_commit = KATAGO_REFERENCE_COMMIT

    args.gocube_win_loss_utility_factor = defaults["win_loss_utility_factor"]
    args.gocube_static_score_utility_factor = defaults["static_score_utility_factor"]
    args.gocube_dynamic_score_utility_factor = defaults["dynamic_score_utility_factor"]
    args.gocube_dynamic_score_center_zero_weight = defaults["dynamic_score_center_zero_weight"]
    args.gocube_dynamic_score_center_scale = defaults["dynamic_score_center_scale"]
    args.gocube_cpuct_exploration = defaults["cpuct_exploration"]
    args.gocube_cpuct_exploration_log = defaults["cpuct_exploration_log"]
    args.gocube_cpuct_exploration_base = defaults["cpuct_exploration_base"]
    args.gocube_root_fpu_reduction = defaults["root_fpu_reduction_max"]
    args.gocube_fpu_parent_weight_by_visited_policy = defaults[
        "fpu_parent_weight_by_visited_policy"
    ]
    args.gocube_fpu_parent_weight_by_visited_policy_pow = defaults[
        "fpu_parent_weight_by_visited_policy_pow"
    ]
    args.gocube_root_ending_bonus_points = defaults["root_ending_bonus_points"]
    args.gocube_fill_dame_before_pass = defaults["fill_dame_before_pass"]
    args.gocube_conservative_pass = defaults["conservative_pass"]

    # KataGo's self-play path dedicates a small fraction of territory-scoring
    # games to cleanup/encore training. Its policy-init gamma shape defaults to
    # 1.0 and cleanup uses a 2/3 policy temperature at the pinned commit.
    args.gocube_cleanup_training_prob = float(cli.cleanup_training_prob)
    args.gocube_cleanup_training_prelude_area_prop = float(cli.cleanup_training_prelude_area_prop)
    args.gocube_cleanup_training_gamma_shape = cleanup_defaults["prelude_gamma_shape"]
    args.gocube_cleanup_training_policy_temperature = cleanup_defaults["policy_temperature"]

    # Keep the framework fields aligned with the pinned KataGo values that are
    # actually consumed by MCTS. The experiment dimensions requested by the user
    # remain 50 regular sims and 0.25 fast-search probability.
    args.cpuct = defaults["cpuct_exploration"]
    args.fpu_reduction = defaults["fpu_reduction_max"]
    args.numMCTSSims = int(cli.sims)
    args.arenaMCTSSims = int(cli.arena_sims)
    args.probFastSim = float(cli.fast_game_prob)

    # KataGo trains from policy-guided self-play rather than this framework's
    # historical random-MCTS warmup. A fresh random network still supplies the
    # initial policy/value/score/ownership predictions.
    args.numWarmupIters = 0
    return game_cls, args


def run_paths(args):
    return (
        os.path.join(args.checkpoint, args.run_name),
        os.path.join(args.data, args.run_name),
        os.path.join("runs", args.run_name),
    )


def assert_fresh_run(args):
    existing = [path for path in run_paths(args) if os.path.exists(path)]
    if existing:
        joined = "\n  - ".join(existing)
        raise RuntimeError(
            "Refusing to reuse an existing training namespace. "
            "This entrypoint is from-scratch by default. Existing paths:\n  - " + joined
        )


def print_katago_search_configuration(args):
    print_training_configuration(args)
    print("Pinned KataGo search:")
    print(f"  reference commit = {args.gocube_katago_search_reference_commit}")
    print(f"  utility mode = {args.search_utility_mode}")
    print(f"  win/loss factor = {args.gocube_win_loss_utility_factor:g}")
    print(f"  dynamic score factor = {args.gocube_dynamic_score_utility_factor:g}")
    print(f"  dynamic center zero weight = {args.gocube_dynamic_score_center_zero_weight:g}")
    print(f"  dynamic center scale = {args.gocube_dynamic_score_center_scale:g}")
    print(f"  cpuct = {args.gocube_cpuct_exploration:g}")
    print(f"  fpu reduction = {args.fpu_reduction:g}")
    print(f"  root fpu reduction = {args.gocube_root_fpu_reduction:g}")
    print(f"  root ending bonus points = {args.gocube_root_ending_bonus_points:g}")
    print(f"  fill dame before pass = {args.gocube_fill_dame_before_pass}")
    print(f"  conservative pass = {args.gocube_conservative_pass}")
    print(f"  cleanup training probability = {args.gocube_cleanup_training_prob:g}")
    print(f"  cleanup prelude area proportion = {args.gocube_cleanup_training_prelude_area_prop:g}")
    print(f"  cleanup prelude gamma shape = {args.gocube_cleanup_training_gamma_shape:g}")
    print(f"  cleanup policy temperature = {args.gocube_cleanup_training_policy_temperature:g}")


def main(argv=None):
    cli = parse_args(argv)
    game_cls, args = build_katago_training_args(cli)
    if not cli.allow_existing_run:
        assert_fresh_run(args)
    print_katago_search_configuration(args)
    ensure_training_manifest(args.checkpoint, args.run_name, game_cls)
    network = NNetWrapper(game_cls, args)
    coach = KataGoSearchCoach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
