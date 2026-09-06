from __future__ import annotations

import argparse
import json
import os
from time import time

import torch
from torch import multiprocessing as mp

from alphazero.Coach import TrainState, _set_state
from alphazero.envs.gocube.diversified_game import diversified_pinned_game_class
from alphazero.envs.gocube.diversified_selfplay import (
    DiversifiedPinnedSelfPlayAgent,
    KATAGO_PINNED_DIVERSIFICATION_DEFAULTS,
)
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.sample_clock import SampleClockNNetWrapper, TRAINING_CONTRACT
from alphazero.envs.gocube.selfplay_semantics import (
    KATAGO_CLEANUP_TRAINING_DEFAULTS,
    KATAGO_PINNED_SELFPLAY_DEFAULTS,
)
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
DEFAULT_LR_WARMUP_SAMPLES = 2_000_000
DEFAULT_LR_MILESTONE_SAMPLES = (20_000_000, 40_000_000)
DEFAULT_LR_WARMUP_START_FACTOR = 0.05
DEFAULT_LR_DECAY_GAMMA = 0.1
DEFAULT_GRADIENT_CLIP_NORM = 5.0

_DIVERSIFICATION_COUNTER_KEYS = (
    "normal_starts",
    "early_forks",
    "ordinary_forks",
    "policy_initialized_starts",
    "fork_depth_sum",
    "fork_depth_count",
)


def _sample_milestones(value: str) -> tuple[int, ...]:
    value = str(value).strip()
    if not value:
        return ()
    milestones = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(item <= 0 for item in milestones):
        raise argparse.ArgumentTypeError("LR sample milestones must be positive integers")
    if tuple(sorted(set(milestones))) != milestones:
        raise argparse.ArgumentTypeError("LR sample milestones must be strictly increasing")
    return milestones


class KataGoSearchCoach(GoCubeCoach):
    """GoCube coach with pinned search, diversified self-play, and sample-clock training."""

    def __init__(self, game_cls, nnet, args):
        super().__init__(game_cls, nnet, args)
        self.score_tensors = []
        self.ownership_tensors = []

    def _reset_selfplay_telemetry(self):
        telemetry = super()._reset_selfplay_telemetry()
        for key in _DIVERSIFICATION_COUNTER_KEYS:
            if key not in telemetry:
                telemetry[key] = mp.Value('q', 0)
        self._iteration_telemetry.update({
            "normal_starts": 0,
            "early_forks": 0,
            "ordinary_forks": 0,
            "policy_initialized_starts": 0,
            "fork_depth_sum": 0,
            "fork_depth_count": 0,
            "average_fork_depth": 0.0,
        })
        return telemetry

    def _snapshot_selfplay_telemetry(self, iteration):
        super()._snapshot_selfplay_telemetry(iteration)
        snapshot = self._iteration_telemetry
        fork_count = int(snapshot.get("fork_depth_count", 0))
        average_depth = (
            float(snapshot.get("fork_depth_sum", 0)) / fork_count if fork_count else 0.0
        )
        snapshot["average_fork_depth"] = average_depth
        for key in (
            "normal_starts",
            "early_forks",
            "ordinary_forks",
            "policy_initialized_starts",
        ):
            self.writer.add_scalar(f"selfplay/{key}", int(snapshot.get(key, 0)), iteration)
        self.writer.add_scalar("selfplay/average_fork_depth", average_depth, iteration)

    def _record_training_metrics(self, iteration, *, history_iterations, window_samples,
                                 latest_iteration_samples, planned_steps):
        super()._record_training_metrics(
            iteration,
            history_iterations=history_iterations,
            window_samples=window_samples,
            latest_iteration_samples=latest_iteration_samples,
            planned_steps=planned_steps,
        )
        scheduler = self.train_net.scheduler
        extras = {
            "total_training_samples": int(self.train_net.total_training_samples),
            "total_optimizer_updates": int(self.train_net.total_optimizer_updates),
            "effective_lr": float(self.train_net.last_train_learning_rate),
            "samples_since_lr_change": int(scheduler.samples_since_last_lr_change),
            "gradient_norm_pre_clip": float(self.train_net.last_train_gradient_norm),
            "gradient_norm_pre_clip_max": float(self.train_net.last_train_gradient_norm_max),
            "gradient_clip_events": int(self.train_net.last_train_clipping_events),
            "gradient_clip_frequency": float(self.train_net.last_train_clipping_frequency),
        }
        self._training_telemetry.update(extras)
        for key, value in extras.items():
            self.writer.add_scalar(f"training/{key}", value, iteration)

    def _print_iteration_summary(self, iteration):
        super()._print_iteration_summary(iteration)
        sample = self._iteration_telemetry
        training = getattr(self, "_training_telemetry", {})
        print()
        print("Diversification:")
        print(f"  normal starts:           {sample.get('normal_starts', 0)}")
        print(f"  early forks:             {sample.get('early_forks', 0)}")
        print(f"  ordinary forks:          {sample.get('ordinary_forks', 0)}")
        print(f"  policy-init starts:      {sample.get('policy_initialized_starts', 0)}")
        print(f"  average fork depth:      {sample.get('average_fork_depth', 0.0):.2f}")
        print("Sample-clock training:")
        print(f"  total training samples:  {training.get('total_training_samples', 0)}")
        print(f"  total optimizer updates: {training.get('total_optimizer_updates', 0)}")
        print(f"  effective LR:            {training.get('effective_lr', self.args.lr):g}")
        print(f"  samples since LR change: {training.get('samples_since_lr_change', 0)}")
        print(f"  pre-clip grad norm:      {training.get('gradient_norm_pre_clip', 0.0):.4f}")
        print(f"  grad clip frequency:     {training.get('gradient_clip_frequency', 0.0):.2%}")

    def _patch_iteration_manifest(self):
        path = getattr(self, "_iteration_record_manifest_path", None)
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        aggregate = manifest.setdefault("aggregate_metrics", {})
        aggregate["training"] = dict(getattr(self, "_training_telemetry", {}))
        sample = self._iteration_telemetry
        fork_count = int(sample.get("fork_depth_count", 0))
        aggregate["selfplay_diversification"] = {
            "normal_starts": int(sample.get("normal_starts", 0)),
            "early_forks": int(sample.get("early_forks", 0)),
            "ordinary_forks": int(sample.get("ordinary_forks", 0)),
            "policy_initialized_starts": int(sample.get("policy_initialized_starts", 0)),
            "average_fork_depth": (
                float(sample.get("fork_depth_sum", 0)) / fork_count if fork_count else 0.0
            ),
        }
        temporary = path + ".training.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        result = super().train(iteration)
        self._patch_iteration_manifest()
        return result

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
                DiversifiedPinnedSelfPlayAgent(
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

    @_set_state(TrainState.KILL_AGENTS)
    def killSelfPlayAgents(self):
        result = super().killSelfPlayAgents()
        # Coach clears the legacy three tensors, while the pinned four-head
        # path owns two additional shared-tensor arrays. They must be cleared
        # between iterations or index 0 on the next iteration reuses stale buffers.
        self.score_tensors = []
        self.ownership_tensors = []
        return result

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
    diverse = KATAGO_PINNED_DIVERSIFICATION_DEFAULTS
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
        "--cleanup-training-prob", type=float, default=cleanup_defaults["probability"]
    )
    parser.add_argument(
        "--cleanup-training-prelude-area-prop",
        type=float,
        default=cleanup_defaults["prelude_area_prop"],
    )
    parser.add_argument("--early-fork-prob", type=float, default=diverse["early_fork_game_prob"])
    parser.add_argument(
        "--early-fork-expected-move-prop",
        type=float,
        default=diverse["early_fork_game_expected_move_prop"],
    )
    parser.add_argument("--fork-prob", type=float, default=diverse["fork_game_prob"])
    parser.add_argument(
        "--policy-init-area-prop", type=float, default=diverse["policy_init_area_prop"]
    )
    parser.add_argument("--lr-warmup-samples", type=int, default=DEFAULT_LR_WARMUP_SAMPLES)
    parser.add_argument(
        "--lr-warmup-start-factor", type=float, default=DEFAULT_LR_WARMUP_START_FACTOR
    )
    parser.add_argument(
        "--lr-milestone-samples",
        type=_sample_milestones,
        default=DEFAULT_LR_MILESTONE_SAMPLES,
        help="Comma-separated persistent sample thresholds for LR decay.",
    )
    parser.add_argument("--lr-decay-gamma", type=float, default=DEFAULT_LR_DECAY_GAMMA)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--no-arena", action="store_true")
    parser.add_argument("--model-gating", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--allow-existing-run",
        action="store_true",
        help="Allow resuming an existing namespace. The checkpoint must satisfy the new training contract.",
    )
    return parser.parse_args(argv)


def build_katago_training_args(cli):
    base_game_cls, args = build_training_args(cli)
    # get_args() historically returns a mutable global dotdict. Never install
    # the new training contract into that shared object.
    args = args.copy()
    game_cls = diversified_pinned_game_class(base_game_cls)
    defaults = KATAGO_SEARCH_DEFAULTS
    cleanup_defaults = KATAGO_CLEANUP_TRAINING_DEFAULTS
    selfplay_defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
    diverse = KATAGO_PINNED_DIVERSIFICATION_DEFAULTS

    args.search_utility_mode = KATAGO_PINNED_SEARCH_UTILITY_MODE
    args.gocube_katago_search_contract = KATAGO_SEARCH_CONTRACT
    args.gocube_katago_search_reference_commit = KATAGO_REFERENCE_COMMIT
    args.gocube_observation_schema = game_cls.OBSERVATION_SCHEMA

    args.gocube_win_loss_utility_factor = defaults["win_loss_utility_factor"]
    args.gocube_static_score_utility_factor = defaults["static_score_utility_factor"]
    args.gocube_dynamic_score_utility_factor = defaults["dynamic_score_utility_factor"]
    args.gocube_dynamic_score_center_zero_weight = defaults["dynamic_score_center_zero_weight"]
    args.gocube_dynamic_score_center_scale = defaults["dynamic_score_center_scale"]
    args.gocube_cpuct_exploration = defaults["cpuct_exploration"]
    args.gocube_cpuct_exploration_log = defaults["cpuct_exploration_log"]
    args.gocube_cpuct_exploration_base = defaults["cpuct_exploration_base"]
    args.gocube_root_fpu_reduction = defaults["root_fpu_reduction_max"]
    args.gocube_fpu_parent_weight_by_visited_policy = defaults["fpu_parent_weight_by_visited_policy"]
    args.gocube_fpu_parent_weight_by_visited_policy_pow = defaults["fpu_parent_weight_by_visited_policy_pow"]
    args.gocube_root_ending_bonus_points = defaults["root_ending_bonus_points"]
    args.gocube_fill_dame_before_pass = defaults["fill_dame_before_pass"]
    args.gocube_conservative_pass = defaults["conservative_pass"]

    args.gocube_cleanup_training_prob = float(cli.cleanup_training_prob)
    args.gocube_cleanup_training_prelude_area_prop = float(cli.cleanup_training_prelude_area_prop)
    args.gocube_cleanup_training_gamma_shape = cleanup_defaults["prelude_gamma_shape"]
    args.gocube_cleanup_training_policy_temperature = cleanup_defaults["policy_temperature"]

    args.gocube_pass_alive_auto_end_probability = selfplay_defaults["pass_alive_auto_end_probability"]
    args.gocube_root_prune_useless_moves = selfplay_defaults["root_prune_useless_moves"]
    args.gocube_seki_fork_hack_probability = selfplay_defaults["seki_fork_hack_probability"]

    args.gocube_early_fork_game_prob = float(cli.early_fork_prob)
    args.gocube_early_fork_game_expected_move_prop = float(cli.early_fork_expected_move_prop)
    args.gocube_fork_game_prob = float(cli.fork_prob)
    args.gocube_fork_game_min_choices = diverse["fork_game_min_choices"]
    args.gocube_early_fork_game_max_choices = diverse["early_fork_game_max_choices"]
    args.gocube_fork_game_max_choices = diverse["fork_game_max_choices"]
    args.gocube_init_games_with_policy = diverse["init_games_with_policy"]
    args.gocube_policy_init_area_prop = float(cli.policy_init_area_prop)
    args.gocube_policy_init_gamma_shape = diverse["policy_init_gamma_shape"]
    args.gocube_policy_init_temperature = diverse["policy_init_temperature"]
    args.gocube_plain_fork_pool_capacity = diverse["plain_fork_pool_capacity"]

    args.gocube_training_contract = TRAINING_CONTRACT
    args.gocube_lr_warmup_samples = int(cli.lr_warmup_samples)
    args.gocube_lr_warmup_start_factor = float(cli.lr_warmup_start_factor)
    args.gocube_lr_milestone_samples = tuple(int(x) for x in cli.lr_milestone_samples)
    args.gocube_lr_decay_gamma = float(cli.lr_decay_gamma)
    args.gocube_gradient_clip_norm = float(cli.gradient_clip_norm)

    args.cpuct = defaults["cpuct_exploration"]
    args.fpu_reduction = defaults["fpu_reduction_max"]
    args.numMCTSSims = int(cli.sims)
    args.arenaMCTSSims = int(cli.arena_sims)
    args.probFastSim = float(cli.fast_game_prob)
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
    print(f"  observation schema = {args.gocube_observation_schema}")
    print(f"  win/loss factor = {args.gocube_win_loss_utility_factor:g}")
    print(f"  dynamic score factor = {args.gocube_dynamic_score_utility_factor:g}")
    print(f"  cpuct = {args.gocube_cpuct_exploration:g}")
    print(f"  root fpu reduction = {args.gocube_root_fpu_reduction:g}")
    print(f"  cleanup training probability = {args.gocube_cleanup_training_prob:g}")
    print(f"  seki fork hack probability = {args.gocube_seki_fork_hack_probability:g}")
    print("Pinned KataGo diversification:")
    print(f"  early fork probability = {args.gocube_early_fork_game_prob:g}")
    print(f"  early fork depth proportion = {args.gocube_early_fork_game_expected_move_prop:g}")
    print(f"  ordinary fork probability = {args.gocube_fork_game_prob:g}")
    print(
        f"  fork choices = {args.gocube_fork_game_min_choices}.."
        f"{args.gocube_early_fork_game_max_choices} early / "
        f"{args.gocube_fork_game_min_choices}..{args.gocube_fork_game_max_choices} ordinary"
    )
    print(f"  policy-init area proportion = {args.gocube_policy_init_area_prop:g}")
    print(f"  policy-init gamma shape = {args.gocube_policy_init_gamma_shape:g}")
    print(f"  policy-init temperature = {args.gocube_policy_init_temperature:g}")
    print("Sample-clock training:")
    print(f"  contract = {args.gocube_training_contract}")
    print(f"  warmup samples = {args.gocube_lr_warmup_samples}")
    print(f"  warmup start factor = {args.gocube_lr_warmup_start_factor:g}")
    print(f"  LR sample milestones = {args.gocube_lr_milestone_samples}")
    print(f"  LR decay gamma = {args.gocube_lr_decay_gamma:g}")
    print(f"  gradient clip norm = {args.gocube_gradient_clip_norm:g}")


def main(argv=None):
    cli = parse_args(argv)
    game_cls, args = build_katago_training_args(cli)
    if not cli.allow_existing_run:
        assert_fresh_run(args)
    print_katago_search_configuration(args)
    ensure_training_manifest(args.checkpoint, args.run_name, game_cls)
    network = SampleClockNNetWrapper(game_cls, args)
    coach = KataGoSearchCoach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
