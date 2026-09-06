from __future__ import annotations

import argparse
import json
import os
from time import time

import torch
from torch import multiprocessing as mp

from alphazero.Coach import TrainState, _set_state
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.pinned_game import pinned_game_class
from alphazero.envs.gocube.pinned_selfplay import PinnedSelfPlayAgent
from alphazero.envs.gocube.selfplay_semantics import (
    KATAGO_CLEANUP_TRAINING_DEFAULTS,
    KATAGO_PINNED_SELFPLAY_DEFAULTS,
)
from alphazero.envs.gocube.train import GoCubeCoach, build_training_args, print_training_configuration
from alphazero.envs.gocube.training_contract import (
    LR_CLOCK,
    TRAINING_CONTRACT,
    SampleClockLRScheduler,
    SampleClockNNetWrapper,
)
from alphazero.inference_batching import collect_ready_worker_ids, process_coalesced_inference
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.search_contract import (
    KATAGO_PINNED_SEARCH_UTILITY_MODE,
    KATAGO_REFERENCE_COMMIT,
    KATAGO_SEARCH_CONTRACT,
    KATAGO_SEARCH_DEFAULTS,
)


DEFAULT_RUN_NAME = "gocube-cube-4-katago-pinned-s50-fast025-20260906"
_DIVERSIFICATION_TELEMETRY_KEYS = (
    "normal_starts",
    "early_forks",
    "ordinary_forks",
    "policy_initialized_starts",
    "fork_depth_sum",
    "fork_depth_count",
)


def _parse_sample_milestones(value: str) -> tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    milestones = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if any(value <= 0 for value in milestones):
        raise ValueError("LR sample milestones must be positive")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("LR sample milestones must be strictly increasing and unique")
    return milestones


class KataGoSearchCoach(GoCubeCoach):
    """GoCube coach with pinned search, diversified self-play, and sample-clock telemetry."""

    def __init__(self, game_cls, nnet, args):
        super().__init__(game_cls, nnet, args)
        self.score_tensors = []
        self.ownership_tensors = []

    def _reset_selfplay_telemetry(self):
        telemetry = super()._reset_selfplay_telemetry()
        for key in _DIVERSIFICATION_TELEMETRY_KEYS:
            if key not in telemetry:
                telemetry[key] = mp.Value('q', 0)
        return telemetry

    def _snapshot_selfplay_telemetry(self, iteration):
        super()._snapshot_selfplay_telemetry(iteration)
        data = self._iteration_telemetry
        depth_count = int(data.get("fork_depth_count", 0))
        average_depth = float(data.get("fork_depth_sum", 0)) / depth_count if depth_count else 0.0
        data["average_fork_depth"] = average_depth
        for key in ("normal_starts", "early_forks", "ordinary_forks", "policy_initialized_starts"):
            self.writer.add_scalar(f"selfplay/{key}", int(data.get(key, 0)), iteration)
        self.writer.add_scalar("selfplay/average_fork_depth", average_depth, iteration)

    def _record_training_metrics(self, iteration, **kwargs):
        super()._record_training_metrics(iteration, **kwargs)
        net = self.train_net
        metrics = {
            "training_contract": TRAINING_CONTRACT,
            "lr_clock": LR_CLOCK,
            "total_training_samples": int(net.total_training_samples),
            "total_optimizer_updates": int(net.total_optimizer_updates),
            "effective_lr": float(net.optimizer.param_groups[0]["lr"]),
            "samples_since_lr_change": int(net.samples_since_lr_change),
            "gradient_norm_pre_clip": float(net.last_train_gradient_norm),
            "gradient_norm_pre_clip_avg": float(net.last_train_gradient_norm_avg),
            "gradient_clip_events": int(net.last_train_clip_events),
            "gradient_clip_checks": int(net.last_train_clip_checks),
            "gradient_clip_frequency": float(net.last_train_clip_frequency),
        }
        self._training_telemetry.update(metrics)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"training/{key}", value, iteration)
        self._write_training_metrics_manifest(iteration)

    def _write_training_metrics_manifest(self, iteration):
        folder = os.path.join(
            self.args.data,
            self.args.run_name,
            "records",
            f"iteration-{int(iteration):04d}",
        )
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "training-metrics.json")
        temporary = path + ".tmp"
        payload = {
            "schema_version": 1,
            "run_name": self.args.run_name,
            "iteration": int(iteration),
            "training_contract": TRAINING_CONTRACT,
            "metrics": self._training_telemetry,
        }
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._training_telemetry_manifest_path = path

    def _print_iteration_summary(self, iteration):
        super()._print_iteration_summary(iteration)
        sample = self._iteration_telemetry
        training = getattr(self, "_training_telemetry", {})
        print()
        print(f"Normal starts:             {sample.get('normal_starts', 0)}")
        print(f"Early forks:               {sample.get('early_forks', 0)}")
        print(f"Ordinary forks:            {sample.get('ordinary_forks', 0)}")
        print(f"Policy-initialized starts: {sample.get('policy_initialized_starts', 0)}")
        print(f"Average fork depth:        {sample.get('average_fork_depth', 0.0):.2f}")
        print()
        print(f"Total training samples:    {training.get('total_training_samples', 0)}")
        print(f"Optimizer updates total:   {training.get('total_optimizer_updates', 0)}")
        print(f"Effective LR:              {training.get('effective_lr', self.args.lr):g}")
        print(f"Samples since LR change:   {training.get('samples_since_lr_change', 0)}")
        print(f"Gradient norm pre-clip:    {training.get('gradient_norm_pre_clip', 0.0):.4g}")
        print(f"Gradient clipping freq:    {training.get('gradient_clip_frequency', 0.0):.2%}")

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
                PinnedSelfPlayAgent(
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
    parser.add_argument("--lr-warmup-samples", type=int, default=100_000)
    parser.add_argument("--lr-warmup-start-factor", type=float, default=0.05)
    parser.add_argument(
        "--lr-sample-milestones",
        default="",
        help="Comma-separated total-training-sample thresholds. Empty means no post-warmup decay.",
    )
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
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
    base_game_cls, inherited_args = build_training_args(cli)
    # The framework get_args() object is globally shared. Never attach the new
    # sample scheduler to that object: copy before adding this training contract.
    args = inherited_args.copy()
    game_cls = pinned_game_class(base_game_cls)
    defaults = KATAGO_SEARCH_DEFAULTS
    cleanup_defaults = KATAGO_CLEANUP_TRAINING_DEFAULTS
    selfplay_defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS

    if cli.lr_warmup_samples < 0:
        raise ValueError("lr-warmup-samples must be non-negative")
    if not 0.0 < cli.lr_warmup_start_factor <= 1.0:
        raise ValueError("lr-warmup-start-factor must be within (0,1]")
    if cli.lr_gamma <= 0.0:
        raise ValueError("lr-gamma must be positive")
    if cli.gradient_clip_norm <= 0.0:
        raise ValueError("gradient-clip-norm must be positive")
    sample_milestones = _parse_sample_milestones(cli.lr_sample_milestones)

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
    args.gocube_early_fork_game_probability = selfplay_defaults["early_fork_game_probability"]
    args.gocube_early_fork_expected_move_prop = selfplay_defaults["early_fork_expected_move_prop"]
    args.gocube_fork_game_probability = selfplay_defaults["fork_game_probability"]
    args.gocube_fork_game_min_choices = selfplay_defaults["fork_game_min_choices"]
    args.gocube_early_fork_game_max_choices = selfplay_defaults["early_fork_game_max_choices"]
    args.gocube_fork_game_max_choices = selfplay_defaults["fork_game_max_choices"]
    args.gocube_init_games_with_policy = selfplay_defaults["init_games_with_policy"]
    args.gocube_policy_init_area_prop = selfplay_defaults["policy_init_area_prop"]
    args.gocube_policy_init_gamma_shape = selfplay_defaults["policy_init_gamma_shape"]
    args.gocube_policy_init_temperature = selfplay_defaults["policy_init_temperature"]

    args.gocube_training_contract = TRAINING_CONTRACT
    args.gocube_lr_clock = LR_CLOCK
    args.gocube_lr_sample_milestones = sample_milestones
    args.gocube_lr_gamma = float(cli.lr_gamma)
    args.gocube_lr_warmup_samples = int(cli.lr_warmup_samples)
    args.gocube_lr_warmup_start_factor = float(cli.lr_warmup_start_factor)
    args.gocube_gradient_clip_norm = float(cli.gradient_clip_norm)
    args.scheduler = SampleClockLRScheduler
    args.scheduler_args = {
        "milestones": sample_milestones,
        "gamma": args.gocube_lr_gamma,
        "warmup_samples": args.gocube_lr_warmup_samples,
        "warmup_start_factor": args.gocube_lr_warmup_start_factor,
    }

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
    print(f"  dynamic center zero weight = {args.gocube_dynamic_score_center_zero_weight:g}")
    print(f"  dynamic center scale = {args.gocube_dynamic_score_center_scale:g}")
    print(f"  cpuct = {args.gocube_cpuct_exploration:g}")
    print(f"  fpu reduction = {args.fpu_reduction:g}")
    print(f"  root fpu reduction = {args.gocube_root_fpu_reduction:g}")
    print(f"  root ending bonus points = {args.gocube_root_ending_bonus_points:g}")
    print(f"  fill dame before pass = {args.gocube_fill_dame_before_pass}")
    print(f"  conservative pass = {args.gocube_conservative_pass}")
    print(f"  cleanup training probability = {args.gocube_cleanup_training_prob:g}")
    print(f"  pass-alive auto-end probability = {args.gocube_pass_alive_auto_end_probability:g}")
    print(f"  seki fork hack probability = {args.gocube_seki_fork_hack_probability:g}")
    print("Pinned KataGo diversification:")
    print(f"  early fork probability = {args.gocube_early_fork_game_probability:g}")
    print(f"  early fork expected move prop = {args.gocube_early_fork_expected_move_prop:g}")
    print(f"  ordinary fork probability = {args.gocube_fork_game_probability:g}")
    print(f"  fork choices = {args.gocube_fork_game_min_choices}..{args.gocube_fork_game_max_choices}")
    print(f"  early fork max choices = {args.gocube_early_fork_game_max_choices}")
    print(f"  policy-init area prop = {args.gocube_policy_init_area_prop:g}")
    print(f"  policy-init gamma shape = {args.gocube_policy_init_gamma_shape:g}")
    print(f"  policy-init temperature = {args.gocube_policy_init_temperature:g}")
    print("Training safety contract:")
    print(f"  contract = {args.gocube_training_contract}")
    print(f"  LR clock = {args.gocube_lr_clock}")
    print(f"  warmup samples = {args.gocube_lr_warmup_samples}")
    print(f"  warmup start factor = {args.gocube_lr_warmup_start_factor:g}")
    print(f"  sample milestones = {list(args.gocube_lr_sample_milestones)}")
    print(f"  LR gamma = {args.gocube_lr_gamma:g}")
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
