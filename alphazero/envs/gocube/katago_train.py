from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from glob import glob
from time import time

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, TensorDataset

from alphazero.Arena import Arena
from alphazero.Coach import TrainState, _set_state
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.envs.gocube.diversified_game import diversified_pinned_game_class
from alphazero.envs.gocube.diversified_selfplay import (
    DiversifiedPinnedSelfPlayAgent,
    KATAGO_PINNED_DIVERSIFICATION_DEFAULTS,
)
from alphazero.envs.gocube.exploration_contract import (
    KATAGO_PINNED_EXPLORATION_CONTRACT,
    KATAGO_PINNED_EXPLORATION_DEFAULTS,
)
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.production_training import (
    anchor_checkpoint_iteration,
    arena_regression_signals,
    build_replay_training_plan,
    summarize_arena_outcomes,
)
from alphazero.envs.gocube.records import ITERATION_MANIFEST_FILENAME
from alphazero.envs.gocube.sample_clock import SampleClockNNetWrapper, TRAINING_CONTRACT
from alphazero.envs.gocube.selfplay_semantics import (
    KATAGO_CLEANUP_TRAINING_DEFAULTS,
    KATAGO_PINNED_SELFPLAY_DEFAULTS,
)
from alphazero.envs.gocube.train import (
    GoCubeCoach,
    build_training_args,
    print_training_configuration,
    validate_tensor_row_counts,
)
from alphazero.inference_batching import collect_ready_worker_ids, process_coalesced_inference
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.search_contract import (
    KATAGO_PINNED_SEARCH_UTILITY_MODE,
    KATAGO_REFERENCE_COMMIT,
    KATAGO_SEARCH_CONTRACT,
    KATAGO_SEARCH_DEFAULTS,
)
from alphazero.utils import get_iter_file


DEFAULT_RUN_NAME = "gocube-cube-4-katago-pinned-s50-ratioexp-arena-20260906"
DEFAULT_LR_WARMUP_SAMPLES = 2_000_000
DEFAULT_LR_MILESTONE_SAMPLES = (20_000_000, 40_000_000)
DEFAULT_LR_WARMUP_START_FACTOR = 0.05
DEFAULT_LR_DECAY_GAMMA = 0.1
DEFAULT_GRADIENT_CLIP_NORM = 5.0
DEFAULT_TRAIN_SAMPLES_PER_NEW_SAMPLE = 1.0
DEFAULT_ARENA_GAMES_PER_OPPONENT = 64
DEFAULT_ARENA_ANCHOR_PERIOD = 10
DEFAULT_ARENA_REGRESSION_WIN_RATE = 0.45
DEFAULT_ARENA_SEED = 20260906

_SWEEP_FLAG_TO_ARG_KEYS = {
    "--chosen-move-temperature-halflife": ("gocube_chosen_move_temperature_halflife",),
    "--root-dirichlet-noise-weight": (
        "gocube_root_dirichlet_noise_weight",
        "root_noise_frac",
    ),
    "--replay-window-iters": ("gocube_replay_window_iters",),
    "--fast-game-prob": ("probFastSim",),
    "--train-samples-per-new-sample": ("gocube_train_samples_per_new_sample",),
    "--arena-batched": ("gocube_arena_batched", "arenaBatched"),
}

_DIVERSIFICATION_COUNTER_KEYS = (
    "normal_starts",
    "early_forks",
    "ordinary_forks",
    "policy_initialized_starts",
    "fork_depth_sum",
    "fork_depth_count",
)
_EXPLORATION_COUNTER_KEYS = (
    "exploration_telemetry_positions",
    "exploration_raw_visits",
    "exploration_forced_visits",
    "exploration_target_visits",
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


def checkpoint_arg_overrides(cli, args) -> dict[str, object]:
    """Return only sweep values that the user explicitly supplied on the CLI.

    Ordinary production resume keeps its historical strict saved-args behavior.
    Explicit sweep flags opt a cloned run into retaining just those configured
    overrides while all checkpoint compatibility validation remains active.
    """

    explicit = getattr(cli, "_explicit_sweep_flags", frozenset())
    overrides = {}
    for flag, keys in _SWEEP_FLAG_TO_ARG_KEYS.items():
        if flag not in explicit:
            continue
        for key in keys:
            overrides[key] = args[key]
    return overrides


class KataGoSearchCoach(GoCubeCoach):
    """Pinned GoCube search with sample-ratio training and observational Arena."""

    def __init__(self, game_cls, nnet, args):
        super().__init__(game_cls, nnet, args)
        self.score_tensors = []
        self.ownership_tensors = []
        self._arena_telemetry = None

    def _load_model(self, model, iteration):
        overrides = getattr(self.train_net, "_gocube_checkpoint_arg_overrides", None)
        if not overrides:
            return super()._load_model(model, iteration)

        folder = os.path.join(self.args.checkpoint, self.args.run_name)
        filename = get_iter_file(iteration)
        checkpoint = torch.load(os.path.join(folder, filename), map_location="cpu")
        saved_args = checkpoint.get("args")
        if not isinstance(saved_args, dict):
            return super()._load_model(model, iteration)

        restored = {}
        for key in overrides:
            if key not in saved_args:
                continue
            restored[key] = (key in model.args, model.args.get(key))
            model.args[key] = saved_args[key]
        try:
            model.load_checkpoint(
                folder=folder,
                filename=filename,
                use_saved_args=False,
            )
        finally:
            for key, (was_present, value) in restored.items():
                if was_present:
                    model.args[key] = value
                else:
                    model.args.pop(key, None)
        for key, value in overrides.items():
            model.args[key] = value

    def _reset_selfplay_telemetry(self):
        # Arena results belong to the iteration that produced them. Clear the
        # prior iteration before the current manifest is patched during train().
        self._arena_telemetry = None
        telemetry = super()._reset_selfplay_telemetry()
        for key in _DIVERSIFICATION_COUNTER_KEYS + _EXPLORATION_COUNTER_KEYS:
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
            "exploration_telemetry_positions": 0,
            "exploration_raw_visits": 0,
            "exploration_forced_visits": 0,
            "exploration_target_visits": 0,
            "exploration_forced_visit_fraction": 0.0,
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

        raw_visits = int(snapshot.get("exploration_raw_visits", 0))
        forced_visits = int(snapshot.get("exploration_forced_visits", 0))
        forced_fraction = forced_visits / raw_visits if raw_visits else 0.0
        snapshot["exploration_forced_visit_fraction"] = forced_fraction
        for key in _EXPLORATION_COUNTER_KEYS:
            self.writer.add_scalar(f"exploration/{key}", int(snapshot.get(key, 0)), iteration)
        self.writer.add_scalar("exploration/forced_visit_fraction", forced_fraction, iteration)

    def _record_training_metrics(
        self,
        iteration,
        *,
        history_iterations,
        window_samples,
        latest_iteration_samples,
        planned_steps,
        planned_training_samples=0,
    ):
        super()._record_training_metrics(
            iteration,
            history_iterations=history_iterations,
            window_samples=window_samples,
            latest_iteration_samples=latest_iteration_samples,
            planned_steps=planned_steps,
        )
        scheduler = self.train_net.scheduler
        actual_samples = int(self.train_net.last_train_examples_seen)
        new_samples = int(latest_iteration_samples)
        effective_ratio = actual_samples / new_samples if new_samples else 0.0
        effective_passes = actual_samples / window_samples if window_samples else 0.0
        extras = {
            "new_selfplay_samples": new_samples,
            "replay_window_samples": int(window_samples),
            "train_samples_per_new_sample": float(self.args.gocube_train_samples_per_new_sample),
            "planned_training_samples": int(planned_training_samples),
            "actual_training_samples": actual_samples,
            "optimizer_steps": int(self.train_net.last_train_actual_steps),
            "effective_train_new_data_ratio": effective_ratio,
            "effective_passes_over_replay_window": effective_passes,
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
        print("Exploration target correction:")
        print(f"  telemetry positions:     {sample.get('exploration_telemetry_positions', 0)}")
        print(f"  raw root visits:         {sample.get('exploration_raw_visits', 0)}")
        print(f"  forced visits removed:   {sample.get('exploration_forced_visits', 0)}")
        print(f"  forced visit fraction:   {sample.get('exploration_forced_visit_fraction', 0.0):.2%}")
        print("Sample-based replay training:")
        print(f"  new self-play samples:   {training.get('new_selfplay_samples', 0)}")
        print(f"  replay window samples:   {training.get('replay_window_samples', 0)}")
        print(f"  configured train/new:    {training.get('train_samples_per_new_sample', 0.0):g}")
        print(f"  planned train samples:   {training.get('planned_training_samples', 0)}")
        print(f"  actual train samples:    {training.get('actual_training_samples', 0)}")
        print(f"  effective train/new:     {training.get('effective_train_new_data_ratio', 0.0):.3f}")
        print(f"  passes over replay:      {training.get('effective_passes_over_replay_window', 0.0):.3f}")
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
        aggregate["exploration_policy_target"] = {
            "telemetry_positions": int(sample.get("exploration_telemetry_positions", 0)),
            "raw_root_visits": int(sample.get("exploration_raw_visits", 0)),
            "forced_exploration_visits_removed": int(sample.get("exploration_forced_visits", 0)),
            "target_visits_after_correction": int(sample.get("exploration_target_visits", 0)),
            "forced_visit_fraction": float(sample.get("exploration_forced_visit_fraction", 0.0)),
            "per_position_detail": "game records -> moves[].search_telemetry",
        }
        if self._arena_telemetry is not None:
            aggregate["arena"] = self._arena_telemetry
        temporary = path + ".training.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _replay_iterations(self, iteration):
        explicit_window = (
            self.args.get("gocube_replay_window_iters", None)
            if isinstance(self.args, dict)
            else getattr(self.args, "gocube_replay_window_iters", None)
        )
        if explicit_window is not None:
            start = max(1, int(iteration) - int(explicit_window) + 1)
            return range(start, int(iteration) + 1)

        current_history_size = min(
            max(
                self.args.minTrainHistoryWindow,
                (iteration + self.args.minTrainHistoryWindow) // self.args.trainHistoryIncrementIters,
            ),
            self.args.maxTrainHistoryWindow,
        )
        return range(max(1, iteration - current_history_size), iteration + 1)

    def _load_replay_datasets(self, iteration):
        datasets = []
        loaded_samples = {}
        for train_iter in self._replay_iterations(iteration):
            filename = os.path.join(
                self.args.data,
                self.args.run_name,
                get_iter_file(train_iter).replace('.pkl', ''),
            )
            try:
                tensors = [
                    torch.load(filename + suffix)
                    for suffix in (
                        '-data.pkl', '-policy.pkl', '-value.pkl', '-score.pkl',
                        '-ownership.pkl', '-ownership-mask.pkl',
                    )
                ]
            except FileNotFoundError as exc:
                print('Warning: could not find complete V3 tensor data. ' + str(exc))
                continue
            row_count = validate_tensor_row_counts(tensors)
            if tensors[0].shape[1:] != self.game_cls.observation_size():
                raise ValueError("V3 dataset observation schema/shape mismatch")
            datasets.append(TensorDataset(*tensors))
            loaded_samples[train_iter] = row_count
        return datasets, loaded_samples

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        if self.args.train_on_past_data:
            raise ValueError(
                "Pinned production training does not support train_on_past_data; replay must come from this run"
            )
        datasets, loaded_samples = self._load_replay_datasets(iteration)
        replay_window_samples = sum(len(dataset) for dataset in datasets)
        new_samples = int(loaded_samples.get(iteration, 0))
        plan = build_replay_training_plan(
            new_selfplay_samples=new_samples,
            replay_window_samples=replay_window_samples,
            train_samples_per_new_sample=self.args.gocube_train_samples_per_new_sample,
            batch_size=self.args.train_batch_size,
        )

        if not datasets or replay_window_samples == 0 or plan.planned_training_samples == 0:
            print("No planned production training samples; skipping optimizer step.")
            self.train_net.last_train_planned_steps = 0
            self.train_net.last_train_actual_steps = 0
            self.train_net.last_train_examples_seen = 0
            self.train_net.last_train_learning_rate = float(self.train_net.optimizer.param_groups[0]["lr"])
            self.loss_pi, self.loss_v = self.train_net.l_pi, self.train_net.l_v
        else:
            dataset = ConcatDataset(datasets)
            sampler = RandomSampler(
                dataset,
                replacement=True,
                num_samples=plan.planned_training_samples,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                num_workers=self.args.workers,
                pin_memory=True,
            )
            self.loss_pi, self.loss_v = self.train_net.train(
                dataloader,
                plan.planned_optimizer_steps,
            )
            del dataloader
            del dataset

        self._record_training_metrics(
            iteration,
            history_iterations=len(datasets),
            window_samples=replay_window_samples,
            latest_iteration_samples=new_samples,
            planned_steps=plan.planned_optimizer_steps,
            planned_training_samples=plan.planned_training_samples,
        )
        self.writer.add_scalar('loss/policy', self.loss_pi, iteration)
        self.writer.add_scalar('loss/value', self.loss_v, iteration)
        self.writer.add_scalar('loss/ownership', self.train_net.l_ownership, iteration)
        self.writer.add_scalar('loss/score', self.train_net.l_score, iteration)
        self.writer.add_scalar('loss/total', self.train_net.l_total, iteration)
        self._print_iteration_summary(iteration)
        self._save_model(self.train_net, iteration)
        self._patch_iteration_manifest()

    def _checkpoint_descriptor(self, iteration):
        return {
            "id": f"{self.args.run_name}@{int(iteration)}",
            "iteration": int(iteration),
            "path": os.path.relpath(
                os.path.abspath(os.path.join(
                    self.args.checkpoint,
                    self.args.run_name,
                    get_iter_file(int(iteration)),
                )),
                os.getcwd(),
            ),
        }

    def _arena_history(self, *, anchor_iteration=None):
        pattern = os.path.join(
            self.args.data,
            self.args.run_name,
            "records",
            "iteration-*",
            ITERATION_MANIFEST_FILENAME,
        )
        previous_rates = []
        anchor_rates = []
        for path in sorted(glob(pattern)):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                arena = manifest.get("aggregate_metrics", {}).get("arena")
                if not isinstance(arena, dict):
                    continue
                previous = arena.get("current_vs_previous", {})
                if "win_rate" in previous:
                    previous_rates.append(float(previous["win_rate"]))
                anchor = arena.get("current_vs_anchor", {})
                opponent = anchor.get("opponent_checkpoint", {})
                if (
                    anchor_iteration is not None
                    and int(opponent.get("iteration", -1)) == int(anchor_iteration)
                    and "win_rate" in anchor
                ):
                    anchor_rates.append(float(anchor["win_rate"]))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return previous_rates, anchor_rates

    def _run_checkpoint_arena(self, current_iteration, opponent_iteration):
        games = int(self.args.gocube_arena_games_per_opponent)
        self._load_model(self.self_play_net, int(opponent_iteration))
        arena_args = self.args.copy()
        arena_args.numMCTSSims = int(self.args.arenaMCTSSims)
        arena_args.arenaMCTSSims = int(self.args.arenaMCTSSims)
        arena_args.probFastSim = 0.0
        arena_args.add_root_noise = False
        arena_args.add_root_temp = False
        arena_args.startTemp = 0.0
        arena_args.arenaTemp = 0.0
        arena_args.arena_batch_size = max(
            1,
            math.ceil(games / max(1, int(arena_args.get("workers", 1)))),
        )
        current_player = MCTSPlayer(self.train_net, self.game_cls, arena_args)
        opponent_player = MCTSPlayer(self.self_play_net, self.game_cls, arena_args)
        use_batched = bool(
            self.args.get("gocube_arena_batched", False)
            if isinstance(self.args, dict)
            else getattr(self.args, "gocube_arena_batched", False)
        )
        arena = Arena(
            [current_player, opponent_player],
            self.game_cls,
            use_batched_mcts=use_batched,
            args=arena_args,
        )

        if use_batched:
            seed = (
                int(self.args.gocube_arena_seed)
                + int(current_iteration) * 100_003
                + int(opponent_iteration) * 1_009
            )
            np.random.seed(seed & 0xFFFFFFFF)
            random.seed(seed)
            torch.manual_seed(seed & 0x7FFFFFFF)
            wins, draws, _ = arena.play_games(games, verbose=False, shuffle_players=True)
            no_results = int(arena.no_results)
            scored = int(sum(wins) + draws)
            summary = {
                "games": int(scored + no_results),
                "scored_games": scored,
                "wins": int(wins[0]),
                "losses": int(wins[1]),
                "draws": int(draws),
                "no_results": no_results,
                "win_rate": (
                    (float(wins[0]) + 0.5 * float(draws)) / scored if scored else 0.0
                ),
                "by_color": arena.player_color_results(0),
            }
        else:
            outcomes = []
            for game_index in range(games):
                seed = (
                    int(self.args.gocube_arena_seed)
                    + int(current_iteration) * 100_003
                    + int(opponent_iteration) * 1_009
                    + game_index
                )
                np.random.seed(seed & 0xFFFFFFFF)
                random.seed(seed)
                order = [0, 1] if game_index % 2 == 0 else [1, 0]
                current_color = "black" if order[0] == 0 else "white"
                final_state, winstate = arena.play_game(False, order)
                has_draw_slot = len(winstate) > self.game_cls.num_players()
                if has_draw_slot and bool(winstate[-1]):
                    result = "no_result" if getattr(final_state, "terminal_kind", None) == "no_result" else "draw"
                else:
                    winner_color = next(
                        (idx for idx, won in enumerate(winstate[:self.game_cls.num_players()]) if bool(won)),
                        None,
                    )
                    if winner_color is None:
                        result = "no_result"
                    else:
                        result = "win" if order[winner_color] == 0 else "loss"
                outcomes.append((current_color, result))
                if self.stop_train.is_set():
                    break
            summary = summarize_arena_outcomes(outcomes)

        summary["opponent_checkpoint"] = self._checkpoint_descriptor(opponent_iteration)
        summary["evaluation_contract"] = {
            "deterministic": not use_batched,
            "search_sims": int(self.args.arenaMCTSSims),
            "fast_search": False,
            "root_noise": False,
            "root_policy_temperature": False,
            "move_temperature": 0.0,
            "alternating_colors": not use_batched,
            "batched": use_batched,
            "workers": int(self.args.get("workers", 1)) if isinstance(self.args, dict) else int(getattr(self.args, "workers", 1)),
            "seed": int(self.args.gocube_arena_seed),
            "rules_fingerprint": self.args.gocube_rules_fingerprint,
            "komi": float(self.args.gocube_komi),
        }
        return summary

    @_set_state(TrainState.COMPARE_PAST)
    def compareToPast(self, model_iter):
        if self.args.model_gating:
            raise RuntimeError("Production checkpoint Arena is observational and must never gate training")
        previous_iteration = max(0, int(model_iter) - 1)
        anchor_iteration = anchor_checkpoint_iteration(
            int(model_iter),
            int(self.args.gocube_arena_anchor_period),
        )
        print(f"ARENA current {model_iter} vs previous {previous_iteration}")
        previous = self._run_checkpoint_arena(model_iter, previous_iteration)
        if self.stop_train.is_set():
            return
        if anchor_iteration == previous_iteration:
            anchor = dict(previous)
        else:
            print(f"ARENA current {model_iter} vs anchor {anchor_iteration}")
            anchor = self._run_checkpoint_arena(model_iter, anchor_iteration)
        if self.stop_train.is_set():
            return

        previous_history, anchor_history = self._arena_history(anchor_iteration=anchor_iteration)
        regression = arena_regression_signals(
            previous["win_rate"],
            previous_history,
            material_threshold=float(self.args.gocube_arena_regression_win_rate),
        )
        anchor_series = anchor_history + [float(anchor["win_rate"])]
        self._arena_telemetry = {
            "schema_version": 1,
            "model_gating": False,
            "current_checkpoint": self._checkpoint_descriptor(model_iter),
            "current_vs_previous": previous,
            "current_vs_anchor": anchor,
            "cumulative_progress_relative_to_anchor": {
                "anchor_iteration": int(anchor_iteration),
                "current_win_rate": float(anchor["win_rate"]),
                "delta_from_even": float(anchor["win_rate"]) - 0.5,
                "win_rate_history_for_anchor": anchor_series[-10:],
            },
            "regression_signals": regression,
        }
        self.writer.add_scalar("arena/previous_win_rate", previous["win_rate"], model_iter)
        self.writer.add_scalar("arena/anchor_win_rate", anchor["win_rate"], model_iter)
        self.writer.add_scalar(
            "arena/material_regression",
            1 if regression["material_regression"] else 0,
            model_iter,
        )
        self._patch_iteration_manifest()
        print(
            f"Arena previous: W/L/D={previous['wins']}/{previous['losses']}/{previous['draws']} "
            f"winrate={previous['win_rate']:.3f}"
        )
        print(
            f"Arena anchor {anchor_iteration}: W/L/D={anchor['wins']}/{anchor['losses']}/{anchor['draws']} "
            f"winrate={anchor['win_rate']:.3f}"
        )

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
    exploration = KATAGO_PINNED_EXPLORATION_DEFAULTS
    raw_argv = list(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument(
        "--train-samples-per-new-sample",
        type=float,
        default=DEFAULT_TRAIN_SAMPLES_PER_NEW_SAMPLE,
    )
    parser.add_argument("--replay-window-iters", type=int, default=None)
    parser.add_argument("--fast-game-prob", type=float, default=0.25)
    parser.add_argument(
        "--chosen-move-temperature-halflife",
        type=float,
        default=exploration["chosen_move_temperature_halflife"],
    )
    parser.add_argument(
        "--root-dirichlet-noise-weight",
        type=float,
        default=exploration["root_dirichlet_noise_weight"],
    )
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
    parser.add_argument(
        "--arena-games-per-opponent",
        type=int,
        default=DEFAULT_ARENA_GAMES_PER_OPPONENT,
    )
    parser.add_argument("--arena-anchor-period", type=int, default=DEFAULT_ARENA_ANCHOR_PERIOD)
    parser.add_argument(
        "--arena-regression-win-rate",
        type=float,
        default=DEFAULT_ARENA_REGRESSION_WIN_RATE,
    )
    parser.add_argument("--arena-seed", type=int, default=DEFAULT_ARENA_SEED)
    parser.add_argument("--arena-batched", action="store_true")
    parser.add_argument("--no-arena", action="store_true")
    parser.add_argument("--model-gating", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--allow-existing-run",
        action="store_true",
        help="Allow resuming an existing namespace. The checkpoint must satisfy the new training contract.",
    )
    parsed = parser.parse_args(raw_argv)
    parsed._explicit_sweep_flags = frozenset(
        flag
        for flag in _SWEEP_FLAG_TO_ARG_KEYS
        if any(token == flag or token.startswith(flag + "=") for token in raw_argv)
    )
    return parsed


def build_katago_training_args(cli):
    if cli.train_steps_per_iteration is not None:
        raise ValueError(
            "Pinned production training uses --train-samples-per-new-sample; fixed per-iteration steps are disabled"
        )
    if cli.model_gating:
        raise ValueError("Production checkpoint Arena is observational; model gating is disabled")
    if cli.arena_games_per_opponent < 1:
        raise ValueError("arena-games-per-opponent must be positive")
    if cli.arena_anchor_period < 1:
        raise ValueError("arena-anchor-period must be positive")
    if not 0.0 <= cli.arena_regression_win_rate <= 0.5:
        raise ValueError("arena-regression-win-rate must be within [0,0.5]")
    if not math.isfinite(float(cli.chosen_move_temperature_halflife)) or float(
        cli.chosen_move_temperature_halflife
    ) <= 0.0:
        raise ValueError("chosen-move-temperature-halflife must be finite and positive")
    if not math.isfinite(float(cli.root_dirichlet_noise_weight)) or not 0.0 <= float(
        cli.root_dirichlet_noise_weight
    ) <= 1.0:
        raise ValueError("root-dirichlet-noise-weight must be finite and within [0,1]")
    if cli.replay_window_iters is not None and int(cli.replay_window_iters) < 1:
        raise ValueError("replay-window-iters must be positive")
    build_replay_training_plan(
        new_selfplay_samples=1,
        replay_window_samples=1,
        train_samples_per_new_sample=cli.train_samples_per_new_sample,
        batch_size=cli.train_batch_size,
    )

    base_game_cls, args = build_training_args(cli)
    args = args.copy()
    game_cls = diversified_pinned_game_class(base_game_cls)
    defaults = KATAGO_SEARCH_DEFAULTS
    cleanup_defaults = KATAGO_CLEANUP_TRAINING_DEFAULTS
    selfplay_defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
    diverse = KATAGO_PINNED_DIVERSIFICATION_DEFAULTS
    exploration = KATAGO_PINNED_EXPLORATION_DEFAULTS

    if not math.isclose(float(args.gocube_komi), 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Production GoCube contract requires komi 0.5, got {args.gocube_komi}")

    args.search_utility_mode = KATAGO_PINNED_SEARCH_UTILITY_MODE
    args.gocube_katago_search_contract = KATAGO_SEARCH_CONTRACT
    args.gocube_katago_search_reference_commit = KATAGO_REFERENCE_COMMIT
    args.gocube_katago_exploration_contract = KATAGO_PINNED_EXPLORATION_CONTRACT
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

    args.gocube_root_dirichlet_noise_total_concentration = exploration[
        "root_dirichlet_noise_total_concentration"
    ]
    args.gocube_root_dirichlet_noise_weight = float(cli.root_dirichlet_noise_weight)
    args.gocube_root_policy_temperature_early = exploration["root_policy_temperature_early"]
    args.gocube_root_policy_temperature = exploration["root_policy_temperature"]
    args.gocube_root_policy_temperature_halflife = exploration[
        "root_policy_temperature_halflife"
    ]
    args.gocube_root_desired_per_child_visits_coeff = exploration[
        "root_desired_per_child_visits_coeff"
    ]
    args.gocube_chosen_move_temperature_halflife = float(cli.chosen_move_temperature_halflife)
    args.root_noise_frac = float(cli.root_dirichlet_noise_weight)
    args.root_policy_temp = exploration["root_policy_temperature"]

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
    args.gocube_train_samples_per_new_sample = float(cli.train_samples_per_new_sample)
    args.gocube_replay_window_iters = (
        None if cli.replay_window_iters is None else int(cli.replay_window_iters)
    )
    args.gocube_lr_warmup_samples = int(cli.lr_warmup_samples)
    args.gocube_lr_warmup_start_factor = float(cli.lr_warmup_start_factor)
    args.gocube_lr_milestone_samples = tuple(int(x) for x in cli.lr_milestone_samples)
    args.gocube_lr_decay_gamma = float(cli.lr_decay_gamma)
    args.gocube_gradient_clip_norm = float(cli.gradient_clip_norm)

    args.gocube_arena_games_per_opponent = int(cli.arena_games_per_opponent)
    args.gocube_arena_anchor_period = int(cli.arena_anchor_period)
    args.gocube_arena_regression_win_rate = float(cli.arena_regression_win_rate)
    args.gocube_arena_seed = int(cli.arena_seed)
    args.gocube_arena_batched = bool(cli.arena_batched)

    args.cpuct = defaults["cpuct_exploration"]
    args.fpu_reduction = defaults["fpu_reduction_max"]
    args.numMCTSSims = int(cli.sims)
    args.arenaMCTSSims = int(cli.arena_sims)
    args.probFastSim = float(cli.fast_game_prob)
    args.numWarmupIters = 0
    args.autoTrainSteps = True
    args.train_steps_per_iteration = None
    args.compareWithBaseline = False
    args.compareWithPast = not cli.no_arena and not cli.smoke
    args.pastCompareFreq = 1
    args.model_gating = False
    args.arenaBatched = bool(cli.arena_batched)
    args.arenaTemp = 0.0
    args.startTemp = 1.0
    args.pop("train_sample_ratio", None)

    if int(args.numFastSims) != 20:
        raise ValueError(f"Pinned production contract requires 20 fast sims, got {args.numFastSims}")
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
    print("Pinned KataGo exploration/policy-target block:")
    print(f"  contract = {args.gocube_katago_exploration_contract}")
    print(f"  Dirichlet total concentration = {args.gocube_root_dirichlet_noise_total_concentration:g}")
    print(f"  Dirichlet weight = {args.gocube_root_dirichlet_noise_weight:g}")
    print(f"  chosen move temperature halflife = {args.gocube_chosen_move_temperature_halflife:g}")
    print(f"  root policy temp early/normal = {args.gocube_root_policy_temperature_early:g}/{args.gocube_root_policy_temperature:g}")
    print(f"  root desired child visits coeff = {args.gocube_root_desired_per_child_visits_coeff:g}")
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
    print("Sample-based replay training:")
    print(f"  contract = {args.gocube_training_contract}")
    print(f"  train samples/new sample = {args.gocube_train_samples_per_new_sample:g}")
    if args.gocube_replay_window_iters is None:
        print("  replay window = production schedule")
    else:
        print(f"  replay window = last {args.gocube_replay_window_iters} iterations")
    print(f"  warmup samples = {args.gocube_lr_warmup_samples}")
    print(f"  warmup start factor = {args.gocube_lr_warmup_start_factor:g}")
    print(f"  LR sample milestones = {args.gocube_lr_milestone_samples}")
    print(f"  LR decay gamma = {args.gocube_lr_decay_gamma:g}")
    print(f"  gradient clip norm = {args.gocube_gradient_clip_norm:g}")
    print("Checkpoint Arena:")
    print(f"  games/opponent = {args.gocube_arena_games_per_opponent}")
    print(f"  anchor period = {args.gocube_arena_anchor_period}")
    print(f"  arena sims = {args.arenaMCTSSims}")
    print(f"  batched = {'ON' if args.gocube_arena_batched else 'OFF'}")
    print("  fast/noise/root-temp = OFF/OFF/OFF")
    print("  model gating = OFF")


def main(argv=None):
    cli = parse_args(argv)
    game_cls, args = build_katago_training_args(cli)
    if not cli.allow_existing_run:
        assert_fresh_run(args)
    print_katago_search_configuration(args)
    ensure_training_manifest(args.checkpoint, args.run_name, game_cls)
    network = SampleClockNNetWrapper(game_cls, args)
    network._gocube_checkpoint_arg_overrides = checkpoint_arg_overrides(cli, args)
    coach = KataGoSearchCoach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
