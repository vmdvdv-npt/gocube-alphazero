import argparse
import json
import math
import os
import pickle
from glob import glob
from math import ceil
from time import time

import pyximport
import torch
from torch import multiprocessing as mp
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

pyximport.install()

from alphazero.Coach import Coach, TrainState, _set_state, get_args
from alphazero.NNetWrapper import NNetWrapper
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.envs.gocube.game import game_class
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.records import (
    build_game_record,
    effective_parameter_snapshot,
    game_id_prefix,
    write_game_record,
    write_iteration_manifest,
)
from alphazero.envs.gocube.training_guard import evaluate_selfplay_guard, phase_fractions
from alphazero.inference_batching import collect_ready_worker_ids, process_coalesced_inference
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.search_contract import (
    GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE,
    GOCUBE_SEARCH_CONTRACT,
)
from alphazero.utils import get_iter_file


_INT_TELEMETRY_COUNTER_KEYS = (
    "regular_decisions",
    "fast_decisions",
    "base_positions",
    "base_endgame_positions",
    "endgame_extra_samples",
    "pass_main_black",
    "pass_main_white",
    "pass_cleanup1_black",
    "pass_cleanup1_white",
    "pass_cleanup2_black",
    "pass_cleanup2_white",
    "main_second_pass_count",
    "main_second_pass_turn_sum",
    "main_double_pass_within_2",
    "main_double_pass_within_4",
    "main_double_pass_within_8",
    "phase_main_decisions",
    "phase_cleanup1_decisions",
    "phase_cleanup2_decisions",
    "phase_main_placements",
    "phase_cleanup1_placements",
    "phase_cleanup2_placements",
    "phase_main_passes",
    "phase_cleanup1_passes",
    "phase_cleanup2_passes",
    "samples_main",
    "samples_main_after_one_pass",
    "samples_cleanup1",
    "samples_cleanup2",
    "weighted_samples_main",
    "weighted_samples_main_after_one_pass",
    "weighted_samples_cleanup1",
    "weighted_samples_cleanup2",
    "search_audited_positions",
    "search_score_dominated_pass",
)

_FLOAT_TELEMETRY_COUNTER_KEYS = (
    "search_pass_root_prior_sum",
    "search_pass_visit_fraction_sum",
    "search_pass_win_utility_sum",
    "search_pass_score_utility_sum",
    "search_pass_combined_utility_sum",
    "search_best_nonpass_score_gain_sum",
    "search_best_nonpass_win_delta_sum",
)

_TELEMETRY_COUNTER_KEYS = _INT_TELEMETRY_COUNTER_KEYS + _FLOAT_TELEMETRY_COUNTER_KEYS
_PHASE_BUCKETS = ("main", "main_after_one_pass", "cleanup1", "cleanup2")


def validate_tensor_row_counts(tensors, expected=None):
    if not tensors:
        raise ValueError("V3 tensor set must not be empty")
    counts = [int(tensor.size(0)) for tensor in tensors]
    if len(set(counts)) != 1:
        raise ValueError(f"V3 tensor row-count mismatch: {counts}")
    if expected is not None and counts[0] != int(expected):
        raise ValueError(f"V3 tensor row count {counts[0]} != expected {int(expected)}")
    return counts[0]


def expected_saved_samples(base_positions, base_endgame_positions, endgame_weight):
    """Legacy accounting helper retained for diagnostics/tests.

    Production V3 no longer uses a blanket endgame weight. New runs keep that
    compatibility value fixed at one and account phase-specific weights from
    the actual queued rows instead.
    """
    if endgame_weight < 1:
        raise ValueError("endgame weight must be at least 1")
    return int(base_positions) + int(base_endgame_positions) * (int(endgame_weight) - 1)


def expected_phase_saved_samples(telemetry):
    return sum(int(telemetry.get(f"weighted_samples_{bucket}", 0)) for bucket in _PHASE_BUCKETS)


def resolve_train_steps(*, dataset_size, sample_budget, batch_size, auto_train_steps,
                        fixed_steps=None, train_on_all=False):
    dataset_size = int(dataset_size)
    batch_size = int(batch_size)
    if dataset_size <= 0:
        return 0
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if train_on_all:
        return max(1, dataset_size // batch_size)
    if auto_train_steps:
        return max(1, int(sample_budget) // batch_size)
    if fixed_steps is None or int(fixed_steps) < 1:
        raise ValueError("fixed training steps must be at least 1")
    return int(fixed_steps)


class GoCubeCoach(Coach):
    """Coach variant for GoCube V3 search, batching, targets, and guards."""

    def __init__(self, game_cls, nnet, args):
        super().__init__(game_cls, nnet, args)
        self.score_tensors = []
        self.ownership_tensors = []
        self._selfplay_guard_result = None

    def _save_model(self, model, iteration):
        previous_self_play_iter = getattr(self, "self_play_iter", max(0, iteration - 1))
        super()._save_model(model, iteration)
        if hasattr(self, "args") and not self.args.model_gating:
            self._arena_previous_iter = previous_self_play_iter
            self.self_play_iter = iteration

    def compareToPast(self, model_iter):
        if self.args.model_gating:
            return super().compareToPast(model_iter)

        current_self_play_iter = self.self_play_iter
        previous_iter = getattr(self, "_arena_previous_iter", max(0, model_iter - 1))
        self.self_play_iter = previous_iter
        try:
            return super().compareToPast(model_iter)
        finally:
            self.self_play_iter = current_self_play_iter

    def _reset_selfplay_telemetry(self):
        telemetry = getattr(self, "selfplay_telemetry", None)
        if telemetry is None or set(telemetry) != set(_TELEMETRY_COUNTER_KEYS):
            telemetry = {
                key: mp.Value('d' if key in _FLOAT_TELEMETRY_COUNTER_KEYS else 'q', 0)
                for key in _TELEMETRY_COUNTER_KEYS
            }
            self.selfplay_telemetry = telemetry
        else:
            for counter in telemetry.values():
                with counter.get_lock():
                    counter.value = 0
        self._iteration_telemetry = {
            "games": 0,
            "regular_decisions": 0,
            "fast_decisions": 0,
            "total_decisions": 0,
            "realized_fast_fraction": 0.0,
            "base_positions": 0,
            "base_endgame_positions": 0,
            "endgame_weight": 1,
            "endgame_extra_samples": 0,
            "saved_total": 0,
            "main_after_pass_weight": int(self.args.gocube_main_after_pass_weight),
            "cleanup1_weight": int(self.args.gocube_cleanup1_weight),
            "cleanup2_weight": int(self.args.gocube_cleanup2_weight),
        }
        self._selfplay_guard_result = None
        return telemetry

    def _writer_scalar(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)

    def _snapshot_selfplay_telemetry(self, iteration):
        snapshot = {
            key: (float(counter.value) if key in _FLOAT_TELEMETRY_COUNTER_KEYS else int(counter.value))
            for key, counter in self.selfplay_telemetry.items()
        }
        regular = snapshot["regular_decisions"]
        fast = snapshot["fast_decisions"]
        total = regular + fast
        realized = fast / total if total else 0.0
        self._iteration_telemetry.update(snapshot)
        self._iteration_telemetry["games"] = int(self.games_played.value)
        self._iteration_telemetry["total_decisions"] = total
        self._iteration_telemetry["realized_fast_fraction"] = realized

        self._writer_scalar("selfplay/regular_decisions", regular, iteration)
        self._writer_scalar("selfplay/fast_decisions", fast, iteration)
        self._writer_scalar("selfplay/total_decisions", total, iteration)
        self._writer_scalar("selfplay/realized_fast_fraction", realized, iteration)

        for phase in ("main", "cleanup1", "cleanup2"):
            self._writer_scalar(f"pass/{phase}_black", snapshot[f"pass_{phase}_black"], iteration)
            self._writer_scalar(f"pass/{phase}_white", snapshot[f"pass_{phase}_white"], iteration)
            self._writer_scalar(f"phase/{phase}_decisions", snapshot[f"phase_{phase}_decisions"], iteration)
            self._writer_scalar(f"phase/{phase}_placements", snapshot[f"phase_{phase}_placements"], iteration)
            self._writer_scalar(f"phase/{phase}_passes", snapshot[f"phase_{phase}_passes"], iteration)

        fractions = phase_fractions(snapshot)
        for phase, fraction in fractions.items():
            self._iteration_telemetry[f"phase_{phase}_fraction"] = fraction
            self._writer_scalar(f"phase/{phase}_fraction", fraction, iteration)

        second_count = snapshot["main_second_pass_count"]
        second_turn = snapshot["main_second_pass_turn_sum"] / second_count if second_count else 0.0
        games = max(1, int(self.games_played.value))
        early_rate = snapshot["main_double_pass_within_2"] / games
        self._iteration_telemetry["main_second_pass_turn"] = second_turn
        self._iteration_telemetry["main_early_double_pass_rate"] = early_rate
        self._writer_scalar("main/second_pass_turn", second_turn, iteration)
        self._writer_scalar("main/double_pass_within_2", snapshot["main_double_pass_within_2"], iteration)
        self._writer_scalar("main/double_pass_within_4", snapshot["main_double_pass_within_4"], iteration)
        self._writer_scalar("main/double_pass_within_8", snapshot["main_double_pass_within_8"], iteration)
        self._writer_scalar("main/early_double_pass_rate", early_rate, iteration)

        audited = snapshot["search_audited_positions"]
        search_mean_keys = {
            "pass_root_prior": "search_pass_root_prior_sum",
            "pass_visit_fraction": "search_pass_visit_fraction_sum",
            "pass_win_utility": "search_pass_win_utility_sum",
            "pass_score_utility": "search_pass_score_utility_sum",
            "pass_combined_utility": "search_pass_combined_utility_sum",
            "best_nonpass_score_gain": "search_best_nonpass_score_gain_sum",
            "best_nonpass_win_delta": "search_best_nonpass_win_delta_sum",
        }
        for public_key, sum_key in search_mean_keys.items():
            mean = snapshot[sum_key] / audited if audited else 0.0
            self._iteration_telemetry[f"search_{public_key}"] = mean
            self._writer_scalar(f"search/{public_key}", mean, iteration)
        self._writer_scalar("search/score_dominated_pass", snapshot["search_score_dominated_pass"], iteration)
        self._writer_scalar("search/audited_positions", audited, iteration)

    @_set_state(TrainState.INIT_AGENTS)
    def generateSelfPlayAgents(self):
        telemetry = self._reset_selfplay_telemetry()
        self._iteration_record_context = self._build_iteration_record_context(self.model_iter)
        self.stop_agents = mp.Event()
        self.ready_queue = mp.Queue()
        self.score_tensors = []
        self.ownership_tensors = []
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

            self.score_tensors.append(torch.zeros([self.args.process_batch_size, 1]))
            self.score_tensors[i].share_memory_()
            self.ownership_tensors.append(torch.zeros([self.args.process_batch_size, point_count, 3]))
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
                    i, self.game_cls, self.ready_queue, self.batch_ready[i],
                    self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                    self.result_queue, self.completed, self.games_played, self.stop_agents, self.pause_train,
                    self.args, _is_warmup=self.warmup, telemetry=telemetry,
                    score_tensor=self.score_tensors[i], ownership_tensor=self.ownership_tensors[i],
                )
            )
            self.agents[i].daemon = True
            self.agents[i].start()

    def _build_iteration_record_context(self, iteration):
        self_play_iteration = int(getattr(self, "self_play_iter", max(0, iteration - 1)))
        checkpoint_path = os.path.join(
            self.args.checkpoint,
            self.args.run_name,
            get_iter_file(self_play_iteration),
        )
        return {
            "run_name": self.args.run_name,
            "iteration": int(iteration),
            "checkpoint": {
                "id": f"{self.args.run_name}@{self_play_iteration}",
                "iteration": self_play_iteration,
                "path": os.path.relpath(os.path.abspath(checkpoint_path), os.getcwd()),
                "model_role": "self_play",
            },
            "parameters": effective_parameter_snapshot(self.args),
        }

    @_set_state(TrainState.SELF_PLAY)
    def processSelfPlayBatches(self, iteration):
        sample_time = AverageMeter()
        inference_batch_size = AverageMeter()
        inference_calls = 0
        positions_evaluated = 0
        bar = Bar("Generating Samples", max=self.args.gamesPerIteration)
        end = time()
        n = 0
        while self.completed.value != self.args.workers:
            if self.stop_train.is_set() and not self.stop_agents.is_set():
                self.stop_agents.set()
            worker_ids = collect_ready_worker_ids(
                self.ready_queue, self.args.workers, self.args.inference_batch_wait_ms,
            )
            if worker_ids:
                nnet = self.self_play_net if self.args.model_gating else self.train_net
                rows = process_coalesced_inference(
                    nnet, worker_ids, self.input_tensors, self.policy_tensors,
                    self.value_tensors, self.batch_ready,
                    score_tensors=self.score_tensors,
                    ownership_tensors=self.ownership_tensors,
                )
                inference_batch_size.update(rows)
                inference_calls += 1
                positions_evaluated += rows
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
        self._writer_scalar("performance/sample_time", sample_time.avg, iteration)
        self._writer_scalar("performance/inference_batch_size", inference_batch_size.avg, iteration)
        self._writer_scalar("performance/gpu_inference_calls", inference_calls, iteration)
        self._writer_scalar("performance/positions_per_inference_call",
                            positions_evaluated / inference_calls if inference_calls else 0.0, iteration)
        self._iteration_telemetry["performance_inference_batch_size"] = inference_batch_size.avg
        self._iteration_telemetry["performance_gpu_inference_calls"] = inference_calls
        self._iteration_telemetry["performance_positions_evaluated"] = positions_evaluated
        self._snapshot_selfplay_telemetry(iteration)
        print()

    @_set_state(TrainState.SAVE_SAMPLES)
    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        expected_total = expected_phase_saved_samples(self._iteration_telemetry)
        if not self.args.symmetricSamples and num_samples != expected_total:
            raise ValueError(
                f"V3 phase sample accounting mismatch: queue={num_samples}, expected={expected_total}"
            )
        print(f"Saving {num_samples} KataGo Japanese V3 scored samples")
        data_tensor = torch.zeros([num_samples, *self.game_cls.observation_size()])
        policy_tensor = torch.zeros([num_samples, self.game_cls.action_size()])
        value_tensor = torch.zeros([num_samples, self.game_cls.num_players() + 1])
        score_tensor = torch.zeros([num_samples, 1])
        ownership_tensor = torch.zeros([num_samples, self.game_cls.logical_topology().point_count, 3])
        ownership_mask_tensor = torch.zeros([num_samples, self.game_cls.logical_topology().point_count])
        for i in range(num_samples):
            sample = self.file_queue.get()
            if len(sample) != 6:
                raise ValueError(f"V3 training sample must contain 6 tensors, got {len(sample)}")
            data, policy, value, score, ownership, ownership_mask = sample
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.from_numpy(policy)
            value_tensor[i] = torch.from_numpy(value)
            score_tensor[i] = torch.from_numpy(score)
            ownership_tensor[i] = torch.from_numpy(ownership)
            ownership_mask_tensor[i] = torch.from_numpy(ownership_mask)
        tensors = (
            data_tensor, policy_tensor, value_tensor, score_tensor,
            ownership_tensor, ownership_mask_tensor,
        )
        validate_tensor_row_counts(tensors, expected=num_samples)
        folder = os.path.join(self.args.data, self.args.run_name)
        filename = os.path.join(folder, get_iter_file(iteration).replace('.pkl', ''))
        os.makedirs(folder, exist_ok=True)
        torch.save(data_tensor, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(policy_tensor, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(value_tensor, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(score_tensor, filename + '-score.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(ownership_tensor, filename + '-ownership.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(ownership_mask_tensor, filename + '-ownership-mask.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        self._iteration_telemetry["saved_total"] = num_samples

        base_total = sum(int(self._iteration_telemetry.get(f"samples_{bucket}", 0)) for bucket in _PHASE_BUCKETS)
        for bucket in _PHASE_BUCKETS:
            base = int(self._iteration_telemetry.get(f"samples_{bucket}", 0))
            weighted = int(self._iteration_telemetry.get(f"weighted_samples_{bucket}", 0))
            self._writer_scalar(f"samples/{bucket}", base, iteration)
            self._writer_scalar(f"weighted_samples/{bucket}", weighted, iteration)
            self._writer_scalar(f"samples/{bucket}_fraction", base / base_total if base_total else 0.0, iteration)
            self._writer_scalar(f"weighted_samples/{bucket}_fraction", weighted / num_samples if num_samples else 0.0, iteration)
        self._writer_scalar("samples/saved_total", num_samples, iteration)

    @_set_state(TrainState.PROCESS_RESULTS)
    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize()
        self._iteration_telemetry["games"] = num_games
        wins = [0] * self.game_cls.num_players()
        draws = 0
        length_sum = 0
        counters = {
            "terminal/scored_games": 0,
            "terminal/no_result_games": 0,
            "terminal/pass_alive_early_end": 0,
            "terminal/entered_cleanup1": 0,
            "terminal/entered_cleanup2": 0,
            "terminal/cleanup1_moves": 0,
            "terminal/cleanup2_moves": 0,
            "terminal/cleanup_captures": 0,
            "terminal/ko_unblock_actions": 0,
            "terminal/cycle_no_result": 0,
            "terminal/training_valid_fraction": 0.0,
        }
        record_entries = []
        context = getattr(self, "_iteration_record_context", self._build_iteration_record_context(iteration))
        record_dir = os.path.join(
            self.args.data,
            self.args.run_name,
            "records",
            f"iteration-{int(iteration):04d}",
        )
        for fallback_game_number in range(1, num_games + 1):
            result = self.result_queue.get()
            if len(result) == 4:
                state, winstate, _agent_id, record_payload = result
            else:
                state, winstate, _agent_id = result
                record_payload = None
            length_sum += state.turns
            if winstate[-1]:
                draws += 1
            else:
                for player in range(self.game_cls.num_players()):
                    wins[player] += int(bool(winstate[player]))
            if hasattr(state, "diagnostic_counters"):
                for key, value in state.diagnostic_counters().items():
                    counters[key] += value
            if record_payload is not None:
                game_id = record_payload["game_id"]
                target_path = os.path.join(record_dir, f"{game_id}.json")
                record = build_game_record(
                    game=state,
                    game_id=game_id,
                    run_name=context["run_name"],
                    iteration=iteration,
                    game_number=int(record_payload.get("game_number_inside_iteration", fallback_game_number)),
                    checkpoint=context["checkpoint"],
                    parameters=context["parameters"],
                    moves=record_payload["moves"],
                    start_time=record_payload["start_time"],
                    end_time=record_payload["end_time"],
                    winstate=winstate,
                    record_path=os.path.relpath(os.path.abspath(target_path), os.getcwd()),
                )
                entry = write_game_record(record_dir, record)
                entry["game_number_inside_iteration"] = record["game_number_inside_iteration"]
                record_entries.append(entry)

        denominator = max(1, num_games)
        for i in range(len(wins)):
            self._writer_scalar(
                f'win_rate/player{i}',
                (wins[i] + (0.5 * draws if self.args.use_draws_for_winrate else 0)) / denominator,
                iteration,
            )
        self._writer_scalar('win_rate/draws', draws / denominator, iteration)
        self._writer_scalar('win_rate/avg_game_length', length_sum / denominator, iteration)
        for key, value in counters.items():
            if key == "terminal/training_valid_fraction":
                value /= denominator
            self._writer_scalar(key, value, iteration)

        guard = evaluate_selfplay_guard(
            self._iteration_telemetry,
            games=num_games,
            args=self.args,
        )
        self._selfplay_guard_result = guard
        self._iteration_telemetry["selfplay_status"] = guard.status
        self._iteration_telemetry["guard_warnings"] = list(guard.warnings)
        self._iteration_telemetry["guard_fatal_reasons"] = list(guard.fatal_reasons)
        self._writer_scalar("selfplay/training_allowed", 1 if guard.training_allowed else 0, iteration)
        for key, value in guard.metrics.items():
            self._writer_scalar(f"guard/{key}", value, iteration)

        record_entries.sort(key=lambda entry: int(entry["game_number_inside_iteration"]))
        aggregate_metrics = {
            "games": int(num_games),
            "black_wins": int(wins[0]) if wins else 0,
            "white_wins": int(wins[1]) if len(wins) > 1 else 0,
            "draws": int(draws),
            "average_game_length": length_sum / denominator,
            **counters,
            "selfplay_status": guard.status,
            "training_allowed": guard.training_allowed,
            "guard_warnings": list(guard.warnings),
            "abort_reasons": list(guard.fatal_reasons),
            "runtime_guard": guard.metrics,
            "selfplay_telemetry": self._iteration_telemetry,
        }
        aggregate_metrics["terminal/training_valid_fraction"] = (
            counters["terminal/training_valid_fraction"] / denominator
        )
        manifest_path = write_iteration_manifest(
            record_dir,
            run_name=context["run_name"],
            iteration=iteration,
            checkpoint=context["checkpoint"],
            parameters=context["parameters"],
            records=record_entries,
            aggregate_metrics=aggregate_metrics,
        )
        self._iteration_record_manifest_path = manifest_path

    def killSelfPlayAgents(self):
        result = super().killSelfPlayAgents()
        self.score_tensors = []
        self.ownership_tensors = []
        return result

    def _record_training_metrics(self, iteration, *, history_iterations, window_samples,
                                 latest_iteration_samples, planned_steps):
        actual_steps = int(getattr(self.train_net, "last_train_actual_steps", 0))
        examples_seen = int(getattr(self.train_net, "last_train_examples_seen", 0))
        learning_rate = float(
            getattr(
                self.train_net,
                "last_train_learning_rate",
                self.train_net.optimizer.param_groups[0]["lr"],
            )
        )
        effective_passes = examples_seen / window_samples if window_samples else 0.0
        metrics = {
            "history_iterations": int(history_iterations),
            "window_samples": int(window_samples),
            "latest_iteration_samples": int(latest_iteration_samples),
            "batch_size": int(self.args.train_batch_size),
            "planned_optimizer_steps": int(planned_steps),
            "actual_optimizer_steps": actual_steps,
            "examples_seen": examples_seen,
            "effective_sample_passes": effective_passes,
            "learning_rate": learning_rate,
        }
        self._training_telemetry = metrics
        for key, value in metrics.items():
            self._writer_scalar(f"training/{key}", value, iteration)

    def _print_iteration_summary(self, iteration):
        sample = self._iteration_telemetry
        training = getattr(self, "_training_telemetry", {})
        print(f"=== V3 iteration {iteration} training summary ===")
        print(f"Status:                   {sample.get('selfplay_status', 'valid')}")
        print(f"Games:                    {sample.get('games', 0)}")
        print(f"Regular decisions:        {sample.get('regular_decisions', 0)}")
        print(f"Fast decisions:           {sample.get('fast_decisions', 0)}")
        print(f"Fast fraction:            {sample.get('realized_fast_fraction', 0.0):.2%}")
        print()
        print(f"Base positions:           {sample.get('base_positions', 0)}")
        print(f"Main samples:             {sample.get('samples_main', 0)}")
        print(f"Main-after-pass samples:  {sample.get('samples_main_after_one_pass', 0)}")
        print(f"Cleanup1 samples:         {sample.get('samples_cleanup1', 0)}")
        print(f"Cleanup2 samples:         {sample.get('samples_cleanup2', 0)}")
        print(f"Extra weighted samples:   {sample.get('endgame_extra_samples', 0)}")
        print(f"Saved samples:            {sample.get('saved_total', 0)}")
        print()
        print(f"History datasets:         {training.get('history_iterations', 0)}")
        print(f"Window samples:           {training.get('window_samples', 0)}")
        print(f"Latest iteration samples: {training.get('latest_iteration_samples', 0)}")
        print(f"Batch size:               {training.get('batch_size', self.args.train_batch_size)}")
        print()
        print(f"Optimizer steps planned:  {training.get('planned_optimizer_steps', 0)}")
        print(f"Optimizer steps actual:   {training.get('actual_optimizer_steps', 0)}")
        print(f"Examples seen:            {training.get('examples_seen', 0)}")
        print(f"Effective passes:         {training.get('effective_sample_passes', 0.0):.3f}")
        print(f"LR:                       {training.get('learning_rate', self.args.lr):g}")

    def _iteration_is_marked_invalid(self, run_name, train_iter):
        path = os.path.join(
            self.args.data,
            run_name,
            "records",
            f"iteration-{int(train_iter):04d}",
            "iteration-manifest.json",
        )
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            return False
        status = manifest.get("aggregate_metrics", {}).get("selfplay_status")
        return status == "invalid_selfplay"

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        guard = getattr(self, "_selfplay_guard_result", None)
        if guard is not None and not guard.training_allowed:
            print("Self-play validation failed. Samples and diagnostics were preserved; optimizer training is forbidden.")
            for reason in guard.fatal_reasons:
                print(f"  FATAL: {reason}")
            self.train_net.last_train_planned_steps = 0
            self.train_net.last_train_actual_steps = 0
            self.train_net.last_train_examples_seen = 0
            self._training_telemetry = {
                "history_iterations": 0,
                "window_samples": 0,
                "latest_iteration_samples": 0,
                "batch_size": int(self.args.train_batch_size),
                "planned_optimizer_steps": 0,
                "actual_optimizer_steps": 0,
                "examples_seen": 0,
                "effective_sample_passes": 0.0,
                "learning_rate": float(self.train_net.optimizer.param_groups[0]["lr"]),
            }
            self._print_iteration_summary(iteration)
            self.stop_train.set()
            return self.loss_pi, self.loss_v

        if self.args.train_on_past_data and self.args.past_data_run_name != self.args.run_name:
            raise ValueError("V3 refuses cross-run past-data loading; V2/V3 samples must never be mixed")
        num_train_steps = 0
        sample_counter = 0
        loaded_iteration_samples = {}

        def add_tensor_dataset(train_iter, tensor_dataset_list, run_name=self.args.run_name):
            nonlocal num_train_steps, sample_counter
            if self._iteration_is_marked_invalid(run_name, train_iter):
                print(f"Skipping invalid_selfplay iteration {train_iter}; it is diagnostic-only data.")
                return
            filename = os.path.join(self.args.data, run_name, get_iter_file(train_iter).replace('.pkl', ''))
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
                return
            row_count = validate_tensor_row_counts(tensors)
            if tensors[0].shape[1:] != self.game_cls.observation_size():
                raise ValueError("V3 dataset observation schema/shape mismatch")
            tensor_dataset_list.append(TensorDataset(*tensors))
            loaded_iteration_samples[train_iter] = row_count
            if self.args.averageTrainSteps:
                num_train_steps += row_count
                sample_counter += 1
            else:
                num_train_steps = row_count

        def train_data(tensor_dataset_list, train_on_all=False):
            window_samples = sum(len(dataset) for dataset in tensor_dataset_list)
            history_iterations = len(tensor_dataset_list)
            latest_iteration_samples = loaded_iteration_samples.get(iteration, 0)
            if not tensor_dataset_list or window_samples == 0:
                print('No valid scored V3 samples in this window; skipping optimizer step.')
                self.train_net.last_train_planned_steps = 0
                self.train_net.last_train_actual_steps = 0
                self.train_net.last_train_examples_seen = 0
                self.train_net.last_train_learning_rate = float(self.train_net.optimizer.param_groups[0]["lr"])
                self._record_training_metrics(
                    iteration,
                    history_iterations=history_iterations,
                    window_samples=window_samples,
                    latest_iteration_samples=latest_iteration_samples,
                    planned_steps=0,
                )
                return self.train_net.l_pi, self.train_net.l_v
            dataset = ConcatDataset(tensor_dataset_list)
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                num_workers=self.args.workers,
                pin_memory=True,
            )
            sample_budget = num_train_steps
            if self.args.averageTrainSteps and sample_counter:
                sample_budget //= sample_counter
            train_steps = resolve_train_steps(
                dataset_size=len(dataset),
                sample_budget=sample_budget,
                batch_size=self.args.train_batch_size,
                auto_train_steps=self.args.autoTrainSteps,
                fixed_steps=self.args.train_steps_per_iteration,
                train_on_all=train_on_all,
            )
            result = self.train_net.train(dataloader, train_steps)
            self._record_training_metrics(
                iteration,
                history_iterations=history_iterations,
                window_samples=len(dataset),
                latest_iteration_samples=latest_iteration_samples,
                planned_steps=train_steps,
            )
            del dataloader
            del dataset
            return result

        if self.args.train_on_past_data and iteration == self.args.startIter:
            next_start_iter = 1
            total_iters = len(glob(os.path.join(self.args.data, self.args.past_data_run_name, '*-data.pkl')))
            num_chunks = ceil(total_iters / self.args.past_data_chunk_size)
            for _ in range(num_chunks):
                datasets = []
                i = next_start_iter
                for i in range(next_start_iter, min(next_start_iter + self.args.past_data_chunk_size, total_iters + 1)):
                    add_tensor_dataset(i, datasets, run_name=self.args.past_data_run_name)
                next_start_iter = i + 1
                self.loss_pi, self.loss_v = train_data(datasets, train_on_all=True)
        else:
            datasets = []
            current_history_size = min(
                max(self.args.minTrainHistoryWindow,
                    (iteration + self.args.minTrainHistoryWindow) // self.args.trainHistoryIncrementIters),
                self.args.maxTrainHistoryWindow,
            )
            for i in range(max(1, iteration - current_history_size), iteration + 1):
                add_tensor_dataset(i, datasets)
            self.loss_pi, self.loss_v = train_data(datasets)
        self._writer_scalar('loss/policy', self.loss_pi, iteration)
        self._writer_scalar('loss/value', self.loss_v, iteration)
        self._writer_scalar('loss/ownership', self.train_net.l_ownership, iteration)
        self._writer_scalar('loss/score', self.train_net.l_score, iteration)
        self._writer_scalar('loss/total', self.train_net.l_total, iteration)
        self._print_iteration_summary(iteration)
        self._save_model(self.train_net, iteration)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaZero on a GoCube topology")
    parser.add_argument("--topology", choices=("torus", "cube"), default="torus")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--arena-sims", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--train-steps-per-iteration", type=int, default=None)
    parser.add_argument("--fast-game-prob", type=float, default=0.25)
    parser.add_argument("--endgame-sample-weight", type=int, default=1,
                        help="deprecated V3 blanket weight; only value 1 is accepted")
    parser.add_argument("--main-after-pass-weight", type=int, default=1)
    parser.add_argument("--cleanup1-weight", type=int, default=1)
    parser.add_argument("--cleanup2-weight", type=int, default=1)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--win-loss-utility-factor", type=float, default=1.0)
    parser.add_argument("--static-score-utility-factor", type=float, default=0.0)
    parser.add_argument("--dynamic-score-utility-factor", type=float, default=0.40)
    parser.add_argument("--dynamic-score-center-zero-weight", type=float, default=0.25)
    parser.add_argument("--dynamic-score-center-scale", type=float, default=0.50)
    parser.add_argument("--root-ending-bonus-points", type=float, default=0.50)
    parser.add_argument("--score-improvement-threshold", type=float, default=1.0)
    parser.add_argument("--win-probability-tolerance", type=float, default=0.005)
    parser.add_argument("--search-audit-probability", type=float, default=0.25)
    parser.add_argument("--guard-min-games", type=int, default=32)
    parser.add_argument("--early-double-pass-warning-rate", type=float, default=0.01)
    parser.add_argument("--early-double-pass-fatal-rate", type=float, default=0.05)
    parser.add_argument("--cleanup2-warning-fraction", type=float, default=0.50)
    parser.add_argument("--cleanup2-fatal-fraction", type=float, default=0.70)
    parser.add_argument("--score-dominated-pass-fatal-rate", type=float, default=0.25)
    parser.add_argument("--score-audit-min-positions", type=int, default=16)
    parser.add_argument("--no-fill-dame-before-pass", action="store_true")
    parser.add_argument("--no-conservative-pass", action="store_true")
    parser.add_argument("--no-arena", action="store_true")
    parser.add_argument("--model-gating", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _cli(cli, name, default):
    return getattr(cli, name, default)


def _rate(value, name):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def build_training_args(cli):
    if cli.workers < 1:
        raise ValueError("workers must be at least 1")
    if cli.games_per_iteration < 1:
        raise ValueError("games-per-iteration must be at least 1")
    if cli.iterations < 1:
        raise ValueError("iterations must be at least 1")
    if cli.train_batch_size < 1:
        raise ValueError("train-batch-size must be at least 1")
    if cli.train_steps_per_iteration is not None and cli.train_steps_per_iteration < 1:
        raise ValueError("train-steps-per-iteration must be at least 1")
    if cli.sims < 1:
        raise ValueError("sims must be at least 1")
    if cli.arena_sims < 1:
        raise ValueError("arena-sims must be at least 1")
    if not 0.0 <= cli.fast_game_prob <= 1.0:
        raise ValueError("fast-game-prob must be between 0 and 1")
    if cli.inference_batch_wait_ms < 0:
        raise ValueError("inference-batch-wait-ms must be non-negative")
    if cli.endgame_sample_weight < 1:
        raise ValueError("endgame-sample-weight must be at least 1")
    if cli.endgame_sample_weight != 1:
        raise ValueError("V3 blanket endgame oversampling is disabled; endgame-sample-weight must be 1")

    phase_weights = {
        "main_after_pass": int(_cli(cli, "main_after_pass_weight", 1)),
        "cleanup1": int(_cli(cli, "cleanup1_weight", 1)),
        "cleanup2": int(_cli(cli, "cleanup2_weight", 1)),
    }
    if any(weight < 1 for weight in phase_weights.values()):
        raise ValueError("phase sample weights must be at least 1")

    dynamic_center_zero = _rate(_cli(cli, "dynamic_score_center_zero_weight", 0.25),
                                "dynamic-score-center-zero-weight")
    win_tolerance = _rate(_cli(cli, "win_probability_tolerance", 0.005),
                          "win-probability-tolerance")
    audit_probability = _rate(_cli(cli, "search_audit_probability", 0.25),
                              "search-audit-probability")
    early_warning = _rate(_cli(cli, "early_double_pass_warning_rate", 0.01),
                          "early-double-pass-warning-rate")
    early_fatal = _rate(_cli(cli, "early_double_pass_fatal_rate", 0.05),
                        "early-double-pass-fatal-rate")
    cleanup_warning = _rate(_cli(cli, "cleanup2_warning_fraction", 0.50),
                            "cleanup2-warning-fraction")
    cleanup_fatal = _rate(_cli(cli, "cleanup2_fatal_fraction", 0.70),
                          "cleanup2-fatal-fraction")
    dominated_fatal = _rate(_cli(cli, "score_dominated_pass_fatal_rate", 0.25),
                            "score-dominated-pass-fatal-rate")
    if early_warning > early_fatal:
        raise ValueError("early double-pass warning threshold cannot exceed fatal threshold")
    if cleanup_warning > cleanup_fatal:
        raise ValueError("cleanup2 warning threshold cannot exceed fatal threshold")
    if _cli(cli, "guard_min_games", 32) < 1:
        raise ValueError("guard-min-games must be at least 1")
    if _cli(cli, "score_audit_min_positions", 16) < 1:
        raise ValueError("score-audit-min-positions must be at least 1")
    if _cli(cli, "dynamic_score_center_scale", 0.50) <= 0:
        raise ValueError("dynamic-score-center-scale must be positive")
    if _cli(cli, "root_ending_bonus_points", 0.50) < 0:
        raise ValueError("root-ending-bonus-points must be non-negative")
    if _cli(cli, "score_improvement_threshold", 1.0) < 0:
        raise ValueError("score-improvement-threshold must be non-negative")

    game_cls = game_class(cli.topology, cli.size, "japanese")
    run_name = cli.run_name or f"gocube-{cli.topology}-{cli.size}-japanese75-katago-v3-pilot"
    process_batch_size = max(1, math.ceil(cli.games_per_iteration / cli.workers))
    iterations = 1 if cli.smoke else cli.iterations
    arena_enabled = not (cli.smoke or cli.no_arena)
    model_gating = bool(cli.model_gating)
    if model_gating and not arena_enabled:
        raise ValueError("model gating requires arena evaluation")
    if cli.smoke:
        auto_train_steps = False
        train_steps_per_iteration = 1
    elif cli.train_steps_per_iteration is None:
        auto_train_steps = True
        train_steps_per_iteration = None
    else:
        auto_train_steps = False
        train_steps_per_iteration = cli.train_steps_per_iteration

    args = get_args(
        run_name=run_name,
        workers=cli.workers,
        gamesPerIteration=cli.games_per_iteration,
        numIters=iterations,
        numMCTSSims=cli.sims,
        arenaMCTSSims=cli.arena_sims,
        arenaTemp=0.0,
        arenaBatched=False,
        process_batch_size=process_batch_size,
        train_batch_size=cli.train_batch_size,
        inference_batch_wait_ms=cli.inference_batch_wait_ms,
        compareWithBaseline=arena_enabled,
        compareWithPast=arena_enabled,
        model_gating=model_gating,
        autoTrainSteps=auto_train_steps,
        train_steps_per_iteration=train_steps_per_iteration,
        probFastSim=0.0 if cli.smoke else cli.fast_game_prob,
        nnet_type="graph",
        symmetricSamples=False,
        num_channels=64,
        depth=6,
        value_dense_layers=[128, 64],
        policy_dense_layers=[128],
        score_dense_layers=[64],
        gocube_auxiliary_targets=True,
        ownership_loss_weight=0.5,
        score_loss_weight=0.5,
        gocube_endgame_sample_weight=1,
        gocube_main_after_pass_weight=phase_weights["main_after_pass"],
        gocube_cleanup1_weight=phase_weights["cleanup1"],
        gocube_cleanup2_weight=phase_weights["cleanup2"],
        search_utility_mode=GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE,
        gocube_search_contract=GOCUBE_SEARCH_CONTRACT,
        gocube_win_loss_utility_factor=float(_cli(cli, "win_loss_utility_factor", 1.0)),
        gocube_static_score_utility_factor=float(_cli(cli, "static_score_utility_factor", 0.0)),
        gocube_dynamic_score_utility_factor=float(_cli(cli, "dynamic_score_utility_factor", 0.40)),
        gocube_dynamic_score_center_zero_weight=dynamic_center_zero,
        gocube_dynamic_score_center_scale=float(_cli(cli, "dynamic_score_center_scale", 0.50)),
        gocube_root_ending_bonus_points=float(_cli(cli, "root_ending_bonus_points", 0.50)),
        gocube_fill_dame_before_pass=not bool(_cli(cli, "no_fill_dame_before_pass", False)),
        gocube_conservative_pass=not bool(_cli(cli, "no_conservative_pass", False)),
        gocube_score_improvement_threshold_points=float(_cli(cli, "score_improvement_threshold", 1.0)),
        gocube_win_probability_tolerance=win_tolerance,
        gocube_search_audit_probability=audit_probability,
        gocube_guard_min_games=int(_cli(cli, "guard_min_games", 32)),
        gocube_early_double_pass_warning_rate=early_warning,
        gocube_early_double_pass_fatal_rate=early_fatal,
        gocube_cleanup2_warning_fraction=cleanup_warning,
        gocube_cleanup2_fatal_fraction=cleanup_fatal,
        gocube_score_dominated_pass_fatal_rate=dominated_fatal,
        gocube_score_audit_min_positions=int(_cli(cli, "score_audit_min_positions", 16)),
        gocube_topology=game_cls.topology_kind(),
        gocube_size=game_cls.board_size(),
        gocube_rule_set=game_cls.RULESET,
        gocube_komi=float(game_cls.KOMI),
        gocube_terminal_adjudicator=game_cls.TERMINAL_ADJUDICATOR_ID,
        gocube_observation_schema=game_cls.OBSERVATION_SCHEMA,
        gocube_rules_fingerprint=game_cls.rules_fingerprint(),
        gocube_katago_rules_version=game_cls.KATAGO_RULES_VERSION,
        gocube_katago_reference_commit=game_cls.KATAGO_REFERENCE_COMMIT,
        gocube_recording_enabled=True,
        gocube_record_root=os.path.join("data", run_name, "records"),
        gocube_game_id_registry=os.path.join("data", ".gocube-game-ids"),
        gocube_game_id_prefix=game_id_prefix(game_cls),
    )
    return game_cls, args


def print_training_configuration(args):
    arena_enabled = bool(args.compareWithBaseline or args.compareWithPast)
    print("Search:")
    print(f"  contract = {args.gocube_search_contract}")
    print(f"  utility mode = {args.search_utility_mode}")
    print(f"  utility factors = win {args.gocube_win_loss_utility_factor:g}, "
          f"static score {args.gocube_static_score_utility_factor:g}, "
          f"dynamic score {args.gocube_dynamic_score_utility_factor:g}")
    print(f"  root ending bonus = {args.gocube_root_ending_bonus_points:g} points")
    print(f"  second-pass threshold = {args.gocube_score_improvement_threshold_points:g} point, "
          f"win tolerance {args.gocube_win_probability_tolerance:.2%}")
    print("Self-play:")
    print(f"  regular sims = {args.numMCTSSims}")
    print(f"  fast sims = {args.numFastSims}")
    print(f"  fast probability = {args.probFastSim:.0%}")
    print("Training:")
    print(f"  games/iteration = {args.gamesPerIteration}")
    print(f"  batch size = {args.train_batch_size}")
    if args.autoTrainSteps:
        print("  step mode = auto")
    else:
        print(f"  step mode = fixed ({args.train_steps_per_iteration})")
    print("  endgame weight = 1 (blanket V3 oversampling disabled)")
    print(f"  phase weights = main-after-pass {args.gocube_main_after_pass_weight}, "
          f"cleanup1 {args.gocube_cleanup1_weight}, cleanup2 {args.gocube_cleanup2_weight}")
    if arena_enabled:
        print(
            f"Arena: fixed {args.arenaMCTSSims} sims, fast OFF, root noise OFF, "
            f"root temp OFF, action temp {args.arenaTemp:g}, games {args.arenaCompare}, "
            "color-balanced scheduling"
        )
    else:
        print("Arena: OFF")
    if args.model_gating:
        print(f"Model gating: ON @ threshold {args.min_next_model_winrate:.3f}")
    else:
        print("Model gating: OFF")


def main():
    cli = parse_args()
    game_cls, args = build_training_args(cli)
    print_training_configuration(args)
    ensure_training_manifest(args.checkpoint, args.run_name, game_cls)
    network = NNetWrapper(game_cls, args)
    coach = GoCubeCoach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
