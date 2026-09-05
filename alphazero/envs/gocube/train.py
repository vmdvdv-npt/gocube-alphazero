import argparse
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
from alphazero.inference_batching import collect_ready_worker_ids, process_coalesced_inference
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.utils import get_iter_file


_TELEMETRY_COUNTER_KEYS = (
    "regular_decisions",
    "fast_decisions",
    "base_positions",
    "base_endgame_positions",
    "endgame_extra_samples",
)


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
    if endgame_weight < 1:
        raise ValueError("endgame weight must be at least 1")
    return int(base_positions) + int(base_endgame_positions) * (int(endgame_weight) - 1)


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
    """Coach variant for GoCube V3 batching, targets, and budget diagnostics."""

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
        if telemetry is None:
            telemetry = {key: mp.Value('q', 0) for key in _TELEMETRY_COUNTER_KEYS}
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
            "endgame_weight": int(self.args.gocube_endgame_sample_weight),
            "endgame_extra_samples": 0,
            "saved_total": 0,
        }
        return telemetry

    def _snapshot_selfplay_telemetry(self, iteration):
        snapshot = {
            key: int(counter.value)
            for key, counter in self.selfplay_telemetry.items()
        }
        regular = snapshot["regular_decisions"]
        fast = snapshot["fast_decisions"]
        total = regular + fast
        realized = fast / total if total else 0.0
        self._iteration_telemetry.update(snapshot)
        self._iteration_telemetry["total_decisions"] = total
        self._iteration_telemetry["realized_fast_fraction"] = realized
        self.writer.add_scalar("selfplay/regular_decisions", regular, iteration)
        self.writer.add_scalar("selfplay/fast_decisions", fast, iteration)
        self.writer.add_scalar("selfplay/total_decisions", total, iteration)
        self.writer.add_scalar("selfplay/realized_fast_fraction", realized, iteration)

    @_set_state(TrainState.INIT_AGENTS)
    def generateSelfPlayAgents(self):
        telemetry = self._reset_selfplay_telemetry()
        self._iteration_record_context = self._build_iteration_record_context(self.model_iter)
        self.stop_agents = mp.Event()
        self.ready_queue = mp.Queue()
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
            self.batch_ready.append(mp.Event())

            if self.args.cuda:
                self.input_tensors[i].pin_memory()
                self.policy_tensors[i].pin_memory()
                self.value_tensors[i].pin_memory()

            self.agents.append(
                SelfPlayAgent(
                    i, self.game_cls, self.ready_queue, self.batch_ready[i],
                    self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                    self.result_queue, self.completed, self.games_played, self.stop_agents, self.pause_train,
                    self.args, _is_warmup=self.warmup, telemetry=telemetry,
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

    @_set_state(TrainState.SAVE_SAMPLES)
    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        base_positions = int(self._iteration_telemetry["base_positions"])
        base_endgame = int(self._iteration_telemetry["base_endgame_positions"])
        extra_samples = int(self._iteration_telemetry["endgame_extra_samples"])
        endgame_weight = int(self.args.gocube_endgame_sample_weight)
        if not self.args.symmetricSamples:
            expected_extra = base_endgame * (endgame_weight - 1)
            if extra_samples != expected_extra:
                raise ValueError(
                    f"V3 endgame weighting mismatch: extra={extra_samples}, expected={expected_extra}"
                )
            expected_total = expected_saved_samples(base_positions, base_endgame, endgame_weight)
            if num_samples != expected_total:
                raise ValueError(
                    f"V3 saved sample accounting mismatch: queue={num_samples}, expected={expected_total}"
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
        self.writer.add_scalar("samples/base_positions", base_positions, iteration)
        self.writer.add_scalar("samples/base_endgame_positions", base_endgame, iteration)
        self.writer.add_scalar("samples/endgame_weight", endgame_weight, iteration)
        self.writer.add_scalar("samples/endgame_extra_samples", extra_samples, iteration)
        self.writer.add_scalar("samples/saved_total", num_samples, iteration)

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
            self.writer.add_scalar(
                f'win_rate/player{i}',
                (wins[i] + (0.5 * draws if self.args.use_draws_for_winrate else 0)) / denominator,
                iteration,
            )
        self.writer.add_scalar('win_rate/draws', draws / denominator, iteration)
        self.writer.add_scalar('win_rate/avg_game_length', length_sum / denominator, iteration)
        for key, value in counters.items():
            if key == "terminal/training_valid_fraction":
                value /= denominator
            self.writer.add_scalar(key, value, iteration)
        if record_entries:
            record_entries.sort(key=lambda entry: int(entry["game_number_inside_iteration"]))
            aggregate_metrics = {
                "games": int(num_games),
                "black_wins": int(wins[0]) if wins else 0,
                "white_wins": int(wins[1]) if len(wins) > 1 else 0,
                "draws": int(draws),
                "average_game_length": length_sum / denominator,
                **counters,
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
            self.writer.add_scalar(f"training/{key}", value, iteration)

    def _print_iteration_summary(self, iteration):
        sample = self._iteration_telemetry
        training = getattr(self, "_training_telemetry", {})
        print(f"=== V3 iteration {iteration} training summary ===")
        print(f"Games:                    {sample.get('games', 0)}")
        print(f"Regular decisions:        {sample.get('regular_decisions', 0)}")
        print(f"Fast decisions:           {sample.get('fast_decisions', 0)}")
        print(f"Fast fraction:            {sample.get('realized_fast_fraction', 0.0):.2%}")
        print()
        print(f"Base positions:           {sample.get('base_positions', 0)}")
        print(f"Base endgame:             {sample.get('base_endgame_positions', 0)}")
        print(f"Endgame weight:           {sample.get('endgame_weight', 1)}")
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

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        if self.args.train_on_past_data and self.args.past_data_run_name != self.args.run_name:
            raise ValueError("V3 refuses cross-run past-data loading; V2/V3 samples must never be mixed")
        num_train_steps = 0
        sample_counter = 0
        loaded_iteration_samples = {}

        def add_tensor_dataset(train_iter, tensor_dataset_list, run_name=self.args.run_name):
            nonlocal num_train_steps, sample_counter
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
        self.writer.add_scalar('loss/policy', self.loss_pi, iteration)
        self.writer.add_scalar('loss/value', self.loss_v, iteration)
        self.writer.add_scalar('loss/ownership', self.train_net.l_ownership, iteration)
        self.writer.add_scalar('loss/score', self.train_net.l_score, iteration)
        self.writer.add_scalar('loss/total', self.train_net.l_total, iteration)
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
    parser.add_argument("--endgame-sample-weight", type=int, default=1)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--no-arena", action="store_true")
    parser.add_argument("--model-gating", action="store_true")
    parser.add_argument("--smoke", action="store_true")
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
        gocube_endgame_sample_weight=cli.endgame_sample_weight,
        gocube_topology=game_cls.topology_kind(),
        gocube_size=game_cls.board_size(),
        gocube_rule_set=game_cls.RULESET,
        gocube_komi=float(game_cls.KOMI),
        gocube_terminal_adjudicator=game_cls.TERMINAL_ADJUDICATOR_ID,
        gocube_observation_schema=game_cls.OBSERVATION_SCHEMA,
        gocube_rules_fingerprint=game_cls.rules_fingerprint(),
        gocube_katago_rules_version=game_cls.KATAGO_RULES_VERSION,
        gocube_katago_reference_commit=game_cls.KATAGO_REFERENCE_COMMIT,
        # These are observability settings only.  They do not participate in
        # game rules, training targets, or optimizer behavior.
        gocube_recording_enabled=True,
        gocube_record_root=os.path.join("data", run_name, "records"),
        gocube_game_id_registry=os.path.join("data", run_name, "records", ".gocube-game-ids"),
        gocube_game_id_prefix=game_id_prefix(game_cls),
    )
    return game_cls, args


def print_training_configuration(args):
    arena_enabled = bool(args.compareWithBaseline or args.compareWithPast)
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
    print(f"  endgame weight = {args.gocube_endgame_sample_weight}")
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
