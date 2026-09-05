import argparse
import math
import os
import pickle
from glob import glob
from math import ceil
from time import time

import pyximport
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

pyximport.install()

from alphazero.Coach import Coach, TrainState, _set_state, get_args
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import game_class
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.inference_batching import collect_ready_worker_ids, process_coalesced_inference
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.utils import get_iter_file


class GoCubeCoach(Coach):
    """Coach variant for GoCube V3 batching, targets, and terminal diagnostics."""

    def _save_model(self, model, iteration):
        super()._save_model(model, iteration)
        if hasattr(self, "args") and not self.args.model_gating:
            self.self_play_iter = iteration

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
        self.writer.add_scalar("loss/sample_time", sample_time.avg, iteration)
        self.writer.add_scalar("performance/inference_batch_size", inference_batch_size.avg, iteration)
        print()

    @_set_state(TrainState.SAVE_SAMPLES)
    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
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
        folder = os.path.join(self.args.data, self.args.run_name)
        filename = os.path.join(folder, get_iter_file(iteration).replace('.pkl', ''))
        os.makedirs(folder, exist_ok=True)
        torch.save(data_tensor, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(policy_tensor, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(value_tensor, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(score_tensor, filename + '-score.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(ownership_tensor, filename + '-ownership.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(ownership_mask_tensor, filename + '-ownership-mask.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)

    @_set_state(TrainState.PROCESS_RESULTS)
    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize()
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
        for _ in range(num_games):
            state, winstate, _agent_id = self.result_queue.get()
            length_sum += state.turns
            if winstate[-1]:
                draws += 1
            else:
                for player in range(self.game_cls.num_players()):
                    wins[player] += int(bool(winstate[player]))
            if hasattr(state, "diagnostic_counters"):
                for key, value in state.diagnostic_counters().items():
                    counters[key] += value
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

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        if self.args.train_on_past_data and self.args.past_data_run_name != self.args.run_name:
            raise ValueError("V3 refuses cross-run past-data loading; V2/V3 samples must never be mixed")
        num_train_steps = 0
        sample_counter = 0

        def add_tensor_dataset(train_iter, tensor_dataset_list, run_name=self.args.run_name):
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
            if tensors[0].shape[1:] != self.game_cls.observation_size():
                raise ValueError("V3 dataset observation schema/shape mismatch")
            tensor_dataset_list.append(TensorDataset(*tensors))
            nonlocal num_train_steps
            if self.args.averageTrainSteps:
                nonlocal sample_counter
                num_train_steps += tensors[0].size(0)
                sample_counter += 1
            else:
                num_train_steps = tensors[0].size(0)

        def train_data(tensor_dataset_list, train_on_all=False):
            if not tensor_dataset_list or sum(len(dataset) for dataset in tensor_dataset_list) == 0:
                print('No valid scored V3 samples in this window; skipping optimizer step.')
                return self.train_net.l_pi, self.train_net.l_v
            dataset = ConcatDataset(tensor_dataset_list)
            dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                    num_workers=self.args.workers, pin_memory=True)
            if self.args.averageTrainSteps and sample_counter:
                nonlocal num_train_steps
                num_train_steps //= sample_counter
            train_steps = len(dataset) // self.args.train_batch_size if train_on_all else (
                num_train_steps // self.args.train_batch_size
                if self.args.autoTrainSteps else self.args.train_steps_per_iteration
            )
            result = self.train_net.train(dataloader, train_steps)
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
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--fast-game-prob", type=float, default=0.75)
    parser.add_argument("--endgame-sample-weight", type=int, default=3)
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
    run_name = cli.run_name or f"gocube-{cli.topology}-{cli.size}-japanese75-katago-v3"
    process_batch_size = max(1, math.ceil(cli.games_per_iteration / cli.workers))
    iterations = 1 if cli.smoke else cli.iterations
    arena_enabled = not (cli.smoke or cli.no_arena)
    model_gating = bool(cli.model_gating)
    if model_gating and not arena_enabled:
        raise ValueError("model gating requires arena evaluation")
    args = get_args(
        run_name=run_name,
        workers=cli.workers,
        gamesPerIteration=cli.games_per_iteration,
        numIters=iterations,
        numMCTSSims=cli.sims,
        arenaMCTSSims=cli.arena_sims,
        arenaTemp=0.0,
        process_batch_size=process_batch_size,
        train_batch_size=cli.train_batch_size,
        inference_batch_wait_ms=cli.inference_batch_wait_ms,
        compareWithBaseline=arena_enabled,
        compareWithPast=arena_enabled,
        model_gating=model_gating,
        autoTrainSteps=not cli.smoke,
        train_steps_per_iteration=1 if cli.smoke else 64,
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
    )
    return game_cls, args


def print_training_configuration(args):
    arena_enabled = bool(args.compareWithBaseline or args.compareWithPast)
    print(
        f"Self-play: {args.numMCTSSims} sims, fast {args.numFastSims} @ "
        f"{args.probFastSim:.0%}"
    )
    if arena_enabled:
        print(
            f"Arena: fixed {args.arenaMCTSSims} sims, fast OFF, root noise OFF, "
            f"root temp OFF, action temp {args.arenaTemp:g}, games {args.arenaCompare}"
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
