from pyximport import install as pyxinstall
from numpy import get_include
pyxinstall(setup_args={'include_dirs': get_include()})

from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.utils import get_iter_file, dotdict, get_game_results, default_temp_scaling
from alphazero.Arena import Arena
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.worker_errors import SelfPlayWorkerError, unexpected_worker_exit_payload

from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tensorboardX import SummaryWriter
from glob import glob
from queue import Empty
from time import time
from math import ceil
from enum import Enum

import copy
import numpy as np
import torch
import pickle
import os

DEFAULT_ARGS = dotdict({
    'run_name': 'boardgame',
    'cuda': torch.cuda.is_available(),
    'workers': mp.cpu_count(),
    'startIter': 0,
    'numIters': 1000,
    'process_batch_size': 256,
    'train_batch_size': 1024,
    'arena_batch_size': 64,
    'train_steps_per_iteration': 64,
    'train_sample_ratio': 1,
    'averageTrainSteps': False,
    'autoTrainSteps': True,   # Calculates the average number of samples in the training window
                              # if averageTrainSteps set to True, otherwise uses the latest
                              # number of samples, and does train_sample_ratio * avg_num_steps 
                              # or last_num_train_steps // train_batch_size
                              # training steps.
    'train_on_past_data': False,
    'past_data_chunk_size': 25,
    'past_data_run_name': 'boardgame',
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 256 * mp.cpu_count(),
    'minTrainHistoryWindow': 4,
    'maxTrainHistoryWindow': 20,
    'trainHistoryIncrementIters': 2,
    '_num_players': None,  # Doesn't have to be changed, set automatically by the env.
    'min_discount': 1,
    'fpu_reduction': 0.2,
    'num_stacked_observations': 8,  # TODO: built-in stacked observations (arg does nothing right now)
    'numWarmupIters': 1,  # Iterations where games are played randomly, 0 for none
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'symmetricSamples': True,
    'numMCTSSims': 100,
    'numFastSims': 20,
    'numWarmupSims': 5,
    'probFastSim': 0.75,
    'mctsResetThreshold': None,
    'startTemp': 1,
    'temp_scaling_fn': default_temp_scaling,
    'root_policy_temp': 1.1,
    'root_noise_frac': 0.1,
    'add_root_noise': True,
    'add_root_temp': True,
    'compareWithBaseline': True,
    'baselineTester': RawMCTSPlayer,
    'arenaCompareBaseline': 128,
    'arenaCompare': 128,
    'arenaTemp': 0.25,
    'arenaMCTS': True,
    'arenaBatched': True,
    'baselineCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'model_gating': True,
    'max_gating_iters': None,
    'min_next_model_winrate': 0.52,
    'use_draws_for_winrate': True,
    'load_model': True,
    'cpuct': 1.25,
    'value_loss_weight': 1.5,
    'checkpoint': 'checkpoint',
    'data': 'data',

    'scheduler': torch.optim.lr_scheduler.MultiStepLR,
    'scheduler_args': dotdict({
        'milestones': [75, 125],
        'gamma': 0.1

        # 'min_lr': 1e-4,
        # 'patience': 3,
        # 'cooldown': 1,
        # 'verbose': False
    }),

    'lr': 1e-2,
    'optimizer': torch.optim.SGD,
    'optimizer_args': dotdict({
        'momentum': 0.9,
        'weight_decay': 1e-4
    }),

    'nnet_type': 'resnet',  # 'resnet' or 'fc'
    'num_channels': 32,
    'depth': 4,
    'value_head_channels': 16,
    'policy_head_channels': 16,

    # fc only uses the following
    'input_fc_layers': [1024] * 4,  # only for fc networks
    'value_dense_layers': [512, 64],
    'policy_dense_layers': [512, 256]
})


def get_args(args=None, **kwargs):
    # DEFAULT_ARGS contains nested mutable values (scheduler/optimizer
    # settings and architecture lists).  Returning the module-level object
    # made sequential config resolution order-dependent: a later B1 build
    # could silently mutate the already-resolved B0 config.
    new_args = dotdict(copy.deepcopy(dict(DEFAULT_ARGS)))
    if args:
        new_args.update(args)
    for key, value in kwargs.items():
        setattr(new_args, key, value)
    return new_args


class TrainState(Enum):
    STANDBY = 0
    INIT = 1
    INIT_AGENTS = 2
    SELF_PLAY = 3
    SAVE_SAMPLES = 4
    PROCESS_RESULTS = 5
    KILL_AGENTS = 6
    TRAIN = 7
    COMPARE_BASELINE = 8
    COMPARE_PAST = 9


def _set_state(state: TrainState):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.state = state
            ret = func(self, *args, **kwargs)
            self.state = TrainState.STANDBY
            return ret
        return wrapper
    return decorator


class Coach:
    @_set_state(TrainState.INIT)
    def __init__(self, game_cls, nnet, args):
        # ``dotdict.__getattr__`` raises KeyError for a missing optional key,
        # so use mapping lookup for the argument objects produced by get_args.
        master_seed = args.get("master_seed") if isinstance(args, dict) else getattr(args, "master_seed", None)
        if master_seed is not None:
            from alphazero.envs.gocube.reproducibility import seed_process
            seed_process(int(master_seed))
        self.game_cls = game_cls
        self.train_net = nnet
        self.self_play_net = nnet.__class__(game_cls, args)
        self.args = args
        self.args._num_players = self.game_cls.num_players() + self.game_cls.has_draw()
        
        train_iter = self.args.startIter

        if self.args.load_model:
            networks = sorted(glob(self.args.checkpoint + '/' + self.args.run_name + '/iteration-*.pkl'))
            self.args.startIter = len(networks)
            if self.args.startIter == 0:
                self._save_model(self.train_net, 0)
                self.args.startIter = 1

            train_iter = self.args.startIter - 1
            self._load_model(self.train_net, train_iter)

        if self.args.selfPlayModelIter == 0:
            self.self_play_iter = 0
        else:
            self.self_play_iter = self.args.selfPlayModelIter or train_iter

        if self.args.model_gating:
            self._load_model(self.self_play_net, self.self_play_iter)
        
        self.gating_counter = 0
        self.warmup = False
        self.loss_pi = 0
        self.loss_v = 0
        self.sample_time = 0
        self.iter_time = 0
        self.eta = 0
        self.arena = None
        self.model_iter = self.args.startIter
        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.last_worker_error = None
        self.stop_train = mp.Event()
        self.pause_train = mp.Event()
        self.stop_agents = mp.Event()
        self.train_net.stop_train = self.stop_train
        self.train_net.pause_train = self.pause_train
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker_error_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/' + self.args.run_name)
        else:
            self.writer = SummaryWriter()
        # self.args.expertValueWeight.current = self.args.expertValueWeight.start
    
    def _load_model(self, model, iteration):
        model.load_checkpoint(
            folder=os.path.join(self.args.checkpoint, self.args.run_name),
            filename=get_iter_file(iteration)
        )
    
    def _save_model(self, model, iteration):
        model.save_checkpoint(
            folder=os.path.join(self.args.checkpoint, self.args.run_name),
            filename=get_iter_file(iteration)
        )

    def learn(self):
        print('Because of batching, it can take a long time before any games finish.')

        try:

            while self.model_iter <= self.args.numIters:
                if self._training_budget_reached():
                    print('Scientific training budget already reached; stopping before another generation chunk.')
                    break
                print(f'------ITER {self.model_iter}------')

                if (
                    (not self.args.skipSelfPlayIters
                        or self.model_iter > self.args.skipSelfPlayIters)
                    and not (self.args.train_on_past_data and self.model_iter == self.args.startIter)
                ):
                    if self.model_iter <= self.args.numWarmupIters:
                        print('Warmup: random policy and value')
                        self.warmup = True
                    elif self.self_play_iter == 0:
                        self.warmup = True
                    elif self.warmup:
                        self.warmup = False

                    self.generateSelfPlayAgents()
                    self.processSelfPlayBatches(self.model_iter)
                    if self.stop_train.is_set():
                        break
                    self.saveIterationSamples(self.model_iter)
                    if self.stop_train.is_set():
                        break
                    self.processGameResults(self.model_iter)
                    if self.stop_train.is_set():
                        break
                    self.killSelfPlayAgents()
                    if self.stop_train.is_set():
                        break

                self.train(self.model_iter)
                if self.stop_train.is_set():
                    break

                if self._training_budget_reached():
                    print('Scientific training budget reached; stopping after this generation chunk.')
                    break

                if self.args.compareWithBaseline and (self.model_iter - 1) % self.args.baselineCompareFreq == 0:
                    self.compareToBaseline(self.model_iter)
                    if self.stop_train.is_set():
                        break

                if self.args.compareWithPast and (self.model_iter - 1) % self.args.pastCompareFreq == 0:
                    self.compareToPast(self.model_iter)
                    if self.stop_train.is_set():
                        break

                # z = self.args.expertValueWeight
                # self.args.expertValueWeight.current = min(
                #     self.model_iter, z.iterations) / z.iterations * (z.end - z.start) + z.start

                self.writer.add_scalar('win_rate/self_play_model', self.self_play_iter, self.model_iter)
                self.model_iter += 1
                print()

        except KeyboardInterrupt:
            pass
        finally:
            print()
            self.writer.close()
            if self.agents:
                # Preserve the original exception while ensuring a failed
                # training iteration never leaves child processes behind.
                self._abort_selfplay_agents()

    def _training_budget_reached(self):
        """Hook for production coaches with a sample-based stopping target."""

        return False

    def _drain_worker_error_queue(self):
        payloads = []
        while True:
            try:
                payload = self.worker_error_queue.get_nowait()
            except Empty:
                break
            if isinstance(payload, dict):
                payloads.append(payload)
            else:
                payloads.append({
                    "worker_id": -1,
                    "pid": None,
                    "iteration": int(getattr(self, "model_iter", 0)),
                    "game_slot": None,
                    "game_id": None,
                    "stage": "shutdown",
                    "exception_type": "InvalidWorkerErrorPayload",
                    "exception_message": repr(payload),
                    "traceback": "",
                })
        return payloads

    def _check_selfplay_workers(self, iteration):
        """Raise as soon as a child reports or suffers an abnormal exit."""

        payloads = self._drain_worker_error_queue()
        if payloads:
            self.last_worker_error = payloads[0]
            raise SelfPlayWorkerError(payloads[0])

        for agent in self.agents:
            exitcode = agent.exitcode
            if exitcode is not None and exitcode != 0:
                payload = unexpected_worker_exit_payload(
                    worker_id=int(agent.id),
                    iteration=int(iteration),
                    exitcode=int(exitcode),
                    pid=agent.pid,
                )
                self.last_worker_error = payload
                raise SelfPlayWorkerError(payload)

    @staticmethod
    def _annotate_parent_exception(exc, iteration, worker_ids):
        message = (
            f"parent inference failure: iteration={iteration}, "
            f"worker_ids={list(worker_ids)}"
        )
        add_note = getattr(exc, "add_note", None)
        if add_note is not None:
            add_note(message)
        else:
            # Keep the original exception type on Python versions without
            # BaseException.add_note while still exposing the context.
            exc.args = tuple(getattr(exc, "args", (str(exc),))) + (message,)

    @staticmethod
    def _close_ipc_queue(queue):
        if queue is None:
            return
        try:
            queue.cancel_join_thread()
        except (AttributeError, OSError):
            pass
        try:
            queue.close()
        except (AttributeError, OSError):
            pass

    def _reset_selfplay_ipc(self):
        for name in (
            "ready_queue", "file_queue", "result_queue", "worker_error_queue",
        ):
            self._close_ipc_queue(getattr(self, name, None))
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker_error_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)

    def _shutdown_selfplay_agents(self):
        """Bounded process shutdown shared by normal and emergency paths."""

        agents = list(self.agents)
        for agent in agents:
            try:
                agent.join(timeout=0.5)
            except (AssertionError, OSError):
                pass

        alive = []
        for agent in agents:
            try:
                if agent.is_alive():
                    alive.append(agent)
                    agent.terminate()
            except (AssertionError, OSError):
                pass
        for agent in alive:
            try:
                agent.join(timeout=0.5)
            except (AssertionError, OSError):
                pass

        still_alive = []
        for agent in alive:
            try:
                if agent.is_alive():
                    still_alive.append(agent)
                    kill = getattr(agent, "kill", None)
                    if kill is not None:
                        kill()
            except (AssertionError, OSError):
                pass
        for agent in still_alive:
            try:
                agent.join(timeout=0.5)
            except (AssertionError, OSError):
                pass

        exitcodes = [(int(agent.id), agent.exitcode) for agent in agents]
        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        if hasattr(self, "score_tensors"):
            self.score_tensors = []
        if hasattr(self, "ownership_tensors"):
            self.ownership_tensors = []
        self._reset_selfplay_ipc()
        return exitcodes

    def _abort_selfplay_agents(self):
        """Abort every self-play child in a bounded, fixed order."""

        if not self.agents:
            return
        self.stop_agents.set()
        self.pause_train.clear()
        for event in self.batch_ready:
            event.set()
        # stop_agents prevents further ready-queue/inference issuance. The
        # bounded shutdown below then handles children in every lifecycle state.
        self._shutdown_selfplay_agents()

    @_set_state(TrainState.INIT_AGENTS)
    def generateSelfPlayAgents(self):
        self.stop_agents = mp.Event()
        self.ready_queue = mp.Queue()
        self._close_ipc_queue(getattr(self, "worker_error_queue", None))
        self.worker_error_queue = mp.Queue()
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
                SelfPlayAgent(i, self.game_cls, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.result_queue, self.completed, self.games_played, self.stop_agents, self.pause_train,
                              self.args, _is_warmup=self.warmup,
                              worker_error_queue=self.worker_error_queue,
                              iteration=int(self.model_iter))
            )
            self.agents[i].daemon = True
            self.agents[i].start()

    @_set_state(TrainState.SELF_PLAY)
    def processSelfPlayBatches(self, iteration):
        sample_time = AverageMeter()
        bar = Bar('Generating Samples', max=self.args.gamesPerIteration)
        end = time()

        n = 0
        try:
            while self.completed.value != self.args.workers:
                self._check_selfplay_workers(iteration)
                if self.stop_train.is_set():
                    self.stop_agents.set()
                    break

                try:
                    id = self.ready_queue.get(timeout=1)
                    self._check_selfplay_workers(iteration)
                    nnet = self.self_play_net if self.args.model_gating else self.train_net
                    try:
                        policy, value = nnet.process(self.input_tensors[id])
                    except Exception as exc:
                        self._annotate_parent_exception(exc, iteration, [id])
                        raise
                    self.policy_tensors[id].copy_(policy)
                    self.value_tensors[id].copy_(value)
                    self.batch_ready[id].set()
                except Empty:
                    pass

                size = self.games_played.value
                if size > n:
                    sample_time.update((time() - end) / (size - n), size - n)
                    n = size
                    end = time()
                bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
                bar.goto(size)
                self.sample_time = sample_time.avg
                self.iter_time = bar.elapsed_td
                self.eta = bar.eta_td
            self._check_selfplay_workers(iteration)
        except BaseException:
            self._abort_selfplay_agents()
            raise

        if not self.stop_agents.is_set(): self.stop_agents.set()
        bar.update()
        bar.finish()
        self.writer.add_scalar('loss/sample_time', sample_time.avg, iteration)
        print()

    @_set_state(TrainState.SAVE_SAMPLES)
    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')

        data_tensor = torch.zeros([num_samples, *self.game_cls.observation_size()])
        policy_tensor = torch.zeros([num_samples, self.game_cls.action_size()])
        value_tensor = torch.zeros([num_samples, self.game_cls.num_players() + 1])
        for i in range(num_samples):
            data, policy, value = self.file_queue.get()
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.from_numpy(policy)
            value_tensor[i] = torch.from_numpy(value)

        folder = os.path.join(self.args.data, self.args.run_name)
        filename = os.path.join(folder, get_iter_file(iteration).replace('.pkl', ''))
        if not os.path.exists(folder): os.makedirs(folder)

        torch.save(data_tensor, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(policy_tensor, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(value_tensor, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        del data_tensor
        del policy_tensor
        del value_tensor

    @_set_state(TrainState.PROCESS_RESULTS)
    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize()
        wins, draws, avg_game_length = get_game_results(self.result_queue, self.game_cls)

        for i in range(len(wins)):
            self.writer.add_scalar(f'win_rate/player{i}', (
                    wins[i] + (0.5 * draws if self.args.use_draws_for_winrate else 0)
            ) / num_games, iteration)
        self.writer.add_scalar('win_rate/draws', draws / num_games, iteration)
        self.writer.add_scalar('win_rate/avg_game_length', avg_game_length, iteration)

    @_set_state(TrainState.KILL_AGENTS)
    def killSelfPlayAgents(self):
        if not self.agents:
            return
        payloads = self._drain_worker_error_queue()
        self.stop_agents.set()
        self.pause_train.clear()
        for event in self.batch_ready:
            event.set()
        agent_pids = {int(agent.id): agent.pid for agent in self.agents}
        exitcodes = self._shutdown_selfplay_agents()
        if payloads:
            self.last_worker_error = payloads[0]
            raise SelfPlayWorkerError(payloads[0])
        abnormal = [(worker_id, code) for worker_id, code in exitcodes if code not in (0, None)]
        if abnormal:
            worker_id, exitcode = abnormal[0]
            payload = unexpected_worker_exit_payload(
                worker_id=worker_id,
                iteration=int(getattr(self, "model_iter", 0)),
                exitcode=int(exitcode),
                pid=agent_pids.get(worker_id),
            )
            self.last_worker_error = payload
            raise SelfPlayWorkerError(payload)

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        num_train_steps = 0
        sample_counter = 0
        
        def add_tensor_dataset(train_iter, tensor_dataset_list, run_name=self.args.run_name):
            filename = os.path.join(
                os.path.join(self.args.data, run_name), get_iter_file(train_iter).replace('.pkl', '')
            )
            
            try:
                data_tensor = torch.load(filename + '-data.pkl')
                policy_tensor = torch.load(filename + '-policy.pkl')
                value_tensor = torch.load(filename + '-value.pkl')
            except FileNotFoundError as e:
                print('Warning: could not find tensor data. ' + str(e))
                return
            
            tensor_dataset_list.append(
                TensorDataset(data_tensor, policy_tensor, value_tensor)
            )
            nonlocal num_train_steps
            if self.args.averageTrainSteps:
                nonlocal sample_counter
                num_train_steps += data_tensor.size(0)
                sample_counter += 1
            else:
                num_train_steps = data_tensor.size(0)

        def train_data(tensor_dataset_list, train_on_all=False):
            dataset = ConcatDataset(tensor_dataset_list)
            dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                    num_workers=self.args.workers, pin_memory=True)
            
            if self.args.averageTrainSteps:
                nonlocal num_train_steps
                num_train_steps //= sample_counter

            train_steps = len(dataset) // self.args.train_batch_size \
               if train_on_all else (num_train_steps // self.args.train_batch_size
                   if self.args.autoTrainSteps else self.args.train_steps_per_iteration)

            result = self.train_net.train(dataloader, train_steps)

            del dataloader
            del dataset

            return result

        if self.args.train_on_past_data and iteration == self.args.startIter:
            next_start_iter = 1
            total_iters = len(
                glob(os.path.join(os.path.join(self.args.data, self.args.past_data_run_name), '*.pkl'))
            ) // 3
            num_chunks = ceil(total_iters / self.args.past_data_chunk_size)
            print(f'Training on past data from run "{self.args.past_data_run_name}" in {num_chunks} chunks of '
                  f'{self.args.past_data_chunk_size} iterations ({total_iters} iterations in total).')

            for _ in range(num_chunks):
                datasets = []
                i = next_start_iter
                for i in range(next_start_iter, min(
                    next_start_iter + self.args.past_data_chunk_size, total_iters + 1
                )):
                    add_tensor_dataset(i, datasets, run_name=self.args.past_data_run_name)
                next_start_iter = i + 1

                self.loss_pi, self.loss_v = train_data(datasets, train_on_all=True)
                del datasets
        else:
            datasets = []

            # current_history_size = self.args.numItersForTrainExamplesHistory
            current_history_size = min(
                max(
                    self.args.minTrainHistoryWindow,
                    (iteration + self.args.minTrainHistoryWindow) // self.args.trainHistoryIncrementIters
                ),
                self.args.maxTrainHistoryWindow
            )

            [add_tensor_dataset(i, datasets) for i in range(max(1, iteration - current_history_size), iteration + 1)]
            self.loss_pi, self.loss_v = train_data(datasets)

        self.writer.add_scalar('loss/policy', self.loss_pi, iteration)  # TODO: policy loss not showing up in tensorboard
        self.writer.add_scalar('loss/value', self.loss_v, iteration)
        self.writer.add_scalar('loss/total', self.loss_pi + self.loss_v, iteration)

        self._save_model(self.train_net, iteration)

    @_set_state(TrainState.COMPARE_PAST)
    def compareToPast(self, model_iter):
        self._load_model(self.self_play_net, self.self_play_iter)

        print(f'PITTING AGAINST ITERATION {self.self_play_iter}')
        # if self.args.arenaBatched:
        #     if not self.args.arenaMCTS:
        #         self.args.arenaMCTS = True
        #         print('WARNING: Batched arena comparison is enabled which uses MCTS, but arena MCTS is set to False.'
        #                           ' Ignoring this, and continuing with batched MCTS in arena.')

        #     nplayer = self.train_net.process
        #     pplayer = self.self_play_net.process
        # else:
        #     cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        #     nplayer = cls(self.game_cls, self.args, self.train_net)
        #     pplayer = cls(self.game_cls, self.args, self.self_play_net)
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        nplayer = cls(self.train_net, self.game_cls, self.args)
        pplayer = cls(self.self_play_net, self.game_cls, self.args)

        players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
        self.arena = Arena(players, self.game_cls, use_batched_mcts=self.args.arenaBatched, args=self.args)
        wins, draws, winrates = self.arena.play_games(self.args.arenaCompare)
        if self.stop_train.is_set(): return
        winrate = winrates[0]

        print(f'NEW/PAST WINS : {wins[0]} / {sum(wins[1:])} ; DRAWS : {draws}\n')
        print(f'NEW MODEL WINRATE : {round(winrate, 3)}')
        self.writer.add_scalar('win_rate/past', winrate, model_iter)

        ### Model gating ###
        if (
            self.args.model_gating
            and winrate < self.args.min_next_model_winrate
            and (self.args.max_gating_iters is None
                 or self.gating_counter < self.args.max_gating_iters)
        ):
            self.gating_counter += 1
        elif self.args.model_gating:
            self.self_play_iter = model_iter
            self._load_model(self.self_play_net, self.self_play_iter)
            self.gating_counter = 0

        if self.args.model_gating:
            print(f'Using model version {self.self_play_iter} for self play.')

    @_set_state(TrainState.COMPARE_BASELINE)
    def compareToBaseline(self, iteration):
        test_player = self.args.baselineTester(self.game_cls, self.args)
        can_process = test_player.supports_process() and self.args.arenaBatched
        nnplayer = (MCTSPlayer if self.args.arenaMCTS else NNPlayer)(self.train_net, self.game_cls, self.args)

        print('PITTING AGAINST BASELINE: ' + self.args.baselineTester.__name__)

        players = [nnplayer] + [test_player] * (self.game_cls.num_players() - 1)
        self.arena = Arena(players, self.game_cls, use_batched_mcts=can_process, args=self.args)
        wins, draws, winrates = self.arena.play_games(self.args.arenaCompare)
        if self.stop_train.is_set(): return
        winrate = winrates[0]

        print(f'NEW/BASELINE WINS : {wins[0]} / {sum(wins[1:])} ; DRAWS : {draws}\n')
        print(f'NEW MODEL WINRATE : {round(winrate, 3)}')
        self.writer.add_scalar('win_rate/baseline', winrate, iteration)
