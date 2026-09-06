# cython: language_level=3
from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.search_contract import GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE, SearchOutput
from alphazero.utils import dotdict

from typing import Callable, List, Tuple, Optional
from enum import Enum
from queue import Empty

import torch.multiprocessing as mp
import numpy as np
import torch
import random
import time


class _PlayerStats:
    def __init__(self, index):
        self.index = index
        self.wins = 0
        self.winrate = 0

    def reset_wins(self):
        self.wins = 0
        self.winrate = 0

    def add_win(self):
        self.wins += 1

    def update(self, num_games, draws):
        if not num_games:
            self.winrate = 0
        else:
            self.winrate = (self.wins + 0.5 * draws) / num_games


class ArenaState(Enum):
    STANDBY = 0
    INIT = 1
    PLAY_GAMES = 2
    SINGLE_GAME = 3


def _set_state(state: ArenaState):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'state') or self.state == ArenaState.STANDBY:
                self.state = state
            ret = func(self, *args, **kwargs)
            self.state = ArenaState.STANDBY
            return ret
        return wrapper
    return decorator


class Arena:
    """An Arena class where any game's agents can be pitted against each other."""

    @_set_state(ArenaState.INIT)
    def __init__(
            self,
            players: List[BasePlayer],
            game_cls,
            use_batched_mcts=True,
            display: Callable[[GameState, Optional[int]], None] = None,
            args: dotdict = None
    ):
        num_players = game_cls.num_players()
        if len(players) != num_players:
            raise ValueError('Argument `players` must have the same amount of players as the game supports. '
                             f'Got {len(players)} player agents, while the game requires {num_players}')

        self.game_cls = game_cls
        self.display = display
        self.args = args.copy()
        if hasattr(self.args, 'get'):
            arena_sims = self.args.get('arenaMCTSSims', self.args.numMCTSSims)
            arena_temp = self.args.get('arenaTemp', 0.0)
        else:
            arena_sims = getattr(self.args, 'arenaMCTSSims', self.args.numMCTSSims)
            arena_temp = getattr(self.args, 'arenaTemp', 0.0)
        self.args.arenaMCTSSims = arena_sims
        self.args.numMCTSSims = arena_sims
        self.args.probFastSim = 0.0
        self.args.add_root_noise = False
        self.args.add_root_temp = False
        self.args.arenaTemp = arena_temp
        self.args.startTemp = arena_temp
        self.use_batched_mcts = use_batched_mcts
        self.__player_stats = None
        self.__players = None
        self.players = players
        self.games_played = 0
        self.total_games = 0
        self.eps_time = 0
        self.total_time = 0
        self.eta = 0
        self.game_state = None
        self.draws = 0
        self.no_results = 0
        self._agents = []
        self.stop_event = mp.Event()
        self.pause_event = mp.Event()

    @property
    def players(self) -> List[BasePlayer]:
        return self.__players

    @players.setter
    def players(self, value: List[BasePlayer]):
        self.__players = value
        for player in self.__players:
            player.args = self.args
            if hasattr(player, 'temp'):
                player.temp = self.args.arenaTemp
        self.__player_stats = [_PlayerStats(i) for i in range(len(self.players))]
        self.__check_players_valid()

    def __check_players_valid(self):
        if self.use_batched_mcts and not all(p.supports_process() for p in self.players):
            raise ValueError('Batched MCTS is not supported for players that do not support batch processing.')

    def __reset_stats(self):
        self.draws = 0
        self.no_results = 0
        [s.reset_wins() for s in self.__player_stats]

    def __update_winrates(self):
        num_games = sum([s.wins for s in self.__player_stats]) + (
            self.draws if self.args.use_draws_for_winrate else 0
        )
        [s.update(
            num_games, self.draws if self.args.use_draws_for_winrate else 0
        ) for s in self.__player_stats]

    def __collect_batched_results(self, result_queue):
        num_games = result_queue.qsize()
        wins = [0] * self.game_cls.num_players()
        draws = 0
        no_results = 0
        for _ in range(num_games):
            state, winstate, agent_id = result_queue.get()
            has_draw_slot = len(winstate) > self.game_cls.num_players()
            if has_draw_slot and winstate[-1]:
                if getattr(state, 'terminal_kind', None) == 'no_result':
                    no_results += 1
                else:
                    draws += 1
                continue
            for player, is_win in enumerate(winstate[:self.game_cls.num_players()]):
                if is_win:
                    index = self._agents[agent_id].player_to_index[player]
                    wins[index] += 1
        return wins, draws, no_results

    def wins(self) -> List[int]:
        return [s.wins for s in self.__player_stats]

    def winrates(self) -> List[float]:
        return [s.winrate for s in self.__player_stats]

    @_set_state(ArenaState.SINGLE_GAME)
    def play_game(self, verbose=False, _player_to_index: List[int] = None) -> Tuple[GameState, np.ndarray]:
        if verbose: assert self.display

        self.stop_event = mp.Event()
        self.pause_event = mp.Event()
        [p.reset() for p in self.players]
        self.game_state = self.game_cls()
        player_to_index = _player_to_index or list(range(self.game_state.num_players()))

        while not self.stop_event.is_set():
            while self.pause_event.is_set():
                time.sleep(.1)

            action = self.players[player_to_index[self.game_state.player]](self.game_state)
            if self.stop_event.is_set() or not isinstance(action, int):
                break

            if verbose:
                print(f'Turn {self.game_state.turns}, Player {self.game_state.player}')

            [p.update(self.game_state, action) for p in self.players]
            self.game_state.play_action(action)

            if verbose:
                self.display(self.game_state, action)

            winstate = self.game_state.win_state()
            if winstate.any():
                if verbose:
                    print(f'Game over: Turn {self.game_state.turns}, Result {winstate}')
                    self.display(self.game_state)
                return self.game_state, winstate

        return self.game_state, self.game_state.win_state()

    @_set_state(ArenaState.PLAY_GAMES)
    def play_games(self, num: int, verbose=False, shuffle_players=True) -> Tuple[List[int], int, List[float]]:
        self.total_games = num
        self.stop_event = mp.Event()
        self.pause_event = mp.Event()
        eps_time = AverageMeter()
        bar = Bar('Arena.play_games', max=num)
        end = time.time()
        self.__reset_stats()

        if self.use_batched_mcts:
            self.__check_players_valid()

            def empty_queue(q: mp.Queue):
                for _ in range(q.qsize()):
                    try:
                        q.get_nowait()
                    except Empty:
                        break

            self.args.gamesPerIteration = num
            self._agents = []
            policy_tensors = []
            value_tensors = []
            score_tensors = []
            ownership_tensors = []
            batch_ready = []
            batch_queues = []
            self.stop_event = mp.Event()
            self.pause_event = mp.Event()
            ready_queue = mp.Queue()
            result_queue = mp.Queue()
            completed = mp.Value('i', 0)
            games_played = mp.Value('i', 0)
            score_aware = (
                getattr(self.args, 'search_utility_mode', 'legacy')
                == GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE
            )
            point_count = self.game_cls.action_size() - 1

            for i in range(self.args.workers):
                input_tensors = [[] for _ in range(self.game_cls.num_players())]
                batch_queues.append(mp.Queue())

                policy_tensors.append(torch.zeros(
                    [self.args.arena_batch_size, self.game_cls.action_size()]
                ))
                policy_tensors[i].share_memory_()

                value_tensors.append(torch.zeros([self.args.arena_batch_size, self.game_cls.num_players() + 1]))
                value_tensors[i].share_memory_()

                if score_aware:
                    score_tensors.append(torch.zeros([self.args.arena_batch_size, 1]))
                    score_tensors[i].share_memory_()
                    ownership_tensors.append(torch.zeros([self.args.arena_batch_size, point_count, 3]))
                    ownership_tensors[i].share_memory_()
                else:
                    score_tensors.append(None)
                    ownership_tensors.append(None)

                batch_ready.append(mp.Event())
                if self.args.cuda:
                    policy_tensors[i].pin_memory()
                    value_tensors[i].pin_memory()
                    if score_aware:
                        score_tensors[i].pin_memory()
                        ownership_tensors[i].pin_memory()

                self._agents.append(
                    SelfPlayAgent(
                        i, self.game_cls, ready_queue, batch_ready[i],
                        input_tensors, policy_tensors[i], value_tensors[i], batch_queues[i],
                        result_queue, completed, games_played, self.stop_event, self.pause_event, self.args,
                        _is_arena=True,
                        score_tensor=score_tensors[i], ownership_tensor=ownership_tensors[i],
                    )
                )
                self._agents[i].daemon = True
                self._agents[i].start()

            sample_time = AverageMeter()
            end = time.time()
            n = 0
            while completed.value != self.args.workers:
                try:
                    id = ready_queue.get(timeout=1)
                    policy = []
                    value = []
                    score = []
                    ownership = []
                    data = batch_queues[id].get()
                    for player in range(len(self.players)):
                        batch = data[player]
                        if not isinstance(batch, list):
                            if score_aware:
                                output = self.players[player].process_for_search(batch)
                                if not isinstance(output, SearchOutput) or output.score is None or output.ownership is None:
                                    raise RuntimeError('score-aware Arena player must return all four search heads')
                                policy.append(output.policy.to(policy_tensors[id].device))
                                value.append(output.value.to(value_tensors[id].device))
                                score.append(output.score.to(score_tensors[id].device))
                                ownership.append(output.ownership.to(ownership_tensors[id].device))
                            else:
                                p, v = self.players[player].process(batch)
                                policy.append(p.to(policy_tensors[id].device))
                                value.append(v.to(value_tensors[id].device))

                    policy_tensors[id].copy_(torch.cat(policy))
                    value_tensors[id].copy_(torch.cat(value))
                    if score_aware:
                        score_tensors[id].copy_(torch.cat(score))
                        ownership_tensors[id].copy_(torch.cat(ownership))
                    batch_ready[id].set()
                except Empty:
                    pass

                size = games_played.value
                if size > n:
                    sample_time.update((time.time() - end) / (size - n), size - n)
                    n = size
                    end = time.time()

                wins, draws, no_results = self.__collect_batched_results(result_queue)
                for i, w in enumerate(wins):
                    self.__player_stats[i].wins += w
                self.draws += draws
                self.no_results += no_results
                self.__update_winrates()

                bar.suffix = '({eps}/{maxeps}) Winrates: {wr} | No-result: {nr} | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}' \
                    .format(
                        eps=size, maxeps=num, et=sample_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                        wr=[round(w, 3) for w in self.winrates()], nr=self.no_results
                    )
                bar.goto(size)

                self.games_played = size
                self.eps_time = sample_time.avg
                self.total_time = bar.elapsed_td
                self.eta = bar.eta_td

            self.stop_event.set()
            bar.update()
            bar.finish()

            empty_queue(ready_queue)
            empty_queue(result_queue)
            for q in batch_queues:
                empty_queue(q)

            for agent in self._agents:
                agent.join()
                del policy_tensors[0]
                del value_tensors[0]
                del score_tensors[0]
                del ownership_tensors[0]
                del batch_ready[0]

        else:
            players = list(range(self.game_cls.num_players()))

            def get_player_order():
                if not shuffle_players: return
                if len(players) == 2:
                    players.reverse()
                else:
                    random.shuffle(players)

            for eps in range(1, num + 1):
                if self.stop_event.is_set():
                    break

                get_player_order()
                final_state, winstate = self.play_game(verbose, players)
                if self.stop_event.is_set():
                    break

                for player, is_win in enumerate(winstate):
                    if is_win:
                        if player >= self.game_cls.num_players():
                            if getattr(final_state, 'terminal_kind', None) == 'no_result':
                                self.no_results += 1
                            else:
                                self.draws += 1
                        else:
                            self.__player_stats[players[player]].add_win()

                self.__update_winrates()
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix = '({eps}/{maxeps}) Winrates: {wr} | No-result: {nr} | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}' \
                    .format(
                        eps=eps, maxeps=num, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                        wr=[round(w, 3) for w in self.winrates()], nr=self.no_results
                    )
                bar.next()
                self.games_played = eps
                self.eps_time = eps_time.avg
                self.total_time = bar.elapsed_td
                self.eta = bar.eta_td

            bar.update()
            bar.finish()

        return self.wins(), self.draws, self.winrates()
