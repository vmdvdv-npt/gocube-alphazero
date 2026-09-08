# cython: language_level=3
from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.search_contract import KATAGO_PINNED_SEARCH_UTILITY_MODE, SearchOutput
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
    """
    An Arena class where any game's agents can be pitted against each other.
    """

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
        self._player_color_results = []
        self._agents = []
        self.stop_event = mp.Event()
        self.pause_event = mp.Event()
        self.__reset_color_results()

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
        score_aware = (
            self.args.get('search_utility_mode', 'legacy') == KATAGO_PINNED_SEARCH_UTILITY_MODE
            if hasattr(self.args, 'get')
            else getattr(self.args, 'search_utility_mode', 'legacy') == KATAGO_PINNED_SEARCH_UTILITY_MODE
        )
        if self.use_batched_mcts and score_aware:
            if not all(hasattr(p, 'nn') and hasattr(p.nn, 'process_for_search') for p in self.players):
                raise ValueError('KataGo-derived batched Arena requires four-head process_for_search().')

    def __reset_color_results(self):
        self._player_color_results = [
            {
                'black': {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'no_results': 0},
                'white': {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'no_results': 0},
            }
            for _ in range(len(self.players))
        ]

    def __reset_stats(self):
        self.draws = 0
        self.no_results = 0
        [s.reset_wins() for s in self.__player_stats]
        self.__reset_color_results()

    def __update_winrates(self):
        num_games = sum([s.wins for s in self.__player_stats]) + (
            self.draws if self.args.use_draws_for_winrate else 0
        )
        [s.update(
            num_games, self.draws if self.args.use_draws_for_winrate else 0
        ) for s in self.__player_stats]

    def __record_batched_color_result(self, state, winstate, agent_id):
        if self.game_cls.num_players() != 2:
            return
        player_to_index = list(self._agents[agent_id].player_to_index)
        if len(player_to_index) < 2:
            return
        color_by_index = {
            int(player_to_index[0]): 'black',
            int(player_to_index[1]): 'white',
        }
        has_draw_slot = len(winstate) > self.game_cls.num_players()
        is_draw_slot = has_draw_slot and bool(winstate[-1])
        if is_draw_slot:
            result_key = 'no_results' if getattr(state, 'terminal_kind', None) == 'no_result' else 'draws'
            for player_index in range(2):
                bucket = self._player_color_results[player_index][color_by_index[player_index]]
                bucket['games'] += 1
                bucket[result_key] += 1
            return

        winner_color = next(
            (color for color, won in enumerate(winstate[:2]) if bool(won)),
            None,
        )
        winner_index = None if winner_color is None else int(player_to_index[winner_color])
        for player_index in range(2):
            bucket = self._player_color_results[player_index][color_by_index[player_index]]
            bucket['games'] += 1
            if winner_index is None:
                bucket['no_results'] += 1
            elif player_index == winner_index:
                bucket['wins'] += 1
            else:
                bucket['losses'] += 1

    def __collect_batched_results(self, result_queue, expected_count=None, timeout=None):
        wins = [0] * self.game_cls.num_players()
        draws = 0
        no_results = 0
        result_count = 0

        if expected_count is None:
            while True:
                try:
                    result = result_queue.get_nowait()
                except Empty:
                    break
                state, winstate, agent_id = result
                self.__record_batched_color_result(state, winstate, agent_id)
                has_draw_slot = len(winstate) > self.game_cls.num_players()
                if has_draw_slot and winstate[-1]:
                    if getattr(state, 'terminal_kind', None) == 'no_result':
                        no_results += 1
                    else:
                        draws += 1
                else:
                    for player, is_win in enumerate(winstate[:self.game_cls.num_players()]):
                        if is_win:
                            index = self._agents[agent_id].player_to_index[player]
                            wins[index] += 1
                result_count += 1
        else:
            expected_count = int(expected_count)
            for _ in range(expected_count):
                try:
                    result = result_queue.get(timeout=timeout) if timeout is not None else result_queue.get()
                except Empty as exc:
                    raise RuntimeError(
                        'Batched Arena result finalization expected '
                        f'{expected_count} remaining accepted results, but only received {result_count}'
                    ) from exc
                state, winstate, agent_id = result
                self.__record_batched_color_result(state, winstate, agent_id)
                has_draw_slot = len(winstate) > self.game_cls.num_players()
                if has_draw_slot and winstate[-1]:
                    if getattr(state, 'terminal_kind', None) == 'no_result':
                        no_results += 1
                    else:
                        draws += 1
                else:
                    for player, is_win in enumerate(winstate[:self.game_cls.num_players()]):
                        if is_win:
                            index = self._agents[agent_id].player_to_index[player]
                            wins[index] += 1
                result_count += 1

        return wins, draws, no_results, result_count

    def __account_batched_results(self, result_queue, expected_count=None, timeout=None):
        wins, draws, no_results, result_count = self.__collect_batched_results(
            result_queue, expected_count, timeout
        )
        for i, w in enumerate(wins):
            self.__player_stats[i].wins += w
        self.draws += draws
        self.no_results += no_results
        self.__update_winrates()
        return result_count

    def __finalize_batched_results(
            self, result_queue, games_played, results_accounted, num, cancelled=False
    ):
        final_games_played = int(games_played.value)
        if final_games_played < 0:
            raise RuntimeError(
                f'Batched Arena accounting invariant violated: final games_played={final_games_played} < 0'
            )
        if final_games_played > int(num):
            raise RuntimeError(
                'Batched Arena accounting invariant violated: '
                f'final games_played={final_games_played} > requested num={num}'
            )
        if results_accounted < 0 or results_accounted > final_games_played:
            raise RuntimeError(
                'Batched Arena accounting invariant violated: '
                f'results_accounted={results_accounted}, final_games_played={final_games_played}'
            )

        remaining_results = final_games_played - results_accounted
        results_accounted += self.__account_batched_results(
            result_queue,
            expected_count=remaining_results,
            timeout=5.0,
        )
        if results_accounted != final_games_played:
            raise RuntimeError(
                'Batched Arena accounting invariant violated after finalization: '
                f'results_accounted={results_accounted}, final_games_played={final_games_played}'
            )

        aggregate_outcomes = sum(self.wins()) + self.draws + self.no_results
        if aggregate_outcomes != final_games_played:
            raise RuntimeError(
                'Batched Arena outcome accounting invariant violated: '
                f'outcomes={aggregate_outcomes}, final_games_played={final_games_played}'
            )
        if not cancelled and final_games_played != int(num):
            raise RuntimeError(
                'Batched Arena completed without cancellation before reaching requested games: '
                f'final_games_played={final_games_played}, requested num={num}'
            )
        if self.game_cls.num_players() == 2:
            for player_index in range(2):
                color_games = sum(
                    self._player_color_results[player_index][color]['games']
                    for color in ('black', 'white')
                )
                if color_games != final_games_played:
                    raise RuntimeError(
                        'Batched Arena color accounting invariant violated: '
                        f'player={player_index}, games={color_games}, '
                        f'final_games_played={final_games_played}'
                    )

        self.games_played = final_games_played
        return final_games_played, results_accounted

    def __drain_batched_results_while_workers_exit(
            self, agents, result_queue, games_played, results_accounted
    ):
        """Drain result IPC while workers finish their Queue feeder threads."""
        initial_games_played = int(games_played.value)
        deadline = time.monotonic() + 30.0

        while True:
            results_accounted += self.__account_batched_results(result_queue)
            for agent in agents:
                if agent.is_alive():
                    agent.join(timeout=0.05)
                    results_accounted += self.__account_batched_results(result_queue)

            if not any(agent.is_alive() for agent in agents):
                break
            if time.monotonic() >= deadline:
                active_workers = [int(agent.id) for agent in agents if agent.is_alive()]
                raise RuntimeError(
                    'Batched Arena workers did not exit while result_queue was being drained: '
                    f'active_workers={active_workers}, results_accounted={results_accounted}, '
                    f'games_played={int(games_played.value)}'
                )

        # Every worker is known to have exited, so this join is non-blocking;
        # all potentially blocking joins above were bounded and interleaved
        # with result consumption.
        for agent in agents:
            agent.join()

        final_games_played = int(games_played.value)
        if final_games_played < initial_games_played:
            raise RuntimeError(
                'Batched Arena accounting invariant violated during worker shutdown: '
                f'initial games_played={initial_games_played}, '
                f'final games_played={final_games_played}'
            )
        return final_games_played, results_accounted

    def wins(self) -> List[int]:
        return [s.wins for s in self.__player_stats]

    def winrates(self) -> List[float]:
        return [s.winrate for s in self.__player_stats]

    def player_color_results(self, player_index: int):
        player_index = int(player_index)
        if player_index < 0 or player_index >= len(self._player_color_results):
            raise IndexError('Arena player index out of range')
        return {
            color: dict(values)
            for color, values in self._player_color_results[player_index].items()
        }

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
                while True:
                    try:
                        q.get_nowait()
                    except Empty:
                        break

            score_aware = (
                self.args.get('search_utility_mode', 'legacy') == KATAGO_PINNED_SEARCH_UTILITY_MODE
                if hasattr(self.args, 'get')
                else getattr(self.args, 'search_utility_mode', 'legacy') == KATAGO_PINNED_SEARCH_UTILITY_MODE
            )
            point_count = int(self.game_cls.logical_topology().point_count) if score_aware else 0
            if score_aware:
                # The historical batched Arena's row->game mapping becomes
                # ambiguous once multiple game slots in one worker finish at
                # different times. One active game per worker preserves the
                # existing scheduling while making the mapping exact.
                self.args.arena_batch_size = 1

            self.args.gamesPerIteration = num
            self._agents = []
            observation_adapters = [
                getattr(player, 'observation_adapter', None) for player in self.players
            ]
            # Keep the legacy worker path byte-for-byte compatible when no
            # player has a model-specific adapter.  Cross-profile GoCube
            # Arena supplies one immutable adapter per network.
            if any(adapter is None for adapter in observation_adapters):
                if any(adapter is not None for adapter in observation_adapters):
                    raise ValueError(
                        'Arena requires an observation adapter for every model '
                        'when cross-profile observation routing is enabled'
                    )
                observation_adapters = None
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
                    ownership_tensors.append(torch.zeros([
                        self.args.arena_batch_size, point_count, 3
                    ]))
                    ownership_tensors[i].share_memory_()

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
                        score_tensor=score_tensors[i] if score_aware else None,
                        ownership_tensor=ownership_tensors[i] if score_aware else None,
                        observation_adapters=observation_adapters,
                    )
                )
                self._agents[i].daemon = True
                self._agents[i].start()

            sample_time = AverageMeter()
            end = time.time()

            n = 0
            results_accounted = 0
            while completed.value != self.args.workers and not self.stop_event.is_set():
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
                                out = self.players[player].nn.process_for_search(batch)
                                if not isinstance(out, SearchOutput):
                                    raise RuntimeError('process_for_search() must return SearchOutput')
                                if out.score is None or out.ownership is None:
                                    raise RuntimeError(
                                        'KataGo-derived batched Arena requires score and ownership heads'
                                    )
                                policy.append(out.policy.to(policy_tensors[id].device))
                                value.append(out.value.to(value_tensors[id].device))
                                score.append(out.score.to(score_tensors[id].device))
                                ownership.append(out.ownership.to(ownership_tensors[id].device))
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

                results_accounted += self.__account_batched_results(result_queue)

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

            cancelled = self.stop_event.is_set()
            self.stop_event.set()
            _, results_accounted = self.__drain_batched_results_while_workers_exit(
                self._agents,
                result_queue,
                games_played,
                results_accounted,
            )

            final_games_played, results_accounted = self.__finalize_batched_results(
                result_queue,
                games_played,
                results_accounted,
                num,
                cancelled=cancelled,
            )
            if final_games_played > n:
                sample_time.update(
                    (time.time() - end) / (final_games_played - n), final_games_played - n
                )
            self.__update_winrates()
            bar.suffix = '({eps}/{maxeps}) Winrates: {wr} | No-result: {nr} | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}' \
                .format(
                    eps=final_games_played, maxeps=num, et=sample_time.avg, total=bar.elapsed_td,
                    eta=bar.eta_td, wr=[round(w, 3) for w in self.winrates()], nr=self.no_results
                )
            bar.goto(final_games_played)
            self.eps_time = sample_time.avg
            self.total_time = bar.elapsed_td
            self.eta = bar.eta_td
            bar.update()
            bar.finish()

            empty_queue(ready_queue)
            empty_queue(result_queue)
            for q in batch_queues:
                empty_queue(q)

            for _ in self._agents:
                del policy_tensors[0]
                del value_tensors[0]
                if score_aware:
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
