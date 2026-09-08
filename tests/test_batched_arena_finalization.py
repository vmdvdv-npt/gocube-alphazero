from queue import Empty, Queue
from types import SimpleNamespace

import numpy as np
import pytest
from torch import multiprocessing as mp

from alphazero.Arena import Arena
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.utils import dotdict


class _TwoPlayerGame:
    @staticmethod
    def num_players():
        return 2


class _ArenaPlayer:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def supports_process():
        return False


class _ResultState:
    def __init__(self, terminal_kind='scored'):
        self.terminal_kind = terminal_kind


def _new_arena():
    args = dotdict({
        'numMCTSSims': 1,
        'arenaMCTSSims': 1,
        'arenaTemp': 0.0,
        'use_draws_for_winrate': True,
    })
    arena = Arena(
        [_ArenaPlayer(args), _ArenaPlayer(args)],
        _TwoPlayerGame,
        use_batched_mcts=False,
        args=args,
    )
    arena._agents = [SimpleNamespace(player_to_index=[0, 1])]
    return arena


def _seed_one_accounted_win(arena):
    arena._Arena__player_stats[0].wins = 1
    arena._player_color_results[0]['black']['games'] = 1
    arena._player_color_results[0]['black']['wins'] = 1
    arena._player_color_results[1]['white']['games'] = 1
    arena._player_color_results[1]['white']['losses'] = 1


def test_batched_arena_finalization_drains_pending_accepted_result():
    arena = _new_arena()
    _seed_one_accounted_win(arena)
    result_queue = Queue()
    result_queue.put((_ResultState(), np.array([False, True, False]), 0))
    games_played = mp.Value('i', 2)

    final_games_played, results_accounted = arena._Arena__finalize_batched_results(
        result_queue,
        games_played,
        results_accounted=1,
        num=2,
    )

    assert final_games_played == 2
    assert results_accounted == 2
    assert arena.games_played == 2
    assert arena.wins() == [1, 1]
    assert arena.draws == 0
    assert arena.no_results == 0
    assert sum(arena.wins()) + arena.draws + arena.no_results == 2
    assert all(
        arena.player_color_results(player)['black']['games']
        + arena.player_color_results(player)['white']['games'] == 2
        for player in range(2)
    )
    with pytest.raises(Empty):
        result_queue.get_nowait()


@pytest.mark.parametrize(
    ('terminal_kind', 'expected_draws', 'expected_no_results'),
    [('scored', 1, 0), ('no_result', 0, 1)],
)
def test_batched_arena_finalization_preserves_draw_and_no_result_accounting(
        terminal_kind, expected_draws, expected_no_results
):
    arena = _new_arena()
    result_queue = Queue()
    result_queue.put((_ResultState(), np.array([True, False, False]), 0))
    result_queue.put(
        (_ResultState(terminal_kind), np.array([False, False, True]), 0)
    )
    games_played = mp.Value('i', 2)

    final_games_played, results_accounted = arena._Arena__finalize_batched_results(
        result_queue,
        games_played,
        results_accounted=0,
        num=2,
    )

    assert (final_games_played, results_accounted) == (2, 2)
    assert arena.wins() == [1, 0]
    assert arena.draws == expected_draws
    assert arena.no_results == expected_no_results
    assert sum(arena.wins()) + arena.draws + arena.no_results == 2


def test_non_recording_rejected_terminal_game_is_not_published():
    agent = SelfPlayAgent.__new__(SelfPlayAgent)
    agent.games_played = mp.Value('i', 1)
    agent.args = SimpleNamespace(gamesPerIteration=1)
    agent.result_queue = Queue()
    agent.id = 0
    agent._current_game_slot = 0
    agent._current_stage = 'finish_game'

    accepted = agent._publish_non_recording_result(
        _ResultState(), np.array([True, False, False])
    )

    assert accepted is False
    assert agent.games_played.value == 1
    with pytest.raises(Empty):
        agent.result_queue.get_nowait()
