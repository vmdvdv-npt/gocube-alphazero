from queue import Empty, Queue
from types import SimpleNamespace

import numpy as np
import pytest
from torch import multiprocessing as mp

from alphazero.Arena import Arena
from alphazero.arena_bookkeeping import ArenaResult
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.utils import dotdict


class _TwoPlayerGame:
    @staticmethod
    def num_players():
        return 2

    @staticmethod
    def action_size():
        return 1


class _ArenaPlayer:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def supports_process():
        return False


class _BatchedArenaPlayer(_ArenaPlayer):
    @staticmethod
    def supports_process():
        return True


class _ResultState:
    def __init__(self, terminal_kind='scored'):
        self.terminal_kind = terminal_kind


class _LargeResultState(_ResultState):
    def __init__(self):
        super().__init__()
        self.payload = b'x' * (2 * 1024 * 1024)


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


def test_result_attribution_uses_immutable_game_snapshot_not_current_worker_mapping():
    arena = _new_arena()
    result_queue = Queue()
    result_queue.put(ArenaResult(
        final_state=_ResultState(),
        winstate=np.array([True, False, False]),
        game_id=0,
        worker_id=0,
        slot_id=1,
        generation=3,
        model_a_color="white",
        player_to_index=(1, 0),
    ))
    # The worker can already have recycled the slot and changed this mutable
    # compatibility attribute by the time the parent accounts for the result.
    arena._agents[0].player_to_index = [0, 1]
    games_played = mp.Value('i', 1)

    final_games_played, results_accounted = arena._Arena__finalize_batched_results(
        result_queue,
        games_played,
        results_accounted=0,
        num=1,
    )

    assert (final_games_played, results_accounted) == (1, 1)
    assert arena.wins() == [0, 1]
    assert arena.player_color_results(0)["white"]["losses"] == 1
    assert arena.player_color_results(1)["black"]["wins"] == 1


def test_out_of_order_game_results_keep_color_and_winner_attribution():
    arena = _new_arena()
    result_queue = Queue()
    # Results intentionally arrive in a different order from their global IDs.
    for game_id in (3, 1, 2, 0):
        mapping = (1, 0) if game_id % 2 else (0, 1)
        result_queue.put(ArenaResult(
            final_state=_ResultState(),
            winstate=np.array([True, False, False]),
            game_id=game_id,
            worker_id=game_id % 2,
            slot_id=game_id % 2,
            generation=0,
            model_a_color="white" if game_id % 2 else "black",
            player_to_index=mapping,
        ))
    games_played = mp.Value('i', 4)

    final_games_played, results_accounted = arena._Arena__finalize_batched_results(
        result_queue,
        games_played,
        results_accounted=0,
        num=4,
    )

    assert (final_games_played, results_accounted) == (4, 4)
    assert arena.wins() == [2, 2]
    assert arena.player_color_results(0)["black"]["games"] == 2
    assert arena.player_color_results(0)["white"]["games"] == 2


def test_duplicate_game_result_is_rejected_before_double_accounting():
    arena = _new_arena()
    result_queue = Queue()
    result = ArenaResult(
        final_state=_ResultState(),
        winstate=np.array([True, False, False]),
        game_id=0,
        worker_id=0,
        slot_id=0,
        generation=0,
        model_a_color="black",
        player_to_index=(0, 1),
    )
    result_queue.put(result)
    result_queue.put(result)

    with pytest.raises(RuntimeError, match="duplicate game_id"):
        arena._Arena__finalize_batched_results(
            result_queue,
            mp.Value('i', 2),
            results_accounted=0,
            num=2,
        )


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


def _produce_large_results(result_queue):
    for _ in range(2):
        result_queue.put(
            (_LargeResultState(), np.array([True, False, False]), 0)
        )


def _run_large_result_shutdown():
    arena = _new_arena()
    result_queue = mp.Queue(maxsize=1)
    games_played = mp.Value('i', 2)
    producer = mp.Process(target=_produce_large_results, args=(result_queue,))
    producer.id = 0
    producer.player_to_index = [0, 1]
    arena._agents = [producer]
    producer.start()

    final_games_played, results_accounted = (
        arena._Arena__drain_batched_results_while_workers_exit(
            arena._agents,
            result_queue,
            games_played,
            results_accounted=0,
        )
    )
    final_games_played, results_accounted = arena._Arena__finalize_batched_results(
        result_queue,
        games_played,
        results_accounted,
        num=2,
    )
    assert (final_games_played, results_accounted) == (2, 2)
    assert sum(arena.wins()) + arena.draws + arena.no_results == 2
    result_queue.close()
    result_queue.join_thread()


def test_batched_shutdown_drains_real_multiprocessing_queue_before_join_deadlock():
    runner = mp.Process(target=_run_large_result_shutdown)
    runner.start()
    runner.join(timeout=10.0)
    if runner.is_alive():
        runner.terminate()
        runner.join()
        pytest.fail('batched Arena shutdown hung while joining a blocked result producer')
    assert runner.exitcode == 0


class _LargeResultSelfPlayAgent(mp.Process):
    def __init__(self, *args, **_kwargs):
        super().__init__()
        self.id = int(args[0])
        self.result_queue = args[8]
        self.complete_count = args[9]
        self.games_played = args[10]
        self.player_to_index = [0, 1]

    def run(self):
        for _ in range(2):
            self.result_queue.put(
                (_LargeResultState(), np.array([True, False, False]), self.id)
            )
        with self.games_played.get_lock():
            self.games_played.value = 2
        with self.complete_count.get_lock():
            self.complete_count.value += 1


def _run_arena_with_large_result_producer():
    import alphazero.Arena as arena_module

    arena_module.SelfPlayAgent = _LargeResultSelfPlayAgent
    args = dotdict({
        'numMCTSSims': 1,
        'arenaMCTSSims': 1,
        'arenaTemp': 0.0,
        'workers': 1,
        'cuda': False,
        'arena_batch_size': 1,
        'use_draws_for_winrate': True,
    })
    arena = Arena(
        [_BatchedArenaPlayer(args), _BatchedArenaPlayer(args)],
        _TwoPlayerGame,
        use_batched_mcts=True,
        args=args,
    )
    wins, draws, _ = arena.play_games(2)
    assert arena.games_played == 2
    assert sum(wins) + draws + arena.no_results == 2


def test_batched_arena_play_games_drains_producer_queue_during_shutdown():
    runner = mp.Process(target=_run_arena_with_large_result_producer)
    runner.start()
    runner.join(timeout=10.0)
    if runner.is_alive():
        runner.terminate()
        runner.join()
        pytest.fail('Arena.play_games hung while joining a producer before draining result_queue')
    assert runner.exitcode == 0
