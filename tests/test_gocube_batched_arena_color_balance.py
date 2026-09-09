from __future__ import annotations

import pytest

from alphazero.arena_bookkeeping import (
    arena_game_ids_by_worker,
    model_a_color_for_game_id,
    player_to_index_for_game_id,
)
from tools import gocube_checkpoint_arena as checkpoint_arena
from tools import gocube_checkpoint_arena_complete as checkpoint_arena_impl
from tools.gocube_balanced_arena import (
    BalancedArenaSelfPlayAgent,
    _validate_color_balance,
    arena_worker_assignments,
)


@pytest.mark.parametrize("workers", (4, 8, 16))
@pytest.mark.parametrize("games", (64, 128, 256, 512))
def test_batched_arena_schedule_is_exactly_color_balanced(workers, games):
    assignments = arena_worker_assignments(games, workers)
    assert len(assignments) == workers
    assert sum(item.quota for item in assignments) == games
    game_ids = [game_id for item in assignments for game_id in item.game_ids]
    assert game_ids == list(range(games))
    assert sum(model_a_color_for_game_id(game_id) == "black" for game_id in game_ids) == games // 2
    assert sum(model_a_color_for_game_id(game_id) == "white" for game_id in game_ids) == games // 2
    assert all(
        player_to_index_for_game_id(game_id)
        == ((0, 1) if model_a_color_for_game_id(game_id) == "black" else (1, 0))
        for game_id in game_ids
    )


def test_batched_arena_schedule_has_minimal_imbalance_for_odd_game_count():
    assignments = arena_worker_assignments(129, 8)
    game_ids = [game_id for item in assignments for game_id in item.game_ids]
    black = sum(model_a_color_for_game_id(game_id) == "black" for game_id in game_ids)
    white = sum(model_a_color_for_game_id(game_id) == "white" for game_id in game_ids)
    assert black == 65
    assert white == 64


def test_balanced_batched_arena_allows_multiple_games_on_one_worker():
    assignments = arena_worker_assignments(8, 1)
    assert assignments[0].game_ids == tuple(range(8))
    assert assignments[0].model_a_colors == (
        "black", "white", "black", "white", "black", "white", "black", "white"
    )


def test_game_schedule_is_worker_speed_independent():
    schedule = arena_game_ids_by_worker(64, 4)
    fast_worker_completion = [game_id for group in schedule for game_id in group]
    slow_worker_completion = [game_id for group in reversed(schedule) for game_id in reversed(group)]
    assert set(fast_worker_completion) == set(slow_worker_completion) == set(range(64))
    assert sum(model_a_color_for_game_id(game_id) == "black" for game_id in fast_worker_completion) == 32
    assert sum(model_a_color_for_game_id(game_id) == "black" for game_id in slow_worker_completion) == 32


def test_checkpoint_arena_entrypoint_installs_balanced_worker_without_batch_guard():
    assert checkpoint_arena_impl.SelfPlayAgent is BalancedArenaSelfPlayAgent
    assert checkpoint_arena._coalesced_batched_summary is checkpoint_arena_impl._coalesced_batched_summary


def test_color_balance_guard_rejects_skewed_results():
    good = {
        "by_color": {
            "black": {"games": 64},
            "white": {"games": 64},
        }
    }
    _validate_color_balance(good, 128)

    bad = {
        "by_color": {
            "black": {"games": 80},
            "white": {"games": 48},
        }
    }
    with pytest.raises(RuntimeError, match="color-balance contract failed"):
        _validate_color_balance(bad, 128)
