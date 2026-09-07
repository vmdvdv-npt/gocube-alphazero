from __future__ import annotations

import pytest

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
    assert sum(item.quota for item in assignments if item.model_a_color == "black") == games // 2
    assert sum(item.quota for item in assignments if item.model_a_color == "white") == games // 2
    assert all(item.player_to_index == ((0, 1) if item.model_a_color == "black" else (1, 0)) for item in assignments)


def test_batched_arena_schedule_has_minimal_imbalance_for_odd_game_count():
    assignments = arena_worker_assignments(129, 8)
    black = sum(item.quota for item in assignments if item.model_a_color == "black")
    white = sum(item.quota for item in assignments if item.model_a_color == "white")
    assert black == 65
    assert white == 64


def test_balanced_batched_arena_fails_closed_for_single_worker_multi_game_run():
    with pytest.raises(ValueError, match="at least two workers"):
        arena_worker_assignments(2, 1)


def test_checkpoint_arena_entrypoint_installs_balanced_worker_and_guard():
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
