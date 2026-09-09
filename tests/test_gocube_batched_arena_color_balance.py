from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

import pytest

from tools import gocube_checkpoint_arena as checkpoint_arena
from tools import gocube_checkpoint_arena_complete as checkpoint_arena_impl
from tools.gocube_balanced_arena import (
    BalancedArenaSelfPlayAgent,
    _validate_color_balance,
    arena_worker_assignments,
)


def _fake_final_state(*, turns=1, margin=None, terminal_kind="scored"):
    terminal_adjudication = None
    if margin is not None:
        terminal_adjudication = SimpleNamespace(score=SimpleNamespace(margin=float(margin)))
    return SimpleNamespace(
        turns=int(turns),
        terminal_kind=terminal_kind,
        terminal_adjudication=terminal_adjudication,
        semantic_state=None,
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


@pytest.mark.parametrize(
    ("player_to_index", "winstate", "expected"),
    (
        ((0, 1), (True, False, False), ("black", "win", None)),
        ((0, 1), (False, True, False), ("black", "loss", None)),
        ((1, 0), (True, False, False), ("white", "loss", None)),
        ((1, 0), (False, True, False), ("white", "win", None)),
    ),
)
def test_result_for_model_a_is_correct_for_both_colors_and_winners(
    player_to_index, winstate, expected
):
    result = checkpoint_arena_impl._result_for_model_a(
        _fake_final_state(),
        winstate,
        list(player_to_index),
    )
    assert result == expected


def test_result_for_model_a_score_margin_follows_model_a_color():
    final_state = _fake_final_state(margin=3.0)

    assert checkpoint_arena_impl._result_for_model_a(
        final_state, (True, False, False), [0, 1]
    ) == ("black", "win", 3.0)
    assert checkpoint_arena_impl._result_for_model_a(
        final_state, (True, False, False), [1, 0]
    ) == ("white", "loss", -3.0)


def test_deterministic_result_queue_bookkeeping_uses_emitting_workers_color_mapping():
    agents = [
        SimpleNamespace(player_to_index=[0, 1]),  # model A is black
        SimpleNamespace(player_to_index=[1, 0]),  # model A is white
    ]
    result_queue = Queue()

    # Known outcomes, deliberately covering both model-A colors and both results.
    result_queue.put((_fake_final_state(turns=11), (True, False, False), 0))
    result_queue.put((_fake_final_state(turns=12), (False, True, False), 1))
    result_queue.put((_fake_final_state(turns=13), (False, True, False), 0))
    result_queue.put((_fake_final_state(turns=14), (True, False, False), 1))

    outcomes = []
    lengths = []
    diagnostics = []
    checkpoint_arena_impl._drain_results(
        result_queue,
        agents,
        outcomes,
        lengths,
        diagnostics,
    )

    assert outcomes == [
        ("black", "win", None),
        ("white", "win", None),
        ("black", "loss", None),
        ("white", "loss", None),
    ]
    assert lengths == [11, 12, 13, 14]

    summary = checkpoint_arena_impl._summarize_outcomes(outcomes)
    assert summary["wins"] == 2
    assert summary["losses"] == 2
    assert summary["draws"] == 0
    assert summary["no_results"] == 0
    assert summary["win_rate"] == 0.5
    assert summary["by_color"] == {
        "black": {
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "no_results": 0,
        },
        "white": {
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "no_results": 0,
        },
    }
