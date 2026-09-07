from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alphazero.envs.gocube.pinned_game as pinned_game_module
from alphazero.envs.gocube.diversified_game import (
    DiversifiedPinnedCube4JapaneseGame,
)
from alphazero.envs.gocube.pinned_game import PinnedCube4JapaneseGame
from alphazero.envs.gocube.katago_v3 import MAIN, SCORED, V3State


def _play_deterministic_points(game, count):
    for _ in range(count):
        valid = np.flatnonzero(game.valid_moves())
        valid = valid[valid != game.pass_action()]
        assert len(valid), "fixture ran out of deterministic legal non-pass moves"
        game.play_action(int(valid[0]))


def _assert_state_fields_equal(left: V3State, right: V3State):
    assert left == right
    for field in (
        "board", "current_player", "turns", "phase", "consecutive_passes",
        "previous_board", "ko_recap_blocked", "phase_history", "history_since_pass",
        "black_pass_states", "white_pass_states", "ko_capture_history",
        "second_cleanup_start_colors", "main_moves", "cleanup1_moves", "cleanup2_moves",
        "captures", "terminal_kind", "no_result_reason",
    ):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            assert np.array_equal(left_value, right_value), field
        else:
            assert left_value == right_value, field


def test_initial_and_ordinary_state_history_match_each_prefix():
    game = PinnedCube4JapaneseGame()
    game.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    expected = PinnedCube4JapaneseGame()
    expected.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )

    for _ in range(12):
        valid = np.flatnonzero(game.valid_moves())
        valid = valid[valid != game.pass_action()]
        action = int(valid[0])
        game.play_action(action)
        expected.play_action(action)
        assert game._pinned_state_history_offset == 0
        assert len(game._pinned_state_history) == len(game._pinned_move_history) + 1

    for local_state_index, state in enumerate(game._pinned_state_history):
        _assert_state_fields_equal(
            state,
            game._pinned_state_for_history_len(
                game._pinned_state_history_offset + local_state_index
            ),
        )
        replay = PinnedCube4JapaneseGame()
        replay.configure_pinned_selfplay(
            auto_end_pass_alive=False,
            root_prune_useless_moves=False,
            seki_fork_hack_prob=0.0,
        )
        for _, action in game._pinned_move_history[:local_state_index]:
            replay.play_action(action)
        _assert_state_fields_equal(state, replay.semantic_state)

    clone = game.clone()
    assert clone._pinned_state_history_offset == game._pinned_state_history_offset


def test_twenty_moves_plain_fork_then_six_passes_keep_absolute_alignment(monkeypatch):
    pool = DiversifiedPinnedCube4JapaneseGame._plain_fork_pool()
    pool.clear()
    source = DiversifiedPinnedCube4JapaneseGame()
    source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    source.configure_diversification(
        early_fork_prob=0.0,
        ordinary_fork_prob=1.0,
        early_expected_move_prop=0.025,
    )
    _play_deterministic_points(source, 21)
    monkeypatch.setattr(
        "alphazero.envs.gocube.diversified_game.sample_fork_depth",
        lambda *args, **kwargs: 20,
    )
    monkeypatch.setattr(source, "_has_unowned_final_spot", lambda: True)
    source._state = source.semantic_state.__class__(
        **{**source.semantic_state.__dict__, "terminal_kind": SCORED}
    )
    source._maybe_store_plain_fork()

    target = DiversifiedPinnedCube4JapaneseGame()
    target.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    assert target.maybe_start_plain_fork() == {"mode": "ordinary_fork", "fork_depth": 20}
    assert len(target._pinned_move_history) == 20
    assert len(target._pinned_state_history) == 1
    assert target._pinned_state_history_offset == 20

    expected = DiversifiedPinnedCube4JapaneseGame(target.semantic_state)
    expected.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    for pass_number in range(1, 7):
        target.play_action(target.pass_action())
        expected.play_action(expected.pass_action())
        assert target._pinned_state_history_offset == 20
        assert len(target._pinned_state_history) - 1 == (
            len(target._pinned_move_history) - target._pinned_state_history_offset
        )
        if pass_number in (1, 2, 4, 6):
            state = target._pinned_state_for_history_len(20 + pass_number)
            _assert_state_fields_equal(state, expected.semantic_state)
            assert target._pinned_move_history[:20 + pass_number] == (
                target._pinned_move_history[:20] +
                target._pinned_move_history[20:20 + pass_number]
            )
    assert target.semantic_state.phase == "scored"
    assert target.semantic_state.terminal_kind == SCORED


@pytest.mark.parametrize("kind", ["early_fork", "ordinary_fork"])
def test_early_and_ordinary_fork_restore_the_same_full_state(kind):
    pool = DiversifiedPinnedCube4JapaneseGame._plain_fork_pool()
    pool.clear()
    source = DiversifiedPinnedCube4JapaneseGame()
    source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    _play_deterministic_points(source, 8)
    candidate_state = source.semantic_state
    candidate_history = source._pinned_move_history
    pool.append((kind, candidate_state, candidate_history, len(candidate_history)))

    target = DiversifiedPinnedCube4JapaneseGame()
    assert target.maybe_start_plain_fork()["mode"] == kind
    _assert_state_fields_equal(target.semantic_state, candidate_state)
    assert target._pinned_move_history == candidate_history
    assert target.player == candidate_state.current_player
    assert target.last_action == candidate_history[-1][1]


def test_plain_fork_seki_sampling_uses_only_the_saved_local_segment(monkeypatch):
    pool = DiversifiedPinnedCube4JapaneseGame._seki_pool()
    pool.clear()
    DiversifiedPinnedCube4JapaneseGame._plain_fork_pool().clear()
    source = DiversifiedPinnedCube4JapaneseGame()
    source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    _play_deterministic_points(source, 21)
    candidate_state = source.semantic_state
    candidate_history = source._pinned_move_history
    DiversifiedPinnedCube4JapaneseGame._plain_fork_pool().append(
        ("ordinary_fork", candidate_state, candidate_history, 20)
    )

    target = DiversifiedPinnedCube4JapaneseGame()
    assert target.maybe_start_plain_fork() is not None
    target.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=1.0,
    )
    target._has_unowned_final_spot = lambda: True
    fake_random = SimpleNamespace(exponential=lambda: 0.0, randint=lambda low, high: low)
    monkeypatch.setattr(pinned_game_module.np, "random", fake_random)
    for _ in range(6):
        target.play_action(target.pass_action())

    assert pool
    candidate_state, candidate_history = pool[-1]
    assert len(candidate_history) >= target._pinned_state_history_offset
    local_state_index = len(candidate_history) - target._pinned_state_history_offset
    assert candidate_state == target._pinned_state_for_history_len(len(candidate_history))
    assert 0 <= local_state_index <= len(target._pinned_state_history) - 1


def test_state_lookup_rejects_invalid_absolute_history_lengths():
    game = PinnedCube4JapaneseGame()
    game.play_action(0)
    game._pinned_move_history = ((0, 0), (1, 1))
    game._pinned_state_history = (game.semantic_state,)
    game._pinned_state_history_offset = 1
    with pytest.raises(RuntimeError, match="requested_absolute_history_len=0"):
        game._pinned_state_for_history_len(0)
    with pytest.raises(RuntimeError, match="move_history_length=2"):
        game._pinned_state_for_history_len(3)
