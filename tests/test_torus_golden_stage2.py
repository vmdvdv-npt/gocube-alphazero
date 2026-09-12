from __future__ import annotations

import pytest

import gocube_golden as g


def board(*, black=(), white=(), points=25):
    stones = [g.EMPTY] * points
    for p in black:
        stones[p] = g.BLACK
    for p in white:
        stones[p] = g.WHITE
    return tuple(stones)


# ---------------------------------------------------------------------------
# Stage-1 hardening
# ---------------------------------------------------------------------------

def test_live_history_must_end_at_current_board():
    state = g.initial_state()
    unrelated = list(state.board_key)
    unrelated[0] = int(g.BLACK)
    with pytest.raises(ValueError, match="must end"):
        g.GoldenState(
            stones=state.stones,
            side_to_move=state.side_to_move,
            superko_history=(state.board_key, tuple(unrelated)),
            consecutive_passes=0,
            topology=state.topology,
            rules_id=state.rules_id,
            rules_fingerprint=state.rules_fingerprint,
            komi=state.komi,
        )


def test_live_history_rejects_duplicate_stone_positions():
    state = g.initial_state()
    with pytest.raises(ValueError, match="duplicate"):
        g.GoldenState(
            stones=state.stones,
            side_to_move=state.side_to_move,
            superko_history=(state.board_key, state.board_key),
            consecutive_passes=0,
            topology=state.topology,
            rules_id=state.rules_id,
            rules_fingerprint=state.rules_fingerprint,
            komi=state.komi,
        )


def test_pass_preserves_live_history_tail_invariant():
    state = g.apply_action(g.initial_state(), g.PASS).after
    assert state.is_canonical_live
    assert state.superko_history[-1] == state.board_key
    assert len(set(state.superko_history)) == len(state.superko_history)


def test_synthetic_fixture_is_explicit_and_arena_rejects_it():
    synthetic = g.research_state_from_stones(board(black=(0,)))
    assert not synthetic.is_canonical_live
    arena = g.SequentialGoldenArena()
    with pytest.raises(ValueError, match="synthetic"):
        arena.play_game(
            game_id="synthetic",
            pair_id="p",
            player_A=g.BadPlayer("A"),
            player_B=g.BadPlayer("B"),
            black_player="A",
            start_state=synthetic,
        )


def test_arena_rejects_terminal_start_even_if_history_is_live():
    terminal = g.initial_state()
    terminal = g.apply_action(terminal, g.PASS).after
    terminal = g.apply_action(terminal, g.PASS).after
    with pytest.raises(ValueError, match="terminal"):
        g.SequentialGoldenArena().play_game(
            game_id="terminal",
            pair_id="p",
            player_A=g.BadPlayer("A"),
            player_B=g.BadPlayer("B"),
            black_player="A",
            start_state=terminal,
            start_trace=(g.PASS, g.PASS),
        )


def test_noninitial_arena_start_requires_exact_legal_replay_trace():
    state = g.apply_action(g.initial_state(), 0).after
    arena = g.SequentialGoldenArena()
    with pytest.raises(ValueError, match="reproducible"):
        arena.play_game(
            game_id="bad-start-trace", pair_id="p",
            player_A=g.BadPlayer("A"), player_B=g.BadPlayer("B"),
            black_player="A", start_state=state, start_trace=(),
        )
