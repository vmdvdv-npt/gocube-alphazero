from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gocube_golden import (
    BASELINE_KOMI,
    BLACK,
    EMPTY,
    PASS,
    STAGE0_RULES_FINGERPRINT,
    TORUS_5X5,
    TORUS_5X5_TOPOLOGY_FINGERPRINT,
    WHITE,
    GoldenState,
    IllegalMoveError,
    IllegalMoveReason,
    LegacyKomiError,
    Ownership,
    Winner,
    apply_action,
    board_key,
    demonstration_text,
    group_from_board,
    initial_state,
    legal_actions,
    liberties_from_board,
    permute_topology,
    replay,
    research_topology,
    result_from_terminal,
    score_terminal,
    state_from_stones,
)


def board(*, black=(), white=(), points=25):
    stones = [EMPTY] * points
    for point in black:
        stones[point] = BLACK
    for point in white:
        if stones[point] != EMPTY:
            raise AssertionError(f"fixture collision at {point}")
        stones[point] = WHITE
    return tuple(stones)


def terminal_state(stones, **kwargs):
    return state_from_stones(stones, consecutive_passes=2, **kwargs)


def assert_illegal(state, action, reason):
    before = state.state_key
    with pytest.raises(IllegalMoveError) as caught:
        apply_action(state, action)
    assert caught.value.reason == reason
    assert state.state_key == before


def permute_stones(stones, old_to_new):
    result = [EMPTY] * len(stones)
    for old, stone in enumerate(stones):
        result[old_to_new[old]] = stone
    return tuple(result)


def swap_board(position):
    return tuple(
        int(WHITE) if value == int(BLACK) else int(BLACK) if value == int(WHITE) else int(EMPTY)
        for value in position
    )


def swap_stones(stones):
    return tuple(
        WHITE if stone == BLACK else BLACK if stone == WHITE else EMPTY for stone in stones
    )


# ---------------------------------------------------------------------------
# Exact graph-area scoring
# ---------------------------------------------------------------------------


def test_black_owned_empty_region():
    stones = [BLACK] * 25
    stones[12] = EMPTY
    score = score_terminal(terminal_state(stones))
    assert score.black_stones == 24
    assert score.black_territory == 1
    assert score.black_area == 25
    assert score.white_area == 0
    assert score.ownership[12] == Ownership.BLACK


def test_white_owned_empty_region():
    stones = [WHITE] * 25
    stones[12] = EMPTY
    score = score_terminal(terminal_state(stones))
    assert score.white_stones == 24
    assert score.white_territory == 1
    assert score.white_area == 25
    assert score.ownership[12] == Ownership.WHITE


def test_neutral_region_touching_both_colors():
    score = score_terminal(terminal_state(board(black=(0,), white=(2,))))
    assert score.black_territory == 0
    assert score.white_territory == 0
    assert score.neutral_points == 23


def test_neutral_component_touching_no_stones():
    score = score_terminal(terminal_state(board()))
    assert score.neutral_points == 25
    assert set(score.ownership) == {Ownership.NEUTRAL}


def test_territory_component_crosses_torus_wrap():
    stones = [BLACK] * 25
    stones[0] = EMPTY
    stones[4] = EMPTY
    score = score_terminal(terminal_state(stones))
    assert score.black_territory == 2
    assert score.ownership[0] == Ownership.BLACK
    assert score.ownership[4] == Ownership.BLACK


def test_mixed_color_wrap_component_is_neutral():
    stones = [BLACK] * 25
    stones[0] = EMPTY
    stones[4] = EMPTY
    stones[1] = WHITE
    score = score_terminal(terminal_state(stones))
    assert score.neutral_points == 2
    assert score.ownership[0] == Ownership.NEUTRAL
    assert score.ownership[4] == Ownership.NEUTRAL


def test_exact_komi_applied_only_to_final_margin():
    score = score_terminal(terminal_state(board()))
    assert score.black_area == 0
    assert score.white_area == 0
    assert score.komi == 0.5
    assert score.margin_black == -0.5
    assert result_from_terminal(terminal_state(board())).winner == Winner.WHITE


def test_forbidden_legacy_komi_7_5_fails_closed():
    with pytest.raises(LegacyKomiError, match="7.5"):
        initial_state(komi=7.5)
    with pytest.raises(LegacyKomiError, match="contact the project owner"):
        state_from_stones(board(), komi=7.5)


def test_scorer_does_not_remove_synthetic_zero_liberty_stones():
    # Scoring is literal graph-area over the terminal board, not a dead-stone
    # cleanup engine. This fixture is intentionally not claimed as a legal trace.
    stones = board(black=(12,), white=(7, 11, 13, 17))
    score = score_terminal(terminal_state(stones))
    assert score.black_stones == 1
    assert score.white_stones == 4


# ---------------------------------------------------------------------------
# Canonical state identity / illegal-move immutability
# ---------------------------------------------------------------------------


def test_same_board_with_different_superko_history_has_different_full_state_identity():
    stones = board(black=(0,))
    current = board_key(stones)
    empty = board_key(board())
    left = state_from_stones(stones, side_to_move=WHITE, superko_history=(current,))
    right = state_from_stones(stones, side_to_move=WHITE, superko_history=(empty, current))
    assert left.board_key == right.board_key
    assert left.state_key != right.state_key


def test_occupied_invalid_and_suicide_checks_do_not_mutate_state():
    occupied = state_from_stones(board(black=(0,)), side_to_move=WHITE)
    assert_illegal(occupied, 0, IllegalMoveReason.OCCUPIED)
    assert_illegal(occupied, -1, IllegalMoveReason.INVALID_ACTION_ID)
    assert_illegal(occupied, 25, IllegalMoveReason.INVALID_ACTION_ID)
    assert_illegal(occupied, True, IllegalMoveReason.INVALID_ACTION_ID)
    suicide = state_from_stones(board(white=(7, 11, 13, 17)), side_to_move=BLACK)
    assert_illegal(suicide, 12, IllegalMoveReason.SUICIDE)


