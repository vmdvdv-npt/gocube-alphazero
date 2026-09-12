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
# Canonical topology and Stage-0 identity
# ---------------------------------------------------------------------------


def test_canonical_torus_5x5_identity_and_graph_invariants():
    topology = TORUS_5X5
    assert topology.topology_id == "torus-5x5-row-major-v1"
    assert topology.fingerprint == TORUS_5X5_TOPOLOGY_FINGERPRINT
    assert topology.point_count == 25
    assert topology.neighbors(0) == (20, 1, 5, 4)  # N/E/S/W with both wraps
    for point, neighbors in enumerate(topology.adjacency):
        assert len(neighbors) == 4
        assert len(set(neighbors)) == 4
        assert point not in neighbors
        for neighbor in neighbors:
            assert point in topology.neighbors(neighbor)


def test_canonical_state_uses_stage0_rules_fingerprint_and_komi():
    state = initial_state()
    assert state.komi == BASELINE_KOMI == 0.5
    assert state.rules_fingerprint == STAGE0_RULES_FINGERPRINT
    assert state.topology.fingerprint == TORUS_5X5_TOPOLOGY_FINGERPRINT
    assert state.superko_history == (state.board_key,)


def test_stage1_identity_matches_stage0_passport_data_without_importing_production_package():
    profile_path = Path(__file__).resolve().parents[1] / "configs/gocube/torus_golden_v1.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["profile_id"] == "gocube-torus-golden-graph-area-v1"
    assert profile["topology"]["topology_id"] == TORUS_5X5.topology_id
    assert profile["topology"]["fingerprint"] == TORUS_5X5_TOPOLOGY_FINGERPRINT
    assert profile["rules"]["profile_id"] == "graph-area-v1"
    assert profile["rules"]["fingerprint"] == STAGE0_RULES_FINGERPRINT
    assert profile["rules"]["komi"] == BASELINE_KOMI
    assert profile["rules"]["ko"] == "positional-superko"
    assert profile["rules"]["suicide"] == "forbidden"
    assert profile["rules"]["terminal"]["rule_terminal_condition"] == "two-consecutive-passes"
    assert profile["rules"]["scoring"]["method"] == "graph-area"


# ---------------------------------------------------------------------------
# Groups and liberties
# ---------------------------------------------------------------------------


def test_group_single_stone_and_multiple_liberties():
    state = state_from_stones(board(black=(12,)))
    group = group_from_board(state, 12)
    assert group == frozenset({12})
    assert liberties_from_board(state, group) == frozenset({7, 11, 13, 17})


def test_multistone_group_and_one_liberty():
    stones = board(black=(12, 13), white=(7, 11, 17, 8, 18, 14))
    state = state_from_stones(stones)
    group = group_from_board(state, 12)
    assert group == frozenset({12, 13})
    assert liberties_from_board(state, group) == frozenset()


def test_group_connects_through_torus_wrap():
    state = state_from_stones(board(black=(0, 4)))
    assert group_from_board(state, 0) == frozenset({0, 4})
    assert len(liberties_from_board(state, group_from_board(state, 0))) >= 1


def test_group_with_exactly_one_liberty():
    state = state_from_stones(board(black=(12,), white=(7, 11, 13)))
    assert liberties_from_board(state, group_from_board(state, 12)) == frozenset({17})


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------


def test_single_stone_capture():
    # White 12 has sole liberty 7; Black plays 7.
    state = state_from_stones(
        board(black=(11, 13, 17), white=(12, 2, 6, 8)), side_to_move=BLACK
    )
    transition = apply_action(state, 7)
    assert transition.captured == (12,)
    assert transition.after.stones[12] == EMPTY
    assert transition.after.stones[7] == BLACK


def test_multi_stone_capture():
    # White chain 12-13 has sole liberty 7.
    state = state_from_stones(
        board(black=(11, 17, 8, 14, 18), white=(12, 13)), side_to_move=BLACK
    )
    transition = apply_action(state, 7)
    assert transition.captured == (12, 13)


def test_simultaneous_capture_of_multiple_opponent_groups():
    # Separate White groups 7 and 13 each have sole liberty 12.
    state = state_from_stones(
        board(black=(2, 8, 6, 14, 18), white=(7, 13)), side_to_move=BLACK
    )
    transition = apply_action(state, 12)
    assert transition.captured == (7, 13)
    assert transition.after.stones[7] == EMPTY
    assert transition.after.stones[13] == EMPTY


def test_capture_across_torus_wrap():
    # White 0 has neighbors 20,1,5,4; the wrapped west liberty is 4.
    state = state_from_stones(
        board(black=(20, 1, 5), white=(0,)), side_to_move=BLACK
    )
    transition = apply_action(state, 4)
    assert transition.captured == (0,)
    assert transition.after.stones[0] == EMPTY


# ---------------------------------------------------------------------------
# Suicide: captures happen first
# ---------------------------------------------------------------------------


def test_pure_suicide_is_illegal_and_state_is_immutable():
    state = state_from_stones(
        board(white=(7, 11, 13, 17)), side_to_move=BLACK
    )
    assert_illegal(state, 12, IllegalMoveReason.SUICIDE)


def test_apparent_suicide_that_captures_is_legal():
    # Every adjacent White stone has its final liberty at 12. Black 12 appears
    # surrounded, but simultaneous captures create liberties before suicide is checked.
    state = state_from_stones(
        board(
            black=(2, 6, 8, 10, 14, 16, 18, 22),
            white=(7, 11, 13, 17),
        ),
        side_to_move=BLACK,
    )
    transition = apply_action(state, 12)
    assert transition.captured == (7, 11, 13, 17)
    assert liberties_from_board(
        transition.after, group_from_board(transition.after, 12)
    )


# ---------------------------------------------------------------------------
# Positional superko
# ---------------------------------------------------------------------------


def ko_fixture_before_capture():
    # White 12 has sole liberty 7. Black 7 captures it; then White 12 would
    # capture Black 7 and exactly recreate this board.
    return state_from_stones(
        board(black=(11, 13, 17), white=(12, 2, 6, 8)), side_to_move=BLACK
    )


def test_simple_ko_capture_shape():
    initial = ko_fixture_before_capture()
    captured = apply_action(initial, 7)
    assert captured.captured == (12,)
    assert captured.after.side_to_move == WHITE
    assert liberties_from_board(
        captured.after, group_from_board(captured.after, 7)
    ) == frozenset({12})


def test_immediate_ko_recapture_rejected_by_positional_superko():
    initial = ko_fixture_before_capture()
    after_capture = apply_action(initial, 7).after
    assert_illegal(after_capture, 12, IllegalMoveReason.SUPERKO)


def test_longer_history_repetition_rejected_even_not_immediate():
    initial = ko_fixture_before_capture()
    after_capture = apply_action(initial, 7).after
    unrelated = list(initial.board_key)
    unrelated[24] = int(BLACK)
    long_history = (initial.board_key, tuple(unrelated), after_capture.board_key)
    state = state_from_stones(
        after_capture.stones,
        side_to_move=WHITE,
        superko_history=long_history,
    )
    assert_illegal(state, 12, IllegalMoveReason.SUPERKO)


def test_side_to_move_does_not_change_board_superko_identity():
    initial = ko_fixture_before_capture()
    after_capture = apply_action(initial, 7).after
    same_board_other_side = state_from_stones(
        after_capture.stones,
        side_to_move=BLACK,
        superko_history=after_capture.superko_history,
    )
    # The forbidden recapture board is an exact arrangement match regardless of
    # which side-to-move field accompanied the earlier position.
    same_board_white = state_from_stones(
        same_board_other_side.stones,
        side_to_move=WHITE,
        superko_history=same_board_other_side.superko_history,
    )
    assert_illegal(same_board_white, 12, IllegalMoveReason.SUPERKO)


# ---------------------------------------------------------------------------
# PASS and terminal
# ---------------------------------------------------------------------------


def test_first_pass_is_nonterminal_and_does_not_extend_superko_history():
    state = initial_state()
    transition = apply_action(state, PASS)
    assert not transition.after.is_terminal
    assert transition.after.consecutive_passes == 1
    assert transition.after.stones == state.stones
    assert transition.after.superko_history == state.superko_history
    assert transition.after.side_to_move == WHITE


def test_stone_move_after_pass_resets_counter():
    after_pass = apply_action(initial_state(), PASS).after
    after_stone = apply_action(after_pass, 0).after
    assert after_stone.consecutive_passes == 0
    assert not after_stone.is_terminal


def test_pass_pass_is_only_formal_terminal_and_produces_result():
    state = initial_state()
    state = apply_action(state, PASS).after
    state = apply_action(state, PASS).after
    assert state.is_terminal
    result = result_from_terminal(state)
    assert result.terminal_reason == "DOUBLE_PASS"
    assert result.winner == Winner.WHITE
    assert result.margin_black == -0.5


def test_move_after_terminal_is_rejected_fail_closed():
    state = apply_action(apply_action(initial_state(), PASS).after, PASS).after
    assert_illegal(state, 0, IllegalMoveReason.MOVE_AFTER_TERMINAL)
    assert_illegal(state, PASS, IllegalMoveReason.MOVE_AFTER_TERMINAL)


def test_full_board_is_not_a_separate_terminal_rule():
    # A synthetically full board is still nonterminal until pass/pass. No
    # board-full or no-legal-move heuristic is a Golden rule terminal.
    state = state_from_stones(tuple(BLACK for _ in range(25)), consecutive_passes=0)
    assert not state.is_terminal
    assert legal_actions(state) == (PASS,)


