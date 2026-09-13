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
# Replay
# ---------------------------------------------------------------------------


def test_replay_legal_trace_is_deterministic_and_result_is_stable():
    actions = (0, 1, 2, 10, 6, 11, 21, 3, PASS, PASS)
    initial = initial_state()
    first = replay(initial, actions)
    second = replay(initial, actions)
    assert first == second
    assert first.ok
    assert first.final_state.is_terminal
    assert first.result is not None
    assert first.result.winner == Winner.BLACK
    assert first.result.margin_black == 1.5
    assert any(transition.captured == (1,) for transition in first.transitions)


def test_replay_stops_on_first_illegal_action_with_reason():
    initial = initial_state()
    report = replay(initial, (0, 0, 1, PASS, PASS))
    assert not report.ok
    assert report.illegal_action_index == 1
    assert report.illegal_action == 0
    assert report.illegal_reason == IllegalMoveReason.OCCUPIED
    assert len(report.transitions) == 1
    assert not report.final_state.is_terminal
    assert report.result is None


# ---------------------------------------------------------------------------
# Metamorphic graph isomorphism
# ---------------------------------------------------------------------------


def test_vertex_permutation_preserves_legality_capture_score_and_winner():
    old_to_new = tuple((old * 7 + 3) % 25 for old in range(25))
    topology2 = permute_topology(TORUS_5X5, old_to_new, topology_id="stage1-permutation-a")

    state1 = state_from_stones(
        board(black=(11, 13, 17), white=(12, 2, 6, 8)), side_to_move=BLACK
    )
    state2 = state_from_stones(
        permute_stones(state1.stones, old_to_new),
        side_to_move=BLACK,
        topology=topology2,
    )
    tr1 = apply_action(state1, 7)
    tr2 = apply_action(state2, old_to_new[7])
    assert tr2.captured == tuple(sorted(old_to_new[p] for p in tr1.captured))
    assert tr2.after.stones == permute_stones(tr1.after.stones, old_to_new)

    end1 = apply_action(apply_action(tr1.after, PASS).after, PASS).after
    end2 = apply_action(apply_action(tr2.after, PASS).after, PASS).after
    score1 = score_terminal(end1)
    score2 = score_terminal(end2)
    result1 = result_from_terminal(end1)
    result2 = result_from_terminal(end2)
    assert (score1.black_area, score1.white_area, score1.neutral_points) == (
        score2.black_area,
        score2.white_area,
        score2.neutral_points,
    )
    assert result1.winner == result2.winner
    assert result1.margin_black == result2.margin_black


# ---------------------------------------------------------------------------
# Correct colour swap including history and komi sign
# ---------------------------------------------------------------------------


def test_colour_swap_transforms_history_side_and_komi_and_negates_margin():
    original = terminal_state(board(black=(0, 1), white=(2,)), komi=0.5)
    swapped = state_from_stones(
        swap_stones(original.stones),
        side_to_move=WHITE if original.side_to_move == BLACK else BLACK,
        topology=original.topology,
        komi=-original.komi,
        superko_history=tuple(swap_board(position) for position in original.superko_history),
        consecutive_passes=2,
    )
    score1 = score_terminal(original)
    score2 = score_terminal(swapped)
    assert score2.black_area == score1.white_area
    assert score2.white_area == score1.black_area
    assert score2.margin_black == -score1.margin_black
    assert result_from_terminal(swapped).winner == (
        Winner.WHITE if result_from_terminal(original).winner == Winner.BLACK else Winner.BLACK
    )


# ---------------------------------------------------------------------------
# Tiny explicit research graph checks
# ---------------------------------------------------------------------------


def test_tiny_graph_legal_enumeration_transition_immutability_and_scorer_determinism():
    cycle4 = research_topology(
        ((1, 3), (0, 2), (1, 3), (2, 0)), topology_id="tiny-cycle4-stage1"
    )
    initial = initial_state(topology=cycle4)
    assert legal_actions(initial) == (0, 1, 2, 3, PASS)
    before = initial.state_key
    t1 = apply_action(initial, 0)
    t2 = apply_action(initial, 0)
    assert initial.state_key == before
    assert t1.after == t2.after
    terminal = apply_action(apply_action(t1.after, PASS).after, PASS).after
    assert score_terminal(terminal) == score_terminal(terminal)


# ---------------------------------------------------------------------------
# Dependency isolation and demonstration artifact
# ---------------------------------------------------------------------------


def test_golden_source_imports_no_legacy_engine_mcts_nn_or_parallel_stack():
    root = Path(__file__).resolve().parents[1] / "gocube_golden"
    forbidden_roots = {
        "alphazero",
        "torch",
        "pyximport",
        "multiprocessing",
        "Cython",
    }
    forbidden_names = {
        "katago_v3",
        "Benson",
        "Coach",
        "MCTS",
        "valid_moves",
        "play_action",
        "terminal_kind",
        "torus_topology",
        "make_topology",
    }
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_roots
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_roots
        source = path.read_text(encoding="utf-8")
        for name in forbidden_names:
            assert name not in source, f"forbidden legacy dependency marker {name!r} in {path}"


def test_demonstration_trace_is_human_readable_and_complete():
    text = demonstration_text()
    assert "captured': (1,)" in text
    assert "terminal reason: DOUBLE_PASS" in text
    assert "black area: 5" in text
    assert "white area: 3" in text
    assert "neutral points: 17" in text
    assert "komi: 0.5" in text
    assert "margin_black: 1.5" in text
    assert "winner: BLACK" in text
