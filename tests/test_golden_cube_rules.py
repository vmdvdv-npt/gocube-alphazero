from __future__ import annotations

import pytest

from gocube_golden.cube_topology import CUBE4_TOPOLOGY, CROSS_FACE_SEAM
from gocube_golden.cube_training import CUBE_WATCHDOG, cube_initial_state, cube_state_identity
from gocube_golden.rules import IllegalMoveError, IllegalMoveReason, apply_action, group_from_board, legal_actions, liberties_from_board
from gocube_golden.scoring import score_terminal
from gocube_golden.state import BLACK, EMPTY, WHITE, LegacyKomiError, research_state_from_stones


def state_with(*, black=(), white=(), side=BLACK, passes=0, history=None):
    stones = [EMPTY] * 96
    for point in black:
        stones[point] = BLACK
    for point in white:
        if stones[point] != EMPTY:
            raise AssertionError("fixture point collision")
        stones[point] = WHITE
    return research_state_from_stones(
        stones,
        side_to_move=side,
        topology=CUBE4_TOPOLOGY,
        consecutive_passes=passes,
        superko_history=history,
    )


def test_single_stone_capture_across_seam():
    target = 1  # front:0:1
    seam_neighbor = next(
        neighbor for neighbor, relation in zip(CUBE4_TOPOLOGY.adjacency[target], CUBE4_TOPOLOGY.relation_types[target])
        if relation == CROSS_FACE_SEAM
    )
    black = [neighbor for neighbor in CUBE4_TOPOLOGY.adjacency[target] if neighbor != seam_neighbor]
    state = state_with(black=black, white=(target,))
    transition = apply_action(state, seam_neighbor)
    assert transition.captured == (target,)
    assert transition.after.stones[target] == EMPTY


def test_multiface_group_and_liberties_follow_real_game_adjacency():
    left, right = CUBE4_TOPOLOGY.seams[0].point_pairs[1]
    state = state_with(black=(left, right), side=WHITE)
    group = group_from_board(state, left)
    liberties = liberties_from_board(state, group)
    assert group == frozenset((left, right))
    assert liberties
    assert CUBE4_TOPOLOGY.relation(left, right) == CROSS_FACE_SEAM


def test_suicide_involving_seam_neighbor_is_rejected():
    point = 0
    state = state_with(white=CUBE4_TOPOLOGY.adjacency[point])
    with pytest.raises(IllegalMoveError) as caught:
        apply_action(state, point)
    assert caught.value.reason == IllegalMoveReason.SUICIDE


def test_superko_rejects_a_seam_recreation():
    point = CUBE4_TOPOLOGY.seams[0].point_pairs[0][0]
    current = tuple([EMPTY] * 96)
    target = list(current)
    target[point] = BLACK
    state = state_with(black=(), white=(), side=BLACK, history=(current, tuple(target)))
    with pytest.raises(IllegalMoveError) as caught:
        apply_action(state, point)
    assert caught.value.reason == IllegalMoveReason.SUPERKO


def test_pass_and_double_pass_are_formal_terminal_and_pass_is_history_exempt():
    state = cube_initial_state()
    first = apply_action(state, "PASS").after
    second = apply_action(first, "PASS").after
    assert first.superko_history == state.superko_history
    assert second.is_terminal
    assert second.superko_history == state.superko_history
    with pytest.raises(IllegalMoveError, match="after formal"):
        apply_action(second, 0)


def test_graph_area_spans_faces_and_mixed_boundary_is_neutral():
    empty_pair = CUBE4_TOPOLOGY.seams[0].point_pairs[1]
    stones = [BLACK] * 96
    for point in empty_pair:
        stones[point] = EMPTY
    black_state = research_state_from_stones(stones, topology=CUBE4_TOPOLOGY, consecutive_passes=2)
    black_score = score_terminal(black_state)
    assert black_score.black_area == 96 and black_score.white_area == 0

    mixed = list(stones)
    boundary = next(neighbor for neighbor in CUBE4_TOPOLOGY.adjacency[empty_pair[0]] if neighbor not in empty_pair)
    mixed[boundary] = WHITE
    mixed_state = research_state_from_stones(mixed, topology=CUBE4_TOPOLOGY, consecutive_passes=2)
    mixed_score = score_terminal(mixed_state)
    assert mixed_score.neutral_points >= 2


def test_komi_is_hard_fail_closed():
    with pytest.raises(LegacyKomiError):
        cube_initial_state(komi=7.5)
    assert cube_initial_state().komi == 0.5


def test_cube_watchdog_is_20_times_points():
    assert CUBE_WATCHDOG == 1920
