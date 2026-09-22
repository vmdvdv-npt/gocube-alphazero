from __future__ import annotations

import copy
import json

import pytest

from gocube_golden.cube_family import (
    CROSS_FACE_SEAM,
    FACE_CORNER,
    cube_family_topology,
)
from gocube_golden.cube_game_contract_v2 import (
    FORMAL_DOUBLE_PASS,
    action_count_for_size,
    action_index_to_rules_action,
    classify_completion,
    completion_is_formal_result,
    concrete_game_fingerprint,
    concrete_game_identity,
    contract_fingerprint,
    load_contract,
    point_count_for_size,
    project_ownership,
    project_score,
    project_wdl,
    rules_action_to_action_index,
    validate_contract,
    validate_cube_size,
)
from gocube_golden.rules import (
    IllegalMoveError,
    IllegalMoveReason,
    apply_action,
    group_from_board,
    legal_actions,
    liberties_from_board,
)
from gocube_golden.scoring import Ownership, score_terminal
from gocube_golden.state import (
    BLACK,
    EMPTY,
    PASS,
    STAGE0_RULES_FINGERPRINT,
    WHITE,
    initial_state,
    research_state_from_stones,
    rules_fingerprint_for,
)


CUBE4 = cube_family_topology(4)


def state_with(*, black=(), white=(), side=BLACK, passes=0, history=None):
    stones = [EMPTY] * CUBE4.point_count
    for point in black:
        stones[point] = BLACK
    for point in white:
        if stones[point] != EMPTY:
            raise AssertionError("fixture point collision")
        stones[point] = WHITE
    return research_state_from_stones(
        stones,
        side_to_move=side,
        topology=CUBE4,
        consecutive_passes=passes,
        superko_history=history,
    )


@pytest.mark.parametrize("size", range(2, 8))
def test_family_counts_and_canonical_pass_index(size):
    contract = load_contract()
    point_count = 6 * size * size
    assert validate_cube_size(size) == size
    assert point_count_for_size(size) == point_count
    assert action_count_for_size(size) == point_count + 1
    profile = next(item for item in contract["profiles"] if item["size"] == size)
    assert profile == {
        "profile": f"cube{size}",
        "size": size,
        "point_count": point_count,
        "action_count": point_count + 1,
        "initial_komi": 0.5,
    }
    assert action_index_to_rules_action(point_count, size) == PASS
    assert rules_action_to_action_index(PASS, size) == point_count


@pytest.mark.parametrize("bad_size", [True, False, 1, 8, 2.0, 4.5, "4", None])
def test_size_validation_rejects_bool_fraction_coercion_and_out_of_range(bad_size):
    with pytest.raises(ValueError):
        validate_cube_size(bad_size)


def test_cube4_policy_size_97_is_not_a_game_length_limit():
    contract = load_contract()
    assert point_count_for_size(4) == 96
    assert action_count_for_size(4) == 97
    assert contract["actions"]["action_count_is_not_game_length_limit"] is True
    assert contract["results"]["game_length"]["limited_by_action_count"] is False
    assert contract["results"]["technical_termination"]["watchdog_limit"] is None
    assert contract["results"]["technical_termination"]["historical_1920_inherited"] is False


def test_cube4_geometry_has_degree_four_real_seams_and_corner_triangles():
    topology = CUBE4
    assert topology.point_count == 96
    assert all(len(neighbors) == 4 for neighbors in topology.adjacency)
    assert len(topology.seams) == 12
    assert sum(len(seam.point_pairs) for seam in topology.seams) == 48
    assert len(topology.physical_corners) == 8

    face_corners = {
        point.point_id for point in topology.points if point.geometry_class == FACE_CORNER
    }
    assert len(face_corners) == 24
    assert {point for triple in topology.physical_corners for point in triple} == face_corners

    for triple in topology.physical_corners:
        assert len(triple) == 3
        for left in triple:
            for right in triple:
                if left == right:
                    continue
                assert right in topology.adjacency[left]
                assert topology.relation(left, right) == CROSS_FACE_SEAM


def test_capture_across_seam_and_multiface_liberties_are_unique():
    left, right = CUBE4.seams[0].point_pairs[1]
    group_state = state_with(black=(left, right), side=WHITE)
    group = group_from_board(group_state, left)
    expected_liberties = (
        set(CUBE4.adjacency[left]) | set(CUBE4.adjacency[right])
    ) - {left, right}
    assert group == frozenset((left, right))
    assert liberties_from_board(group_state, group) == frozenset(expected_liberties)

    target = left
    seam_neighbor = right
    black = [neighbor for neighbor in CUBE4.adjacency[target] if neighbor != seam_neighbor]
    capture_state = state_with(black=black, white=(target,))
    transition = apply_action(capture_state, seam_neighbor)
    assert transition.captured == (target,)
    assert transition.after.stones[target] == EMPTY


def test_capture_is_resolved_before_suicide_check():
    point = CUBE4.physical_corners[0][0]
    white = set(CUBE4.adjacency[point])
    black = set()
    for neighbor in white:
        black.update(CUBE4.adjacency[neighbor])
    black.discard(point)
    black.difference_update(white)

    state = state_with(black=black, white=white, side=BLACK)
    for neighbor in white:
        opponent_group = group_from_board(state, neighbor)
        assert liberties_from_board(state, opponent_group) == frozenset((point,))

    transition = apply_action(state, point)
    assert set(transition.captured) == white
    assert transition.after.stones[point] == BLACK
    assert liberties_from_board(transition.after, group_from_board(transition.after, point))


def test_captured_point_is_not_reserved_and_can_be_reused_when_legal():
    target = 1
    capture_from = next(
        neighbor
        for neighbor, relation in zip(CUBE4.adjacency[target], CUBE4.relation_types[target])
        if relation == CROSS_FACE_SEAM
    )
    black = [neighbor for neighbor in CUBE4.adjacency[target] if neighbor != capture_from]
    captured = apply_action(state_with(black=black, white=(target,)), capture_from)
    assert captured.after.stones[target] == EMPTY

    later_stones = list(captured.after.stones)
    opened_neighbor = next(neighbor for neighbor in CUBE4.adjacency[target] if neighbor != capture_from)
    later_stones[opened_neighbor] = EMPTY
    later = research_state_from_stones(later_stones, side_to_move=WHITE, topology=CUBE4)
    assert target in legal_actions(later)
    assert apply_action(later, target).after.stones[target] == WHITE


def test_suicide_is_still_rejected_without_capture():
    point = CUBE4.physical_corners[0][0]
    state = state_with(white=CUBE4.adjacency[point])
    with pytest.raises(IllegalMoveError) as caught:
        apply_action(state, point)
    assert caught.value.reason == IllegalMoveReason.SUICIDE


def test_positional_superko_uses_only_stones_not_side_to_move():
    point = 5
    empty = tuple([EMPTY] * CUBE4.point_count)
    repeated = list(empty)
    repeated[point] = BLACK
    state = state_with(side=BLACK, history=(tuple(repeated), empty))
    assert all(
        isinstance(position, tuple) and len(position) == CUBE4.point_count
        for position in state.superko_history
    )
    with pytest.raises(IllegalMoveError) as caught:
        apply_action(state, point)
    assert caught.value.reason == IllegalMoveReason.SUPERKO


def test_pass_adapter_history_and_double_pass_terminal():
    state = initial_state(topology=CUBE4)
    pass_index = point_count_for_size(4)
    assert action_index_to_rules_action(pass_index, 4) == PASS

    first = apply_action(state, action_index_to_rules_action(pass_index, 4)).after
    assert first.superko_history == state.superko_history
    assert first.side_to_move == WHITE
    assert first.consecutive_passes == 1

    second = apply_action(first, action_index_to_rules_action(pass_index, 4)).after
    assert second.is_terminal
    assert second.superko_history == state.superko_history
    with pytest.raises(IllegalMoveError) as caught:
        apply_action(second, 0)
    assert caught.value.reason == IllegalMoveReason.MOVE_AFTER_TERMINAL


def test_graph_area_territory_mixed_boundary_and_empty_board_are_exact():
    empty_pair = CUBE4.seams[0].point_pairs[1]
    stones = [BLACK] * CUBE4.point_count
    for point in empty_pair:
        stones[point] = EMPTY
    black_terminal = research_state_from_stones(stones, topology=CUBE4, consecutive_passes=2)
    black_score = score_terminal(black_terminal)
    assert black_score.black_area == 96
    assert black_score.white_area == 0
    assert black_score.neutral_points == 0

    mixed = list(stones)
    boundary = next(neighbor for neighbor in CUBE4.adjacency[empty_pair[0]] if neighbor not in empty_pair)
    mixed[boundary] = WHITE
    mixed_terminal = research_state_from_stones(mixed, topology=CUBE4, consecutive_passes=2)
    mixed_score = score_terminal(mixed_terminal)
    assert mixed_score.neutral_points == len(empty_pair)
    assert all(mixed_score.ownership[point] == Ownership.NEUTRAL for point in empty_pair)

    empty_terminal = research_state_from_stones(
        [EMPTY] * CUBE4.point_count,
        topology=CUBE4,
        consecutive_passes=2,
    )
    empty_score = score_terminal(empty_terminal)
    assert empty_score.black_area == 0
    assert empty_score.white_area == 0
    assert empty_score.neutral_points == 96
    assert empty_score.margin_black == -0.5


def test_terminal_scoring_does_not_remove_dead_stones_or_add_prisoner_points():
    dead_point = 0
    stones = [WHITE] * CUBE4.point_count
    stones[dead_point] = BLACK
    terminal = research_state_from_stones(stones, topology=CUBE4, consecutive_passes=2)
    score = score_terminal(terminal)
    assert score.black_stones == 1
    assert score.white_stones == 95
    assert score.black_area == 1
    assert score.white_area == 95
    assert score.ownership[dead_point] == Ownership.BLACK
    assert not hasattr(score, "prisoner_points")


def test_targets_project_from_each_saved_positions_player_not_terminal_player():
    assert project_wdl("BLACK", "BLACK") == (1, 0, 0)
    assert project_wdl("BLACK", "WHITE") == (0, 0, 1)
    assert project_wdl("DRAW", "BLACK") == (0, 1, 0)
    absolute = (Ownership.BLACK, Ownership.WHITE, Ownership.NEUTRAL)
    assert project_ownership(absolute, "BLACK") == ("OWN", "OPPONENT", "NEUTRAL")
    assert project_ownership(absolute, "WHITE") == ("OPPONENT", "OWN", "NEUTRAL")
    assert project_score(3.5, "BLACK") == 3.5
    assert project_score(3.5, "WHITE") == -3.5


def test_technical_completion_is_not_a_draw_and_formal_double_pass_wins_boundary():
    technical = classify_completion(formal_double_pass=False, technical_reason="MOVE_LIMIT")
    assert technical == "TECHNICAL_MOVE_LIMIT"
    assert technical != "DRAW"
    assert not completion_is_formal_result(technical)

    boundary = classify_completion(formal_double_pass=True, technical_reason="MOVE_LIMIT")
    assert boundary == FORMAL_DOUBLE_PASS
    assert completion_is_formal_result(boundary)


def test_contract_fingerprint_is_canonical_and_semantic_drift_fails_closed():
    contract = load_contract()
    assert contract["contract_fingerprint"] == contract_fingerprint(contract)

    reordered = json.loads(json.dumps(contract, ensure_ascii=True, sort_keys=False))
    assert contract_fingerprint(reordered) == contract["contract_fingerprint"]

    mutated = copy.deepcopy(contract)
    mutated["rules"]["suicide"] = "allowed"
    with pytest.raises(ValueError):
        validate_contract(mutated)

    mutated["contract_fingerprint"] = contract_fingerprint(mutated)
    with pytest.raises(ValueError, match="suicide"):
        validate_contract(mutated)


def test_concrete_identity_distinguishes_family_size_topology_rules_and_komi():
    contract = load_contract()
    cube4_rules = rules_fingerprint_for(CUBE4, 0.5)
    identity = concrete_game_identity(
        contract,
        4,
        topology_id=CUBE4.topology_id,
        topology_fingerprint=CUBE4.fingerprint,
        rules_fingerprint=cube4_rules,
        komi=0.5,
    )
    baseline = concrete_game_fingerprint(identity)

    changed = dict(identity)
    changed["komi"] = 1.5
    assert concrete_game_fingerprint(changed) != baseline
    changed = dict(identity)
    changed["size"] = 5
    assert concrete_game_fingerprint(changed) != baseline
    assert identity["family_contract_fingerprint"] == contract["contract_fingerprint"]
    assert identity["rules_fingerprint"] != identity["family_contract_fingerprint"]


def test_existing_torus_identity_is_unchanged_and_v2_does_not_inherit_v1_watchdog():
    assert STAGE0_RULES_FINGERPRINT == (
        "sha256:8eac3337443a70893fa5ad359580f7ba92b18958e06f0d775c29f08791796842"
    )
    assert initial_state().rules_fingerprint == STAGE0_RULES_FINGERPRINT
    assert load_contract()["results"]["technical_termination"]["historical_1920_inherited"] is False
