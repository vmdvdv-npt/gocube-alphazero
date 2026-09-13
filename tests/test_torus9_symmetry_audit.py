from __future__ import annotations

import numpy as np

from tools.torus9_symmetry_audit import (
    build_transformations,
    check_topology,
    transform_policy,
    transform_point_axis,
    transform_state,
)


def test_torus9_group_has_648_unique_adjacency_automorphisms():
    transformations = build_transformations()
    assert len(transformations) == 648
    assert check_topology(transformations)["passed"] is True


def test_state_transform_moves_every_superko_board_and_preserves_rule_scalars():
    from gocube_golden.rules import apply_action, prepare_legal_actions
    from gocube_golden.state import PASS, initial_state
    from gocube_golden.topology import TORUS_9X9

    state = initial_state(topology=TORUS_9X9, komi=0.5)
    for action in (0, 1, 2, 3, 4, 5, 6):
        state = apply_action(state, action).after
    transform = next(
        item for item in build_transformations()
        if item["d4"] == "rotation_90" and item["dx"] == 2 and item["dy"] == 3
    )
    transformed = transform_state(state, transform["permutation"])

    assert transformed.side_to_move == state.side_to_move
    assert transformed.consecutive_passes == state.consecutive_passes
    assert transformed.komi == state.komi
    assert len(transformed.superko_history) == len(state.superko_history)
    assert transformed.superko_history[-1] == transformed.board_key

    original_mask = prepare_legal_actions(state).action_mask
    transformed_mask = prepare_legal_actions(transformed).action_mask
    inverse = tuple(next(i for i, value in enumerate(transform["permutation"]) if value == target) for target in range(81))
    assert transformed_mask[:81] == tuple(original_mask[inverse[index]] for index in range(81))
    assert transformed_mask[81] == original_mask[81]

    terminal = apply_action(apply_action(state, PASS).after, PASS).after
    transformed_terminal = apply_action(apply_action(transformed, PASS).after, PASS).after
    assert transform_state(terminal, transform["permutation"]) == transformed_terminal


def test_point_and_action_targets_use_the_same_source_to_destination_convention():
    transformation = next(
        item for item in build_transformations()
        if item["d4"] == "reflection_then_rotation_90" and item["dx"] == 1 and item["dy"] == 2
    )
    permutation = transformation["permutation"]
    points = list(range(81))
    ownership = [[point, -point] for point in points]
    policy = points + [123]
    transformed_ownership = transform_point_axis(ownership, permutation, axis=0)
    transformed_policy = transform_policy(policy, permutation)
    for source, destination in enumerate(permutation):
        assert np.array_equal(transformed_ownership[destination], ownership[source])
        assert transformed_policy[destination] == policy[source]
    assert transformed_policy[81] == 123
