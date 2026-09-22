from __future__ import annotations

import copy

import pytest
import torch

from gocube_golden.cube_family import (
    FACE_CORNER,
    FACE_EDGE,
    FACE_INTERIOR,
    cube_family_topology,
    cube_state_from_stones,
    deserialize_cube_state,
    initial_cube_state,
    rotate_cube_state,
    serialize_cube_state,
)
from gocube_golden.cube_observation_v2 import (
    CHANNELS,
    CHANNEL_COUNT,
    CHANNEL_INDEX,
    CORNER_DISTANCE_FAMILY_SCALE,
    HISTORY_DEPTH,
    SCHEMA_FINGERPRINT,
    SCHEMA_ID,
    SEAM_DISTANCE_FAMILY_SCALE,
    advance_cube_observation_context,
    build_cube_observation,
    concrete_observation_identity,
    cube_observation_schema,
    deserialize_cube_observation_context,
    initial_cube_observation_context,
    load_cube_observation_schema,
    make_cube_observation_context,
    rotate_cube_observation_context,
    serialize_cube_observation_context,
    validate_concrete_observation_identity,
    validate_cube_observation_schema,
    write_cube_observation,
)
from gocube_golden.rules import apply_action, legal_actions
from gocube_golden.state import BLACK, EMPTY, PASS, WHITE

EXPECTED_CHANNELS = (
    "own_stones",
    "opponent_stones",
    "side_to_move_is_black",
    "previous_action_was_pass",
    "legal_point_mask",
    "komi_stm_normalized",
    "previous_move_point",
    "own_liberties_1",
    "own_liberties_2",
    "own_liberties_3_plus",
    "opponent_liberties_1",
    "opponent_liberties_2",
    "opponent_liberties_3_plus",
    "history_1_own",
    "history_1_opponent",
    "history_2_own",
    "history_2_opponent",
    "history_3_own",
    "history_3_opponent",
    "history_4_own",
    "history_4_opponent",
    "is_face_interior",
    "is_face_edge",
    "is_face_corner",
    "corner_distance_family_scaled",
    "seam_distance_family_scaled",
    "corner_distance_topology_relative",
    "seam_distance_topology_relative",
    "cross_face_neighbor_count_scaled",
    "face_size_scaled",
)
EXPECTED_SCHEMA_FINGERPRINT = "sha256:c56c878e1acc9847647666426c370a4cda56473b3c4311f310bae225d9ae12e8"
EXPECTED_CONCRETE_FINGERPRINTS = {
    2: "sha256:f1f0b3c74938b1ee041c84e5aed67530a536839499407d9be0d6ddf920c08a0d",
    3: "sha256:7925c153c7348ba6cfc75b7e884f8b55338d6799358af3e1ae35b58a7b4359ed",
    4: "sha256:943511e083779d8ccd9b1cf309263e5e8149f76eee782692afbb5400939cceaf",
    5: "sha256:99d196c950ae7c6690bb66f217a96338848b8b1228232cdf359e017133b0760c",
    6: "sha256:e0626d1adf09233ec79a2d0d71fcede0ab7fe8a0177ddfd9bc47bf75c59e3fe5",
    7: "sha256:d1ba372b4c8d35f33d4d638df9adc5ca693757dcce5ed814ea244a1798d1d0eb",
}


def _advance(state, context, rules_action):
    topology = state.topology
    transition = apply_action(state, rules_action)
    canonical = topology.pass_action if rules_action == PASS else int(rules_action)
    return transition.after, advance_cube_observation_context(context, canonical, transition.after)


def _first_legal_point(state):
    return next(
        action
        for action in legal_actions(state)
        if isinstance(action, int) and not isinstance(action, bool)
    )


def _play_points(state, context, count):
    for _ in range(count):
        action = _first_legal_point(state)
        state, context = _advance(state, context, action)
    return state, context


def _single_stone_liberty_state(n, *, stone_color, side_to_move, liberty_count):
    topology = cube_family_topology(n)
    point = 0
    stones = [EMPTY] * topology.point_count
    stones[point] = stone_color
    neighbors = list(topology.neighbors(point))
    for neighbor in neighbors[liberty_count:]:
        stones[neighbor] = WHITE if stone_color == BLACK else BLACK
    state = cube_state_from_stones(stones, size=n, side_to_move=side_to_move)
    return state, point


def _seam_capture_state(n):
    topology = cube_family_topology(n)
    action, victim = topology.seams[0].point_pairs[min(1, n - 1)]
    stones = [EMPTY] * topology.point_count
    stones[victim] = WHITE
    for neighbor in topology.neighbors(victim):
        if neighbor != action:
            stones[neighbor] = BLACK
    return cube_state_from_stones(stones, size=n, side_to_move=BLACK), action, victim


def _reusable_capture_state(n):
    topology = cube_family_topology(n)
    action, victim = topology.seams[0].point_pairs[min(1, n - 1)]
    group_mate = next(neighbor for neighbor in topology.neighbors(victim) if neighbor != action)
    white_group = {victim, group_mate}
    stones = [EMPTY] * topology.point_count
    for point in white_group:
        stones[point] = WHITE
    external = {
        neighbor
        for point in white_group
        for neighbor in topology.neighbors(point)
        if neighbor not in white_group
    }
    for point in external - {action}:
        stones[point] = BLACK
    state = cube_state_from_stones(stones, size=n, side_to_move=BLACK)
    return state, action, victim, white_group


def _rotation_scenarios(n):
    topology = cube_family_topology(n)

    empty = initial_cube_state(size=n)
    empty_context = initial_cube_observation_context(empty)

    ordinary_state, ordinary_context = _advance(
        empty, empty_context, _first_legal_point(empty)
    )

    seam_a, seam_b = topology.seams[0].point_pairs[0]
    seam_stones = [EMPTY] * topology.point_count
    seam_stones[seam_a] = seam_stones[seam_b] = BLACK
    seam_state = cube_state_from_stones(seam_stones, size=n, side_to_move=WHITE)
    seam_context = make_cube_observation_context(seam_state)

    capture_before, capture_action, _ = _seam_capture_state(n)
    capture_context_before = make_cube_observation_context(capture_before)
    capture_after, capture_context_after = _advance(
        capture_before, capture_context_before, capture_action
    )

    pass_after, pass_context = _advance(ordinary_state, ordinary_context, PASS)

    history_initial = initial_cube_state(size=n)
    history_state, history_context = _play_points(
        history_initial,
        initial_cube_observation_context(history_initial),
        5,
    )

    return (
        (empty, empty_context),
        (ordinary_state, ordinary_context),
        (seam_state, seam_context),
        (capture_after, capture_context_after),
        (pass_after, pass_context),
        (history_state, history_context),
    )


def test_machine_schema_exactly_matches_writer_contract():
    loaded = load_cube_observation_schema()
    assert tuple(loaded["channel_order"]) == EXPECTED_CHANNELS == CHANNELS
    assert loaded["channel_count"] == CHANNEL_COUNT == len(CHANNELS) == 30
    assert loaded["history_depth"] == HISTORY_DEPTH == 4
    assert loaded["dtype"] == "float32"
    assert loaded["layout"] == "[channels,points]"
    assert loaded["schema_id"] == SCHEMA_ID
    assert loaded["schema_fingerprint"] == SCHEMA_FINGERPRINT == EXPECTED_SCHEMA_FINGERPRINT
    assert loaded["normalization"]["corner_distance_family_scale"] == CORNER_DISTANCE_FAMILY_SCALE == 6.0
    assert loaded["normalization"]["seam_distance_family_scale"] == SEAM_DISTANCE_FAMILY_SCALE == 3.0


@pytest.mark.parametrize("n", range(2, 8))
def test_empty_observation_shape_dtype_finite_geometry_and_concrete_identity(n):
    topology = cube_family_topology(n)
    state = initial_cube_state(size=n)
    context = initial_cube_observation_context(state)
    observation = build_cube_observation(state, context)

    assert tuple(observation.shape) == (30, 6 * n * n)
    assert observation.dtype == torch.float32
    assert bool(torch.isfinite(observation).all())
    assert not bool(observation[CHANNEL_INDEX["own_stones"]].any())
    assert not bool(observation[CHANNEL_INDEX["opponent_stones"]].any())
    assert bool(torch.all(observation[CHANNEL_INDEX["side_to_move_is_black"]] == 1.0))
    assert not bool(observation[CHANNEL_INDEX["previous_action_was_pass"]].any())
    assert not bool(observation[CHANNEL_INDEX["previous_move_point"]].any())
    assert bool(torch.all(observation[CHANNEL_INDEX["legal_point_mask"]] == 1.0))
    expected_komi = -0.5 / (topology.point_count + 0.5)
    assert torch.allclose(
        observation[CHANNEL_INDEX["komi_stm_normalized"]],
        torch.full((topology.point_count,), expected_komi, dtype=torch.float32),
    )
    for index in range(1, 5):
        assert not bool(observation[CHANNEL_INDEX[f"history_{index}_own"]].any())
        assert not bool(observation[CHANNEL_INDEX[f"history_{index}_opponent"]].any())

    geometry_sum = (
        observation[CHANNEL_INDEX["is_face_interior"]]
        + observation[CHANNEL_INDEX["is_face_edge"]]
        + observation[CHANNEL_INDEX["is_face_corner"]]
    )
    assert bool(torch.all(geometry_sum == 1.0))
    max_corner = max(point.corner_distance for point in topology.points)
    max_seam = max(point.seam_distance for point in topology.points)
    for point in topology.points:
        p = point.point_id
        expected_class = {
            FACE_INTERIOR: "is_face_interior",
            FACE_EDGE: "is_face_edge",
            FACE_CORNER: "is_face_corner",
        }[point.geometry_class]
        assert observation[CHANNEL_INDEX[expected_class], p].item() == 1.0
        assert observation[CHANNEL_INDEX["corner_distance_family_scaled"], p].item() == pytest.approx(
            point.corner_distance / 6.0
        )
        assert observation[CHANNEL_INDEX["seam_distance_family_scaled"], p].item() == pytest.approx(
            point.seam_distance / 3.0
        )
        corner_relative = 0.0 if max_corner == 0 else point.corner_distance / max_corner
        seam_relative = 0.0 if max_seam == 0 else point.seam_distance / max_seam
        assert observation[CHANNEL_INDEX["corner_distance_topology_relative"], p].item() == pytest.approx(
            corner_relative
        )
        assert observation[CHANNEL_INDEX["seam_distance_topology_relative"], p].item() == pytest.approx(
            seam_relative
        )
        assert observation[CHANNEL_INDEX["cross_face_neighbor_count_scaled"], p].item() == pytest.approx(
            point.num_cross_face_neighbors / 2.0
        )
        assert observation[CHANNEL_INDEX["face_size_scaled"], p].item() == pytest.approx(n / 7.0)

    identity = concrete_observation_identity(topology)
    assert identity["concrete_observation_fingerprint"] == EXPECTED_CONCRETE_FINGERPRINTS[n]


def test_cube2_relative_distance_zero_normalization_has_no_nan_or_inf():
    state = initial_cube_state(size=2)
    observation = build_cube_observation(state, initial_cube_observation_context(state))
    assert bool(torch.all(observation[CHANNEL_INDEX["corner_distance_topology_relative"]] == 0.0))
    assert bool(torch.all(observation[CHANNEL_INDEX["seam_distance_topology_relative"]] == 0.0))
    assert bool(torch.isfinite(observation).all())


@pytest.mark.parametrize("n", range(2, 8))
def test_current_player_perspective_previous_point_pass_history_and_komi(n):
    state = initial_cube_state(size=n, komi=1.5)
    context = initial_cube_observation_context(state)
    first = _first_legal_point(state)
    state, context = _advance(state, context, first)

    point_observation = build_cube_observation(state, context)
    assert state.side_to_move == WHITE
    assert point_observation[CHANNEL_INDEX["opponent_stones"], first].item() == 1.0
    assert point_observation[CHANNEL_INDEX["own_stones"], first].item() == 0.0
    assert point_observation[CHANNEL_INDEX["previous_move_point"], first].item() == 1.0
    assert point_observation[CHANNEL_INDEX["previous_move_point"]].sum().item() == 1.0
    assert not bool(point_observation[CHANNEL_INDEX["previous_action_was_pass"]].any())
    assert bool(torch.all(point_observation[CHANNEL_INDEX["side_to_move_is_black"]] == 0.0))
    expected_white_komi = 1.5 / (state.topology.point_count + 1.5)
    assert point_observation[CHANNEL_INDEX["komi_stm_normalized"], 0].item() == pytest.approx(
        expected_white_komi
    )

    state, context = _advance(state, context, PASS)
    pass_observation = build_cube_observation(state, context)
    assert state.side_to_move == BLACK
    assert bool(torch.all(pass_observation[CHANNEL_INDEX["previous_action_was_pass"]] == 1.0))
    assert not bool(pass_observation[CHANNEL_INDEX["previous_move_point"]].any())
    assert pass_observation[CHANNEL_INDEX["history_1_own"], first].item() == 1.0
    assert context.previous_boards[0] == context.current_board
    expected_black_komi = -1.5 / (state.topology.point_count + 1.5)
    assert pass_observation[CHANNEL_INDEX["komi_stm_normalized"], 0].item() == pytest.approx(
        expected_black_komi
    )


@pytest.mark.parametrize("n", range(2, 8))
def test_four_step_real_history_is_bounded_and_uses_current_player_perspective(n):
    state = initial_cube_state(size=n)
    context = initial_cube_observation_context(state)
    states_before = []
    for _ in range(5):
        states_before.append(tuple(int(stone) for stone in state.stones))
        state, context = _advance(state, context, _first_legal_point(state))
    assert context.previous_boards == tuple(reversed(states_before[-4:]))

    observation = build_cube_observation(state, context)
    own = int(state.side_to_move)
    other = int(WHITE if state.side_to_move == BLACK else BLACK)
    for history_index, board in enumerate(context.previous_boards, start=1):
        for point, stone in enumerate(board):
            assert observation[CHANNEL_INDEX[f"history_{history_index}_own"], point].item() == float(
                stone == own
            )
            assert observation[CHANNEL_INDEX[f"history_{history_index}_opponent"], point].item() == float(
                stone == other
            )


@pytest.mark.parametrize("n", range(2, 8))
@pytest.mark.parametrize("liberty_count,channel_suffix", ((1, "1"), (2, "2"), (4, "3_plus")))
def test_own_and_opponent_liberty_buckets_are_group_features(n, liberty_count, channel_suffix):
    own_state, own_point = _single_stone_liberty_state(
        n, stone_color=BLACK, side_to_move=BLACK, liberty_count=liberty_count
    )
    own_observation = build_cube_observation(own_state, make_cube_observation_context(own_state))
    assert own_observation[CHANNEL_INDEX[f"own_liberties_{channel_suffix}"], own_point].item() == 1.0

    opponent_state, opponent_point = _single_stone_liberty_state(
        n, stone_color=WHITE, side_to_move=BLACK, liberty_count=liberty_count
    )
    opponent_observation = build_cube_observation(
        opponent_state, make_cube_observation_context(opponent_state)
    )
    assert opponent_observation[
        CHANNEL_INDEX[f"opponent_liberties_{channel_suffix}"], opponent_point
    ].item() == 1.0


@pytest.mark.parametrize("n", range(2, 8))
def test_group_spanning_seam_is_counted_once_and_filled_across_whole_group(n):
    topology = cube_family_topology(n)
    a, b = topology.seams[0].point_pairs[0]
    stones = [EMPTY] * topology.point_count
    stones[a] = stones[b] = BLACK
    state = cube_state_from_stones(stones, size=n, side_to_move=BLACK)
    observation = build_cube_observation(state, make_cube_observation_context(state))
    active = [
        name
        for name in ("own_liberties_1", "own_liberties_2", "own_liberties_3_plus")
        if observation[CHANNEL_INDEX[name], a].item() == 1.0
    ]
    assert len(active) == 1
    assert observation[CHANNEL_INDEX[active[0]], b].item() == 1.0


@pytest.mark.parametrize("n", range(2, 8))
def test_capture_updates_legal_mask_and_captured_point_can_become_legal_again(n):
    state, action, victim = _seam_capture_state(n)
    before_context = make_cube_observation_context(state)
    before = build_cube_observation(state, before_context)
    assert before[CHANNEL_INDEX["legal_point_mask"], action].item() == 1.0
    state, context = _advance(state, before_context, action)
    assert state.stones[victim] == EMPTY
    assert context.previous_action == action

    reusable, reuse_action, reuse_victim, white_group = _reusable_capture_state(n)
    reusable_context = make_cube_observation_context(reusable)
    transition = apply_action(reusable, reuse_action)
    assert white_group <= set(transition.captured)
    after_context = advance_cube_observation_context(
        reusable_context, reuse_action, transition.after
    )
    after = build_cube_observation(transition.after, after_context)
    assert after[CHANNEL_INDEX["legal_point_mask"], reuse_victim].item() == 1.0


@pytest.mark.parametrize("n", range(2, 8))
def test_state_and_observation_context_round_trip_reproduces_exact_tensor(n):
    state = initial_cube_state(size=n)
    context = initial_cube_observation_context(state)
    state, context = _play_points(state, context, 3)
    state, context = _advance(state, context, PASS)
    expected = build_cube_observation(state, context)

    restored_state = deserialize_cube_state(serialize_cube_state(state))
    restored_context = deserialize_cube_observation_context(
        serialize_cube_observation_context(context)
    )
    actual = build_cube_observation(restored_state, restored_context)
    assert torch.equal(actual, expected)
    assert restored_context.previous_boards == context.previous_boards
    assert restored_context.previous_action == context.previous_action


def test_schema_and_concrete_identity_fail_closed_for_historical_and_mismatched_inputs():
    for mutation in (
        lambda value: value.__setitem__("schema_id", "gocube-cube4-golden-observation-v1"),
        lambda value: value.__setitem__("channel_count", 15),
    ):
        schema = copy.deepcopy(cube_observation_schema())
        mutation(schema)
        with pytest.raises(ValueError):
            validate_cube_observation_schema(schema)

    topology = cube_family_topology(4)
    base = concrete_observation_identity(topology)
    mutations = (
        lambda value: value.__setitem__("schema_id", "gocube-cube4-golden-observation-v1"),
        lambda value: value.__setitem__("geometry_fingerprint", "sha256:" + "0" * 64),
        lambda value: value.__setitem__("game_graph_fingerprint", "sha256:" + "1" * 64),
        lambda value: value.__setitem__("size", 5),
    )
    for mutation in mutations:
        identity = copy.deepcopy(base)
        mutation(identity)
        with pytest.raises(ValueError):
            validate_concrete_observation_identity(identity, topology)

    with pytest.raises(ValueError):
        validate_concrete_observation_identity(base, cube_family_topology(5))

    state = initial_cube_state(size=4)
    payload = serialize_cube_observation_context(initial_cube_observation_context(state))
    payload["observation_schema_id"] = "gocube-cube4-golden-observation-v1"
    with pytest.raises(ValueError):
        deserialize_cube_observation_context(payload)


@pytest.mark.parametrize("n", range(2, 8))
def test_all_24_rotations_are_observation_equivariant_for_required_scenarios(n):
    topology = cube_family_topology(n)
    scenarios = _rotation_scenarios(n)
    assert len(topology.rotations) == 24
    assert len(scenarios) == 6
    for state, context in scenarios:
        base = build_cube_observation(state, context)
        for rotation in topology.rotations:
            rotated_state = rotate_cube_state(state, rotation)
            rotated_context = rotate_cube_observation_context(context, rotation)
            actual = build_cube_observation(rotated_state, rotated_context)
            expected = torch.empty_like(base)
            expected[:, torch.as_tensor(rotation.point_permutation, dtype=torch.long)] = base
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("n", range(2, 8))
def test_allocating_and_preallocated_writer_are_identical_and_layout_fails_closed(n):
    state = initial_cube_state(size=n)
    context = initial_cube_observation_context(state)
    built = build_cube_observation(state, context, cube_observation_schema())

    destination = torch.empty_like(built)
    returned = write_cube_observation(destination, state, context, cube_observation_schema())
    assert returned.data_ptr() == destination.data_ptr()
    assert torch.equal(destination, built)

    with pytest.raises(ValueError, match="shape"):
        write_cube_observation(
            torch.empty((CHANNEL_COUNT - 1, state.topology.point_count), dtype=torch.float32),
            state,
            context,
        )
    with pytest.raises(ValueError, match="dtype"):
        write_cube_observation(
            torch.empty((CHANNEL_COUNT, state.topology.point_count), dtype=torch.float64),
            state,
            context,
        )
    non_contiguous = torch.empty(
        (state.topology.point_count, CHANNEL_COUNT), dtype=torch.float32
    ).t()
    assert not non_contiguous.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        write_cube_observation(non_contiguous, state, context)


@pytest.mark.parametrize("n", range(2, 8))
def test_terminal_double_pass_fails_closed_for_move_selection_writer(n):
    state = initial_cube_state(size=n)
    context = initial_cube_observation_context(state)
    state, context = _advance(state, context, PASS)
    state, context = _advance(state, context, PASS)
    assert state.is_terminal
    with pytest.raises(ValueError, match="Terminal Cube states"):
        build_cube_observation(state, context)
