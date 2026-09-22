from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
from pathlib import Path

import pytest

from gocube_golden.cube_family import (
    CROSS_FACE_SEAM,
    CUBE_FACES,
    FACE_CORNER,
    FACE_EDGE,
    FACE_INTERIOR,
    SAME_FACE,
    cube_family_topology,
    cube_state_from_stones,
    deserialize_cube_state,
    family_fingerprints,
    geometry_class_counts,
    initial_cube_state,
    rotate_cube_state,
    rotate_rules_action,
    serialize_cube_state,
)
from gocube_golden.rules import (
    IllegalMoveError,
    IllegalMoveReason,
    apply_action,
    group_from_board,
    legal_actions,
    liberties_from_board,
    probe_action,
)
from gocube_golden.scoring import Ownership, score_terminal
from gocube_golden.state import BLACK, EMPTY, PASS, WHITE

CUBE4_V1_REFERENCE = json.loads(
    (Path(__file__).parent / "reference" / "cube4_topology_v1_reference.json").read_text(
        encoding="utf-8"
    )
)

ORACLE_SEAMS = (
    ("front", "top", "top", "bottom", False),
    ("front", "right", "right", "left", False),
    ("front", "bottom", "bottom", "top", False),
    ("front", "left", "left", "right", False),
    ("back", "top", "top", "top", True),
    ("back", "right", "left", "left", False),
    ("back", "bottom", "bottom", "bottom", True),
    ("back", "left", "right", "right", False),
    ("left", "top", "top", "left", False),
    ("left", "bottom", "bottom", "left", True),
    ("right", "top", "top", "right", True),
    ("right", "bottom", "bottom", "right", False),
)

EXPECTED_FINGERPRINTS = {
    2: ("sha256:5d29b9cc1ad0ad4a5cfa5eb8dfe67bece75355c5c93d216d12a4bac0efe46b5b", "sha256:9c7a6932d8950ddfad3f1dd2fa6af0b0debbe466aa48202c28adf691a825efee"),
    3: ("sha256:c3b365cef491c9705f5daa49c687d91efd80b29797f60267d3d6bdc157823d1b", "sha256:fc00c74e386217ced23c132c1b82cdd735367c1e5a3f066a45fc48020b8048a2"),
    4: ("sha256:1dcb3f9d574caf8d68f1f06f25531b64dc2db3629cf719e37f04ba68fa41c2bd", "sha256:0738ef45b686144f09eadac5314f46048d842e2b8dc58a85bd96b6ae9ab795f7"),
    5: ("sha256:2bdddb6b395d1e381ae373649628d787bc5e7188b293d0485ead78d911ebca93", "sha256:19ebe8b6c8fce9709b67dbd5f2784ce0d28fec1808e89fa527bfdcd0f4644515"),
    6: ("sha256:30f1ee043524e214d4a3a06354a9c03f54fdf0aa6d0d2fcacee0fa3d37263d1e", "sha256:0b9a44dffe144ca36d68b810f7c93771c0be7e9110d12ac410f3c35feb87bd78"),
    7: ("sha256:4658bef09a802f14ce9e83fbab455bf0840277fff4b75580a4dff99318a45c29", "sha256:9cb7f251edf44fbd793ff31a2db43071603ac85a0bbbc3f215b720aa402b4d9f"),
}


def _reference_fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _oracle_point(face, row, col, n):
    return CUBE_FACES.index(face) * n * n + row * n + col


def _oracle_edge(face, edge, index, n):
    last = n - 1
    if edge == "top": return _oracle_point(face, 0, index, n)
    if edge == "right": return _oracle_point(face, index, last, n)
    if edge == "bottom": return _oracle_point(face, last, index, n)
    if edge == "left": return _oracle_point(face, index, 0, n)
    raise AssertionError(edge)


def _oracle_pairs(spec, n):
    face_a, edge_a, face_b, edge_b, reverse = spec
    return tuple((_oracle_edge(face_a, edge_a, i, n), _oracle_edge(face_b, edge_b, n - 1 - i if reverse else i, n)) for i in range(n))


def _bfs(adjacency, sources):
    distances = [-1] * len(adjacency); queue = deque()
    for source in sources:
        if distances[source] == -1: distances[source] = 0; queue.append(source)
    while queue:
        point = queue.popleft()
        for neighbor in adjacency[point]:
            if distances[neighbor] == -1: distances[neighbor] = distances[point] + 1; queue.append(neighbor)
    return tuple(distances)


@pytest.mark.parametrize("n", range(2, 8))
def test_parameterized_topology_invariants_and_exact_metadata(n):
    t = cube_family_topology(n)
    assert t.point_count == 6 * n * n and t.action_count == t.point_count + 1 and t.pass_action == t.point_count
    assert len(t.seams) == 12 and all(len(s.point_pairs) == n for s in t.seams)
    assert len(t.physical_corners) == 8 and all(len(c) == 3 and len(set(c)) == 3 for c in t.physical_corners)
    assert all(len(row) == 4 and len(set(row)) == 4 for row in t.adjacency)
    assert all(p not in t.adjacency[p] for p in range(t.point_count))
    assert sum(rel == CROSS_FACE_SEAM for row in t.relation_types for rel in row) // 2 == 12 * n
    assert geometry_class_counts(t) == {FACE_CORNER: 24, FACE_EDGE: 24 * (n - 2), FACE_INTERIOR: 6 * (n - 2) ** 2}
    corner_sources, seam_sources = [], []
    for p in range(t.point_count):
        face, row, col = t.point_address(p); assert t.point_index(face, row, col) == p
        if row in (0, n - 1) and col in (0, n - 1): corner_sources.append(p)
        if row in (0, n - 1) or col in (0, n - 1): seam_sources.append(p)
        for neighbor in t.adjacency[p]:
            assert p in t.adjacency[neighbor] and t.relation(p, neighbor) == t.relation(neighbor, p)
    assert sorted(p for corner in t.physical_corners for p in corner) == sorted(corner_sources)
    assert tuple(p.corner_distance for p in t.points) == _bfs(t.adjacency, corner_sources)
    assert tuple(p.seam_distance for p in t.points) == _bfs(t.adjacency, seam_sources)
    assert all(p.num_cross_face_neighbors == sum(rel == CROSS_FACE_SEAM for rel in t.relation_types[p.point_id]) for p in t.points)


@pytest.mark.parametrize("n", range(2, 8))
def test_independent_explicit_seam_oracle(n):
    t = cube_family_topology(n)
    actual = {(s.face_a, s.edge_a, s.face_b, s.edge_b, s.reversed_index): s.point_pairs for s in t.seams}
    assert set(actual) == set(ORACLE_SEAMS)
    for spec in ORACLE_SEAMS:
        assert actual[spec] == _oracle_pairs(spec, n)
        assert all(t.relation(a, b) == CROSS_FACE_SEAM and t.relation(b, a) == CROSS_FACE_SEAM for a, b in actual[spec])


@pytest.mark.parametrize("n", range(2, 8))
def test_all_24_proper_rotations_preserve_full_geometry_contract(n):
    t = cube_family_topology(n); assert len(t.rotations) == 24
    corners = {frozenset(c) for c in t.physical_corners}; seams = {frozenset(frozenset(pair) for pair in s.point_pairs) for s in t.seams}
    sample = tuple(range(t.point_count))
    for rotation in t.rotations:
        perm = rotation.point_permutation; assert sorted(perm) == list(range(t.point_count)); assert rotation.action_permutation[t.pass_action] == t.pass_action
        values = rotation.permute_points(sample); assert all(values[perm[old]] == old for old in range(t.point_count))
        for face in CUBE_FACES:
            source = [p.point_id for p in t.points if p.face_id == face]; assert len({t.geometry(perm[p]).face_id for p in source}) == 1
        for p, neighbors in enumerate(t.adjacency):
            mapped = perm[p]; assert {perm[nbr] for nbr in neighbors} == set(t.adjacency[mapped])
            a, b = t.geometry(p), t.geometry(mapped); assert (a.geometry_class, a.corner_distance, a.seam_distance) == (b.geometry_class, b.corner_distance, b.seam_distance)
            assert all(t.relation(p, nbr) == t.relation(mapped, perm[nbr]) for nbr in neighbors)
        assert all(frozenset(perm[p] for p in c) in corners for c in t.physical_corners)
        for seam in t.seams:
            assert frozenset(frozenset((perm[a], perm[b])) for a, b in seam.point_pairs) in seams


def test_cube2_is_all_corner_and_rules_core_is_dynamic():
    t = cube_family_topology(2)
    assert all(p.geometry_class == FACE_CORNER and p.corner_distance == 0 and p.seam_distance == 0 and p.num_cross_face_neighbors == 2 for p in t.points)
    state = initial_cube_state(size=2); assert len(legal_actions(state)) == 25
    action = t.physical_corners[0][0]; moved = apply_action(state, action).after
    assert len(moved.stones) == 24 and moved.stones[action] == BLACK
    assert apply_action(apply_action(moved, PASS).after, PASS).after.is_terminal


def test_cube4_parameterized_geometry_matches_frozen_v1_reference():
    generated = cube_family_topology(4)
    components = {
        "point_order": list(generated.point_ids),
        "adjacency": [list(row) for row in generated.adjacency],
        "relation_types": [list(row) for row in generated.relation_types],
        "physical_corners": [list(corner) for corner in generated.physical_corners],
        "seams": [
            {
                "face_a": seam.face_a,
                "edge_a": seam.edge_a,
                "face_b": seam.face_b,
                "edge_b": seam.edge_b,
                "reversed_index": seam.reversed_index,
                "point_pairs": [list(pair) for pair in seam.point_pairs],
            }
            for seam in generated.seams
        ],
        "corner_distances": [point.corner_distance for point in generated.points],
    }
    assert {
        name: _reference_fingerprint(value) for name, value in components.items()
    } == CUBE4_V1_REFERENCE["component_fingerprints"]
    assert generated.topology_id != CUBE4_V1_REFERENCE["topology_id"]
    assert generated.fingerprint != CUBE4_V1_REFERENCE["topology_fingerprint"]


def test_family_fingerprints_are_stable_and_separate():
    assert dict(family_fingerprints()) == EXPECTED_FINGERPRINTS
    assert all(graph != geometry for graph, geometry in EXPECTED_FINGERPRINTS.values())


def test_state_round_trip_preserves_identity_and_superko():
    t = cube_family_topology(4); state = initial_cube_state(size=4)
    first, second = [p.point_id for p in t.points if p.geometry_class == FACE_INTERIOR][:2]
    state = apply_action(state, first).after; state = apply_action(state, second).after; state = apply_action(state, PASS).after
    payload = serialize_cube_state(state); restored = deserialize_cube_state(payload)
    assert restored.state_key == state.state_key and restored.topology is t and restored.superko_history == state.superko_history
    assert payload["topology_fingerprint"] == t.fingerprint and payload["geometry_fingerprint"] == t.geometry_fingerprint


@pytest.mark.parametrize("mutation", (
    lambda p: p.__setitem__("topology_id", "cube4x4x6-golden-topology-v1"),
    lambda p: p.__setitem__("topology_fingerprint", "sha256:" + "0" * 64),
    lambda p: p["rules_identity"].__setitem__("rules_fingerprint", "sha256:" + "f" * 64),
))
def test_state_deserialization_fails_closed_on_identity_mismatch(mutation):
    payload = copy.deepcopy(serialize_cube_state(initial_cube_state(size=4))); mutation(payload)
    with pytest.raises(ValueError): deserialize_cube_state(payload)


def _seam_capture_state():
    t = cube_family_topology(4); action, victim = t.seams[0].point_pairs[1]; stones = [EMPTY] * t.point_count; stones[victim] = WHITE
    for neighbor in t.neighbors(victim):
        if neighbor != action: stones[neighbor] = BLACK
    return t, cube_state_from_stones(stones, size=4, side_to_move=BLACK), action, victim


def test_rules_core_interior_capture_groups_liberties_suicide_superko_and_reuse():
    t = cube_family_topology(4); interior = next(p.point_id for p in t.points if p.geometry_class == FACE_INTERIOR)
    assert apply_action(initial_cube_state(size=4), interior).after.stones[interior] == BLACK
    t, state, action, victim = _seam_capture_state(); transition = apply_action(state, action)
    assert transition.captured == (victim,) and transition.after.stones[victim] == EMPTY
    stones = [EMPTY] * t.point_count; stones[action] = stones[victim] = BLACK; joined = cube_state_from_stones(stones, size=4, side_to_move=WHITE)
    assert {action, victim} <= group_from_board(joined, action)
    stones[victim] = EMPTY; liberty_state = cube_state_from_stones(stones, size=4, side_to_move=WHITE)
    assert victim in liberties_from_board(liberty_state, group_from_board(liberty_state, action))
    suicide_stones = [EMPTY] * t.point_count
    for neighbor in t.neighbors(action): suicide_stones[neighbor] = WHITE
    with pytest.raises(IllegalMoveError) as caught: apply_action(cube_state_from_stones(suicide_stones, size=4, side_to_move=BLACK), action)
    assert caught.value.reason == IllegalMoveReason.SUICIDE
    _, capture_state, capture_action, _ = _seam_capture_state(); result = probe_action(capture_state, capture_action).board_key
    superko = cube_state_from_stones(capture_state.stones, size=4, side_to_move=BLACK, superko_history=(capture_state.board_key, result))
    with pytest.raises(IllegalMoveError) as caught: apply_action(superko, capture_action)
    assert caught.value.reason == IllegalMoveReason.SUPERKO

    group_mate = next(n for n in t.neighbors(victim) if n != action and t.relation(victim, n) == SAME_FACE); white_group = {victim, group_mate}
    stones = [EMPTY] * t.point_count
    for p in white_group: stones[p] = WHITE
    external = {n for p in white_group for n in t.neighbors(p) if n not in white_group}; assert action in external
    for p in external - {action}: stones[p] = BLACK
    captured = apply_action(cube_state_from_stones(stones, size=4, side_to_move=BLACK), action)
    assert set(captured.captured) == white_group and victim in legal_actions(captured.after)
    assert apply_action(captured.after, victim).after.stones[victim] == WHITE


def test_pass_and_graph_area_scoring_across_faces():
    t = cube_family_topology(4); state = initial_cube_state(size=4); history = state.superko_history
    once = apply_action(state, PASS).after; twice = apply_action(once, PASS).after
    assert once.superko_history == history and twice.superko_history == history and twice.is_terminal
    a, b = t.seams[0].point_pairs[1]; stones = [BLACK] * t.point_count; stones[a] = stones[b] = EMPTY
    terminal = cube_state_from_stones(stones, size=4, side_to_move=BLACK, consecutive_passes=2); score = score_terminal(terminal)
    assert score.black_territory == 2 and score.ownership[a] == Ownership.BLACK and score.ownership[b] == Ownership.BLACK


def _quiet_fixture(t, action):
    forbidden = {action, *t.neighbors(action)}; candidates = [p for p in range(t.point_count) if p not in forbidden]; stones = [EMPTY] * t.point_count
    stones[candidates[0]], stones[candidates[-1]] = BLACK, WHITE
    return cube_state_from_stones(stones, size=t.size, side_to_move=BLACK)


@pytest.mark.parametrize("kind,rotation_index", (("face", 5), ("seam", 11), ("corner", 17)))
def test_rotation_commutes_with_rules_and_scoring(kind, rotation_index):
    t = cube_family_topology(4)
    if kind == "face": action = next(p.point_id for p in t.points if p.geometry_class == FACE_INTERIOR); state = _quiet_fixture(t, action)
    elif kind == "seam": t, state, action, _ = _seam_capture_state()
    else: action = t.physical_corners[0][0]; state = _quiet_fixture(t, action)
    rotation = t.rotations[rotation_index]; rotated_state = rotate_cube_state(state, rotation); rotated_action = rotation.point(action)
    assert {rotate_rules_action(t, rotation, a) for a in legal_actions(state)} == set(legal_actions(rotated_state))
    original, rotated = apply_action(state, action), apply_action(rotated_state, rotated_action); expected = rotate_cube_state(original.after, rotation)
    assert rotated.after.stones == expected.stones and set(rotated.captured) == {rotation.point(p) for p in original.captured}
    group_a, group_b = group_from_board(original.after, action), group_from_board(rotated.after, rotated_action)
    assert group_b == frozenset(rotation.point(p) for p in group_a)
    assert liberties_from_board(rotated.after, group_b) == frozenset(rotation.point(p) for p in liberties_from_board(original.after, group_a))
    terminal_a = apply_action(apply_action(original.after, PASS).after, PASS).after; terminal_b = apply_action(apply_action(rotated.after, PASS).after, PASS).after
    score_a, score_b = score_terminal(terminal_a), score_terminal(terminal_b)
    assert (score_a.black_area, score_a.white_area, score_a.margin_black) == (score_b.black_area, score_b.white_area, score_b.margin_black)
    assert rotation.permute_points(score_a.ownership) == score_b.ownership
