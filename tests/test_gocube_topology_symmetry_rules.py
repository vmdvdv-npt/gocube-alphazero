from __future__ import annotations

from dataclasses import replace

import pytest

from alphazero.envs.gocube.core import cube_topology, torus_topology
from alphazero.envs.gocube.katago_v3 import (
    CLEANUP_1,
    SCORED,
    apply_v3_action,
    final_v3_score,
    terminal_from_state,
    v3_state_from_board,
    v3_valid_moves,
)


def _torus_permutations(size):
    result = []
    for transform in (
        lambda x, y: (x, y),
        lambda x, y: (-x, y),
        lambda x, y: (x, -y),
        lambda x, y: (-x, -y),
        lambda x, y: (y, x),
        lambda x, y: (-y, x),
        lambda x, y: (y, -x),
        lambda x, y: (-y, -x),
    ):
        for tx, ty in ((0, 0), (1, 2)):
            permutation = []
            for y in range(size):
                for x in range(size):
                    nx, ny = transform(x, y)
                    nx = (nx + tx) % size
                    ny = (ny + ty) % size
                    permutation.append(ny * size + nx)
            result.append(tuple(permutation))
    return tuple(dict.fromkeys(result))


def _first_graph_automorphism(topology):
    """Find a non-identity automorphism without making networkx a dependency."""

    adjacency = [set(neighbors) for neighbors in topology.neighbors_by_index]
    identity = tuple(range(topology.point_count))
    source_order = sorted(range(topology.point_count), key=lambda p: (-len(adjacency[p]), p))
    mapping = {}
    used = set()

    def compatible(source, target):
        if len(adjacency[source]) != len(adjacency[target]):
            return False
        return all(mapping.get(neighbor) in adjacency[target] for neighbor in adjacency[source] if neighbor in mapping)

    def search(position):
        if position == len(source_order):
            candidate = tuple(mapping[index] for index in range(topology.point_count))
            return candidate if candidate != identity else None
        source = source_order[position]
        candidates = range(topology.point_count)
        for target in candidates:
            if target in used or not compatible(source, target):
                continue
            mapping[source] = target
            used.add(target)
            found = search(position + 1)
            if found is not None:
                return found
            used.remove(target)
            del mapping[source]
        return None

    result = search(0)
    if result is None:
        raise AssertionError(f"No non-identity topology automorphism found for {topology.kind}")
    return result


def _permute_state(state, topology, permutation, *, phase=None, blocked=()):
    black = [permutation[index] for index, value in enumerate(state.board) if int(value) == 1]
    white = [permutation[index] for index, value in enumerate(state.board) if int(value) == 2]
    phase_value = phase or state.phase
    result = v3_state_from_board(
        topology,
        black=black,
        white=white,
        current_player=state.current_player,
        captures=state.captures,
        phase=phase_value,
        ko_recap_blocked=[permutation[index] for index in blocked],
    )
    if phase_value == SCORED:
        result = replace(result, terminal_kind=SCORED)
    return result


def _assert_symmetry(topology, permutation):
    assert sorted(permutation) == list(range(topology.point_count))
    for source, neighbors in enumerate(topology.neighbors_by_index):
        assert {permutation[index] for index in neighbors} == set(topology.neighbor_indices(permutation[source]))

    black = (0, 1, topology.point_count // 2)
    white = (topology.point_count - 1, topology.point_count // 2 + 1)
    state = v3_state_from_board(topology, black=black, white=white, current_player=0)
    action = next(index for index, value in enumerate(v3_valid_moves(state, topology)[:-1]) if value)
    transformed = _permute_state(state, topology, permutation)
    transformed_action = permutation[action]
    next_state = apply_v3_action(state, action, topology)
    transformed_next = apply_v3_action(transformed, transformed_action, topology)
    for source, target in enumerate(permutation):
        assert int(next_state.board[source]) == int(transformed_next.board[target])
    legal = v3_valid_moves(state, topology)
    transformed_legal = v3_valid_moves(transformed, topology)
    expected_legal = legal.copy()
    expected_legal[:-1] = 0
    for source, target in enumerate(permutation):
        expected_legal[target] = legal[source]
    assert (transformed_legal == expected_legal).all()

    scored = replace(
        v3_state_from_board(topology, black=black, white=white, phase=SCORED),
        terminal_kind=SCORED,
    )
    scored_transformed = _permute_state(scored, topology, permutation, phase=SCORED)
    score = final_v3_score(scored, topology, 0.5)[0]
    transformed_score = final_v3_score(scored_transformed, topology, 0.5)[0]
    assert score.black == transformed_score.black
    assert score.white == transformed_score.white
    assert score.winner == transformed_score.winner
    assert terminal_from_state(scored, topology, 0.5).terminal_kind == terminal_from_state(scored_transformed, topology, 0.5).terminal_kind

    blocked = (next(index for index, value in enumerate(state.board) if int(value) == 0),)
    cleanup = v3_state_from_board(topology, black=black, white=white, phase=CLEANUP_1, ko_recap_blocked=blocked)
    cleanup_transformed = _permute_state(cleanup, topology, permutation, phase=CLEANUP_1, blocked=blocked)
    assert set(cleanup_transformed.ko_recap_blocked) == {permutation[blocked[0]]}


@pytest.mark.parametrize("permutation", _torus_permutations(9), ids=lambda p: f"torus-{p[0]}-{p[1]}")
def test_torus_official_dihedral_translation_symmetries(permutation):
    _assert_symmetry(torus_topology(9), permutation)


def test_cube_graph_symmetry_is_bijective_and_preserves_rules():
    topology = cube_topology(2)
    _assert_symmetry(topology, _first_graph_automorphism(topology))


@pytest.mark.parametrize("kind,size", (("cube", 4), ("torus", 9)))
def test_topology_graph_invariants(kind, size):
    topology = cube_topology(size) if kind == "cube" else torus_topology(size)
    assert topology.pass_action == topology.point_count
    for point, neighbors in enumerate(topology.neighbors_by_index):
        assert point not in neighbors
        assert len(neighbors) == len(set(neighbors))
        assert all(0 <= neighbor < topology.point_count for neighbor in neighbors)
        assert all(point in topology.neighbor_indices(neighbor) for neighbor in neighbors)
