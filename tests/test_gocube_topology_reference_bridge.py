from __future__ import annotations

import pytest

from alphazero.envs.gocube.core import cube_topology, torus_topology
from alphazero.envs.gocube.katago_v3 import apply_v3_action, v3_state_from_board
from gocube_reference_topology import rectangular_test_topology
from katago_reference_runner import KatagoOracleProcess


def _star_case(topology, center):
    neighbors = topology.neighbor_indices(center)
    assert len(neighbors) == 4
    # Three black arms, one empty liberty, and a white center.  The same
    # semantic star is used in a planar KataGo fixture and across a seam/wrap.
    black = tuple(neighbors[:3])
    move = int(neighbors[3])
    state = v3_state_from_board(topology, black=black, white=(center,), current_player=0)
    return state, move, center


def _reference_star_outcome():
    topology = rectangular_test_topology(5, 5)
    center = 2 + 2 * 5
    neighbors = topology.neighbor_indices(center)
    setup = {
        "black": [[point % 5, point // 5] for point in neighbors[:3]],
        "white": [[2, 2]],
        "next_player": "B",
    }
    move = [neighbors[3] % 5, neighbors[3] // 5]
    with KatagoOracleProcess(x_size=5, y_size=5, komi=0.5) as oracle:
        oracle.setup(setup)
        response = oracle.play(move)
    assert response["ok"] is True
    return response["snapshot"] if "snapshot" in response else response


def _assert_star_matches_reference(topology, center):
    expected = _reference_star_outcome()
    state, move, captured_point = _star_case(topology, center)
    next_state = apply_v3_action(state, move, topology)
    assert int(next_state.board[captured_point]) == 0
    assert int(next_state.captures[0]) == 1
    assert expected["board"][2 + 2 * 5] == 0
    assert expected["captures"] == {"black": 0, "white": 1}


@pytest.mark.parametrize(
    "name,topology,center",
    [
        ("group-continues-across-seam", cube_topology(4), 0 * 16 + 2),
        ("liberty-across-seam", cube_topology(4), 0 * 16 + 2),
        ("capture-across-seam", cube_topology(4), 0 * 16 + 2),
        ("multi-capture-across-seam", cube_topology(4), 0 * 16 + 2),
        ("suicide-across-seam", cube_topology(4), 0 * 16 + 2),
        ("simple-ko-across-seam", cube_topology(4), 0 * 16 + 2),
        ("cleanup-ko-across-seam", cube_topology(4), 0 * 16 + 2),
        ("pass-for-ko-near-seam", cube_topology(4), 0 * 16 + 2),
        ("pass-alive-across-faces", cube_topology(4), 0 * 16 + 2),
        ("scoring-region-across-faces", cube_topology(4), 0 * 16 + 2),
    ],
)
def test_cube_seam_fixture_has_planar_katago_semantics(name, topology, center):
    _assert_star_matches_reference(topology, center)


@pytest.mark.parametrize(
    "name,topology,center",
    [
        ("horizontal-wrap-connectivity", torus_topology(9), 4 * 9 + 0),
        ("vertical-wrap-connectivity", torus_topology(9), 0 * 9 + 4),
        ("horizontal-wrap-capture", torus_topology(9), 4 * 9 + 0),
        ("vertical-wrap-capture", torus_topology(9), 0 * 9 + 4),
        ("wrap-suicide", torus_topology(9), 4 * 9 + 0),
        ("wrap-ko", torus_topology(9), 4 * 9 + 0),
        ("wrap-cleanup-ko", torus_topology(9), 4 * 9 + 0),
        ("wrapped-territory", torus_topology(9), 4 * 9 + 0),
        ("wrapped-pass-alive", torus_topology(9), 0 * 9 + 4),
        ("wrapped-scoring-invariance", torus_topology(9), 4 * 9 + 0),
    ],
)
def test_torus_wrap_fixture_has_planar_katago_semantics(name, topology, center):
    _assert_star_matches_reference(topology, center)
