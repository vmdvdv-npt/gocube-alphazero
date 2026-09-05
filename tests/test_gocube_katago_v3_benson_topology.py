from dataclasses import replace

import numpy as np

from alphazero.envs.gocube import (
    BLACK, CLEANUP_2, Topology, cube_topology, final_v3_score,
    independent_life_analysis, pass_alive_analysis, torus_topology, v3_state_from_board,
)


def rect_topology(width=5, height=5):
    ids = tuple(f"{x},{y}" for y in range(height) for x in range(width))
    index = {p: i for i, p in enumerate(ids)}
    neighbors = []
    for y in range(height):
        for x in range(width):
            row = []
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    row.append(index[f"{nx},{ny}"])
            neighbors.append(tuple(row))
    return Topology("rect-test", width, ids, tuple(neighbors), index)


def idxs(t, points):
    return tuple(t.point_index(p) for p in points)


def fill_except(t, empty, white=()):
    empty = set(empty); white = set(white)
    black = tuple(p for p in range(t.point_count) if p not in empty and p not in white)
    return v3_state_from_board(t, black=black, white=white).board


def test_unconditional_two_eye_group_is_pass_alive():
    t = rect_topology(5, 5)
    board = fill_except(t, idxs(t, ("1,1", "3,3")))
    analysis = pass_alive_analysis(board, t)
    assert len(analysis.pass_alive_black_groups) == 1
    assert set(analysis.pass_alive_black_territory) == set(idxs(t, ("1,1", "3,3")))


def test_one_eye_and_false_eye_like_groups_are_not_pass_alive():
    t = rect_topology(5, 5)
    one_eye = fill_except(t, idxs(t, ("2,2",)))
    assert not pass_alive_analysis(one_eye, t).pass_alive_black_groups
    # One nominal eye plus an opponent stone inside the second cavity is not a Benson proof.
    e1, e2 = idxs(t, ("1,1", "3,3"))
    false_eye = fill_except(t, (e1,), white=(e2,))
    assert not pass_alive_analysis(false_eye, t).pass_alive_black_groups


def test_opponent_stone_can_be_inside_proven_pass_alive_territory_without_being_called_dead_by_benson():
    t = rect_topology(5, 5)
    e1, e2, intruder = idxs(t, ("1,1", "3,3", "2,2"))
    board = fill_except(t, (e1, e2), white=(intruder,))
    analysis = pass_alive_analysis(board, t)
    # Benson proves life only; absence of proof for White is not a dead declaration.
    assert not analysis.pass_alive_white_groups
    assert analysis.pass_alive_black_groups


def test_seki_is_not_converted_to_ordinary_territory():
    t = rect_topology(4, 3)
    black = idxs(t, ("0,0","0,1","0,2","1,0","1,2"))
    white = idxs(t, ("3,0","3,1","3,2","2,0","2,2"))
    state = v3_state_from_board(t, black=black, white=white)
    life = independent_life_analysis(state.board, t)
    shared = set(idxs(t, ("1,1", "2,1")))
    assert not shared.intersection(life.black_territory)
    assert not shared.intersection(life.white_territory)


def test_benson_equivalent_shapes_work_on_cube_interior_seam_vertex_and_torus_wrap():
    fixtures = [
        (cube_topology(5), ("front:2:2", "back:2:2")),
        (cube_topology(5), ("front:0:2", "top:4:2")),
        (cube_topology(5), ("front:0:0", "top:4:0")),
        (torus_topology(9), ("0,0", "8,8")),
    ]
    for t, eyes in fixtures:
        eye_indices = idxs(t, eyes)
        board = fill_except(t, eye_indices)
        analysis = pass_alive_analysis(board, t)
        assert analysis.pass_alive_black_groups
        assert set(eye_indices).issubset(analysis.covered_points)


def relabel_topology(t, permutation):
    inv = {old: new for new, old in enumerate(permutation)}
    ids = tuple(f"p{new}" for new in range(t.point_count))
    neighbors = tuple(tuple(inv[n] for n in t.neighbor_indices(old)) for old in permutation)
    return Topology("relabel", t.size, ids, neighbors, {p:i for i,p in enumerate(ids)})


def test_graph_isomorphic_relabel_preserves_pass_alive_independent_life_score_and_winner():
    t = torus_topology(9)
    eyes = {t.point_index("0,0"), t.point_index("4,4")}
    state = v3_state_from_board(t, black=tuple(p for p in range(t.point_count) if p not in eyes), phase=CLEANUP_2)
    state = replace(state, second_cleanup_start_colors=bytes(state.board.tolist()))
    permutation = tuple(reversed(range(t.point_count)))
    rt = relabel_topology(t, permutation)
    rboard = np.asarray([state.board[old] for old in permutation], dtype=np.uint8)
    rblack = tuple(i for i, c in enumerate(rboard) if c == BLACK)
    rs = v3_state_from_board(rt, black=rblack, phase=CLEANUP_2)
    rs = replace(rs, second_cleanup_start_colors=bytes(rs.board.tolist()))

    pa = pass_alive_analysis(state.board, t)
    rpa = pass_alive_analysis(rs.board, rt)
    assert len(pa.pass_alive_black_groups) == len(rpa.pass_alive_black_groups)
    life = independent_life_analysis(state.board, t)
    rlife = independent_life_analysis(rs.board, rt)
    assert len(life.black_territory) == len(rlife.black_territory)
    score, _, _ = final_v3_score(state, t, 7.5)
    rscore, _, _ = final_v3_score(rs, rt, 7.5)
    assert score.black == rscore.black
    assert score.white == rscore.white
    assert score.winner == rscore.winner
