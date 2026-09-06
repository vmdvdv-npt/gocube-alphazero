from dataclasses import replace

import numpy as np

from alphazero.envs.gocube import (
    BLACK,
    CLEANUP_2,
    EMPTY,
    WHITE,
    Topology,
    cube_topology,
    final_v3_score,
    independent_life_analysis,
    pass_alive_analysis,
    torus_topology,
    v3_state_from_board,
    v3_valid_moves,
)


def rect_topology(width=5, height=5):
    ids = tuple(f"{x},{y}" for y in range(height) for x in range(width))
    index = {p: i for i, p in enumerate(ids)}
    neighbors = []
    for y in range(height):
        for x in range(width):
            row = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    row.append(index[f"{nx},{ny}"])
            neighbors.append(tuple(row))
    return Topology("rect-test", width, ids, tuple(neighbors), index)


def idxs(t, points):
    return tuple(t.point_index(p) for p in points)


def fill_except(t, empty, white=()):
    empty = set(empty)
    white = set(white)
    black = tuple(p for p in range(t.point_count) if p not in empty and p not in white)
    return v3_state_from_board(t, black=black, white=white).board


def group_sets(groups):
    return {frozenset(group) for group in groups}


def assert_actual_group(board, topology, group, color):
    points = set(group)
    assert points
    assert all(int(board[p]) == color for p in points)
    seen = {next(iter(points))}
    pending = list(seen)
    while pending:
        point = pending.pop()
        for neighbor in topology.neighbor_indices(point):
            if int(board[neighbor]) != color:
                continue
            assert neighbor in points
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    assert seen == points


def assert_black_pass_alive_with_two_regions(topology, first_region, second_region):
    empty = idxs(topology, tuple(first_region) + tuple(second_region))
    board = fill_except(topology, empty)
    analysis = pass_alive_analysis(board, topology)
    assert analysis.pass_alive_black_groups
    assert set(empty).issubset(analysis.covered_points)


def test_unconditional_two_eye_group_is_pass_alive():
    t = rect_topology(5, 5)
    board = fill_except(t, idxs(t, ("1,1", "3,3")))
    analysis = pass_alive_analysis(board, t)
    assert len(analysis.pass_alive_black_groups) == 1
    assert set(analysis.pass_alive_black_territory) == set(idxs(t, ("1,1", "3,3")))


def test_one_eye_group_is_not_pass_alive():
    t = rect_topology(5, 5)
    one_eye = fill_except(t, idxs(t, ("2,2",)))
    assert not pass_alive_analysis(one_eye, t).pass_alive_black_groups


def test_false_eye_like_region_is_not_erroneously_proven_alive():
    t = rect_topology(7, 7)
    large_region = tuple(f"{x},{y}" for y in range(1, 4) for x in range(1, 4))
    nominal_eye = ("5,5",)
    board = fill_except(t, idxs(t, large_region + nominal_eye))

    # The 3x3 region is not vital: its center is not adjacent to the black chain.
    # This fixture contains no impossible zero-liberty opponent stone.
    assert not pass_alive_analysis(board, t).pass_alive_black_groups


def test_opponent_stone_inside_benson_region_does_not_invalidate_region():
    t = rect_topology(5, 5)
    eye_with_intruder, intruder, second_eye = idxs(t, ("1,1", "1,2", "3,3"))
    board = fill_except(t, (eye_with_intruder, second_eye), white=(intruder,))
    analysis = pass_alive_analysis(board, t)

    # The white stone has the empty intersection at 1,1 as a liberty, so this is
    # a physically valid board fixture rather than an uncaptured zero-liberty stone.
    assert analysis.pass_alive_black_groups
    assert intruder in analysis.pass_alive_black_territory
    assert not analysis.pass_alive_white_groups


def test_torus9_opponent_stone_counterexample_is_pass_alive_and_white_has_no_placement():
    t = torus_topology(9)
    empty = {0, 40}
    white = {1}
    black = tuple(p for p in range(t.point_count) if p not in empty and p not in white)
    state = v3_state_from_board(t, black=black, white=white, current_player=1)

    legal = v3_valid_moves(state, t)
    assert np.count_nonzero(legal[:t.point_count]) == 0
    assert legal[t.pass_action] == 1

    analysis = pass_alive_analysis(state.board, t)
    assert group_sets(analysis.pass_alive_black_groups) == {frozenset(black)}
    assert not analysis.pass_alive_white_groups
    assert {0, 1, 40}.issubset(set(analysis.pass_alive_black_territory))


def test_seki_is_not_converted_to_ordinary_territory():
    t = rect_topology(4, 3)
    black = idxs(t, ("0,0", "0,1", "0,2", "1,0", "1,2"))
    white = idxs(t, ("3,0", "3,1", "3,2", "2,0", "2,2"))
    state = v3_state_from_board(t, black=black, white=white)
    life = independent_life_analysis(state.board, t)
    shared = set(idxs(t, ("1,1", "2,1")))
    assert not shared.intersection(life.black_territory)
    assert not shared.intersection(life.white_territory)


def test_cube_interior_benson_shape_is_pass_alive():
    t = cube_topology(5)
    assert_black_pass_alive_with_two_regions(
        t,
        ("front:2:1", "front:2:2"),
        ("back:2:2",),
    )


def test_cube_seam_benson_shape_matches_interior_semantics():
    t = cube_topology(5)
    assert_black_pass_alive_with_two_regions(
        t,
        ("front:0:2", "top:4:2"),
        ("back:2:2",),
    )


def test_cube_vertex_benson_shape_uses_graph_neighbors():
    t = cube_topology(5)
    assert_black_pass_alive_with_two_regions(
        t,
        ("front:0:0", "top:4:0", "left:0:4"),
        ("back:2:2",),
    )


def test_torus_wrap_benson_shape_matches_local_semantics():
    t = torus_topology(9)
    assert_black_pass_alive_with_two_regions(
        t,
        ("0,0", "8,0"),
        ("4,4",),
    )


def relabel_topology(t, permutation):
    inv = {old: new for new, old in enumerate(permutation)}
    ids = tuple(f"p{new}" for new in range(t.point_count))
    neighbors = tuple(tuple(inv[n] for n in t.neighbor_indices(old)) for old in permutation)
    return Topology("relabel", t.size, ids, neighbors, {p: i for i, p in enumerate(ids)})


def test_graph_isomorphic_relabel_preserves_pass_alive_independent_life_score_and_winner():
    t = torus_topology(9)
    eyes = {t.point_index("0,0"), t.point_index("4,4")}
    state = v3_state_from_board(
        t,
        black=tuple(p for p in range(t.point_count) if p not in eyes),
        phase=CLEANUP_2,
    )
    state = replace(state, second_cleanup_start_colors=bytes(state.board.tolist()))

    permutation = tuple(reversed(range(t.point_count)))
    old_to_new = {old: new for new, old in enumerate(permutation)}
    rt = relabel_topology(t, permutation)
    rboard = np.asarray([state.board[old] for old in permutation], dtype=np.uint8)
    rblack = tuple(i for i, c in enumerate(rboard) if c == BLACK)
    rs = v3_state_from_board(rt, black=rblack, phase=CLEANUP_2)
    rs = replace(rs, second_cleanup_start_colors=bytes(rs.board.tolist()))

    pa = pass_alive_analysis(state.board, t)
    rpa = pass_alive_analysis(rs.board, rt)
    mapped_groups = {
        frozenset(old_to_new[p] for p in group)
        for group in pa.pass_alive_black_groups
    }
    assert mapped_groups == group_sets(rpa.pass_alive_black_groups)
    assert {old_to_new[p] for p in pa.pass_alive_black_territory} == set(rpa.pass_alive_black_territory)

    life = independent_life_analysis(state.board, t)
    rlife = independent_life_analysis(rs.board, rt)
    assert len(life.black_territory) == len(rlife.black_territory)
    score, _, _ = final_v3_score(state, t, 0.5)
    rscore, _, _ = final_v3_score(rs, rt, 0.5)
    assert score.black == rscore.black
    assert score.white == rscore.white
    assert score.winner == rscore.winner


def test_pass_alive_groups_are_actual_connected_stone_groups():
    t = rect_topology(5, 5)
    intruder = t.point_index("1,2")
    board = fill_except(
        t,
        idxs(t, ("1,1", "3,3")),
        white=(intruder,),
    )
    analysis = pass_alive_analysis(board, t)
    for group in analysis.pass_alive_black_groups:
        assert_actual_group(board, t, group, BLACK)
    for group in analysis.pass_alive_white_groups:
        assert_actual_group(board, t, group, WHITE)


def test_pass_alive_analysis_is_color_symmetric():
    t = torus_topology(9)
    empty = {0, 40}
    white = {1}
    black = tuple(p for p in range(t.point_count) if p not in empty and p not in white)
    board = v3_state_from_board(t, black=black, white=white).board
    analysis = pass_alive_analysis(board, t)

    swapped = np.asarray(board).copy()
    swapped[board == BLACK] = WHITE
    swapped[board == WHITE] = BLACK
    swapped_analysis = pass_alive_analysis(swapped, t)

    assert group_sets(analysis.pass_alive_black_groups) == group_sets(swapped_analysis.pass_alive_white_groups)
    assert group_sets(analysis.pass_alive_white_groups) == group_sets(swapped_analysis.pass_alive_black_groups)
    assert set(analysis.pass_alive_black_territory) == set(swapped_analysis.pass_alive_white_territory)
    assert set(analysis.pass_alive_white_territory) == set(swapped_analysis.pass_alive_black_territory)


def test_pass_alive_analysis_does_not_mutate_board():
    t = torus_topology(9)
    state = v3_state_from_board(
        t,
        black=tuple(p for p in range(t.point_count) if p not in {0, 1, 40}),
        white=(1,),
    )
    before = state.board.copy()
    writeable = state.board.flags.writeable

    pass_alive_analysis(state.board, t)

    assert np.array_equal(state.board, before)
    assert state.board.flags.writeable == writeable
    assert int(state.board[0]) == EMPTY
