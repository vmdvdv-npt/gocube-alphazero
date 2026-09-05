from dataclasses import replace

from alphazero.envs.gocube import (
    BLACK, CLEANUP_1, CLEANUP_2, WHITE, Topology, apply_v3_action,
    cube_topology, final_v3_score, independent_life_analysis, torus_topology,
    v3_state_from_board,
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


def test_exact_territory_captures_komi_and_winner():
    t = rect_topology(5, 3)
    black = idxs(t, ("0,0","1,0","2,0","0,1","2,1","0,2","1,2","2,2"))
    white = idxs(t, ("4,0","4,1","4,2","3,0","3,2"))
    state = v3_state_from_board(
        t, black=black, white=white, captures=(2, 1), phase=CLEANUP_2,
        second_cleanup_start_colors=None,
    )
    state = replace(state, second_cleanup_start_colors=bytes(state.board.tolist()))
    score, _, _ = final_v3_score(state, t, 0.5)
    assert score.territory.black == 1
    assert score.territory.white == 1
    assert score.captures == (2, 1)
    assert score.black == 3.0
    assert score.white == 2.5
    assert score.margin == 0.5
    assert score.winner == "black"


def test_seki_shared_dame_is_not_ordinary_territory():
    t = rect_topology(4, 3)
    black = idxs(t, ("0,0","0,1","0,2","1,0","1,2"))
    white = idxs(t, ("3,0","3,1","3,2","2,0","2,2"))
    board = v3_state_from_board(t, black=black, white=white).board
    life = independent_life_analysis(board, t)
    center = {t.point_index("1,1"), t.point_index("2,1")}
    assert not center.intersection(life.black_territory)
    assert not center.intersection(life.white_territory)
    assert center.issubset(set(life.dame) | set(life.seki))


def protected_three_eye_state(t, phase):
    eyes = idxs(t, ("1,1","3,1","2,3"))
    black = tuple(p for p in range(t.point_count) if p not in eyes)
    state = v3_state_from_board(t, black=black, phase=phase, current_player=0)
    return replace(state, second_cleanup_start_colors=bytes(state.board.tolist())), eyes


def test_cleanup2_own_territory_fill_is_score_preserving():
    t = rect_topology(5, 5)
    state, eyes = protected_three_eye_state(t, CLEANUP_2)
    before, _, _ = final_v3_score(state, t, 0.0)
    after_state = apply_v3_action(state, eyes[0], t)
    after, _, _ = final_v3_score(after_state, t, 0.0)
    assert after_state.cleanup2_moves == (1, 0)
    assert before.black == after.black


def test_same_fill_in_cleanup1_has_no_cleanup2_compensation():
    t = rect_topology(5, 5)
    state, eyes = protected_three_eye_state(t, CLEANUP_1)
    before = replace(state, phase=CLEANUP_2)
    before_score, _, _ = final_v3_score(before, t, 0.0)
    played = apply_v3_action(state, eyes[0], t)
    assert played.cleanup2_moves == (0, 0)
    comparable = replace(played, phase=CLEANUP_2, second_cleanup_start_colors=bytes(state.board.tolist()))
    after_score, _, _ = final_v3_score(comparable, t, 0.0)
    assert after_score.black == before_score.black - 1


def test_issue_1158_new_stone_outside_independent_life_is_penalized_but_still_counted():
    t = rect_topology(3, 3)
    start = v3_state_from_board(t, phase=CLEANUP_2)
    start = replace(start, second_cleanup_start_colors=bytes(start.board.tolist()))
    played = apply_v3_action(start, t.point_index("1,1"), t)
    score, _, _ = final_v3_score(played, t, 0.0)
    assert score.stones_on_board.black == 1
    assert score.black == 0.0  # +1 cleanup move and -1 new stone outside independent life


def test_issue_1158_unassigned_single_color_empty_component_is_not_dropped():
    t = rect_topology(3, 3)
    center = t.point_index("1,1")
    black = tuple(p for p in range(t.point_count) if p != center)
    state = v3_state_from_board(t, black=black, phase=CLEANUP_2)
    state = replace(state, second_cleanup_start_colors=bytes(state.board.tolist()))
    score, _, _ = final_v3_score(state, t, 0.0)
    assert score.territory.black == 1
    assert score.black == 1.0


def test_cube_seam_and_torus_wrap_territory_score_by_logical_graph():
    cube = cube_topology(4)
    seam_eye = cube.point_index("front:0:0")
    cube_black = tuple(p for p in range(cube.point_count) if p != seam_eye)
    cs = v3_state_from_board(cube, black=cube_black, phase=CLEANUP_2)
    cs = replace(cs, second_cleanup_start_colors=bytes(cs.board.tolist()))
    cube_score, _, _ = final_v3_score(cs, cube, 0.0)
    assert cube_score.territory.black == 1

    torus = torus_topology(9)
    wrap_eye = torus.point_index("0,0")
    torus_black = tuple(p for p in range(torus.point_count) if p != wrap_eye)
    ts = v3_state_from_board(torus, black=torus_black, phase=CLEANUP_2)
    ts = replace(ts, second_cleanup_start_colors=bytes(ts.board.tolist()))
    torus_score, _, _ = final_v3_score(ts, torus, 0.0)
    assert torus_score.territory.black == 1
