import numpy as np

from alphazero.envs.gocube import (
    BLACK,
    WHITE,
    GroupClassification,
    GoState,
    score_position,
    state_from_point_ids,
    stone_groups,
    torus_topology,
)


def full_state(topology, fill, *, empty=(), white=(), black=(), captures=(0, 0)):
    board = np.full(topology.point_count, fill, dtype=np.uint8)
    for point in empty:
        board[topology.point_index(point)] = 0
    for point in white:
        board[topology.point_index(point)] = WHITE
    for point in black:
        board[topology.point_index(point)] = BLACK
    board.flags.writeable = False
    return GoState(
        board=board,
        current_player=0,
        turns=0,
        consecutive_passes=2,
        captures=captures,
        previous_board=None,
        phase="endgame",
    )


def classify_all(state, topology, *, overrides=None):
    overrides = overrides or {}
    result = []
    for group in stone_groups(state, topology):
        point_ids = {topology.point_id(point) for point in group}
        status = "alive"
        for point_id, override in overrides.items():
            if point_id in point_ids:
                status = override
                break
        result.append(GroupClassification(group, status))
    return tuple(result)


def test_fully_enclosed_black_territory_matches_gocube():
    topology = torus_topology(9)
    state = full_state(topology, BLACK, empty=("4,4",))

    score = score_position(state, topology, classify_all(state, topology), "chinese", 0)

    assert score.territory.black == 1
    assert tuple(topology.point_id(i) for i in score.territory_points.black) == ("4,4",)
    assert score.stones_on_board.black == 80
    assert score.black == 81


def test_mixed_boundary_empty_region_is_neutral_matching_gocube():
    topology = torus_topology(9)
    state = full_state(topology, BLACK, empty=("4,4",), white=("3,4",))

    score = score_position(state, topology, classify_all(state, topology), "chinese", 0)

    assert score.territory.black == 0
    assert score.territory.white == 0
    assert score.territory.neutral == 1


def test_connected_territory_through_torus_wraparound_matches_gocube():
    topology = torus_topology(9)
    state = full_state(topology, BLACK, empty=("0,4", "8,4"))

    score = score_position(state, topology, classify_all(state, topology), "chinese", 0)

    assert score.territory.black == 2


def test_chinese_area_score_does_not_add_capture_counters_again():
    topology = torus_topology(9)
    state = full_state(topology, BLACK, empty=("4,4",), captures=(9, 4))

    score = score_position(state, topology, classify_all(state, topology), "chinese", 0)

    assert score.black == 81
    assert score.captures == (9, 4)
    assert score.prisoners is None


def test_japanese_living_stones_are_not_points_and_captures_are_prisoners():
    topology = torus_topology(9)
    state = full_state(topology, BLACK, captures=(3, 2))

    score = score_position(state, topology, classify_all(state, topology), "japanese", 0)

    assert score.stones_on_board.black == 81
    assert score.prisoners == (3, 2)
    assert score.black == 3
    assert score.white == 2


def test_dead_stone_virtual_removal_matches_both_gocube_rulesets():
    topology = torus_topology(9)
    state = full_state(topology, BLACK, white=("4,4",))
    classification = classify_all(state, topology, overrides={"4,4": "dead"})

    chinese = score_position(state, topology, classification, "chinese", 0)
    japanese = score_position(state, topology, classification, "japanese", 0)

    assert chinese.dead_stones.white == 1
    assert chinese.territory.black == 1
    assert chinese.stones_on_board.white == 0
    assert chinese.black == 81
    assert japanese.territory.black == 1
    assert japanese.prisoners == (1, 0)
    assert japanese.black == 2


def test_seki_adjacent_region_stays_seki_neutral_matching_gocube():
    topology = torus_topology(9)
    state = full_state(topology, BLACK, empty=("4,4",))
    classification = tuple(
        GroupClassification(group, "seki")
        for group in stone_groups(state, topology)
    )

    score = score_position(state, topology, classification, "japanese", 0)

    assert score.territory.black == 0
    assert score.territory.seki == 1
    assert tuple(topology.point_id(i) for i in score.territory_points.seki) == ("4,4",)


def test_winner_margin_fractional_komi_and_ruleset_difference_match_gocube():
    topology = torus_topology(9)
    all_black = full_state(topology, BLACK)
    classification = classify_all(all_black, topology)

    white_win = score_position(all_black, topology, classification, "japanese", 7.5)
    draw = score_position(all_black, topology, classification, "japanese", 0)

    assert white_win.winner == "white"
    assert white_win.margin == 7.5
    assert draw.winner == "draw"
    assert draw.margin == 0

    with_empty = full_state(topology, BLACK, empty=("4,4",), captures=(2, 0))
    classification = classify_all(with_empty, topology)
    chinese = score_position(with_empty, topology, classification, "chinese", 0.5)
    japanese = score_position(with_empty, topology, classification, "japanese", 0.5)
    assert chinese.black == 81
    assert japanese.black == 3
