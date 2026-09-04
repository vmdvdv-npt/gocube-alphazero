import numpy as np
import pytest

from alphazero.envs.gocube import (
    BLACK,
    CONSERVATIVE_AREA_ADJUDICATOR_V1,
    ENDGAME,
    WHITE,
    GoState,
    UnsupportedSelfPlayRuleset,
    conservative_area_adjudicate,
    initial_state,
    state_from_point_ids,
    torus_topology,
)


def endgame_state(state, topology):
    from alphazero.envs.gocube import apply_action

    first = apply_action(state, topology.pass_action, topology)
    return apply_action(first, topology.pass_action, topology)


def make_filled_state(topology, occupancy):
    board = np.zeros(topology.point_count, dtype=np.uint8)
    for index, point_id in enumerate(topology.point_ids):
        board[index] = {"empty": 0, "black": BLACK, "white": WHITE}[occupancy(point_id)]
    board.flags.writeable = False
    return GoState(
        board=board,
        current_player=0,
        turns=2,
        consecutive_passes=2,
        captures=(0, 0),
        previous_board=board,
        phase=ENDGAME,
    )


def test_empty_board_adjudication_is_total_and_komi_decides_winner():
    topology = torus_topology(9)
    state = endgame_state(initial_state(topology), topology)

    result = conservative_area_adjudicate(
        state,
        topology,
        ruleset="chinese",
        komi=7.5,
    )

    assert result.adjudicator_id == CONSERVATIVE_AREA_ADJUDICATOR_V1
    assert result.stage_a == ()
    assert result.classification == ()
    assert result.fallback_count == 0
    assert result.score.white == 7.5
    assert result.winner == "white"


def test_unresolved_groups_are_explicitly_retained_alive_not_removed():
    topology = torus_topology(9)
    playing = state_from_point_ids(topology, black=("0,0", "4,4"), current_player=0)
    state = endgame_state(playing, topology)

    result = conservative_area_adjudicate(
        state,
        topology,
        ruleset="chinese",
        komi=0,
    )

    assert result.fallback_count == 2
    assert all(proposal.status == "unresolved" for proposal in result.stage_a)
    assert all(group.status == "alive" for group in result.classification)
    assert all(group.source == "self-play-conservative" for group in result.classification)
    assert all(
        group.evidence["algorithm"] == CONSERVATIVE_AREA_ADJUDICATOR_V1
        for group in result.classification
    )
    assert result.score.dead_stones.black == 0
    assert result.score.stones_on_board.black == 2


def test_stage_a_proven_dead_and_alive_statuses_are_preserved():
    topology = torus_topology(9)
    target = {"4,4", "5,4"}
    liberty = "3,4"
    opponent_eyes = {"0,0", "2,2"}
    empty = {liberty, *opponent_eyes}

    state = make_filled_state(
        topology,
        lambda point: (
            "black" if point in target else ("empty" if point in empty else "white")
        ),
    )

    result = conservative_area_adjudicate(
        state,
        topology,
        ruleset="chinese",
        komi=0,
    )

    assert result.fallback_count == 0
    stage_a_statuses = {tuple(group.points): group.status for group in result.stage_a}
    final_statuses = {tuple(group.points): group.status for group in result.classification}
    assert final_statuses == stage_a_statuses
    assert set(final_statuses.values()) == {"alive", "dead"}


def test_adjudication_is_deterministic_for_same_state_and_configuration():
    topology = torus_topology(9)
    state = endgame_state(
        state_from_point_ids(topology, black=("0,0",), white=("4,4",)),
        topology,
    )

    first = conservative_area_adjudicate(state, topology, ruleset="chinese", komi=7.5)
    second = conservative_area_adjudicate(state, topology, ruleset="chinese", komi=7.5)

    assert first == second


def test_japanese_self_play_fails_closed_instead_of_fabricating_reward():
    topology = torus_topology(9)
    state = endgame_state(initial_state(topology), topology)

    with pytest.raises(UnsupportedSelfPlayRuleset, match="Chinese area scoring only"):
        conservative_area_adjudicate(
            state,
            topology,
            ruleset="japanese",
            komi=6.5,
        )


def test_adjudicator_rejects_state_before_two_passes():
    topology = torus_topology(9)

    with pytest.raises(ValueError, match="two consecutive passes"):
        conservative_area_adjudicate(
            initial_state(topology),
            topology,
            ruleset="chinese",
            komi=7.5,
        )
