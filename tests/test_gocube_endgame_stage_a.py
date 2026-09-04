import numpy as np
import pytest

from alphazero.envs.gocube import (
    BLACK,
    ENDGAME,
    WHITE,
    GoState,
    assisted_endgame_proposal,
    cube_topology,
    proposal_point_ids,
    stone_groups,
    torus_topology,
)


def make_state(topology, occupancy):
    board = np.zeros(topology.point_count, dtype=np.uint8)
    for index, point_id in enumerate(topology.point_ids):
        value = occupancy(point_id)
        board[index] = {"empty": 0, "black": BLACK, "white": WHITE}[value]
    board.flags.writeable = False
    return GoState(
        board=board,
        current_player=0,
        turns=0,
        consecutive_passes=2,
        captures=(0, 0),
        previous_board=None,
        phase=ENDGAME,
    )


def proposal_for_color(proposals, state, color):
    stone = BLACK if color == "black" else WHITE
    return next((proposal for proposal in proposals if state.board[proposal.points[0]] == stone), None)


@pytest.mark.parametrize(
    ("topology", "eyes"),
    (
        (torus_topology(9), ("0,0", "4,4")),
        (cube_topology(5), ("front:2:2", "back:2:2")),
    ),
)
def test_benson_two_eye_fixture_is_automatically_alive(topology, eyes):
    eye_set = set(eyes)
    state = make_state(topology, lambda point: "empty" if point in eye_set else "black")

    assert len(stone_groups(state, topology)) == 1
    result = assisted_endgame_proposal(state, topology)

    assert len(result) == 1
    proposal = result[0]
    assert proposal.status == "alive"
    assert proposal.source == "automatic"
    assert proposal.evidence["algorithm"] == "benson-pass-alive-v1"
    assert proposal.evidence["proof"] == "two-vital-regions"
    assert len(proposal.evidence["vitalRegions"]) == 2


@pytest.mark.parametrize(
    ("topology", "target", "liberty", "opponent_eyes"),
    (
        (
            torus_topology(9),
            ("4,4", "5,4"),
            "3,4",
            ("0,0", "2,2"),
        ),
        (
            cube_topology(5),
            ("front:2:2", "front:2:3"),
            "front:2:1",
            ("back:2:2", "top:2:2"),
        ),
    ),
)
def test_sealed_single_liberty_group_is_dead_only_behind_pass_alive_boundary(
    topology, target, liberty, opponent_eyes
):
    target_set = set(target)
    empty = {liberty, *opponent_eyes}

    def occupancy(point):
        if point in target_set:
            return "black"
        if point in empty:
            return "empty"
        return "white"

    state = make_state(topology, occupancy)
    result = assisted_endgame_proposal(state, topology)
    dead = proposal_for_color(result, state, "black")
    alive = proposal_for_color(result, state, "white")

    assert dead is not None
    assert proposal_point_ids(dead, topology) == tuple(sorted(target))
    assert dead.status == "dead"
    assert dead.source == "automatic"
    assert dead.evidence["algorithm"] == "sealed-single-liberty-dead-v1"
    assert dead.evidence["candidate"] == "single-liberty"
    assert dead.evidence["proof"] == "sealed-liberty-with-pass-alive-boundary"
    assert dead.evidence["liberty"] == liberty
    assert len(dead.evidence["boundaryAliveGroups"]) == 1

    assert alive is not None
    assert alive.status == "alive"
    assert alive.source == "automatic"
    assert alive.evidence["algorithm"] == "benson-pass-alive-v1"


@pytest.mark.parametrize(
    ("topology", "eye"),
    (
        (torus_topology(9), "4,4"),
        (cube_topology(5), "front:2:2"),
    ),
)
def test_single_eye_full_group_remains_unresolved(topology, eye):
    state = make_state(topology, lambda point: "empty" if point == eye else "black")

    result = assisted_endgame_proposal(state, topology)

    assert len(result) == 1
    assert result[0].status == "unresolved"
    assert result[0].source is None
    assert result[0].evidence is None


def test_closed_two_shared_liberty_mutual_life_is_seki_on_torus():
    topology = torus_topology(9)
    liberties = ("3,0", "3,1")
    liberty_set = set(liberties)

    def occupancy(point):
        if point in liberty_set:
            return "empty"
        x = int(point.split(",")[0])
        return "black" if x <= 3 else "white"

    state = make_state(topology, occupancy)
    assert len(stone_groups(state, topology)) == 2

    result = assisted_endgame_proposal(state, topology)

    assert len(result) == 2
    assert all(proposal.status == "seki" for proposal in result)
    assert all(proposal.source == "automatic" for proposal in result)
    for proposal in result:
        assert proposal.evidence["algorithm"] == "closed-mutual-two-liberties-seki-v1"
        assert proposal.evidence["candidate"] == "two-shared-liberties"
        assert proposal.evidence["proof"] == "closed-mutual-capture"
        assert proposal.evidence["sharedLiberties"] == tuple(sorted(liberties))
        assert len(proposal.evidence["groups"]) == 2


def test_closed_two_shared_liberty_mutual_life_is_seki_on_cube():
    topology = cube_topology(5)
    liberties = ("front:2:2", "front:3:2")
    liberty_set = set(liberties)
    white = {"front:2:3", "front:3:3"}

    def occupancy(point):
        if point in liberty_set:
            return "empty"
        return "white" if point in white else "black"

    state = make_state(topology, occupancy)
    assert len(stone_groups(state, topology)) == 2

    result = assisted_endgame_proposal(state, topology)

    assert len(result) == 2
    assert all(proposal.status == "seki" for proposal in result)
    for proposal in result:
        assert proposal.evidence["sharedLiberties"] == tuple(sorted(liberties))


def test_partial_analysis_context_disables_all_automatic_proofs():
    topology = torus_topology(9)
    state = make_state(
        topology,
        lambda point: "black" if point in {"0,0", "4,4"} else "empty",
    )
    groups = stone_groups(state, topology)
    assert len(groups) == 2

    result = assisted_endgame_proposal(state, topology, groups=(groups[0],))

    assert len(result) == 1
    assert result[0].status == "unresolved"
    assert result[0].source is None
