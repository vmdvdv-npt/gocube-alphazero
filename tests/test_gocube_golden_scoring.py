from dataclasses import replace
from types import SimpleNamespace

import pytest

from alphazero.envs.gocube import (
    BLACK,
    CLEANUP_2,
    NO_RESULT,
    SCORED,
    WHITE,
    Topology,
    cube_topology,
    final_v3_score,
    torus_topology,
    v3_state_from_board,
)
from alphazero.envs.gocube.golden_scoring import (
    GOLDEN_KOMI,
    GoldenOutcome,
    GoldenScoreMismatch,
    GoldenScoringError,
    adjudicate_game_state,
    adjudicate_v3_terminal,
)
from alphazero.envs.gocube.production_contract import GOCUBE_KOMI


def rect_topology(width=5, height=5):
    ids = tuple(f"{x},{y}" for y in range(height) for x in range(width))
    index = {point: i for i, point in enumerate(ids)}
    neighbors = []
    for y in range(height):
        for x in range(width):
            adjacent = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    adjacent.append(index[f"{nx},{ny}"])
            neighbors.append(tuple(adjacent))
    return Topology("rect-golden-test", width, ids, tuple(neighbors), index)


def test_golden_komi_is_an_independent_pinned_constant():
    assert GOLDEN_KOMI == 0.5
    assert GOCUBE_KOMI == GOLDEN_KOMI


def test_golden_known_japanese_score_and_winner():
    topology = rect_topology(6, 5)
    black_eyes = {"1,1", "1,3"}
    white_eyes = {"4,1", "4,3"}
    black = tuple(
        topology.point_index(f"{x},{y}")
        for y in range(5)
        for x in range(3)
        if f"{x},{y}" not in black_eyes
    )
    white = tuple(
        topology.point_index(f"{x},{y}")
        for y in range(5)
        for x in range(3, 6)
        if f"{x},{y}" not in white_eyes
    )
    state = v3_state_from_board(
        topology,
        black=black,
        white=white,
        captures=(2, 1),
        phase=CLEANUP_2,
    )
    state = replace(
        state,
        second_cleanup_start_colors=bytes(state.board.tolist()),
    )

    golden = adjudicate_v3_terminal(
        board=state.board,
        adjacency=topology.neighbors_by_index,
        terminal_kind=SCORED,
        white_bonus_score=state.white_bonus_score,
        second_cleanup_start_colors=state.second_cleanup_start_colors,
        captures=state.captures,
        komi=0.5,
    )
    production, _, _ = final_v3_score(state, topology, 0.5)

    assert golden.outcome is GoldenOutcome.BLACK
    assert golden.score is not None
    assert golden.score.black == 15.0
    assert golden.score.white == 14.5
    assert golden.score.margin == 0.5
    assert len(golden.score.area.black_territory) == 2
    assert len(golden.score.area.white_territory) == 2
    assert golden.score.black == production.black
    assert golden.score.white == production.white
    assert golden.score.outcome.value == production.winner


@pytest.mark.parametrize("topology", [cube_topology(4), torus_topology(9)])
def test_golden_scoring_uses_graph_topology_across_seams_and_wraps(topology):
    if topology.kind == "cube":
        eyes = {
            topology.point_index("front:0:0"),
            topology.point_index("back:2:2"),
        }
    else:
        eyes = {
            topology.point_index("0,0"),
            topology.point_index("4,4"),
        }

    board = [BLACK] * topology.point_count
    for point in eyes:
        board[point] = 0

    golden = adjudicate_v3_terminal(
        board=board,
        adjacency=topology.neighbors_by_index,
        terminal_kind=SCORED,
        white_bonus_score=0.0,
        second_cleanup_start_colors=bytes(board),
        captures=(0, 0),
        komi=0.5,
    )

    assert golden.score is not None
    assert len(golden.score.area.black_territory) == 2
    assert golden.outcome is GoldenOutcome.BLACK


def test_no_result_is_never_collapsed_into_draw():
    topology = rect_topology(2, 2)
    golden = adjudicate_v3_terminal(
        board=(0, 0, 0, 0),
        adjacency=topology.neighbors_by_index,
        terminal_kind=NO_RESULT,
        white_bonus_score=0.0,
        second_cleanup_start_colors=None,
        captures=(0, 0),
        komi=0.5,
    )
    assert golden.outcome is GoldenOutcome.NO_RESULT
    assert golden.score is None
    assert golden.outcome is not GoldenOutcome.DRAW


def test_komi_7_5_fails_closed():
    topology = rect_topology(2, 2)
    with pytest.raises(GoldenScoringError, match="komi 0.5"):
        adjudicate_v3_terminal(
            board=(0, 0, 0, 0),
            adjacency=topology.neighbors_by_index,
            terminal_kind=SCORED,
            white_bonus_score=0.0,
            second_cleanup_start_colors=None,
            captures=(0, 0),
            komi=7.5,
        )


def test_nonreciprocal_graph_fails_closed():
    with pytest.raises(GoldenScoringError, match="not reciprocal"):
        adjudicate_v3_terminal(
            board=(BLACK, 0),
            adjacency=((1,), ()),
            terminal_kind=SCORED,
            white_bonus_score=0.0,
            second_cleanup_start_colors=None,
            captures=(0, 0),
            komi=0.5,
        )


def test_production_score_disagreement_is_a_hard_failure():
    topology = rect_topology(2, 2)
    state = v3_state_from_board(topology, phase=CLEANUP_2)
    state = replace(
        state,
        phase=SCORED,
        terminal_kind=SCORED,
        second_cleanup_start_colors=bytes(state.board.tolist()),
    )

    class FakeGame:
        KOMI = 0.5

        @classmethod
        def logical_topology(cls):
            return topology

    game = FakeGame()
    game.semantic_state = state
    game.terminal_adjudication = SimpleNamespace(
        terminal_kind=SCORED,
        score=SimpleNamespace(
            black=999.0,
            white=0.5,
            komi=0.5,
            margin=998.5,
            winner="black",
        ),
    )

    with pytest.raises(GoldenScoreMismatch, match="black mismatch"):
        adjudicate_game_state(game, require_production_agreement=True)
