from __future__ import annotations

from dataclasses import dataclass

import pytest

import gocube_golden as g


LINE3 = g.research_topology(((1,), (0, 2), (1,)), topology_id="golden-qualification-line3-v1")
ONE = g.research_topology(((),), topology_id="golden-qualification-one-v1")


def line3_state(
    stones,
    *,
    side_to_move,
    history=None,
    consecutive_passes=0,
    komi=0.5,
):
    return g.research_state_from_stones(
        stones,
        side_to_move=side_to_move,
        topology=LINE3,
        komi=komi,
        superko_history=history,
        consecutive_passes=consecutive_passes,
    )


@dataclass(frozen=True)
class QualificationCase:
    name: str
    state: g.GoldenState
    expected_utility: float
    optimal_actions: tuple[int | str, ...]


EMPTY3 = (g.EMPTY, g.EMPTY, g.EMPTY)
SUPERKO_MIDDLE = (g.EMPTY, g.BLACK, g.EMPTY)

QUALIFICATION_CASES = (
    QualificationCase(
        "black_to_move_win",
        line3_state((g.EMPTY, g.EMPTY, g.BLACK), side_to_move=g.BLACK),
        1.0,
        (1,),
    ),
    QualificationCase(
        "white_to_move_win",
        line3_state((g.WHITE, g.EMPTY, g.EMPTY), side_to_move=g.WHITE),
        1.0,
        (1,),
    ),
    QualificationCase(
        "side_to_move_loss",
        line3_state((g.EMPTY, g.BLACK, g.BLACK), side_to_move=g.WHITE),
        -1.0,
        (0, g.PASS),
    ),
    QualificationCase(
        "pass_best",
        line3_state((g.EMPTY, g.BLACK, g.EMPTY), side_to_move=g.BLACK),
        1.0,
        (g.PASS,),
    ),
    QualificationCase(
        "multi_ply_tactical",
        line3_state((g.BLACK, g.EMPTY, g.WHITE), side_to_move=g.BLACK),
        1.0,
        (1,),
    ),
    QualificationCase(
        "same_board_open_history",
        line3_state(EMPTY3, side_to_move=g.BLACK, history=(EMPTY3,)),
        1.0,
        (1,),
    ),
    QualificationCase(
        "same_board_superko_blocked",
        line3_state(
            EMPTY3,
            side_to_move=g.BLACK,
            history=(SUPERKO_MIDDLE, EMPTY3),
        ),
        -1.0,
        (0, 2, g.PASS),
    ),
    QualificationCase(
        "preterminal_pass_boundary",
        g.research_state_from_stones(
            (g.BLACK,),
            side_to_move=g.WHITE,
            topology=ONE,
            komi=0.5,
            consecutive_passes=1,
        ),
        -1.0,
        (g.PASS,),
    ),
)


class ExactOracleEvaluator:
    def __init__(self):
        self.calls = 0
        self.terminal_calls = 0

    def evaluate(self, state):
        self.calls += 1
        if state.is_terminal:
            self.terminal_calls += 1
            raise AssertionError("Golden evaluator must never be called on terminal state")
        solved = g.solve_exact(state, node_limit=10_000)
        assert solved.status == g.SolveStatus.EXACT
        if solved.utility > 0:
            wdl = (1.0, 0.0, 0.0)
        elif solved.utility < 0:
            wdl = (0.0, 0.0, 1.0)
        else:
            wdl = (0.0, 1.0, 0.0)
        return g.Evaluation(
            policy=tuple(1.0 for _ in g.GoldenSearchAdapter().action_space(state)),
            wdl=wdl,
        )


def _q_for_action(result, state, action):
    index = g.GoldenSearchAdapter().action_index(state, action)
    return result.root_q[index]


def _assert_same_sign(observed: float, expected: float) -> None:
    if expected > 0:
        assert observed > 0
    elif expected < 0:
        assert observed < 0
    else:
        assert observed == pytest.approx(0.0)


def test_qualification_corpus_has_eight_deterministic_solved_fixtures():
    assert len(QUALIFICATION_CASES) == 8
    assert {case.name for case in QUALIFICATION_CASES} == {
        "black_to_move_win",
        "white_to_move_win",
        "side_to_move_loss",
        "pass_best",
        "multi_ply_tactical",
        "same_board_open_history",
        "same_board_superko_blocked",
        "preterminal_pass_boundary",
    }


@pytest.mark.parametrize("case", QUALIFICATION_CASES, ids=lambda case: case.name)
def test_exact_solver_and_default_64_sim_puct_agree(case):
    solved = g.solve_exact(case.state, node_limit=10_000)
    assert solved.status == g.SolveStatus.EXACT
    assert solved.utility == case.expected_utility
    assert solved.best_actions == case.optimal_actions

    evaluator = ExactOracleEvaluator()
    searched = g.SequentialPUCT().search(case.state, evaluator, seed=0)
    assert searched.simulations == 64
    assert searched.action in solved.best_actions

    child = g.apply_action(case.state, searched.action).after
    child_solved = g.solve_exact(child, node_limit=10_000)
    assert child_solved.status == g.SolveStatus.EXACT
    expected_parent_utility = -child_solved.utility
    assert expected_parent_utility == case.expected_utility

    selected_q = _q_for_action(searched, case.state, searched.action)
    assert selected_q is not None
    _assert_same_sign(selected_q, expected_parent_utility)
    assert evaluator.terminal_calls == 0


def test_multi_ply_fixture_is_not_an_immediate_terminal_choice():
    case = next(item for item in QUALIFICATION_CASES if item.name == "multi_ply_tactical")
    solved = g.solve_exact(case.state, node_limit=10_000)
    assert solved.nodes > 100
    assert all(
        not g.apply_action(case.state, action).after.is_terminal
        for action in g.legal_actions(case.state)
    )
    assert solved.best_actions == (1,)


def test_same_board_different_superko_history_changes_identity_and_legal_continuation():
    open_case = next(item for item in QUALIFICATION_CASES if item.name == "same_board_open_history")
    blocked_case = next(item for item in QUALIFICATION_CASES if item.name == "same_board_superko_blocked")
    open_state = open_case.state
    blocked_state = blocked_case.state

    assert open_state.board_key == blocked_state.board_key
    assert open_state.side_to_move == blocked_state.side_to_move
    assert open_state.state_key != blocked_state.state_key
    assert 1 in g.legal_actions(open_state)
    assert 1 not in g.legal_actions(blocked_state)

    open_solved = g.solve_exact(open_state, node_limit=10_000)
    blocked_solved = g.solve_exact(blocked_state, node_limit=10_000)
    assert open_solved.utility == 1.0 and open_solved.best_actions == (1,)
    assert blocked_solved.utility == -1.0 and blocked_solved.best_actions == (0, 2, g.PASS)


def test_preterminal_pass_uses_exact_terminal_result_without_evaluator_call_on_child():
    case = next(item for item in QUALIFICATION_CASES if item.name == "preterminal_pass_boundary")
    evaluator = ExactOracleEvaluator()
    result = g.SequentialPUCT().search(case.state, evaluator, seed=0)
    assert result.action == g.PASS
    assert evaluator.calls == 1
    assert evaluator.terminal_calls == 0
    child = g.apply_action(case.state, g.PASS).after
    assert child.is_terminal


def test_draw_capable_research_fixture_uses_explicit_zero_komi_only():
    state = g.research_state_from_stones(
        (g.EMPTY,),
        side_to_move=g.BLACK,
        topology=ONE,
        komi=0.0,
        consecutive_passes=1,
    )
    solved = g.solve_exact(state, node_limit=100)
    assert solved.status == g.SolveStatus.EXACT
    assert solved.utility == 0.0
    assert solved.best_actions == (g.PASS,)

    evaluator = ExactOracleEvaluator()
    searched = g.SequentialPUCT().search(state, evaluator, seed=0)
    assert searched.action == g.PASS
    assert _q_for_action(searched, state, g.PASS) == pytest.approx(0.0)
    assert evaluator.terminal_calls == 0
    assert state.komi == 0.0
