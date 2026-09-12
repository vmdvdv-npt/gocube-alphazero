from __future__ import annotations

from pathlib import Path

import pytest

import gocube_golden as g


def board(*, black=(), white=(), points=25):
    stones = [g.EMPTY] * points
    for p in black:
        stones[p] = g.BLACK
    for p in white:
        stones[p] = g.WHITE
    return tuple(stones)


# ---------------------------------------------------------------------------
# Search qualification
# ---------------------------------------------------------------------------

class UniformEvaluator:
    def __init__(self, wdl=(0.5, 0.0, 0.5)):
        self.wdl = wdl
        self.calls = 0
        self.terminal_calls = 0

    def evaluate(self, state):
        self.calls += 1
        if state.is_terminal:
            self.terminal_calls += 1
            raise AssertionError("NN/fake evaluator must never score terminal winner")
        return g.Evaluation(
            policy={action: 1.0 for action in g.legal_actions(state)},
            wdl=self.wdl,
        )


def black_immediate_pass_win():
    state = g.apply_action(g.initial_state(), 0).after
    return g.apply_action(state, g.PASS).after


def white_immediate_pass_win():
    state = g.apply_action(g.initial_state(), g.PASS).after
    state = g.apply_action(state, 0).after
    return g.apply_action(state, g.PASS).after


@pytest.mark.parametrize("factory,expected_side", [
    (black_immediate_pass_win, g.BLACK),
    (white_immediate_pass_win, g.WHITE),
])
def test_immediate_win_both_colors_prefers_pass_without_perspective_inversion(factory, expected_side):
    state = factory()
    assert state.side_to_move == expected_side and state.consecutive_passes == 1
    evaluator = UniformEvaluator()
    result = g.SequentialPUCT(g.SearchSettings(simulations=128)).search(state, evaluator, seed=7)
    assert result.action == g.PASS
    assert evaluator.terminal_calls == 0


@pytest.mark.parametrize("side,stone_color", [
    (g.BLACK, g.WHITE),
    (g.WHITE, g.BLACK),
])
def test_immediate_loss_pass_is_not_preferred_for_either_color(side, stone_color):
    stones = board(
        black=(0,) if stone_color == g.BLACK else (),
        white=(0,) if stone_color == g.WHITE else (),
    )
    state = g.research_state_from_stones(
        stones, side_to_move=side, consecutive_passes=1,
    )
    result = g.SequentialPUCT(g.SearchSettings(simulations=128)).search(
        state, UniformEvaluator(), seed=3
    )
    assert result.action != g.PASS


class TrapEvaluator:
    """Player-relative evaluator that exposes a one-ply sign trap."""
    def __init__(self, root):
        self.root = root
        self.after_bad = g.apply_action(root, 0).after
        self.after_good = g.apply_action(root, 1).after

    def evaluate(self, state):
        legal = g.legal_actions(state)
        if state.state_key == self.root.state_key:
            # Intentionally bias prior toward the trap at 0.
            policy = {a: (8.0 if a == 0 else 2.0 if a == 1 else 0.1) for a in legal}
            return g.Evaluation(policy=policy, wdl=(0.5, 0.0, 0.5))
        if state.state_key == self.after_bad.state_key:
            # Opponent-to-move is winning: parent must see this as bad.
            return g.Evaluation(policy={a: 1.0 for a in legal}, wdl=(1.0, 0.0, 0.0))
        if state.state_key == self.after_good.state_key:
            # Opponent-to-move is losing: parent must see this as good.
            return g.Evaluation(policy={a: 1.0 for a in legal}, wdl=(0.0, 0.0, 1.0))
        return g.Evaluation(policy={a: 1.0 for a in legal}, wdl=(0.5, 0.0, 0.5))


def test_two_ply_player_relative_trap_catches_wrong_sign_backup():
    top = g.research_topology(((1,), (0, 2), (1,)), topology_id="stage2-trap-line3")
    root = g.initial_state(topology=top, komi=0.5)
    result = g.SequentialPUCT(g.SearchSettings(simulations=128)).search(
        root, TrapEvaluator(root), seed=0
    )
    assert result.action == 1


def test_terminal_leaf_uses_exact_golden_result_and_never_calls_evaluator_on_terminal():
    state = black_immediate_pass_win()
    evaluator = UniformEvaluator()
    result = g.SequentialPUCT(g.SearchSettings(simulations=64)).search(state, evaluator)
    assert result.root_visits[-1] > 0
    assert evaluator.terminal_calls == 0


def test_pass_is_in_action_space_and_can_be_selected():
    state = black_immediate_pass_win()
    result = g.SequentialPUCT(g.SearchSettings(simulations=64)).search(
        state, UniformEvaluator()
    )
    assert g.PASS in result.legal_actions
    assert len(result.root_visits) == state.topology.point_count + 1
    assert result.action == g.PASS


class IllegalPolicyEvaluator:
    def evaluate(self, state):
        # Point 0 is occupied in the fixture; all prior mass there must be ignored.
        policy = [0.0] * (state.topology.point_count + 1)
        policy[0] = 1000.0
        return g.Evaluation(policy=policy, wdl=(0.5, 0.0, 0.5))


def test_legal_mask_keeps_illegal_action_at_zero_visits():
    state = g.apply_action(g.initial_state(), 0).after
    result = g.SequentialPUCT(g.SearchSettings(simulations=32)).search(
        state, IllegalPolicyEvaluator()
    )
    assert 0 not in result.legal_actions
    assert result.root_visits[0] == 0


def test_root_visits_and_pi_are_valid_and_normalized():
    result = g.SequentialPUCT(g.SearchSettings(simulations=37)).search(
        g.initial_state(), UniformEvaluator(), seed=4
    )
    assert sum(result.root_visits) == 37
    assert sum(result.root_visits) > 0
    assert sum(result.pi) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in result.pi)


def test_search_does_not_mutate_parent_state():
    state = black_immediate_pass_win()
    before = state.state_key
    g.SequentialPUCT(g.SearchSettings(simulations=64)).search(state, UniformEvaluator(), seed=1)
    assert state.state_key == before


def test_same_contract_evaluator_and_seed_reproduce_root_visits_and_action():
    state = g.initial_state()
    settings = g.SearchSettings(simulations=50)
    one = g.SequentialPUCT(settings).search(state, UniformEvaluator(), seed=123)
    two = g.SequentialPUCT(settings).search(state, UniformEvaluator(), seed=123)
    assert one.root_visits == two.root_visits
    assert one.action == two.action
    assert one.pi == two.pi


def test_wdl_boundary_is_side_to_move_not_absolute_color():
    assert g.wdl_to_side_to_move_utility((1.0, 0.0, 0.0)) == 1.0
    assert g.wdl_to_side_to_move_utility((0.0, 1.0, 0.0)) == 0.0
    assert g.wdl_to_side_to_move_utility((0.0, 0.0, 1.0)) == -1.0
    with pytest.raises(g.SearchError):
        g.wdl_to_side_to_move_utility((1.0, 0.0))


def test_white_to_move_terminal_utility_has_correct_sign():
    state = white_immediate_pass_win()
    terminal = g.apply_action(state, g.PASS).after
    # The second PASS was by White, so terminal side_to_move is Black.
    assert terminal.side_to_move == g.BLACK
    assert g.GoldenSearchAdapter().terminal_utility(terminal) == -1.0


def test_tiny_solved_line3_oracle_and_search_agree():
    top = g.research_topology(((1,), (0, 2), (1,)), topology_id="stage2-solved-line3")
    state = g.initial_state(topology=top, komi=0.5)
    solved = g.solve_exact(state, node_limit=10_000)
    assert solved.status == g.SolveStatus.EXACT
    assert solved.utility == 1.0
    assert solved.best_actions == (1,)
    searched = g.SequentialPUCT(g.SearchSettings(simulations=128)).search(
        state, UniformEvaluator(), seed=0
    )
    assert searched.action == 1


def test_tiny_solver_returns_unknown_instead_of_partial_oracle():
    top = g.research_topology(((1,), (0, 2), (1,)), topology_id="stage2-unknown-line3")
    state = g.initial_state(topology=top, komi=0.5)
    solved = g.solve_exact(state, node_limit=2)
    assert solved.status == g.SolveStatus.UNKNOWN
    assert solved.utility is None
    assert solved.best_actions == ()


def test_search_contract_is_path_b_and_does_not_import_legacy_mcts():
    assert g.SEARCH_PATH == "B"
    assert g.SEARCH_IMPLEMENTATION_ID == "golden-sequential-puct-v1"
    assert g.WDL_SEMANTICS == "side-to-move:[WIN,DRAW,LOSS]"
    assert g.INTERNAL_Q_CONVENTION == "edge-Q-from-parent-side-to-move"
    assert g.SEARCH_IMPLEMENTATION_FINGERPRINT.startswith("sha256:")


def test_game_record_jsonl_persistence_uses_validated_raw_evidence(tmp_path):
    arena = g.SequentialGoldenArena(master_seed=12)
    arena.play_pair(pair_id="persist", player_A=g.GoodPlayer("A"), player_B=g.BadPlayer("B"))
    destination = tmp_path / "records.jsonl"
    g.write_records_jsonl(destination, arena.records)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"game_id": "persist-g1"' in lines[0]
    assert '"absolute_rule_result": "BLACK"' in lines[0]


def test_golden_stage2_python_has_no_production_arena_or_legacy_mcts_imports():
    package = Path(__file__).resolve().parents[1] / "gocube_golden"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "alphazero.Arena" not in sources
    assert "alphazero.MCTS" not in sources
    assert "from alphazero" not in sources
    assert "import alphazero" not in sources


def test_sequential_arena_search_scope_has_no_batch_worker_queue_or_multiprocessing():
    package = Path(__file__).resolve().parents[1] / "gocube_golden"
    scope = "\n".join(
        (package / name).read_text(encoding="utf-8")
        for name in ("arena.py", "players.py", "search.py", "search_adapter.py")
    )
    for forbidden in ("multiprocessing", "asyncio", "worker_queue", "inference_routing"):
        assert forbidden not in scope
