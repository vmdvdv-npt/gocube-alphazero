from types import SimpleNamespace

from torch import multiprocessing as mp

from alphazero.SelfPlayAgent import SelfPlayAgent


class _GameClass:
    @staticmethod
    def pass_action():
        return 1


class _State:
    phase = "main"
    consecutive_passes = 1


class _Game:
    semantic_state = _State()


class _SuppressedPassMCTS:
    @staticmethod
    def pass_diagnostic(_game):
        return {
            "pass_root_prior": 0.99,
            "pass_visit_fraction": 0.80,
            "pass_win_utility": 0.10,
            "pass_score_utility": -0.20,
            "pass_combined_utility": -0.10,
            "best_nonpass_score_gain": 4.5,
            "best_nonpass_win_delta": 0.0,
            "score_dominated_pass": True,
            "pass_suppressed": True,
        }

    @staticmethod
    def best_action(_game):
        # Production best_action sees the post-suppression counts, therefore
        # it is deliberately non-PASS here. The audit must still count the raw
        # score-dominance condition.
        return 0


def _counter(kind="d"):
    return mp.Value(kind, 0)


def test_score_dominated_pass_counter_uses_diagnostic_not_post_suppression_best_action():
    agent = SelfPlayAgent.__new__(SelfPlayAgent)
    agent.score_aware = True
    agent.args = SimpleNamespace(gocube_search_audit_probability=1.0)
    agent.game_cls = _GameClass
    agent.telemetry = {
        "search_audited_positions": _counter("q"),
        "search_pass_root_prior_sum": _counter(),
        "search_pass_visit_fraction_sum": _counter(),
        "search_pass_win_utility_sum": _counter(),
        "search_pass_score_utility_sum": _counter(),
        "search_pass_combined_utility_sum": _counter(),
        "search_best_nonpass_score_gain_sum": _counter(),
        "search_best_nonpass_win_delta_sum": _counter(),
        "search_score_dominated_pass": _counter("q"),
    }
    agent._SelfPlayAgent__mcts_current = _SuppressedPassMCTS()

    agent._record_search_audit(_Game(), action=0)

    assert agent.telemetry["search_audited_positions"].value == 1
    assert agent.telemetry["search_score_dominated_pass"].value == 1
    assert agent.telemetry["search_best_nonpass_score_gain_sum"].value == 4.5
