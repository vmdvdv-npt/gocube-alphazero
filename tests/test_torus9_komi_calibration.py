from __future__ import annotations

from tools.torus9_komi_calibration import (
    Trajectory,
    bootstrap_estimates,
    crossing_interval,
    crossing_point,
    sweep_trajectories,
)
import gocube_golden as g
from gocube_golden.torus9 import summarize_torus9_arena


def test_board_size_scaled_arena_watchdog_contract():
    assert g.resolve_arena_watchdog(5) == 500
    assert g.resolve_arena_watchdog((5, 5)) == 500
    assert g.resolve_arena_watchdog((9, 9)) == 1000
    assert g.resolve_arena_watchdog((9, 13)) == 1500
    assert g.TORUS9_ARENA_MOVE_LIMIT == 1000
    assert g.TORUS9_MOVE_LIMIT == 500


def test_watchdog_is_technical_and_double_pass_before_it_is_formal():
    live = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    assert g.torus9_arena_termination_reason(live, 1000) == "TRUNCATED_MOVE_LIMIT"
    after_one_pass = g.apply_action(live, g.PASS).after
    terminal = g.apply_action(after_one_pass, g.PASS).after
    assert g.torus9_arena_termination_reason(terminal, 2) == "DOUBLE_PASS"


def test_technical_watchdog_row_is_not_reclassified_as_wdl():
    summary = summarize_torus9_arena(
        [{
            "pair_id": "p1",
            "game_id": "p1-g1",
            "mapped_result": None,
            "formal_result": None,
            "technical_termination": "TRUNCATED_MOVE_LIMIT",
        }],
        candidate_label="M8",
        reference_label="M8",
        pairs=1,
    )
    assert summary["W/L/D"] == [0, 0, 0]
    assert summary["technical_games"] == 1
    assert summary["technical_fail_closed"] is True


def test_counterfactual_sweep_reports_crossing_and_wilson_ci():
    rows = [
        Trajectory("test", f"g-{raw}", None, "fixture", "BLACK", raw, 10 + raw, 10, 2)
        for raw in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    ]
    analysis = sweep_trajectories(rows)
    assert analysis["training_komi"] == 0.5
    assert analysis["komi_sweep"]["0.5"]["black_wins"] == 8
    assert analysis["komi_sweep"]["0.5"]["white_wins"] == 1
    assert analysis["komi_sweep"]["0.5"]["wilson_95_ci"][0] >= 0.0
    assert crossing_interval(analysis["komi_sweep"]) == [3.5, 4.5]
    assert 3.5 < crossing_point(analysis["komi_sweep"]) < 4.5


def test_paired_bootstrap_uses_pairs_as_sampling_units():
    rows = [
        Trajectory("paired", "a", None, "fixture", "BLACK", 1.0, 11, 10, 4, pair_id="p1"),
        Trajectory("paired", "b", None, "fixture", "WHITE", 3.0, 13, 10, 4, pair_id="p1"),
        Trajectory("paired", "c", None, "fixture", "BLACK", 5.0, 15, 10, 4, pair_id="p2"),
        Trajectory("paired", "d", None, "fixture", "WHITE", 7.0, 17, 10, 4, pair_id="p2"),
    ]
    result = bootstrap_estimates(rows, seed=17, resamples=32, pairwise=True)
    assert result["pairwise"] is True
    assert result["sampling_unit_count"] == 2
    assert result["resamples"] == 32


def test_torus9_profile_pins_new_arena_watchdog_contract():
    profile = g.load_torus9_profile()
    assert profile["arena"]["contract_id"] == "torus9-golden-arena-search-v2"
    assert profile["arena"]["watchdog"] == 1000
    assert profile["self_play"]["watchdog"] == 500
