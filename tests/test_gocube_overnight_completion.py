from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import c4_overnight_experiment as overnight
from tools import gocube_checkpoint_arena as checkpoint_arena


def test_parameter_grids_and_benchmark_matrix_are_canonical():
    assert [spec["values"] for spec in overnight.PARAMETER_SPECS] == [
        (10.0, 19.0, 32.0),
        (0.15, 0.25, 0.35),
        (0.10, 0.25, 0.40),
        (0.5, 1.0, 2.0),
        (4, 8, 16),
    ]
    assert overnight.SELFPLAY_BENCHMARK_WAITS_MS == (0.5, 1.0, 2.0)
    assert overnight.ARENA_BENCHMARK_WAITS_MS == (0.5, 1.0, 2.0)
    assert overnight.ARENA_BENCHMARK_WORKERS == (4, 8, 16)
    assert max(overnight.ARENA_BENCHMARK_WORKERS) == 16


def test_confidence_decision_stops_only_on_supported_outcomes():
    assert overnight.decision_from_arena({"win_rate_ci95": [0.51, 0.62]}) == "IMPROVED"
    assert overnight.decision_from_arena({"win_rate_ci95": [0.38, 0.49]}) == "REGRESSED"
    assert overnight.decision_from_arena({"win_rate_ci95": [0.45, 0.55]}) == "NO_IMPROVEMENT"


def test_combined_arena_results_use_all_games_and_recompute_ci():
    base = {
        "wins": 70,
        "losses": 50,
        "draws": 8,
        "no_results": 0,
        "number_of_games": 128,
        "wall_time_seconds": 10.0,
        "by_color": {
            "black": {"games": 64, "wins": 35, "losses": 25, "draws": 4, "no_results": 0},
            "white": {"games": 64, "wins": 35, "losses": 25, "draws": 4, "no_results": 0},
        },
        "average_game_length": 60.0,
        "pass_count_mean": 2.0,
        "entered_cleanup1_fraction": 0.5,
        "entered_cleanup2_fraction": 0.25,
        "cleanup_moves_mean": 5.0,
        "cleanup_captures_mean": 1.0,
        "pass_alive_early_end_fraction": 0.1,
        "terminal_kind_counts": {"scored": 128},
    }
    result = overnight.combine_arena_results([base, base])
    assert result["number_of_games"] == 256
    assert result["wins"] == 140
    assert result["pass_count_mean"] == pytest.approx(2.0)
    assert result["terminal_kind_counts"] == {"scored": 256}
    assert result["win_rate_ci95"][0] < result["win_rate"] < result["win_rate_ci95"][1]


def test_markdown_report_contains_final_artifacts_and_resume_command():
    state = {
        "status": "COMPLETE",
        "fixed_contract": {
            "komi": 0.5,
            "workers": 16,
            "regular_sims": 50,
            "fast_sims": 20,
            "games_per_iteration": 256,
            "train_batch_size": 1024,
        },
        "performance_benchmark": {
            "selected": {"selfplay_wait_ms": 0.5, "arena_workers": 8, "arena_wait_ms": 1.0}
        },
        "parameters": [],
        "recommended_champion": {"run": "champ", "iteration": 17, "sweep_overrides": {}},
        "totals": {"training_games": 1000, "benchmark_selfplay_games": 768, "arena_games": 2000},
        "resume_production_command": ".venv/bin/python -m alphazero.envs.gocube.hardened_train",
    }
    report = overnight.render_markdown_report(state)
    assert "komi **0.5**" in report
    assert "Arena workers: **8**" in report
    assert "Recommended champion" in report
    assert "Resume production" in report
    assert "--max-hours" not in report


def test_arena_endgame_summary_reports_pass_cleanup_and_terminal_stats():
    semantic_a = SimpleNamespace(
        black_pass_states=(b"a",),
        white_pass_states=(b"b",),
        entered_cleanup1=True,
        entered_cleanup2=False,
        pass_alive_early_end=False,
        cleanup1_moves=(2, 1),
        cleanup2_moves=(0, 0),
        cleanup_captures=1,
        terminal_kind="scored",
    )
    semantic_b = SimpleNamespace(
        black_pass_states=(),
        white_pass_states=(b"b",),
        entered_cleanup1=True,
        entered_cleanup2=True,
        pass_alive_early_end=True,
        cleanup1_moves=(1, 1),
        cleanup2_moves=(2, 0),
        cleanup_captures=3,
        terminal_kind="scored",
    )
    diagnostics = [
        checkpoint_arena._terminal_diagnostics(SimpleNamespace(semantic_state=semantic_a)),
        checkpoint_arena._terminal_diagnostics(SimpleNamespace(semantic_state=semantic_b)),
    ]
    summary = checkpoint_arena._summarize_endgame(diagnostics)
    assert summary["pass_count_mean"] == pytest.approx(1.5)
    assert summary["entered_cleanup1_fraction"] == pytest.approx(1.0)
    assert summary["entered_cleanup2_fraction"] == pytest.approx(0.5)
    assert summary["pass_alive_early_end_fraction"] == pytest.approx(0.5)
    assert summary["cleanup_captures_mean"] == pytest.approx(2.0)
    assert summary["terminal_kind_counts"] == {"scored": 2}


def test_training_command_keeps_fixed_contract_after_benchmark_selection():
    experiment = object.__new__(overnight.Experiment)
    experiment.python = overnight.Path(".venv/bin/python")
    experiment.selfplay_wait_ms = 0.5
    command = experiment.training_command(
        run_name="champ",
        target_iteration=18,
        sweep_overrides={"--replay-window-iters": 8},
        resume=True,
    )
    joined = " ".join(command)
    assert "--workers 16" in joined
    assert "--sims 50" in joined
    assert "--train-batch-size 1024" in joined
    assert "--inference-batch-wait-ms 0.5" in joined
    assert "--replay-window-iters 8" in joined
    assert "--allow-existing-run" in command
