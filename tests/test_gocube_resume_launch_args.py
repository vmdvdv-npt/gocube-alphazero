from __future__ import annotations

import json

from tools import c4_overnight_experiment as runner
from tools.gocube_experiment_resume import STATE_SCHEMA_VERSION


def test_auto_resume_restores_saved_launch_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / "training_reports" / "night" / "experiment-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_VERSION,
            "experiment_id": "night",
            "status": "INTERRUPTED",
            "started_at_epoch": 1.0,
            "last_update_epoch": 2.0,
            "launch_config": {
                "bootstrap_run": "published-c7",
                "device": "cpu",
                "arena_batch_wait_ms": 2.0,
                "heldout_positions": 24,
                "health_gate_min_win_rate": 0.46,
                "benchmark_games": 96,
                "skip_performance_benchmark": True,
                "seed": 12345,
            },
        }),
        encoding="utf-8",
    )

    cli = runner.parse_args([])

    assert cli.experiment_id == "night"
    assert cli.bootstrap_run == "published-c7"
    assert cli.device == "cpu"
    assert cli.arena_batch_wait_ms == 2.0
    assert cli.heldout_positions == 24
    assert cli.health_gate_min_win_rate == 0.46
    assert cli.benchmark_games == 96
    assert cli.skip_performance_benchmark is True
    assert cli.seed == 12345


def test_explicit_resume_keeps_explicit_override_so_contract_check_can_reject_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / "training_reports" / "night" / "experiment-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_VERSION,
            "experiment_id": "night",
            "status": "INTERRUPTED",
            "started_at_epoch": 1.0,
            "last_update_epoch": 2.0,
            "launch_config": {
                "bootstrap_run": "published-c7",
                "device": "auto",
                "arena_batch_wait_ms": 1.0,
                "heldout_positions": 16,
                "health_gate_min_win_rate": 0.45,
                "benchmark_games": 64,
                "skip_performance_benchmark": False,
                "seed": 20260907,
            },
        }),
        encoding="utf-8",
    )

    cli = runner.parse_args(["--experiment-id", "night", "--device", "cpu"])

    assert cli.bootstrap_run == "published-c7"
    assert cli.device == "cpu"
