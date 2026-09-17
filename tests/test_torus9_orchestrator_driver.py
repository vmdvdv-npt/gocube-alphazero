from __future__ import annotations

from pathlib import Path

import tools.torus9_run_driver as run_driver


ROOT = Path(__file__).resolve().parents[1]


def test_no_checked_in_torus9_execution_policy_preset():
    assert not (ROOT / "configs" / "gocube" / "torus9_training_orchestrator_v1.json").exists()


def test_active_torus9_driver_exposes_no_execution_cli_defaults():
    parser = run_driver.build_parser()
    generation = parser.parse_args(["generation", "--generation", "1"])
    assert vars(generation) == {
        "command": "generation",
        "generation": 1,
        "resume": False,
        "func": run_driver.run_generation,
    }
    arena = parser.parse_args(["arena", "--generation", "3"])
    assert vars(arena) == {
        "command": "arena",
        "generation": 3,
        "func": run_driver.run_arena,
    }


def test_active_torus9_driver_has_no_embedded_historical_execution_preset():
    source = (ROOT / "tools" / "torus9_run_driver.py").read_text(encoding="utf-8")
    assert "PERIODIC_ARENA_PRESET" not in source
    assert "PERIODIC_ARENA_STARTSET" not in source
    assert "LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE" not in source
    assert "torus9_orchestrator_driver" not in source
    assert "monkeypatch" not in source.lower()


def test_torus9_driver_is_not_claimed_as_cube_driver():
    assert run_driver.__name__ == "tools.torus9_run_driver"
    assert "cube" not in run_driver.__doc__.lower()


def test_periodic_arena_uses_resolved_checkpoint_references():
    source = (ROOT / "tools" / "torus9_run_driver.py").read_text(encoding="utf-8")
    assert "resolve_checkpoint(" in source
    assert "evaluation_id_for_comparison(" in source
    assert "ensure_evaluation_layout(" in source
