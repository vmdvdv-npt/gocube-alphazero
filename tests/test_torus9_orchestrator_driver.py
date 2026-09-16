from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import tools.torus9_orchestrator_driver as legacy_driver
import tools.torus9_run_driver as run_driver


ROOT = Path(__file__).resolve().parents[1]


def test_no_checked_in_torus9_execution_policy_preset():
    assert not (ROOT / "configs" / "gocube" / "torus9_training_orchestrator_v1.json").exists()


def test_active_torus9_driver_does_not_expose_legion_execution_cli_defaults():
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


def test_active_torus9_driver_has_no_embedded_periodic_arena_preset():
    assert not hasattr(run_driver, "PERIODIC_ARENA_PRESET")
    assert not hasattr(run_driver, "PERIODIC_ARENA_STARTSET")
    assert not hasattr(run_driver, "LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE")


def test_cube_is_not_claimed_as_a_current_orchestrator_driver():
    assert run_driver.__name__ == "tools.torus9_run_driver"
    assert "cube" not in run_driver.__doc__.lower()


def test_atomic_json_is_safe_for_concurrent_heartbeat_writers(tmp_path: Path):
    path = tmp_path / "heartbeat.json"
    errors: list[BaseException] = []

    def write(worker: int) -> None:
        try:
            for iteration in range(50):
                legacy_driver._atomic_json(
                    path,
                    {"worker": worker, "iteration": iteration},
                )
        except BaseException as exc:  # pragma: no cover - assertion captures thread failure
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload["worker"], int)
    assert isinstance(payload["iteration"], int)
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_lineage_git_commit_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"git_commit": "a" * 40}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        legacy_driver,
        "capture_code_identity",
        lambda _root: SimpleNamespace(
            git_commit_sha="b" * 40,
            git_tree_sha="c" * 40,
            working_tree_clean=True,
        ),
    )
    with pytest.raises(ValueError, match="git commit drift"):
        legacy_driver._validate_code_pin(tmp_path)


def test_lineage_git_commit_pin_accepts_same_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    commit = "a" * 40
    (tmp_path / "manifest.json").write_text(
        json.dumps({"git_commit": commit}),
        encoding="utf-8",
    )
    identity = SimpleNamespace(
        git_commit_sha=commit,
        git_tree_sha="c" * 40,
        working_tree_clean=True,
    )
    monkeypatch.setattr(legacy_driver, "capture_code_identity", lambda _root: identity)
    assert legacy_driver._validate_code_pin(tmp_path) is identity
