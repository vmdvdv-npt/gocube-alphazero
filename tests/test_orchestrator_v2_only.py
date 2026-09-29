from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from gocube_golden.orchestrator_v2 import ArenaRunner
from gocube_golden.orchestrator_v2 import production_entrypoint
from gocube_golden.orchestrator_v2.production_entrypoint import run_spec


ROOT = Path(__file__).resolve().parents[1]


def _run_legacy_entrypoint(path: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, path, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_v2_entrypoint(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gocube_golden.orchestrator_v2.production_entrypoint", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_legacy_training_orchestrator_cli_is_disabled() -> None:
    result = _run_legacy_entrypoint("tools/training_orchestrator.py", "run")

    assert result.returncode != 0
    assert "Legacy Orchestrator V1" in result.stderr
    assert "orchestrator_v2.production_entrypoint" in result.stderr


def test_legacy_training_orchestrator_core_cli_is_disabled() -> None:
    result = _run_legacy_entrypoint("tools/training_orchestrator_core.py", "run")

    assert result.returncode != 0
    assert "Legacy Orchestrator V1" in result.stderr
    assert "orchestrator_v2.production_entrypoint" in result.stderr


def test_legacy_torus9_driver_cli_is_disabled() -> None:
    result = _run_legacy_entrypoint(
        "tools/torus9_run_driver.py", "generation", "--generation", "1"
    )

    assert result.returncode != 0
    assert "Legacy Orchestrator V1" in result.stderr
    assert "orchestrator_v2.production_entrypoint" in result.stderr


def test_direct_v2_arena_runner_requires_production_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZ_ORCHESTRATOR_VERSION", raising=False)

    with pytest.raises(RuntimeError, match="requires .*production_entrypoint"):
        ArenaRunner().run(object())  # type: ignore[arg-type]


def test_standalone_arena_cli_requires_production_entrypoint() -> None:
    result = _run_legacy_entrypoint("tools/arena.py", "--candidate", "/tmp/missing.pt")

    assert result.returncode != 0
    assert "requires" in result.stderr
    assert "orchestrator_v2.production_entrypoint" in result.stderr


@pytest.mark.parametrize("mode", ["arena", "evaluation"])
def test_standalone_v2_arena_run_spec_is_disabled(mode: str) -> None:
    with pytest.raises(RuntimeError, match="Standalone Arena run-specs are disabled"):
        run_spec({
            "schema": "gocube-orchestrator-v2-run-spec-v1",
            "mode": mode,
            mode: {},
        })


@pytest.mark.parametrize(
    "command",
    ["run", "continuous", "performance-tuning", "experiment", "komi-calibration", "workflow"],
)
def test_public_production_launch_commands_are_disabled(command: str) -> None:
    result = _run_v2_entrypoint(
        command,
        "/tmp/configuration-is-not-read.json",
    )

    assert result.returncode != 0
    assert "single gocube-operator-job-v1 file" in result.stderr


def test_hidden_workflow_controller_requires_operator_job_permit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        production_entrypoint,
        "_workflow_spec_from_payload",
        lambda _payload: type("Spec", (), {"topology": "torus9", "workflow_id": "job"})(),
    )

    with pytest.raises(RuntimeError, match="execution permit"):
        production_entrypoint._require_operator_workflow_controller({})
