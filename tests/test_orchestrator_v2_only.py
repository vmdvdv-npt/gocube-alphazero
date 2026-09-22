from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_legacy_entrypoint(path: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, path, *args],
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


def test_legacy_torus9_driver_cli_is_disabled() -> None:
    result = _run_legacy_entrypoint(
        "tools/torus9_run_driver.py", "generation", "--generation", "1"
    )

    assert result.returncode != 0
    assert "Legacy Orchestrator V1" in result.stderr
    assert "orchestrator_v2.production_entrypoint" in result.stderr
