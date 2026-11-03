"""Keep the isolated experimental harness in CI without loading candidate weights."""
from pathlib import Path
import subprocess
import sys


def test_isolated_weightspace_session_contracts():
    root = Path(__file__).resolve().parents[1]
    script = root / 'runs/experiments/alpha-zero-from-scratch/torus_weightspace_starter/v002/test_session.py'
    result = subprocess.run(
        [sys.executable, str(script), '-v'],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
