from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from gocube_golden.production_orchestrator import (
    SupervisionPolicy,
    UniversalProductionTrainingOrchestrator,
)


def _supervisor() -> UniversalProductionTrainingOrchestrator:
    supervisor = object.__new__(UniversalProductionTrainingOrchestrator)
    supervisor.supervision = SupervisionPolicy(
        startup_ack_timeout_seconds=1.0,
        progress_warning_seconds=1.0,
        progress_critical_seconds=2.0,
        critical_child_grace_seconds=0.1,
        restart_backoff_seconds=0.0,
        max_generation_restarts=0,
    )
    supervisor.events = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    return supervisor


def test_process_group_cleanup_reaps_descendants_after_leader_exit(tmp_path: Path):
    child_pid_path = tmp_path / "descendant.pid"
    script = (
        "import pathlib, subprocess, sys; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding='utf-8')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    supervisor = _supervisor()
    try:
        process.wait(timeout=5.0)
        deadline = time.monotonic() + 2.0
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file()
        assert supervisor._process_group_exists(process.pid)

        supervisor._terminate_child_group(
            process,
            reason="regression test: leader exited with live descendant",
        )

        assert process.poll() is not None
        assert not supervisor._process_group_exists(process.pid)
    finally:
        if supervisor._process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
