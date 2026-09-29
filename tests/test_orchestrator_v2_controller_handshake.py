import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gocube_golden.orchestrator_v2 import production_entrypoint as entry
from gocube_golden.orchestrator_v2.execution_permit import (
    PERMIT_ENV,
    PERMIT_KEY_ENV,
    _child_execution_permit,
    _production_authority,
    require_child_execution_permit,
)


def _workflow_config(path: Path, workflow_id: str = "handshake-workflow") -> Path:
    path.write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "topology": "torus9",
                "steps": [
                    {
                        "step_id": "stop",
                        "action": "stop",
                        "config": {"reason": "handshake regression"},
                    }
                ],
            }
        )
    )
    return path


def _permit_context(workflow_id: str):
    code_identity = entry._entrypoint_code_identity()
    authority = _production_authority(
        mode="workflow",
        topology="torus9",
        run_id=workflow_id,
        code_identity=code_identity,
    )
    permit = _child_execution_permit(
        action_type="workflow-controller",
        topology="torus9",
        run_id=workflow_id,
        code_identity=code_identity,
    )
    return authority, permit, code_identity


def test_real_controller_subprocess_sends_ready_before_stop_workflow(tmp_path):
    config = _workflow_config(tmp_path / "workflow.json")
    workflow_id = "handshake-workflow"
    authority, permit, _ = _permit_context(workflow_id)
    with authority:
        with permit:
            result = entry._launch_durable_workflow_controller(
                config,
                runs_root=tmp_path / "runs",
                _startup_timeout_seconds=10,
            )

    assert result["state"] == "STARTED"
    assert result["controller_pid"] > 0
    state = (
        tmp_path
        / "runs"
        / "torus9"
        / "orchestration"
        / "workflows"
        / workflow_id
        / "state.json"
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not state.exists():
        time.sleep(0.01)
    assert json.loads(state.read_text())["state"] == "STOPPED"


def test_handshake_waits_for_real_child_permit_validation(tmp_path, monkeypatch):
    config = _workflow_config(tmp_path / "workflow.json", "delayed-handshake")
    workflow_id = "delayed-handshake"
    authority, permit, code_identity = _permit_context(workflow_id)
    real_popen = entry.subprocess.Popen
    child_holder = {}

    child_code = f"""
import json
import os
import sys
import time
from gocube_golden.orchestrator_v2.execution_permit import require_child_execution_permit

require_child_execution_permit(
    'handshake-test',
    action_type='workflow-controller',
    topology='torus9',
    run_id={workflow_id!r},
    code_identity={code_identity!r},
)
time.sleep(0.3)
os.write(
    int(sys.argv[1]),
    (json.dumps({{
        'schema': {entry.CONTROLLER_READY_SCHEMA!r},
        'pid': os.getpid(),
        'workflow_id': {workflow_id!r},
        'code_identity': {code_identity!r},
    }}) + '\\n').encode(),
)
"""

    def spawn_test_child(command, **kwargs):
        if "--startup-ready-fd" not in command:
            return real_popen(command, **kwargs)
        fd = command[command.index("--startup-ready-fd") + 1]
        child = real_popen([sys.executable, "-c", child_code, fd], **kwargs)
        child_holder["process"] = child
        return child

    monkeypatch.setattr(entry.subprocess, "Popen", spawn_test_child)
    started = time.monotonic()
    with authority:
        with permit:
            result = entry._launch_durable_workflow_controller(
                config,
                runs_root=tmp_path / "runs",
                _startup_timeout_seconds=5,
            )
    elapsed = time.monotonic() - started

    assert result["state"] == "STARTED"
    assert elapsed >= 0.25
    assert child_holder["process"].wait(timeout=2) == 0


def test_controller_without_permit_is_rejected_without_ready(tmp_path):
    config = _workflow_config(tmp_path / "workflow.json", "unauthorized-controller")
    environment = dict(os.environ)
    environment.pop(PERMIT_ENV, None)
    environment.pop(PERMIT_KEY_ENV, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gocube_golden.orchestrator_v2.production_entrypoint",
            "workflow",
            str(config),
            "--runs-root",
            str(tmp_path / "runs"),
            "--controller",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "execution permit" in result.stderr
    assert "READY" not in result.stdout + result.stderr


def test_wrong_parent_pid_permit_still_fails_closed():
    with _production_authority(
        mode="workflow", topology="torus9", run_id="wrong-parent", code_identity="test"
    ):
        with _child_execution_permit(
            action_type="workflow-controller",
            topology="torus9",
            run_id="wrong-parent",
            code_identity="test",
        ):
            with pytest.raises(RuntimeError, match="not bound to this child process"):
                require_child_execution_permit(
                    "handshake-test",
                    action_type="workflow-controller",
                    topology="torus9",
                    run_id="wrong-parent",
                    code_identity="test",
                    parent_pid=os.getpid() + 1,
                )


def test_child_failure_before_ready_is_reaped(tmp_path, monkeypatch):
    config = _workflow_config(tmp_path / "workflow.json", "early-failure")
    authority, permit, _ = _permit_context("early-failure")
    real_popen = entry.subprocess.Popen

    def spawn_failing_child(_command, **kwargs):
        if "--startup-ready-fd" not in _command:
            return real_popen(_command, **kwargs)
        return real_popen([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)

    monkeypatch.setattr(entry.subprocess, "Popen", spawn_failing_child)
    with authority:
        with permit:
            with pytest.raises(RuntimeError, match=r"failed before READY.*exit code: 7"):
                entry._launch_durable_workflow_controller(
                    config,
                    runs_root=tmp_path / "runs",
                    _startup_timeout_seconds=5,
                )


def test_startup_timeout_kills_process_group_and_reaps_child(tmp_path, monkeypatch):
    config = _workflow_config(tmp_path / "workflow.json", "startup-timeout")
    authority, permit, _ = _permit_context("startup-timeout")
    real_popen = entry.subprocess.Popen
    holder = {}

    def spawn_hanging_child(_command, **kwargs):
        if "--startup-ready-fd" not in _command:
            return real_popen(_command, **kwargs)
        process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        holder["process"] = process
        return process

    monkeypatch.setattr(entry.subprocess, "Popen", spawn_hanging_child)
    with authority:
        with permit:
            with pytest.raises(RuntimeError, match="did not become ready within startup timeout"):
                entry._launch_durable_workflow_controller(
                    config,
                    runs_root=tmp_path / "runs",
                    _startup_timeout_seconds=0.1,
                    _startup_grace_seconds=0.1,
                )

    assert holder["process"].poll() is not None


def test_wrong_ready_pid_is_rejected_and_child_is_reaped(tmp_path, monkeypatch):
    config = _workflow_config(tmp_path / "workflow.json", "spoofed-ready")
    authority, permit, _ = _permit_context("spoofed-ready")
    real_popen = entry.subprocess.Popen
    holder = {}
    ready = {
        "schema": entry.CONTROLLER_READY_SCHEMA,
        "pid": 999999,
        "workflow_id": "spoofed-ready",
        "code_identity": entry._entrypoint_code_identity(),
    }
    child_code = (
        "import os, sys, time; "
        "os.write(int(sys.argv[1]), "
        + repr((json.dumps(ready) + "\n").encode())
        + "); time.sleep(60)"
    )

    def spawn_spoofed_child(command, **kwargs):
        if "--startup-ready-fd" not in command:
            return real_popen(command, **kwargs)
        fd = command[command.index("--startup-ready-fd") + 1]
        process = real_popen([sys.executable, "-c", child_code, fd], **kwargs)
        holder["process"] = process
        return process

    monkeypatch.setattr(entry.subprocess, "Popen", spawn_spoofed_child)
    with authority:
        with permit:
            with pytest.raises(RuntimeError, match="READY PID"):
                entry._launch_durable_workflow_controller(
                    config,
                    runs_root=tmp_path / "runs",
                    _startup_timeout_seconds=5,
                    _startup_grace_seconds=0.1,
                )

    assert holder["process"].poll() is not None
