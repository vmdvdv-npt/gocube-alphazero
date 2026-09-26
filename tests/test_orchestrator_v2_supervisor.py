from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from gocube_golden.process_supervision import atomic_write_json, process_group_exists
from gocube_golden.orchestrator_v2 import (
    ACTIVE_CHILD_SCHEMA,
    ActiveChild,
    EXECUTION_INTENT_SCHEMA,
    STOP_SCHEMA,
    SupervisorAction,
    SupervisorIntegrityError,
    SupervisorPolicy,
    SupervisorStatus,
    SupervisorV2,
)


def _supervisor(
    root: Path,
    *,
    execution_id: str = "opaque-execution",
    command: list[str] | None = None,
    policy: SupervisorPolicy | None = None,
    env: dict[str, str] | None = None,
) -> SupervisorV2:
    heartbeat = root / "runtime" / "heartbeat.json"
    return SupervisorV2(
        root,
        execution_id=execution_id,
        liveness_path=heartbeat,
        progress_path=heartbeat,
        command=command or [sys.executable, "-c", "pass"],
        env=env,
        policy=policy
        or SupervisorPolicy(
            heartbeat_grace_seconds=1.0,
            poll_interval_seconds=0.01,
            termination_grace_seconds=0.1,
        ),
    )


def test_success_is_child_exit_zero(tmp_path: Path) -> None:
    result = _supervisor(tmp_path).run_once()

    assert result.success
    assert result.status is SupervisorStatus.SUCCESS
    assert result.returncode == 0
    assert result.attempts == 1
    assert not (tmp_path / "runtime" / "active-child.json").exists()


def test_non_zero_exit_is_failure_and_retries(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...] | None, int]] = []

    def launcher(request):
        calls.append((request.command, request.attempt))
        code = 17 if request.attempt == 1 else 0
        return subprocess.Popen(
            [sys.executable, "-c", f"raise SystemExit({code})"],
            start_new_session=True,
        )

    supervisor = _supervisor(tmp_path)
    supervisor = SupervisorV2(
        tmp_path,
        execution_id="same-execution",
        liveness_path=tmp_path / "runtime" / "live.json",
        progress_path=tmp_path / "runtime" / "progress.json",
        launcher=launcher,
        command=None,
        policy=supervisor.policy,
    )

    result = supervisor.run_once()

    assert result.success
    assert result.attempts == 2
    assert calls == [(None, 1), (None, 2)]


def test_non_zero_exit_after_retry_is_failure(tmp_path: Path) -> None:
    calls: list[int] = []

    def launcher(request):
        calls.append(request.attempt)
        return subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(23)"],
            start_new_session=True,
        )

    supervisor = SupervisorV2(
        tmp_path,
        execution_id="failing-execution",
        liveness_path=tmp_path / "runtime" / "live.json",
        progress_path=tmp_path / "runtime" / "progress.json",
        launcher=launcher,
        policy=SupervisorPolicy(max_retries=1, poll_interval_seconds=0.005),
    )

    result = supervisor.run_once()

    assert not result.success
    assert result.returncode == 23
    assert result.attempts == 2
    assert calls == [1, 2]


def test_stale_liveness_terminates_child(tmp_path: Path) -> None:
    supervisor = _supervisor(
        tmp_path,
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        policy=SupervisorPolicy(
            liveness_timeout_seconds=0.03,
            progress_timeout_seconds=1.0,
            max_retries=0,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.05,
        ),
    )

    result = supervisor.run_once()

    assert not result.success
    assert "liveness" in (result.reason or "")
    assert not supervisor.active_child_path.exists()
    assert supervisor.stop_path.is_file()


def test_stale_progress_terminates_even_when_liveness_changes(tmp_path: Path) -> None:
    heartbeat = tmp_path / "runtime" / "heartbeat.json"
    loop = (
        "while True:\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_text(json.dumps({'liveness_at': time.time(), 'progress_at': started, 'progress_token': 'fixed'}))\n"
        "    time.sleep(0.005)\n"
    )
    script = (
        "import json, time; from pathlib import Path; "
        f"p=Path({str(heartbeat)!r}); started=time.time(); exec({loop!r})"
    )
    supervisor = _supervisor(
        tmp_path,
        command=[sys.executable, "-c", script],
        policy=SupervisorPolicy(
            liveness_timeout_seconds=0.2,
            progress_timeout_seconds=0.03,
            max_retries=0,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.05,
        ),
    )

    result = supervisor.run_once()

    assert not result.success
    assert "progress" in (result.reason or "")
    assert not supervisor.active_child_path.exists()
    events = [
        json.loads(line)
        for line in supervisor.attempts_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["reason"] == "progress heartbeat is missing or stale"


def test_stagnant_progress_is_allowed_when_progress_timeout_is_disabled(tmp_path: Path) -> None:
    heartbeat = tmp_path / "runtime" / "heartbeat.json"
    loop = (
        "def beat():\n"
        "    while True:\n"
        "        p.parent.mkdir(parents=True, exist_ok=True)\n"
        "        p.write_text(json.dumps({'liveness_at': time.time(), 'progress_at': started, 'progress_token': 'fixed'}))\n"
        "        time.sleep(0.005)\n"
    )
    script = (
        "import json, threading, time; from pathlib import Path; "
        f"p=Path({str(heartbeat)!r}); started=time.time(); "
        f"exec({loop!r}); "
        "threading.Thread(target=beat, daemon=True).start(); time.sleep(0.15)"
    )
    policy = SupervisorPolicy(
        heartbeat_grace_seconds=300.0,
        liveness_timeout_seconds=0.2,
        progress_timeout_seconds=None,
        max_retries=0,
        poll_interval_seconds=0.005,
        termination_grace_seconds=0.05,
    )
    supervisor = _supervisor(
        tmp_path,
        command=[sys.executable, "-c", script],
        policy=policy,
    )

    result = supervisor.run_once()

    assert policy.liveness_grace_seconds == 0.2
    assert policy.progress_grace_seconds is None
    assert result.success


def test_retry_repeats_exact_same_command(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "raise SystemExit(19)"]
    seen: list[tuple[str, ...]] = []

    def launcher(request):
        assert request.command is not None
        seen.append(request.command)
        code = 19 if request.attempt == 1 else 0
        return subprocess.Popen(
            [sys.executable, "-c", f"raise SystemExit({code})"],
            start_new_session=True,
        )

    supervisor = SupervisorV2(
        tmp_path,
        execution_id="retry-execution",
        liveness_path=tmp_path / "runtime" / "live.json",
        progress_path=tmp_path / "runtime" / "progress.json",
        launcher=launcher,
        command=command,
        policy=SupervisorPolicy(poll_interval_seconds=0.005),
    )

    result = supervisor.run_once()

    assert result.success
    assert seen == [tuple(command), tuple(command)]


def test_acknowledge_matching_stopped_execution_allows_new_attempt(tmp_path: Path) -> None:
    failed = _supervisor(
        tmp_path,
        execution_id="retryable-execution",
        command=[sys.executable, "-c", "raise SystemExit(19)"],
        policy=SupervisorPolicy(max_retries=0, poll_interval_seconds=0.005),
    )

    result = failed.run_once()
    assert not result.success
    assert failed.stop_path.is_file()
    assert failed.execution_intent_path.is_file()

    acknowledged = failed.acknowledge_stopped_execution()

    assert acknowledged.success
    assert not failed.stop_path.exists()
    assert not failed.execution_intent_path.exists()
    retry = _supervisor(
        tmp_path,
        execution_id="retryable-execution",
        command=[sys.executable, "-c", "pass"],
        policy=failed.policy,
    )
    assert retry.plan().action is SupervisorAction.START
    assert retry.run_once().success


def test_acknowledge_terminates_owned_live_group_before_cleanup(tmp_path: Path) -> None:
    supervisor = _supervisor(
        tmp_path,
        execution_id="live-stopped-execution",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        policy=SupervisorPolicy(
            max_retries=0,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.05,
        ),
    )
    process, active = supervisor._start(1)
    atomic_write_json(
        supervisor.stop_path,
        {
            "schema": STOP_SCHEMA,
            "execution_id": "live-stopped-execution",
            "attempt": 1,
            "pid": active.pid,
            "process_group": active.process_group,
        },
    )

    try:
        acknowledged = supervisor.acknowledge_stopped_execution()
        assert acknowledged.success
        assert not process_group_exists(active.process_group)
        assert not supervisor.active_child_path.exists()
        assert not supervisor.execution_intent_path.exists()
        assert not supervisor.stop_path.exists()
    finally:
        if process_group_exists(active.process_group):
            os.killpg(active.process_group, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except ChildProcessError:
            pass


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema": STOP_SCHEMA,
            "execution_id": "foreign-execution",
            "attempt": 1,
        },
        {
            "schema": "wrong-supervisor-stop-schema",
            "execution_id": "retryable-execution",
            "attempt": 1,
        },
    ],
)
def test_acknowledge_does_not_remove_foreign_or_malformed_stop(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    supervisor = _supervisor(tmp_path, execution_id="retryable-execution")
    atomic_write_json(supervisor.stop_path, payload)

    with pytest.raises(SupervisorIntegrityError, match="supervisor stop is malformed"):
        supervisor.acknowledge_stopped_execution()

    assert supervisor.stop_path.read_text(encoding="utf-8") == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_acknowledge_without_stop_preserves_active_child_for_reattach(tmp_path: Path) -> None:
    first = _supervisor(
        tmp_path,
        execution_id="restartable-execution",
        command=[sys.executable, "-c", "import time; time.sleep(0.25)"],
        policy=SupervisorPolicy(
            max_retries=0,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.05,
        ),
    )
    process, active = first._start(1)
    second = _supervisor(
        tmp_path,
        execution_id="restartable-execution",
        command=[sys.executable, "-c", "pass"],
        policy=first.policy,
    )

    try:
        acknowledged = second.acknowledge_stopped_execution()
        assert acknowledged.success
        assert second.active_child_path.is_file()
        assert second.execution_intent_path.is_file()
        result = second.run_once()
        assert result.success
        assert result.reattached
    finally:
        if process_group_exists(active.process_group):
            os.killpg(active.process_group, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except ChildProcessError:
            pass


def test_acknowledge_fails_closed_for_live_foreign_execution(tmp_path: Path) -> None:
    foreign = _supervisor(
        tmp_path,
        execution_id="foreign-execution",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        policy=SupervisorPolicy(max_retries=0, termination_grace_seconds=0.05),
    )
    process, active = foreign._start(1)
    expected = _supervisor(tmp_path, execution_id="expected-execution")
    atomic_write_json(
        expected.stop_path,
        {
            "schema": STOP_SCHEMA,
            "execution_id": "expected-execution",
            "attempt": 1,
            "pid": active.pid,
            "process_group": active.process_group,
        },
    )

    try:
        with pytest.raises(SupervisorIntegrityError, match="execution intent is malformed"):
            expected.acknowledge_stopped_execution()
        assert expected.stop_path.is_file()
        assert foreign.active_child_path.is_file()
        assert process_group_exists(active.process_group)
    finally:
        if process_group_exists(active.process_group):
            os.killpg(active.process_group, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except ChildProcessError:
            pass


def test_reattach_works_with_only_opaque_execution_identity(tmp_path: Path) -> None:
    first = _supervisor(
        tmp_path,
        execution_id="opaque-id",
        command=[sys.executable, "-c", "import time; time.sleep(0.1)"],
        policy=SupervisorPolicy(
            heartbeat_grace_seconds=1.0,
            max_retries=0,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.1,
        ),
    )
    process, active = first._start(1)
    second = _supervisor(
        tmp_path,
        execution_id="opaque-id",
        command=[sys.executable, "-c", "import time; time.sleep(0.1)"],
        policy=first.policy,
    )

    try:
        plan = second.plan()
        assert plan.action is SupervisorAction.REATTACH
        result = second.run_once()
        assert result.success
        assert result.reattached
    finally:
        if process_group_exists(active.process_group):
            os.killpg(active.process_group, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except ChildProcessError:
            pass


def test_dead_child_recovery_preserves_next_attempt_and_same_execution(tmp_path: Path) -> None:
    first = _supervisor(
        tmp_path,
        execution_id="recoverable-execution",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        policy=SupervisorPolicy(max_retries=2, termination_grace_seconds=0.05),
    )
    process, active = first._start(1)
    os.killpg(active.process_group, signal.SIGKILL)
    process.wait(timeout=5)

    second = _supervisor(
        tmp_path,
        execution_id="recoverable-execution",
        command=None,
        policy=first.policy,
    )
    result = second.reclaim_dead_child()

    assert result.success
    assert result.attempts == 2
    assert not second.active_child_path.exists()
    intent = json.loads(second.execution_intent_path.read_text(encoding="utf-8"))
    assert intent["execution_id"] == "recoverable-execution"
    assert intent["attempt"] == 2


def test_reconcile_completed_execution_stops_owned_live_child(tmp_path: Path) -> None:
    first = _supervisor(
        tmp_path,
        execution_id="completed-execution",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        policy=SupervisorPolicy(
            max_retries=0,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.05,
        ),
    )
    process, active = first._start(1)
    second = _supervisor(
        tmp_path,
        execution_id="completed-execution",
        command=None,
        policy=first.policy,
    )

    try:
        result = second.reconcile_completed_execution()
        assert result.success
        assert not process_group_exists(active.process_group)
        assert not second.active_child_path.exists()
        assert not second.execution_intent_path.exists()
    finally:
        if process_group_exists(active.process_group):
            os.killpg(active.process_group, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except ChildProcessError:
            pass


def test_reconcile_completed_execution_preserves_foreign_intent(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, execution_id="expected-execution")
    atomic_write_json(
        supervisor.execution_intent_path,
        {
            "schema": EXECUTION_INTENT_SCHEMA,
            "execution_id": "foreign-execution",
            "attempt": 1,
        },
    )

    with pytest.raises(SupervisorIntegrityError, match="execution intent is malformed"):
        supervisor.reconcile_completed_execution()

    assert supervisor.execution_intent_path.is_file()


def test_reconcile_completed_execution_preserves_foreign_active_child(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, execution_id="expected-execution")
    atomic_write_json(
        supervisor.execution_intent_path,
        {
            "schema": EXECUTION_INTENT_SCHEMA,
            "execution_id": "expected-execution",
            "attempt": 1,
        },
    )
    atomic_write_json(
        supervisor.active_child_path,
        {
            "schema": ACTIVE_CHILD_SCHEMA,
            "execution_id": "foreign-execution",
            "attempt": 1,
            "pid": 999999,
            "process_group": 999999,
            "started_at": 1.0,
            "liveness_path": "runtime/heartbeat.json",
            "progress_path": "runtime/heartbeat.json",
        },
    )

    with pytest.raises(SupervisorIntegrityError, match="active-child execution ownership mismatch"):
        supervisor.reconcile_completed_execution()

    assert supervisor.execution_intent_path.is_file()
    assert supervisor.active_child_path.is_file()


def test_term_then_kill_cleans_process_group(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]
    supervisor = _supervisor(
        tmp_path,
        command=command,
        policy=SupervisorPolicy(
            liveness_timeout_seconds=0.03,
            progress_timeout_seconds=1.0,
            max_retries=0,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.03,
        ),
    )

    result = supervisor.run_once()

    assert not result.success
    assert not supervisor.active_child_path.exists()
    assert not process_group_exists(json.loads(supervisor.stop_path.read_text())["process_group"])


def test_supervisor_does_not_parse_domain_markers(tmp_path: Path) -> None:
    (tmp_path / "generation-999.complete.json").write_text(
        "not supervisor evidence", encoding="utf-8"
    )

    result = _supervisor(tmp_path).run_once()

    assert result.success


def test_phase_name_does_not_change_progress_policy(tmp_path: Path) -> None:
    heartbeat = tmp_path / "runtime" / "heartbeat.json"
    atomic_write_json(
        heartbeat,
        {
            "liveness_at": time.time(),
            "progress_at": 1.0,
            "progress_token": "stuck",
            "phase": "load-previous-state",
        },
    )
    supervisor = _supervisor(
        tmp_path,
        policy=SupervisorPolicy(
            liveness_timeout_seconds=100.0,
            progress_timeout_seconds=0.01,
            max_retries=0,
        ),
    )
    child = ActiveChild(
        execution_id="opaque-execution",
        attempt=1,
        pid=2,
        process_group=3,
        started_at=1.0,
        liveness_path=heartbeat,
        progress_path=heartbeat,
    )

    status = supervisor.heartbeat_status(child, now=2.0)

    assert status.progress_stale
