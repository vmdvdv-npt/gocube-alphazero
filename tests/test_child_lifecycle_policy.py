from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

import gocube_golden.orchestrator as legacy_orchestrator
import gocube_golden.production_orchestrator as production_orchestrator
from gocube_golden.orchestrator import HealthPolicy, OrchestratorSpec, SoftStopPolicy
from gocube_golden.production_orchestrator import (
    ChildLifecycleContext,
    CriticalHealthError,
    NoOpChildLifecyclePolicy,
    SupervisionPolicy,
    UniversalProductionTrainingOrchestrator,
)


class RecordingPolicy:
    def __init__(self, label: str, order: list[str], events: list[tuple[object, ...]]) -> None:
        self.label = label
        self.order = order
        self.events = events

    def before_child_start(self, context: ChildLifecycleContext) -> None:
        self.order.append(f"{self.label}:before")
        self.events.append(("before", context))

    def after_child_finish(
        self, context: ChildLifecycleContext, exit_code: int
    ) -> None:
        self.order.append(f"{self.label}:after")
        self.events.append(("after", context, exit_code))

    def after_child_abort(
        self, context: ChildLifecycleContext, error: BaseException
    ) -> None:
        self.order.append(f"{self.label}:abort")
        self.events.append(("abort", context, error))


def _spec(tmp_path: Path) -> OrchestratorSpec:
    return OrchestratorSpec(
        path=tmp_path / "orchestrator.json",
        payload={},
        topology="fake",
        profile_path=tmp_path / "profile.json",
        profile_payload={},
        profile_fingerprint="sha256:fake-profile",
        config_fingerprint="sha256:fake-config",
        generation_command=("fake-child",),
        generation_resume_command=("fake-child", "--resume"),
        arena_command=("fake-arena",),
        arena_every_generations=10,
        arena_required=False,
        arena_preset_fingerprint=None,
        arena_startset_fingerprint=None,
        health=HealthPolicy(),
        soft_stop=SoftStopPolicy(),
        performance={},
        learning={},
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lineage_id: str,
    policy: object = ...,
) -> UniversalProductionTrainingOrchestrator:
    monkeypatch.setattr(
        legacy_orchestrator,
        "active_lineage_dir",
        lambda topology, lineage: tmp_path / "runs" / topology / "active" / lineage,
    )
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "spec": _spec(tmp_path),
        "lineage_id": lineage_id,
        "terminal": False,
        "supervision": SupervisionPolicy(
            startup_ack_timeout_seconds=1.0,
            progress_warning_seconds=2.0,
            progress_critical_seconds=3.0,
            critical_child_grace_seconds=0.0,
            restart_backoff_seconds=0.0,
            max_generation_restarts=0,
        ),
    }
    if policy is not ...:
        kwargs["child_lifecycle_policy"] = policy
    run = UniversalProductionTrainingOrchestrator(**kwargs)
    monkeypatch.setattr(
        run,
        "_driver_env",
        lambda generation, *, resume, phase: {},
    )
    monkeypatch.setattr(run, "_process_group_exists", lambda _pid: False)
    return run


def _install_fake_child(
    monkeypatch: pytest.MonkeyPatch,
    order: list[str],
    *,
    exit_code: int,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 424242

        def __init__(self) -> None:
            self.returncode = int(exit_code)
            self._reported_finish = False

        def poll(self) -> int:
            if not self._reported_finish:
                order.append("child:finish")
                self._reported_finish = True
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    def fake_popen(
        argv: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        start_new_session: bool,
    ) -> FakeProcess:
        order.append("child:start")
        captured.update(
            {
                "argv": tuple(argv),
                "cwd": cwd,
                "env": env,
                "start_new_session": start_new_session,
            }
        )
        return FakeProcess()

    monkeypatch.setattr(production_orchestrator.subprocess, "Popen", fake_popen)
    return captured


def _install_running_child(
    monkeypatch: pytest.MonkeyPatch,
    order: list[str],
) -> object:
    class FakeProcess:
        pid = 424243

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

    process = FakeProcess()

    def fake_popen(
        argv: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        start_new_session: bool,
    ) -> FakeProcess:
        del argv, cwd, env, start_new_session
        order.append("child:start")
        return process

    monkeypatch.setattr(production_orchestrator.subprocess, "Popen", fake_popen)
    return process


def _isolate_monitoring(run: UniversalProductionTrainingOrchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "_state", lambda: {})
    monkeypatch.setattr(run, "_heartbeat", lambda _state: None)
    monkeypatch.setattr(run, "_stop_request", lambda: None)


def test_base_orchestrator_without_policy_preserves_child_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _install_fake_child(monkeypatch, order, exit_code=7)
    run = _run(tmp_path, monkeypatch, lineage_id="default-policy")

    assert isinstance(run.child_lifecycle_policy, NoOpChildLifecyclePolicy)
    assert callable(run.child_lifecycle_policy.after_child_abort)
    assert (
        run._run_child(
            ("fake-child", "--generation={generation}"),
            generation=3,
            resume=False,
            phase="generation",
        )
        == 7
    )
    assert order == ["child:start", "child:finish"]
    assert not run.active_child_path.exists()


def test_explicit_noop_policy_does_not_change_child_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    captured = _install_fake_child(monkeypatch, order, exit_code=5)
    policy = NoOpChildLifecyclePolicy()
    run = _run(
        tmp_path,
        monkeypatch,
        lineage_id="explicit-noop",
        policy=policy,
    )

    result = run._run_child(
        ("fake-child", "--generation={generation04}"),
        generation=8,
        resume=True,
        phase="generation",
    )

    assert result == 5
    assert run.child_lifecycle_policy is policy
    assert order == ["child:start", "child:finish"]
    assert captured["argv"] == ("fake-child", "--generation=0008")


def test_explicit_policy_wraps_child_and_receives_typed_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    events: list[tuple[object, ...]] = []
    policy = RecordingPolicy("policy", order, events)
    captured = _install_fake_child(monkeypatch, order, exit_code=4)
    run = _run(
        tmp_path,
        monkeypatch,
        lineage_id="hook-context",
        policy=policy,
    )

    result = run._run_child(
        (
            "fake-child",
            "--generation={generation04}",
            "--lineage={lineage_id}",
        ),
        generation=12,
        resume=True,
        phase="arena",
    )

    assert result == 4
    assert order == [
        "policy:before",
        "child:start",
        "child:finish",
        "policy:after",
    ]
    assert [event[0] for event in events] == ["before", "after"]
    before = events[0][1]
    after = events[1][1]
    assert isinstance(before, ChildLifecycleContext)
    assert before is after
    assert before.orchestrator is run
    assert before.generation == 12
    assert before.phase == "arena"
    assert before.resume is True
    assert before.command == (
        "fake-child",
        "--generation=0012",
        "--lineage=hook-context",
    )
    assert captured["argv"] == before.command
    assert events[1][2] == 4


def test_supervisor_exception_cleans_up_then_aborts_and_propagates_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    events: list[tuple[object, ...]] = []
    policy = RecordingPolicy("policy", order, events)
    process = _install_running_child(monkeypatch, order)
    run = _run(tmp_path, monkeypatch, lineage_id="monitor-abort", policy=policy)
    _isolate_monitoring(run, monkeypatch)
    original = CriticalHealthError("worker/inference health failure")

    def fail_health(_process: object) -> dict[str, object]:
        order.append("monitor")
        raise original

    def cleanup(proc: object, *, reason: str) -> None:
        del reason
        order.append("cleanup")
        setattr(proc, "returncode", -15)

    monkeypatch.setattr(run, "_health_snapshot", fail_health)
    monkeypatch.setattr(run, "_terminate_child_group", cleanup)

    with pytest.raises(CriticalHealthError) as caught:
        run._run_child(("fake-child",), generation=13, resume=False, phase="generation")

    assert caught.value is original
    assert getattr(process, "returncode") == -15
    assert order == [
        "policy:before",
        "child:start",
        "monitor",
        "cleanup",
        "policy:abort",
    ]
    assert [event[0] for event in events] == ["before", "abort"]
    assert events[1][2] is original
    assert not run.active_child_path.exists()


def test_startup_heartbeat_timeout_is_abort_outcome_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    events: list[tuple[object, ...]] = []
    policy = RecordingPolicy("policy", order, events)
    _install_running_child(monkeypatch, order)
    run = _run(tmp_path, monkeypatch, lineage_id="startup-timeout", policy=policy)
    _isolate_monitoring(run, monkeypatch)
    ticks = iter((100.0, 102.0))
    monkeypatch.setattr(production_orchestrator.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(run, "_health_snapshot", lambda _process: {})

    def cleanup(proc: object, *, reason: str) -> None:
        assert "startup heartbeat" in reason
        order.append("cleanup")
        setattr(proc, "returncode", -15)

    monkeypatch.setattr(run, "_terminate_child_group", cleanup)

    with pytest.raises(
        CriticalHealthError,
        match="driver did not publish startup heartbeat before timeout",
    ):
        run._run_child(("fake-child",), generation=14, resume=False, phase="generation")

    assert order == ["policy:before", "child:start", "cleanup", "policy:abort"]
    assert [event[0] for event in events] == ["before", "abort"]
    assert isinstance(events[1][2], CriticalHealthError)
    assert not run.active_child_path.exists()


def test_arena_phase_uses_generic_abort_callback_without_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    events: list[tuple[object, ...]] = []
    policy = RecordingPolicy("policy", order, events)
    _install_running_child(monkeypatch, order)
    run = _run(tmp_path, monkeypatch, lineage_id="arena-abort", policy=policy)
    _isolate_monitoring(run, monkeypatch)
    original = RuntimeError("arena monitoring failed")

    def fail_health(_process: object) -> dict[str, object]:
        raise original

    def cleanup(proc: object, *, reason: str) -> None:
        del reason
        order.append("cleanup")
        setattr(proc, "returncode", -15)

    monkeypatch.setattr(run, "_health_snapshot", fail_health)
    monkeypatch.setattr(run, "_terminate_child_group", cleanup)

    with pytest.raises(RuntimeError) as caught:
        run._run_child(("fake-arena",), generation=15, resume=False, phase="arena")

    assert caught.value is original
    assert [event[0] for event in events] == ["before", "abort"]
    assert events[1][1].phase == "arena"
    assert events[1][2] is original
    assert order[-2:] == ["cleanup", "policy:abort"]


def test_abort_bookkeeping_failure_does_not_mask_original_supervisor_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    events: list[tuple[object, ...]] = []

    class FailingAbortPolicy(RecordingPolicy):
        def after_child_abort(
            self, context: ChildLifecycleContext, error: BaseException
        ) -> None:
            super().after_child_abort(context, error)
            raise RuntimeError("provenance write failed")

    policy = FailingAbortPolicy("policy", order, events)
    _install_running_child(monkeypatch, order)
    run = _run(tmp_path, monkeypatch, lineage_id="abort-bookkeeping", policy=policy)
    _isolate_monitoring(run, monkeypatch)
    original = CriticalHealthError("original supervisor failure")

    def fail_health(_process: object) -> dict[str, object]:
        raise original

    def cleanup(proc: object, *, reason: str) -> None:
        del reason
        order.append("cleanup")
        setattr(proc, "returncode", -15)

    monkeypatch.setattr(run, "_health_snapshot", fail_health)
    monkeypatch.setattr(run, "_terminate_child_group", cleanup)

    with pytest.raises(CriticalHealthError) as caught:
        run._run_child(("fake-child",), generation=16, resume=False, phase="generation")

    assert caught.value is original
    assert [event[0] for event in events] == ["before", "abort"]
    assert order[-2:] == ["cleanup", "policy:abort"]
    if hasattr(BaseException, "add_note"):
        assert any(
            "after_child_abort failed" in note
            for note in getattr(original, "__notes__", [])
        )


def test_child_lifecycle_policy_is_instance_local_and_requires_no_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _install_fake_child(monkeypatch, order, exit_code=0)
    events_a: list[tuple[object, ...]] = []
    events_b: list[tuple[object, ...]] = []
    policy_a = RecordingPolicy("A", order, events_a)
    policy_b = RecordingPolicy("B", order, events_b)
    run_a = _run(tmp_path, monkeypatch, lineage_id="lineage-a", policy=policy_a)
    run_b = _run(tmp_path, monkeypatch, lineage_id="lineage-b", policy=policy_b)

    assert run_a.child_lifecycle_policy is policy_a
    assert run_b.child_lifecycle_policy is policy_b
    run_a._run_child(
        ("fake-child",), generation=1, resume=False, phase="generation"
    )
    run_b._run_child(("fake-child",), generation=2, resume=True, phase="arena")

    assert [event[1].orchestrator for event in events_a] == [run_a, run_a]
    assert [event[1].orchestrator for event in events_b] == [run_b, run_b]
    assert [event[1].generation for event in events_a] == [1, 1]
    assert [event[1].generation for event in events_b] == [2, 2]

    run_default = _run(tmp_path, monkeypatch, lineage_id="lineage-default")
    assert isinstance(run_default.child_lifecycle_policy, NoOpChildLifecyclePolicy)
    assert run_default.child_lifecycle_policy is not run_a.child_lifecycle_policy
    assert run_default.child_lifecycle_policy is not run_b.child_lifecycle_policy
