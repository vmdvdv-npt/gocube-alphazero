from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gocube_golden.orchestrator import atomic_write_json
from gocube_golden.orchestrator_v2 import (
    ActiveChild,
    SupervisorAction,
    SupervisorPolicy,
    SupervisorStatus,
    SupervisorV2,
)


def _commit(root: Path, generation: int, lineage_id: str) -> None:
    atomic_write_json(
        root / f"generation-{generation:02d}.complete.json",
        {
            "schema": "training-generation-commit-v1",
            "generation": generation,
            "label": f"M{generation}",
            "run_id": lineage_id,
        },
    )


def _supervisor(root: Path, lineage_id: str = "lineage") -> SupervisorV2:
    return SupervisorV2(
        root,
        lineage_id=lineage_id,
        command=[sys.executable, "-c", "pass"],
        policy=SupervisorPolicy(
            heartbeat_grace_seconds=5.0,
            poll_interval_seconds=0.01,
            termination_grace_seconds=0.2,
        ),
    )


def test_committed_generation_is_not_rerun_and_next_is_only_planned(tmp_path: Path) -> None:
    lineage_id = "torus9-v2-parity-m93-20260918-v1"
    _commit(tmp_path, 94, lineage_id)
    supervisor = _supervisor(tmp_path, lineage_id)

    plan = supervisor.plan()

    assert plan.last_committed_generation == 94
    assert plan.generation == 95
    assert plan.action is SupervisorAction.START
    assert not (tmp_path / "generation-95.complete.json").exists()
    assert not supervisor.active_child_path.exists()


def test_external_parent_generation_is_the_first_child_baseline(tmp_path: Path) -> None:
    lineage_id = "torus9-v2-ab-acceptance-m95-a-20260919-v1"
    supervisor = SupervisorV2(
        tmp_path,
        lineage_id=lineage_id,
        initial_committed_generation=95,
        command=[sys.executable, "-c", "pass"],
        policy=SupervisorPolicy(poll_interval_seconds=0.01),
    )

    plan = supervisor.plan()

    assert plan.last_committed_generation == 95
    assert plan.generation == 96
    assert plan.action is SupervisorAction.START
    assert not (tmp_path / "generation-96.complete.json").exists()


def test_target_generation_recovers_existing_commit_publication(tmp_path: Path) -> None:
    lineage_id = "lineage"
    _commit(tmp_path, 1, lineage_id)
    supervisor = SupervisorV2(
        tmp_path,
        lineage_id=lineage_id,
        initial_committed_generation=0,
        target_generation=1,
        command=[sys.executable, "-c", "pass"],
        policy=SupervisorPolicy(poll_interval_seconds=0.01),
    )

    plan = supervisor.plan()
    result = supervisor.run_once()

    assert plan.action is SupervisorAction.START
    assert plan.generation == 1
    assert result.status is SupervisorStatus.COMMITTED
    assert result.generation == 1


def test_uncommitted_generation_without_child_is_rerun_same_generation(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    atomic_write_json(
        supervisor.generation_intent_path,
        {
            "schema": "gocube-orchestrator-v2-generation-intent-v1",
            "lineage_id": "lineage",
            "generation": 7,
            "attempt": 1,
        },
    )

    plan = supervisor.plan()

    assert plan.action is SupervisorAction.START
    assert plan.generation == 7
    assert plan.attempt == 1
    assert "same generation" in plan.reason


def test_matching_live_child_is_reattached_without_starting_duplicate(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        child = ActiveChild(
            lineage_id="lineage",
            generation=7,
            attempt=1,
            execution_unit_id="generation-0007-attempt-01-test",
            pid=process.pid,
            process_group=process.pid,
            started_at=time.time(),
            liveness_path=tmp_path / "runtime" / "heartbeats" / "generation-0007.json",
            progress_path=tmp_path / "runtime" / "heartbeats" / "generation-0007.json",
        )
        atomic_write_json(
            supervisor.generation_intent_path,
            {
                "schema": "gocube-orchestrator-v2-generation-intent-v1",
                "lineage_id": "lineage",
                "generation": 7,
                "attempt": 1,
            },
        )
        atomic_write_json(supervisor.active_child_path, child.to_dict(tmp_path))

        plan = supervisor.plan()

        assert plan.action is SupervisorAction.REATTACH
        assert plan.generation == 7
        assert plan.active_child is not None
        assert plan.active_child.process_group == process.pid
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_missing_liveness_and_progress_each_get_five_minute_grace(tmp_path: Path) -> None:
    supervisor = SupervisorV2(
        tmp_path,
        lineage_id="lineage",
        command=[sys.executable, "-c", "pass"],
        clock=lambda: 1_000.0,
    )
    child = ActiveChild(
        lineage_id="lineage",
        generation=7,
        attempt=1,
        execution_unit_id="unit",
        pid=2,
        process_group=3,
        started_at=1_000.0,
        liveness_path=tmp_path / "live.json",
        progress_path=tmp_path / "progress.json",
    )

    within_grace = supervisor.heartbeat_status(child, now=1_299.9)
    after_grace = supervisor.heartbeat_status(child, now=1_300.0)

    assert not within_grace.liveness_stale
    assert not within_grace.progress_stale
    assert after_grace.liveness_stale
    assert after_grace.progress_stale


def test_one_retry_repeats_same_generation_then_commits(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []
    lineage_id = "lineage"

    def launcher(request):
        calls.append((request.generation, request.attempt))
        if request.attempt == 1:
            return subprocess.Popen(
                [sys.executable, "-c", "raise SystemExit(17)"],
                start_new_session=True,
            )
        script = (
            "import json; from pathlib import Path; "
            f"p=Path({str(tmp_path / 'generation-01.complete.json')!r}); "
            f"p.write_text(json.dumps({{'schema':'training-generation-commit-v1','generation':1,'run_id':{lineage_id!r}}}));"
        )
        return subprocess.Popen([sys.executable, "-c", script], start_new_session=True)

    supervisor = SupervisorV2(
        tmp_path,
        lineage_id=lineage_id,
        launcher=launcher,
        policy=SupervisorPolicy(
            heartbeat_grace_seconds=5.0,
            poll_interval_seconds=0.01,
            termination_grace_seconds=0.2,
        ),
    )

    result = supervisor.run_once()

    assert result.status is SupervisorStatus.COMMITTED
    assert result.generation == 1
    assert result.attempts == 2
    assert calls == [(1, 1), (1, 2)]
    assert not supervisor.active_child_path.exists()


def test_commit_marker_drains_child_publication_before_cleanup(tmp_path: Path) -> None:
    lineage_id = "lineage"
    marker = tmp_path / "generation-01.complete.json"
    result = tmp_path / "post-commit-result.json"
    script = (
        "import json, time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(json.dumps({{'schema':'training-generation-commit-v1','generation':1,'run_id':{lineage_id!r}}})); "
        "time.sleep(0.15); "
        f"Path({str(result)!r}).write_text('published')"
    )

    def launcher(_request):
        return subprocess.Popen([sys.executable, "-c", script], start_new_session=True)

    supervisor = SupervisorV2(
        tmp_path,
        lineage_id=lineage_id,
        launcher=launcher,
        policy=SupervisorPolicy(
            heartbeat_grace_seconds=5.0,
            poll_interval_seconds=0.01,
            termination_grace_seconds=1.0,
        ),
    )

    supervision = supervisor.run_once()

    assert supervision.status is SupervisorStatus.COMMITTED
    assert result.read_text(encoding="utf-8") == "published"
    assert not supervisor.active_child_path.exists()


def test_second_technical_failure_stops_without_third_start(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []

    def launcher(request):
        calls.append((request.generation, request.attempt))
        return subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(17)"],
            start_new_session=True,
        )

    supervisor = SupervisorV2(
        tmp_path,
        lineage_id="lineage",
        launcher=launcher,
        policy=SupervisorPolicy(
            heartbeat_grace_seconds=5.0,
            poll_interval_seconds=0.01,
            termination_grace_seconds=0.2,
        ),
    )

    result = supervisor.run_once()

    assert result.status is SupervisorStatus.STOPPED
    assert result.attempts == 2
    assert calls == [(1, 1), (1, 2)]
    assert supervisor.stop_path.is_file()
