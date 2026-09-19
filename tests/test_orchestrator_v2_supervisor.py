from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gocube_golden.orchestrator import atomic_write_json
from gocube_golden.artifact_catalog import sha256_file
from gocube_golden.artifact_graph import (
    ArtifactRef,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    publish_checkpoint_graph,
)
from gocube_golden.orchestrator_v2 import (
    ActiveChild,
    SupervisorAction,
    SupervisorPolicy,
    SupervisorStatus,
    SupervisorV2,
)


def _stage_commit(root: Path, generation: int, lineage_id: str) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "checkpoints",
        "replay",
        "training",
        "metadata/effective-config-v2",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config = EffectiveConfig("torus9", {"topology": "torus9"})
    config_path = root / "metadata/effective-config-v2" / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    parent = CheckpointRef(
        "torus9",
        lineage_id,
        f"M{generation - 1}",
        generation - 1,
        f"checkpoints/M{generation - 1}.pt",
        "sha256:" + "1" * 64,
    )
    manifest = {
        "lineage_id": lineage_id,
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": parent.to_dict(),
        "git_commit": "synthetic",
        "config_fingerprint": config.fingerprint,
        "created_at": "synthetic",
        "checkpoint_hashes": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint_path = root / "checkpoints" / f"M{generation}.pt"
    fresh_path = root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
    rolling_path = root / "replay" / f"rolling-after-{generation:02d}.jsonl"
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    training_path = root / "training" / f"iter-{generation:02d}.json"
    summary_path = root / f"iter-{generation:02d}-summary.json"
    for path, value in (
        (checkpoint_path, b"checkpoint"),
        (fresh_path, b"fresh"),
        (rolling_path, b"rolling"),
        (metadata_path, b"metadata"),
        (training_path, b"training"),
        (summary_path, b"summary"),
    ):
        path.write_bytes(value)
    checkpoint = CheckpointRef(
        "torus9",
        lineage_id,
        f"M{generation}",
        generation,
        checkpoint_path.relative_to(root).as_posix(),
        sha256_file(checkpoint_path),
    )
    fresh = ArtifactRef(fresh_path.relative_to(root).as_posix(), sha256_file(fresh_path))
    config_ref = EffectiveConfigRef(
        ArtifactRef(config_path.relative_to(root).as_posix(), sha256_file(config_path)),
        config.fingerprint,
    )
    marker = {
        "schema": "training-generation-commit-v1",
        "generation": generation,
        "label": f"M{generation}",
        "run_id": lineage_id,
        "checkpoint_sha256": checkpoint.sha256,
        "fresh_replay_sha256": fresh.sha256,
        "rolling_replay_sha256": sha256_file(rolling_path),
        "checkpoint_metadata_sha256": sha256_file(metadata_path),
        "training_metrics_sha256": sha256_file(training_path),
        "summary_sha256": sha256_file(summary_path),
    }
    marker_bytes = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode("utf-8")
    marker_sha = "sha256:" + hashlib.sha256(marker_bytes).hexdigest()
    identities = {
        "checkpoint": {"path": checkpoint.path, "sha256": checkpoint.sha256, "size_bytes": checkpoint_path.stat().st_size},
        "fresh_replay": {"path": fresh.path, "sha256": fresh.sha256, "size_bytes": fresh_path.stat().st_size},
        "rolling_replay": {"path": rolling_path.relative_to(root).as_posix(), "sha256": sha256_file(rolling_path), "size_bytes": rolling_path.stat().st_size},
        "checkpoint_metadata": {"path": metadata_path.relative_to(root).as_posix(), "sha256": sha256_file(metadata_path), "size_bytes": metadata_path.stat().st_size},
        "training_metrics": {"path": training_path.relative_to(root).as_posix(), "sha256": sha256_file(training_path), "size_bytes": training_path.stat().st_size},
        "iteration_summary": {"path": summary_path.relative_to(root).as_posix(), "sha256": sha256_file(summary_path), "size_bytes": summary_path.stat().st_size},
        "completion_marker": {"path": f"generation-{generation:02d}.complete.json", "sha256": marker_sha, "size_bytes": len(marker_bytes)},
    }
    publish_checkpoint_graph(
        root=root,
        parent=parent,
        checkpoint=checkpoint,
        fresh_replay=fresh,
        effective_config=config_ref,
        generation_commit=ArtifactRef(identities["completion_marker"]["path"], marker_sha),  # type: ignore[arg-type]
        artifact_identities=identities,
    )
    return marker


def _commit(root: Path, generation: int, lineage_id: str) -> None:
    atomic_write_json(root / f"generation-{generation:02d}.complete.json", _stage_commit(root, generation, lineage_id))


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


def test_default_policy_preserves_standard_heartbeat_retry_and_drain() -> None:
    policy = SupervisorPolicy()

    assert policy.heartbeat_grace_seconds == 5 * 60.0
    assert policy.max_retries == 1
    assert policy.termination_grace_seconds == 5.0


def test_target_generation_reuses_existing_valid_commit(tmp_path: Path) -> None:
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


def test_partial_graph_without_final_marker_is_not_committed(tmp_path: Path) -> None:
    _stage_commit(tmp_path, 1, "lineage")
    node = json.loads((tmp_path / "metadata/checkpoints/M1.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (tmp_path / "metadata/provenance-v2/M1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert node["parent"]["lineage_id"] == "lineage"
    assert provenance["immediate_parent"] == node["parent"]
    assert manifest["checkpoint_hashes"]["checkpoints/M1.pt"] == node["checkpoint"]["sha256"]
    assert not (tmp_path / "generation-01.complete.json").exists()
    supervisor = _supervisor(tmp_path)

    plan = supervisor.plan()

    assert plan.action is SupervisorAction.START
    assert plan.last_committed_generation is None
    assert plan.generation == 1


def test_marker_without_graph_evidence_fails_closed(tmp_path: Path) -> None:
    atomic_write_json(
        tmp_path / "generation-01.complete.json",
        {
            "schema": "training-generation-commit-v1",
            "generation": 1,
            "run_id": "lineage",
        },
    )
    supervisor = _supervisor(tmp_path)

    plan = supervisor.plan()
    result = supervisor.run_once()

    assert plan.action is SupervisorAction.STOP
    assert "graph/provenance" in plan.reason
    assert result.status is SupervisorStatus.STOPPED


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


def test_long_parent_restore_uses_live_heartbeat_until_progress_resumes(tmp_path: Path) -> None:
    heartbeat = tmp_path / "runtime" / "heartbeats" / "generation-0096.json"
    atomic_write_json(
        heartbeat,
        {
            "schema": "gocube-training-driver-heartbeat-v2",
            "liveness_at": 1_601.0,
            "progress_at": 1.0,
            "phase": "load-previous-state",
        },
    )
    child = ActiveChild(
        lineage_id="lineage",
        generation=96,
        attempt=1,
        execution_unit_id="unit",
        pid=2,
        process_group=3,
        started_at=1_000.0,
        liveness_path=heartbeat,
        progress_path=heartbeat,
    )

    health = _supervisor(tmp_path).heartbeat_status(child, now=1_300.0 + 301.0)

    assert not health.liveness_stale
    assert not health.progress_stale


def test_one_retry_repeats_same_generation_then_commits(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []
    lineage_id = "lineage"
    marker_payload = _stage_commit(tmp_path, 1, lineage_id)
    marker_text = json.dumps(marker_payload, indent=2, sort_keys=True) + "\n"

    def launcher(request):
        calls.append((request.generation, request.attempt))
        if request.attempt == 1:
            return subprocess.Popen(
                [sys.executable, "-c", "raise SystemExit(17)"],
                start_new_session=True,
            )
        script = (
            "from pathlib import Path; "
            f"p=Path({str(tmp_path / 'generation-01.complete.json')!r}); "
            f"p.write_text({marker_text!r});"
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


def test_commit_marker_uses_only_ordinary_process_cleanup(tmp_path: Path) -> None:
    lineage_id = "lineage"
    marker = tmp_path / "generation-01.complete.json"
    marker_payload = _stage_commit(tmp_path, 1, lineage_id)
    script = (
        "import json, time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text({(json.dumps(marker_payload, indent=2, sort_keys=True) + chr(10))!r}); "
        "time.sleep(0.15)"
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
