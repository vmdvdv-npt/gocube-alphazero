from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import ArtifactRef, CheckpointRef, EffectiveConfig, EffectiveConfigRef
from gocube_golden.process_supervision import atomic_write_json, process_group_exists
from gocube_golden.orchestrator_v2 import (
    ActiveChild,
    EXECUTION_INTENT_SCHEMA,
    OutputLineage,
    ProductionTrainOne,
    ResolvedArtifact,
    ResolvedEffectiveConfig,
    SupervisorAction,
    SupervisorIntegrityError,
    STOP_SCHEMA,
    SupervisorV2,
)
from gocube_golden.orchestrator_v2 import production_generation


def _refs(tmp_path: Path):
    config = EffectiveConfig("torus9", {"topology": "torus9"})
    config_path = tmp_path / "lineage" / "metadata" / "effective.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    config_ref = EffectiveConfigRef(
        ArtifactRef("metadata/effective.json", "sha256:" + "1" * 64),
        config.fingerprint,
    )
    parent_ref = CheckpointRef(
        "torus9", "parent", "M7", 7, "checkpoints/M7.pt", "sha256:" + "2" * 64
    )
    child_ref = CheckpointRef(
        "torus9", "child", "M8", 8, "checkpoints/M8.pt", "sha256:" + "3" * 64
    )
    parent = SimpleNamespace(ref=parent_ref, generation=7)
    child = SimpleNamespace(
        ref=child_ref,
        generation=8,
        node=SimpleNamespace(parent=parent_ref),
        topology="torus9",
        lineage_id="child",
        effective_config=SimpleNamespace(ref=config_ref),
    )
    resolved_config = ResolvedEffectiveConfig(
        config_ref,
        ResolvedArtifact(
            config_ref.artifact,
            config_path,
            config_path.parents[2],
            "torus9",
            "child",
            "ACTIVE",
            {"immutable_verified": True},
        ),
        config,
    )
    return parent, child, resolved_config, parent_ref, child_ref


def test_train_one_request_contains_only_immutable_refs(tmp_path: Path):
    parent, _child, config, _parent_ref, _child_ref = _refs(tmp_path)
    output = OutputLineage("torus9", "child", tmp_path / "lineage")
    payload = production_generation._request_payload(
        resolver=SimpleNamespace(runs_root=tmp_path / "runs"),
        parent=parent,
        config=config,
        output_lineage=output,
    )

    assert set(payload) == {
        "schema",
        "runs_root",
        "parent_checkpoint",
        "effective_config",
        "output_lineage",
    }
    assert payload["parent_checkpoint"] == parent.ref.to_dict()
    assert payload["effective_config"] == config.ref.to_dict()
    assert "config" not in payload
    assert "path" not in payload["effective_config"]


def test_train_one_request_carries_only_execution_concurrency_override(tmp_path: Path):
    parent, _child, config, _parent_ref, _child_ref = _refs(tmp_path)
    output = OutputLineage("torus9", "child", tmp_path / "lineage")

    payload = production_generation._request_payload(
        resolver=SimpleNamespace(runs_root=tmp_path / "runs"),
        parent=parent,
        config=config,
        output_lineage=output,
        execution_overrides={
            "active_games_per_worker": 6,
            "total_active_contexts": 96,
        },
    )

    assert payload["execution_overrides"] == {
        "active_games_per_worker": 6,
        "total_active_contexts": 96,
    }


def test_train_one_runs_one_generation_and_returns_immediate_child(tmp_path: Path, monkeypatch):
    parent, child, config, parent_ref, child_ref = _refs(tmp_path)
    output = OutputLineage("torus9", "child", tmp_path / "lineage")
    resolver = SimpleNamespace(runs_root=tmp_path / "runs")

    class FakeSupervisor:
        captured: dict[str, object] = {}

        def __init__(self, root, **kwargs):
            type(self).captured = {"root": root, **kwargs}

        def run_once(self):
            command = self.captured["command"]
            result_path = Path(command[-1])  # type: ignore[index]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "schema": production_generation.TRAIN_ONE_RESULT_SCHEMA,
                        "generation": 8,
                        "checkpoint": child_ref.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(success=True, reason=None)

    def resolve(value):
        assert value == child_ref
        return child

    validation_call: dict[str, object] = {}

    resolver.checkpoint = resolve
    monkeypatch.setattr(production_generation, "SupervisorV2", FakeSupervisor)

    def validate(**kwargs):
        validation_call.update(kwargs)

    monkeypatch.setattr(production_generation, "validate_generation_commit", validate)

    result = ProductionTrainOne(resolver=resolver, repo_root=tmp_path)(
        parent=parent,
        config=config,
        output_lineage=output,
    )

    assert result is child
    request = json.loads(
        (output.root / "runtime" / "requests" / "train-one-0008.json").read_text()
    )
    assert request["parent_checkpoint"] == parent_ref.to_dict()
    assert request["effective_config"] == config.ref.to_dict()
    assert request["output_lineage"]["lineage_id"] == "child"
    assert FakeSupervisor.captured["execution_id"] == "child:generation:8"
    assert validation_call["reuse_committed_rolling_replay_identity"] is True


def test_train_one_reuses_committed_child_without_supervisor(tmp_path: Path, monkeypatch):
    parent, child, config, _parent_ref, child_ref = _refs(tmp_path)
    output = OutputLineage("torus9", "child", tmp_path / "lineage")
    output.root.mkdir(parents=True, exist_ok=True)
    (output.root / "generation-08.complete.json").write_text("{}", encoding="utf-8")
    resolver = SimpleNamespace(runs_root=tmp_path / "runs")
    resolver.checkpoint = lambda value: child if value == child_ref else None

    monkeypatch.setattr(
        production_generation,
        "validate_generation_commit",
        lambda **_kwargs: SimpleNamespace(checkpoint=child_ref),
    )

    class ReconcileOnly:
        reconciled: list[dict[str, object]] = []

        def __init__(self, *_args, **_kwargs):
            type(self).reconciled.append(dict(_kwargs))

        def reconcile_completed_execution(self):
            return SimpleNamespace(success=True)

        def run_once(self):
            raise AssertionError("committed child must not start a generation")

    monkeypatch.setattr(production_generation, "SupervisorV2", ReconcileOnly)
    assert ProductionTrainOne(resolver=resolver)(
        parent=parent,
        config=config,
        output_lineage=output,
    ) is child
    assert ReconcileOnly.reconciled[0]["execution_id"] == "child:generation:8"


def test_train_one_reconciles_committed_execution_for_next_generation(
    tmp_path: Path, monkeypatch
):
    parent, child, config, _parent_ref, child_ref = _refs(tmp_path)
    output = OutputLineage("torus9", "child", tmp_path / "lineage")
    output.root.mkdir(parents=True, exist_ok=True)
    (output.root / "generation-08.complete.json").write_text("{}", encoding="utf-8")
    resolver = SimpleNamespace(runs_root=tmp_path / "runs")
    resolver.checkpoint = lambda value: child if value == child_ref else None

    monkeypatch.setattr(
        production_generation,
        "validate_generation_commit",
        lambda **_kwargs: SimpleNamespace(checkpoint=child_ref),
    )

    execution_id = "child:generation:8"
    runtime = output.root / "runtime"
    atomic_write_json(
        runtime / "execution-intent.json",
        {
            "schema": EXECUTION_INTENT_SCHEMA,
            "execution_id": execution_id,
            "attempt": 1,
            "updated_at": 1.0,
        },
    )
    atomic_write_json(
        runtime / "supervisor-stop.json",
        {
            "schema": STOP_SCHEMA,
            "execution_id": execution_id,
            "attempt": 1,
            "reason": "stale stop from the committed execution",
            "stopped_at": 1.0,
        },
    )
    stale_group = 10_000_000
    while process_group_exists(stale_group):
        stale_group += 1
    heartbeat = runtime / "heartbeats" / "generation-0008.json"
    stale_active = ActiveChild(
        execution_id=execution_id,
        attempt=1,
        pid=stale_group,
        process_group=stale_group,
        started_at=1.0,
        liveness_path=heartbeat,
        progress_path=heartbeat,
    )
    atomic_write_json(runtime / "active-child.json", stale_active.to_dict(output.root))

    result = ProductionTrainOne(resolver=resolver)(
        parent=parent,
        config=config,
        output_lineage=output,
    )

    assert result is child
    assert not (runtime / "execution-intent.json").exists()
    assert not (runtime / "active-child.json").exists()
    assert not (runtime / "supervisor-stop.json").exists()

    next_supervisor = SupervisorV2(
        output.root,
        execution_id="child:generation:9",
        liveness_path=runtime / "heartbeats" / "generation-0009.json",
        progress_path=runtime / "heartbeats" / "generation-0009.json",
        command=[sys.executable, "-c", "pass"],
    )
    assert next_supervisor.plan().action is SupervisorAction.START


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema": EXECUTION_INTENT_SCHEMA,
            "execution_id": "other:generation:8",
            "attempt": 1,
        },
        {
            "schema": EXECUTION_INTENT_SCHEMA,
            "execution_id": "child:generation:8",
            "attempt": 0,
        },
    ],
)
def test_committed_reuse_does_not_delete_foreign_or_malformed_supervisor_intent(
    tmp_path: Path, monkeypatch, payload: dict[str, object]
):
    parent, child, config, _parent_ref, child_ref = _refs(tmp_path)
    output = OutputLineage("torus9", "child", tmp_path / "lineage")
    output.root.mkdir(parents=True, exist_ok=True)
    (output.root / "generation-08.complete.json").write_text("{}", encoding="utf-8")
    resolver = SimpleNamespace(runs_root=tmp_path / "runs")
    resolver.checkpoint = lambda value: child if value == child_ref else None
    monkeypatch.setattr(
        production_generation,
        "validate_generation_commit",
        lambda **_kwargs: SimpleNamespace(checkpoint=child_ref),
    )
    intent_path = output.root / "runtime" / "execution-intent.json"
    atomic_write_json(intent_path, payload)

    with pytest.raises(SupervisorIntegrityError, match="execution intent is malformed"):
        ProductionTrainOne(resolver=resolver)(
            parent=parent,
            config=config,
            output_lineage=output,
        )

    assert intent_path.is_file()


@pytest.mark.parametrize(
    "raw_stop",
    [
        json.dumps(
            {
                "schema": STOP_SCHEMA,
                "execution_id": "other:generation:8",
                "attempt": 1,
            }
        ),
        json.dumps(
            {
                "schema": "wrong-supervisor-stop-schema",
                "execution_id": "child:generation:8",
                "attempt": 1,
            }
        ),
        "not-json",
    ],
)
def test_committed_reuse_does_not_delete_foreign_or_malformed_supervisor_stop(
    tmp_path: Path, monkeypatch, raw_stop: str
):
    parent, child, config, _parent_ref, child_ref = _refs(tmp_path)
    output = OutputLineage("torus9", "child", tmp_path / "lineage")
    output.root.mkdir(parents=True, exist_ok=True)
    (output.root / "generation-08.complete.json").write_text("{}", encoding="utf-8")
    resolver = SimpleNamespace(runs_root=tmp_path / "runs")
    resolver.checkpoint = lambda value: child if value == child_ref else None
    monkeypatch.setattr(
        production_generation,
        "validate_generation_commit",
        lambda **_kwargs: SimpleNamespace(checkpoint=child_ref),
    )
    stop_path = output.root / "runtime" / "supervisor-stop.json"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text(raw_stop, encoding="utf-8")

    with pytest.raises(SupervisorIntegrityError, match="supervisor stop is malformed"):
        ProductionTrainOne(resolver=resolver)(
            parent=parent,
            config=config,
            output_lineage=output,
        )

    assert stop_path.read_text(encoding="utf-8") == raw_stop


def test_removed_arm_execution_path_and_resolved_object_serializer():
    import gocube_golden.orchestrator_v2 as orchestrator_v2

    assert not hasattr(orchestrator_v2, "ProductionArmExecutionPath")
    assert not Path("gocube_golden/orchestrator_v2/production_arm.py").exists()
    assert not hasattr(production_generation, "_serialize_resolved_input")
    assert not hasattr(production_generation, "_deserialize_resolved_input")
