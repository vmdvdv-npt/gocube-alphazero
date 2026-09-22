from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gocube_golden.artifact_graph import ArtifactRef, CheckpointRef, EffectiveConfig, EffectiveConfigRef
from gocube_golden.orchestrator_v2 import (
    OutputLineage,
    ProductionTrainOne,
    ResolvedArtifact,
    ResolvedEffectiveConfig,
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

    class MustNotStart:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("committed child must be reused")

    monkeypatch.setattr(production_generation, "SupervisorV2", MustNotStart)
    assert ProductionTrainOne(resolver=resolver)(
        parent=parent,
        config=config,
        output_lineage=output,
    ) is child


def test_removed_arm_execution_path_and_resolved_object_serializer():
    import gocube_golden.orchestrator_v2 as orchestrator_v2

    assert not hasattr(orchestrator_v2, "ProductionArmExecutionPath")
    assert not Path("gocube_golden/orchestrator_v2/production_arm.py").exists()
    assert not hasattr(production_generation, "_serialize_resolved_input")
    assert not hasattr(production_generation, "_deserialize_resolved_input")
