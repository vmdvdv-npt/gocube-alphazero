from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gocube_golden.artifact_catalog import sha256_file as physical_sha256
from gocube_golden.orchestrator_v2 import (
    ArtifactRef,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    OutputLineage,
    ProductionArmExecutionPath,
    ResolvedGenerationInput,
    SupervisorPolicy,
)
from gocube_golden.orchestrator_v2 import production_arm
from gocube_golden.provenance import canonical_json


def test_child_deserialization_does_not_rehash_replay_artifacts(tmp_path, monkeypatch):
    parent_ref = CheckpointRef(
        "torus9",
        "parent-lineage",
        "M95",
        95,
        "checkpoints/M95.pt",
        "sha256:" + "1" * 64,
    )
    config = EffectiveConfig("torus9", {"topology": "torus9"})
    config_root = tmp_path / "config-owner"
    config_path = config_root / "metadata" / "effective.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(canonical_json(config.to_dict()) + "\n", encoding="utf-8")
    config_ref = EffectiveConfigRef(
        ArtifactRef(
            "metadata/effective.json",
            physical_sha256(config_path),
        ),
        config.fingerprint,
    )

    payload = {
        "runs_root": str(tmp_path / "runs"),
        "generation": 96,
        "parent_checkpoint": parent_ref.to_dict(),
        "effective_config": {
            "ref": config_ref.to_dict(),
            "path": str(config_path),
            "owner_root": str(config_root),
            "owner_topology": "torus9",
            "owner_lineage_id": "child-lineage",
            "owner_status": "ACTIVE",
            "identity": {"immutable_verified": True},
            "config": config.to_dict(),
        },
        "output_lineage": {
            "topology": "torus9",
            "lineage_id": "child-lineage",
            "root": str(tmp_path / "child-lineage"),
        },
    }

    class FakeResolver:
        def __init__(self, runs_root):
            self.runs_root = Path(runs_root)

        def checkpoint(self, value):
            assert value == parent_ref
            return SimpleNamespace(ref=parent_ref)

    monkeypatch.setattr(production_arm, "ArtifactResolver", FakeResolver)

    resolved = production_arm._deserialize_resolved_input(payload)

    assert not hasattr(resolved, "replay_artifacts")
    assert not hasattr(resolved, "replay_identity")
    serialized = production_arm._serialize_resolved_input(
        resolved,
        runs_root=tmp_path / "runs",
        result_path=tmp_path / "result.json",
    )
    assert "replay_artifacts" not in serialized
    assert "replay_identity" not in serialized


def test_production_arm_uses_standard_supervisor_policy_by_default():
    path = ProductionArmExecutionPath()

    # The production seam uses only the ordinary bounded process cleanup.
    assert path.supervisor_policy is None
    assert path._effective_supervisor_policy().termination_grace_seconds == 5.0


def test_production_arm_preserves_explicit_supervisor_policy():
    policy = SupervisorPolicy(termination_grace_seconds=7.0)

    path = ProductionArmExecutionPath(supervisor_policy=policy)

    assert path._effective_supervisor_policy() is policy


def test_production_arm_checks_domain_commit_after_process_success(tmp_path, monkeypatch):
    parent_ref = CheckpointRef(
        "torus9",
        "parent-lineage",
        "M95",
        95,
        "checkpoints/M95.pt",
        "sha256:" + "1" * 64,
    )
    child_ref = CheckpointRef(
        "torus9",
        "child-lineage",
        "M96",
        96,
        "checkpoints/M96.pt",
        "sha256:" + "2" * 64,
    )
    config = EffectiveConfig("torus9", {"topology": "torus9"})
    config_root = tmp_path / "config-owner"
    config_path = config_root / "metadata" / "effective.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(canonical_json(config.to_dict()) + "\n", encoding="utf-8")
    config_ref = EffectiveConfigRef(
        ArtifactRef("metadata/effective.json", physical_sha256(config_path)),
        config.fingerprint,
    )
    resolved = ResolvedGenerationInput(
        parent_checkpoint=SimpleNamespace(ref=parent_ref),  # type: ignore[arg-type]
        generation=96,
        effective_config=SimpleNamespace(
            ref=config_ref,
            path=config_path,
            artifact=SimpleNamespace(
                owner_root=config_root,
                owner_topology="torus9",
                owner_lineage_id="child-lineage",
                owner_status="ACTIVE",
                identity={"immutable_verified": True},
            ),
            config=config,
        ),  # type: ignore[arg-type]
        output_lineage=OutputLineage("torus9", "child-lineage", tmp_path / "output"),
    )
    checked: list[tuple[Path, str, int]] = []

    class FakeResolver:
        runs_root = tmp_path / "runs"

        def checkpoint(self, _value):
            return SimpleNamespace(
                node=SimpleNamespace(parent=parent_ref),
                generation=96,
                ref=child_ref,
            )

    class FakeSupervisor:
        captured: dict[str, object] = {}

        def __init__(self, root, **kwargs):
            self.captured = {"root": root, **kwargs}
            type(self).captured = self.captured

        def run_once(self):
            command = self.captured["command"]
            result_path = Path(command[-1])  # type: ignore[index]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({"checkpoint": child_ref.to_dict()}),
                encoding="utf-8",
            )
            return SimpleNamespace(success=True, reason=None)

    def check_commit(*, root, lineage_id, generation):
        checked.append((root, lineage_id, generation))

    monkeypatch.setattr(production_arm, "SupervisorV2", FakeSupervisor)
    monkeypatch.setattr(production_arm, "validate_generation_commit", check_commit)

    result = ProductionArmExecutionPath(
        resolver=FakeResolver(),
        repo_root=tmp_path,
    )._run_supervised_generation(resolved, tmp_path / "output")

    assert result.ref == child_ref
    assert checked == [(tmp_path / "output", "child-lineage", 96)]
    assert FakeSupervisor.captured["env"]["AZ_DRIVER_HEARTBEAT_PATH"].endswith(  # type: ignore[index]
        "runtime/heartbeats/generation-0096.json"
    )
