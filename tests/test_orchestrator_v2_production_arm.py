from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gocube_golden.artifact_catalog import sha256_file as physical_sha256
from gocube_golden.orchestrator_v2 import (
    ArtifactRef,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    ProductionArmExecutionPath,
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

    # The public override remains unset, while the real production seam gives
    # post-commit result publication enough bounded time to finish.
    assert path.supervisor_policy is None
    assert (
        path._effective_supervisor_policy().committed_drain_seconds
        == 15 * 60.0
    )


def test_production_arm_preserves_explicit_supervisor_policy():
    policy = SupervisorPolicy(committed_drain_seconds=7.0)

    path = ProductionArmExecutionPath(supervisor_policy=policy)

    assert path._effective_supervisor_policy() is policy
