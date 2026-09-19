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
    replay_root = tmp_path / "replay-owner"
    replay_path = replay_root / "replay" / "iter-95-fresh.jsonl"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes(b"replay bytes")
    replay_ref = ArtifactRef(
        "replay/iter-95-fresh.jsonl",
        "sha256:" + "2" * 64,
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
        "replay_artifacts": [
            {
                "ref": replay_ref.to_dict(),
                "path": str(replay_path),
                "owner_root": str(replay_root),
                "owner_topology": "torus9",
                "owner_lineage_id": "parent-lineage",
                "owner_status": "ACTIVE",
                "identity": {"immutable_verified": True, "sha256": replay_ref.sha256},
            }
        ],
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
        "replay_identity": None,
    }

    class FakeResolver:
        def __init__(self, runs_root):
            self.runs_root = Path(runs_root)

        def checkpoint(self, value):
            assert value == parent_ref
            return SimpleNamespace(ref=parent_ref)

    original_sha256 = production_arm.sha256_file
    hashed_paths: list[Path] = []

    def tracked_sha256(path):
        resolved = Path(path).resolve()
        if resolved == replay_path.resolve():
            raise AssertionError("child must not rehash replay artifacts")
        hashed_paths.append(resolved)
        return original_sha256(path)

    monkeypatch.setattr(production_arm, "ArtifactResolver", FakeResolver)
    monkeypatch.setattr(production_arm, "sha256_file", tracked_sha256)

    resolved = production_arm._deserialize_resolved_input(payload)

    assert resolved.replay_artifacts[0].path == replay_path.resolve()
    assert hashed_paths == [config_path.resolve()]


def test_production_arm_uses_standard_supervisor_policy_by_default():
    assert ProductionArmExecutionPath().supervisor_policy is None
