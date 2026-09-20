from __future__ import annotations

import json
from pathlib import Path

import pytest

from gocube_golden.artifact_graph import (
    ArtifactRef,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    publish_checkpoint_graph,
    validate_generation_commit,
)
from gocube_golden.artifact_catalog import sha256_file
from gocube_golden.provenance import canonical_json


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_file(path)


def _committed_graph(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "torus9" / "active" / "child"
    (root / "checkpoints").mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "replay").mkdir()
    (root / "training").mkdir()
    (root / "runtime").mkdir()

    parent = CheckpointRef(
        "torus9", "parent", "M0", 0, "checkpoints/M0.pt", "sha256:" + "0" * 64
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "lineage_id": "child",
                "topology": "torus9",
                "status": "ACTIVE",
                "parent_checkpoint": parent.to_dict(),
                "checkpoint_hashes": {},
            }
        ),
        encoding="utf-8",
    )

    config = EffectiveConfig("torus9", {"topology": "torus9"})
    config_path = root / "metadata" / "effective.json"
    config_path.write_text(canonical_json(config.to_dict()) + "\n", encoding="utf-8")
    effective = EffectiveConfigRef(
        ArtifactRef("metadata/effective.json", sha256_file(config_path)),
        config.fingerprint,
    )

    checkpoint_path = root / "checkpoints" / "M1.pt"
    fresh_path = root / "replay" / "iter-01-fresh.jsonl"
    rolling_path = root / "replay" / "rolling-after-01.jsonl"
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    training_path = root / "training" / "iter-01.json"
    summary_path = root / "iter-01-summary.json"
    checkpoint_sha = _write(checkpoint_path, b"checkpoint")
    fresh_sha = _write(fresh_path, b"fresh\n")
    rolling_sha = _write(rolling_path, b"rolling\n")
    metadata_sha = _write(metadata_path, b"metadata\n")
    training_sha = _write(training_path, b"training\n")
    summary_sha = _write(summary_path, b"summary\n")

    marker_path = root / "generation-01.complete.json"
    marker = {
        "schema": "training-generation-commit-v1",
        "run_id": "child",
        "generation": 1,
        "label": "M1",
        "checkpoint_sha256": checkpoint_sha,
        "fresh_replay_sha256": fresh_sha,
        "rolling_replay_sha256": rolling_sha,
        "rolling_replay_size_bytes": rolling_path.stat().st_size,
        "checkpoint_metadata_sha256": metadata_sha,
        "training_metrics_sha256": training_sha,
        "summary_sha256": summary_sha,
    }
    marker_path.write_text(canonical_json(marker) + "\n", encoding="utf-8")
    marker_sha = sha256_file(marker_path)

    checkpoint = CheckpointRef(
        "torus9", "child", "M1", 1, "checkpoints/M1.pt", checkpoint_sha
    )
    artifact_identities = {
        "fresh_replay": {
            "path": "replay/iter-01-fresh.jsonl",
            "sha256": fresh_sha,
            "size_bytes": fresh_path.stat().st_size,
        },
        "rolling_replay": {
            "path": "replay/rolling-after-01.jsonl",
            "sha256": rolling_sha,
            "size_bytes": rolling_path.stat().st_size,
        },
        "checkpoint": {
            "path": "checkpoints/M1.pt",
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "checkpoint_metadata": {
            "path": "checkpoints/M1.metadata.json",
            "sha256": metadata_sha,
            "size_bytes": metadata_path.stat().st_size,
        },
        "training_metrics": {
            "path": "training/iter-01.json",
            "sha256": training_sha,
            "size_bytes": training_path.stat().st_size,
        },
        "iteration_summary": {
            "path": "iter-01-summary.json",
            "sha256": summary_sha,
            "size_bytes": summary_path.stat().st_size,
        },
    }
    publish_checkpoint_graph(
        root=root,
        parent=parent,
        checkpoint=checkpoint,
        fresh_replay=ArtifactRef("replay/iter-01-fresh.jsonl", fresh_sha),
        effective_config=effective,
        generation_commit=ArtifactRef("generation-01.complete.json", marker_sha),
        artifact_identities=artifact_identities,
    )
    return root, rolling_path


def test_same_transaction_validation_reuses_bound_rolling_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rolling_path = _committed_graph(tmp_path)
    import gocube_golden.artifact_graph as artifact_graph

    original = artifact_graph.sha256_file
    hashed: list[Path] = []

    def spy(path: Path) -> str:
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(artifact_graph, "sha256_file", spy)
    validate_generation_commit(
        root=root,
        lineage_id="child",
        generation=1,
        reuse_committed_rolling_replay_identity=True,
    )

    assert rolling_path not in hashed


def test_same_transaction_identity_reuse_fails_on_size_or_evidence_drift(
    tmp_path: Path,
) -> None:
    root, rolling_path = _committed_graph(tmp_path)
    rolling_path.write_bytes(b"changed-size")
    with pytest.raises(ValueError, match="size evidence"):
        validate_generation_commit(
            root=root,
            lineage_id="child",
            generation=1,
            reuse_committed_rolling_replay_identity=True,
        )

    root, rolling_path = _committed_graph(tmp_path / "evidence")
    provenance_path = root / "metadata" / "provenance-v2" / "M1.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["artifact_identities"]["rolling_replay"]["sha256"] = "sha256:" + "f" * 64
    provenance_path.write_text(canonical_json(provenance) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        validate_generation_commit(
            root=root,
            lineage_id="child",
            generation=1,
            reuse_committed_rolling_replay_identity=True,
        )
