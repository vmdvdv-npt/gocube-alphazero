from __future__ import annotations

import json
from pathlib import Path

from gocube_golden.artifact_graph import (
    ArtifactRef,
    EffectiveConfig,
    EffectiveConfigRef,
    CheckpointRef,
    publish_checkpoint_graph,
)
from gocube_golden.artifact_catalog import sha256_file


def test_publish_allows_subsequent_same_lineage_generation(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "torus9" / "active" / "child"
    (root / "metadata").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "replay").mkdir()
    parent = CheckpointRef(
        "torus9", "parent", "M95", 95, "checkpoints/M95.pt", "sha256:" + "a" * 64
    )
    manifest = {
        "lineage_id": "child",
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": parent.to_dict(),
        "checkpoint_hashes": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    effective_path = root / "metadata" / "effective.json"
    effective_path.write_text(
        json.dumps(EffectiveConfig("torus9", {"topology": "torus9"}).to_dict()),
        encoding="utf-8",
    )
    effective = EffectiveConfigRef(
        ArtifactRef("metadata/effective.json", sha256_file(effective_path)),
        EffectiveConfig("torus9", {"topology": "torus9"}).fingerprint,
    )

    def publish(generation: int, immediate_parent: CheckpointRef) -> None:
        checkpoint_path = root / "checkpoints" / f"M{generation}.pt"
        replay_path = root / "replay" / f"iter-{generation}-fresh.jsonl"
        checkpoint_path.write_bytes(f"checkpoint-{generation}".encode())
        replay_path.write_bytes(f"replay-{generation}".encode())
        checkpoint = CheckpointRef(
            "torus9",
            "child",
            f"M{generation}",
            generation,
            checkpoint_path.relative_to(root).as_posix(),
            sha256_file(checkpoint_path),
        )
        publish_checkpoint_graph(
            root=root,
            parent=immediate_parent,
            checkpoint=checkpoint,
            fresh_replay=ArtifactRef(
                replay_path.relative_to(root).as_posix(), sha256_file(replay_path)
            ),
            effective_config=effective,
            generation_commit=ArtifactRef(
                f"generation-{generation:02d}.complete.json", "sha256:" + "b" * 64
            ),
        )

    publish(96, parent)
    persisted_with_replay_refs = json.loads((root / "manifest.json").read_text())
    persisted_with_replay_refs["parent_checkpoint"]["replay_references"] = [
        {
            "generation": 95,
            "path": "/immutable/parent/replay/iter-95-fresh.jsonl",
            "sha256": "sha256:" + "c" * 64,
        }
    ]
    (root / "manifest.json").write_text(
        json.dumps(persisted_with_replay_refs), encoding="utf-8"
    )
    child_m96 = CheckpointRef(
        "torus9", "child", "M96", 96, "checkpoints/M96.pt", "sha256:" + "0" * 64
    )
    # The second publication must retain the lineage-root parent while
    # recording M96 as the immediate parent in the new CheckpointNode.
    publish(97, child_m96)

    persisted = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["parent_checkpoint"]["checkpoint_id"] == parent.checkpoint_id
    assert persisted["parent_checkpoint"]["replay_references"][0]["generation"] == 95
    node = json.loads((root / "metadata" / "checkpoints" / "M97.json").read_text())
    assert node["parent"] == child_m96.to_dict()
