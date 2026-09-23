from __future__ import annotations

import json
from pathlib import Path

import pytest

from gocube_golden.artifact_resolver import ArtifactResolver
from gocube_golden.cube_m0_publisher import publish_cube_m0
from gocube_golden.provenance import CodeIdentity


def _config(size: int = 2) -> dict[str, object]:
    return {
        "schema": "gocube-effective-config-v2",
        "version": 2,
        "topology": f"cube{size}",
        "compatibility": {"family": "cube-v2", "size": size, "topology": f"cube{size}"},
        "self_play": {"master_seed": 2026092301},
        "training": {
            "learning_rate": 0.001,
            "batch_size": 1,
            "optimizer_steps": 1,
            "weight_decay": 0.0,
        },
        "replay": {"generations": 2, "cap": 16},
        "execution": {},
        "arena": {},
        "supervision": {},
        "extensions": {"master_seed": 2026092301},
    }


def test_cube_family_m0_publisher_writes_canonical_genesis(tmp_path: Path) -> None:
    identity = CodeIdentity("0" * 40, "1" * 40, True)
    publication = publish_cube_m0(
        size=2,
        lineage_id="cube2-stage9-m0",
        effective_config=_config(),
        seed=2026092301,
        runs_root=tmp_path / "runs",
        code_identity=identity,
    )

    root = publication.root
    expected = (
        "manifest.json",
        "checkpoints/M0.pt",
        "checkpoints/M0.metadata.json",
        "replay/rolling-after-00.jsonl",
        "metadata/checkpoints/M0.json",
        "metadata/provenance-v2/M0.json",
        "runtime/artifact-catalog.json",
    )
    assert all((root / path).is_file() for path in expected)
    assert not (root / "data").exists()
    assert not (root / "checkpoint").exists()
    assert not (root / "arena-results").exists()

    node = ArtifactResolver(tmp_path / "runs").checkpoint(publication.checkpoint)
    assert node.generation == 0
    assert node.node.genesis is True
    assert node.node.parent is None
    assert node.node.fresh_replay is None

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_checkpoint"] is None
    assert manifest["checkpoint_hashes"]["checkpoints/M0.pt"] == publication.checkpoint.sha256
    metadata = json.loads((root / "checkpoints/M0.metadata.json").read_text(encoding="utf-8"))
    assert metadata["generation"] == 0
    assert metadata["training_seed"] == 2026092301
    assert metadata["optimizer_updates"] == 0
    assert (root / "replay/rolling-after-00.jsonl").read_bytes() == b""

    with pytest.raises(FileExistsError, match="existing Cube M0 lineage"):
        publish_cube_m0(
            size=2,
            lineage_id="cube2-stage9-m0",
            effective_config=_config(),
            seed=2026092301,
            runs_root=tmp_path / "runs",
            code_identity=identity,
        )
