from __future__ import annotations

import json
from pathlib import Path

import pytest

import gocube_golden.run_storage as run_storage
from gocube_golden.run_storage import (
    active_lineage_dir,
    archived_lineage_dir,
    create_lineage,
    ensure_evaluation_layout,
    ensure_lineage_layout,
    evaluation_dir,
    topology_for_profile,
)


def test_lineage_and_evaluation_paths_use_canonical_layout() -> None:
    assert active_lineage_dir("torus9", "run-01").as_posix().endswith(
        "runs/torus9/active/run-01"
    )
    assert archived_lineage_dir("cube4", "run-01").as_posix().endswith(
        "runs/cube4/archive/run-01"
    )
    assert evaluation_dir("torus9", "arena-01").as_posix().endswith(
        "runs/torus9/evaluations/arena-01"
    )


def test_path_components_cannot_escape_run_root() -> None:
    with pytest.raises(ValueError):
        active_lineage_dir("torus9", "../outside")
    with pytest.raises(ValueError):
        evaluation_dir("torus9", "nested/evaluation")
    with pytest.raises(ValueError):
        active_lineage_dir("../outside", "run-01")


def test_profile_topologies_are_explicit() -> None:
    assert topology_for_profile("gocube-torus9-golden-v3") == "torus9"
    assert topology_for_profile("gocube-torus5-aux") == "torus5"
    assert topology_for_profile("gocube-cube4-golden-training-v1") == "cube4"
    with pytest.raises(ValueError):
        topology_for_profile("unknown-profile")


def test_layout_helpers_create_owned_subdirectories(tmp_path: Path) -> None:
    manifest = {
        "lineage_id": "test-lineage",
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": None,
        "git_commit": "test-commit",
        "config_fingerprint": "sha256:test-config",
        "created_at": "2026-09-16T00:00:00+00:00",
        "checkpoint_hashes": {},
    }
    lineage = ensure_lineage_layout(
        tmp_path / "lineage",
        manifest=manifest,
        extra_directories=("selfplay", "replay"),
    )
    evaluation = ensure_evaluation_layout(tmp_path / "evaluation")
    assert {path.name for path in lineage.iterdir()} == {
        "arena",
        "checkpoints",
        "data",
        "logs",
        "metrics",
        "manifest.json",
        "replay",
        "selfplay",
    }
    assert {path.name for path in evaluation.iterdir()} == {"logs", "metrics", "results"}
    assert json.loads((lineage / "manifest.json").read_text()) == manifest


def test_new_lineage_is_published_with_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(run_storage, "RUNS_ROOT", tmp_path / "runs")
    manifest = {
        "lineage_id": "new-lineage",
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": None,
        "git_commit": "test-commit",
        "config_fingerprint": "sha256:test-config",
        "created_at": "2026-09-16T00:00:00+00:00",
        "checkpoint_hashes": {},
    }

    lineage = create_lineage("torus9", "new-lineage", manifest=manifest)

    assert lineage == tmp_path / "runs" / "torus9" / "active" / "new-lineage"
    assert json.loads((lineage / "manifest.json").read_text()) == manifest


def test_lineage_creation_requires_manifest_and_rejects_manifestless_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        ensure_lineage_layout(tmp_path / "without-manifest")  # type: ignore[call-arg]
    assert not (tmp_path / "without-manifest").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    manifest = {
        "lineage_id": "existing",
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": None,
        "git_commit": "test-commit",
        "config_fingerprint": "sha256:test-config",
        "created_at": "2026-09-16T00:00:00+00:00",
        "checkpoint_hashes": {},
    }
    with pytest.raises(FileExistsError, match="manifest.json"):
        ensure_lineage_layout(existing, manifest=manifest)
