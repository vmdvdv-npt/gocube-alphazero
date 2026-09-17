from __future__ import annotations

import json
from pathlib import Path

import pytest

import gocube_golden.run_storage as run_storage
from gocube_golden.run_storage import CheckpointResolutionError, resolve_checkpoint


def _lineage(
    runs_root: Path,
    *,
    lineage_id: str,
    generation: int,
    content: bytes,
) -> tuple[Path, str]:
    manifest = {
        "lineage_id": lineage_id,
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": None,
        "git_commit": "deadbeef",
        "config_fingerprint": "sha256:config",
        "created_at": "2026-09-17T00:00:00+00:00",
        "checkpoint_hashes": {},
    }
    run_storage.create_lineage(
        "torus9",
        lineage_id,
        manifest=manifest,
        extra_directories=("runtime",),
    )
    root = run_storage.active_lineage_dir("torus9", lineage_id)
    checkpoint = root / "checkpoints" / f"M{generation}.pt"
    checkpoint.write_bytes(content)
    digest = run_storage._sha256_file(checkpoint)
    manifest["checkpoint_hashes"] = {f"checkpoints/M{generation}.pt": digest}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, digest


def test_resolve_external_parent_without_copying(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(run_storage, "RUNS_ROOT", runs_root)
    parent_root, digest = _lineage(
        runs_root,
        lineage_id="parent-lineage",
        generation=17,
        content=b"canonical-parent",
    )
    child_root, _ = _lineage(
        runs_root,
        lineage_id="child-lineage",
        generation=27,
        content=b"child-candidate",
    )

    resolved = resolve_checkpoint(
        {
            "topology": "torus9",
            "lineage_id": "parent-lineage",
            "label": "M17",
            "generation": 17,
            "path": str(parent_root / "checkpoints" / "M17.pt"),
            "sha256": digest,
        },
        runs_root=runs_root,
    )

    assert resolved.path == parent_root / "checkpoints" / "M17.pt"
    assert resolved.sha256 == digest
    assert not (child_root / "checkpoints" / "M17.pt").exists()
    assert len(list(runs_root.rglob("M17.pt"))) == 1


def test_resolve_rejects_sha_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(run_storage, "RUNS_ROOT", runs_root)
    root, _digest = _lineage(
        runs_root,
        lineage_id="lineage",
        generation=17,
        content=b"canonical-parent",
    )

    with pytest.raises(CheckpointResolutionError, match="actual sha256"):
        resolve_checkpoint(
            {
                "topology": "torus9",
                "lineage_id": "lineage",
                "generation": 17,
                "path": str(root / "checkpoints" / "M17.pt"),
                "sha256": "sha256:" + "0" * 64,
            },
            runs_root=runs_root,
        )


def test_resolve_reports_missing_external_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(run_storage, "RUNS_ROOT", runs_root)
    _root, _digest = _lineage(
        runs_root,
        lineage_id="lineage",
        generation=17,
        content=b"canonical-parent",
    )
    missing = runs_root / "torus9" / "active" / "lineage" / "checkpoints" / "M18.pt"

    with pytest.raises(CheckpointResolutionError) as exc_info:
        resolve_checkpoint(
            {
                "topology": "torus9",
                "lineage_id": "lineage",
                "label": "M18",
                "generation": 18,
                "path": str(missing),
                "sha256": "sha256:" + "1" * 64,
            },
            runs_root=runs_root,
        )

    message = str(exc_info.value)
    assert "lineage: lineage" in message
    assert "checkpoint: M18" in message
    assert "exists: false" in message
    assert "expected sha256" in message


def test_resolve_same_lineage_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(run_storage, "RUNS_ROOT", runs_root)
    root, digest = _lineage(
        runs_root,
        lineage_id="same-lineage",
        generation=22,
        content=b"same-lineage-reference",
    )

    resolved = resolve_checkpoint(
        {
            "topology": "torus9",
            "lineage_id": "same-lineage",
            "generation": 22,
            "path": "checkpoints/M22.pt",
            "sha256": digest,
        },
        runs_root=runs_root,
    )

    assert resolved.path == root / "checkpoints" / "M22.pt"
    assert resolved.lineage_id == "same-lineage"
