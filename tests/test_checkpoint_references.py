from __future__ import annotations

import json
from pathlib import Path

import pytest

import gocube_golden.run_storage as run_storage
from gocube_golden.run_storage import CheckpointResolutionError, resolve_checkpoint
from tools.torus9_run_driver import (
    _parent_replay_reference_paths,
    _resolve_parent_replay_reference_paths,
)


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


def test_parent_bootstrap_references_exactly_m42_through_m47(tmp_path: Path) -> None:
    root = tmp_path / "parent"
    paths = _parent_replay_reference_paths(root, 48)
    assert [path.name for path in paths] == [
        "iter-42-fresh.jsonl",
        "iter-43-fresh.jsonl",
        "iter-44-fresh.jsonl",
        "iter-45-fresh.jsonl",
        "iter-46-fresh.jsonl",
        "iter-47-fresh.jsonl",
    ]


def test_parent_bootstrap_resolves_cross_lineage_window_without_copying(tmp_path: Path) -> None:
    older_root = tmp_path / "older-lineage"
    winner_root = tmp_path / "winner-lineage"
    inherited = []
    for generation in range(78, 81):
        path = older_root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"older-{generation}\n", encoding="utf-8")
        inherited.append(
            {
                "generation": generation,
                "path": str(path),
                "sha256": run_storage._sha256_file(path),
            }
        )
    for generation in range(81, 84):
        path = winner_root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"winner-{generation}\n", encoding="utf-8")

    (winner_root / "manifest.json").write_text(
        json.dumps({"parent_checkpoint": {"replay_references": inherited}}),
        encoding="utf-8",
    )
    paths = _resolve_parent_replay_reference_paths(
        parent_root=winner_root,
        generation=84,
        parent_reference={"lineage_id": "winner-lineage"},
    )

    assert paths == tuple(
        [
            older_root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
            for generation in range(78, 81)
        ]
        + [
            winner_root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
            for generation in range(81, 84)
        ]
    )
    assert not (winner_root / "replay" / "iter-78-fresh.jsonl").exists()


def test_parent_bootstrap_rejects_cross_lineage_sha_mismatch(tmp_path: Path) -> None:
    parent_root = tmp_path / "winner-lineage"
    path = parent_root / "replay" / "iter-78-fresh.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("tampered\n", encoding="utf-8")
    (parent_root / "manifest.json").write_text(
        json.dumps(
            {
                "parent_checkpoint": {
                    "replay_references": [
                        {
                            "generation": 78,
                            "path": str(path),
                            "sha256": "sha256:" + "0" * 64,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="M78 SHA-256 mismatch"):
        _resolve_parent_replay_reference_paths(
            parent_root=parent_root,
            generation=84,
            parent_reference=None,
        )


def test_downstream_manifest_protects_parent_from_discard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from gocube_golden.run_lifecycle import discard_lineage

    monkeypatch.setattr(run_storage, "RUNS_ROOT", tmp_path / "runs")
    parent, parent_hash = _lineage(
        tmp_path, lineage_id="parent-to-retain", generation=47, content=b"parent"
    )
    (parent / "runtime" / "state.json").write_text(
        json.dumps({"state": "SOFT_STOPPED"}), encoding="utf-8"
    )
    child = run_storage.active_lineage_dir("torus9", "child-reference")
    child.mkdir(parents=True)
    (child / "manifest.json").write_text(
        json.dumps({
            "lineage_id": "child-reference",
            "topology": "torus9",
            "status": "ACTIVE",
            "parent_checkpoint": {
                "lineage_id": "parent-to-retain",
                "path": str(parent / "checkpoints" / "M47.pt"),
                "sha256": parent_hash,
            },
            "checkpoint_hashes": {},
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="reference"):
        discard_lineage(
            repo_root=tmp_path,
            lineage_id="parent-to-retain",
            reason="test protection",
            useful_result="none",
        )
