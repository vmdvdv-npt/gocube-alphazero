from __future__ import annotations

import json
from pathlib import Path

import pytest

import gocube_golden.run_storage as run_storage
from gocube_golden.run_lifecycle import archive_lineage, discard_lineage


def _lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lineage_id: str = "life-test") -> Path:
    monkeypatch.setattr(run_storage, "RUNS_ROOT", tmp_path / "runs")
    root = run_storage.active_lineage_dir("cube4", lineage_id)
    manifest = {
        "lineage_id": lineage_id,
        "topology": "cube4",
        "status": "ACTIVE",
        "parent_checkpoint": None,
        "git_commit": "deadbeef",
        "config_fingerprint": "sha256:config",
        "created_at": "2026-09-16T00:00:00+00:00",
        "checkpoint_hashes": {"checkpoints/M1.pt": "sha256:model-one"},
    }
    run_storage.create_lineage(
        "cube4",
        lineage_id,
        manifest=manifest,
        extra_directories=("runtime",),
    )
    (root / "runtime" / "state.json").write_text(
        json.dumps({"state":"COMPLETED"}), encoding="utf-8"
    )
    (root / "checkpoints" / "M1.pt").write_text("heavy", encoding="utf-8")
    return root


def test_archive_moves_whole_lineage_without_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    active = _lineage(tmp_path, monkeypatch)
    target = archive_lineage(repo_root=tmp_path, lineage_id="life-test")
    assert not active.exists()
    assert target == run_storage.archived_lineage_dir("cube4", "life-test")
    assert (target / "checkpoints" / "M1.pt").read_text() == "heavy"
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["status"] == "ARCHIVED"


def test_discard_requires_explicit_confirmation_data_and_retains_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    active = _lineage(tmp_path, monkeypatch)
    record = discard_lineage(
        repo_root=tmp_path,
        lineage_id="life-test",
        reason="duplicate broken experiment",
        useful_result="none",
    )
    assert not active.exists()
    assert record.is_file()
    text = record.read_text()
    assert "Status: DISCARDED" in text
    assert "duplicate broken experiment" in text
    assert "Useful result: none" in text


def test_discard_refuses_referenced_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    active = _lineage(tmp_path, monkeypatch)
    evaluation = tmp_path / "runs" / "cube4" / "evaluations" / "compare" / "manifest.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text(
        json.dumps({"reference_sha256":"sha256:model-one"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="reference"):
        discard_lineage(
            repo_root=tmp_path,
            lineage_id="life-test",
            reason="superseded",
            useful_result="none",
        )
    assert active.exists()
