from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gocube_golden.orchestrator_v2.torus9_production as production
from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.orchestrator_v2.torus9_production import Torus9ProductionLineage
from gocube_golden.provenance import CodeIdentity


def _config() -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"topology": "torus9"},
        self_play={"mcts_simulations": 64, "games_per_iteration": 32},
        training={"learning_rate": 0.0003, "optimizer_steps": 40},
        replay={"generations": 1, "cap": 5000},
        arena={"reference_gap": 1},
    )


def _prepare_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_code_rollover: bool,
    second_worktree_clean: bool,
) -> dict[str, object]:
    config = _config()
    parent = SimpleNamespace(
        ref=CheckpointRef(
            "torus9",
            "parent",
            "M95",
            95,
            "checkpoints/M95.pt",
            "sha256:" + "a" * 64,
        )
    )
    identities = iter(
        [
            CodeIdentity("1" * 40, "2" * 40, True),
            CodeIdentity("3" * 40, "4" * 40, second_worktree_clean),
        ]
    )
    monkeypatch.setattr(production, "capture_code_identity", lambda _root: next(identities))
    lineage = Torus9ProductionLineage(tmp_path / "runs")

    lineage.prepare(
        topology="torus9",
        lineage_id="lineage",
        parent=parent,
        effective_config=config,
        experiment_id="experiment",
        arm_id="continuous",
    )
    lineage.prepare(
        topology="torus9",
        lineage_id="lineage",
        parent=parent,
        effective_config=config,
        experiment_id="experiment",
        arm_id="continuous",
        allow_code_rollover=allow_code_rollover,
    )

    return json.loads(
        (tmp_path / "runs" / "torus9" / "active" / "lineage" / "manifest.json").read_text()
    )


def test_clean_code_change_without_allow_code_rollover_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="allow_code_rollover=True"):
        _prepare_lineage(
            tmp_path,
            monkeypatch,
            allow_code_rollover=False,
            second_worktree_clean=True,
        )


def test_lineage_manifest_records_orchestrator_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _prepare_lineage(
        tmp_path,
        monkeypatch,
        allow_code_rollover=True,
        second_worktree_clean=True,
    )

    assert manifest["orchestrator_version"] == "V2"
    assert manifest["orchestrator_entrypoint"] == (
        "gocube_golden.orchestrator_v2.production_entrypoint"
    )


def test_clean_code_change_with_allow_code_rollover_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _prepare_lineage(
        tmp_path,
        monkeypatch,
        allow_code_rollover=True,
        second_worktree_clean=True,
    )

    assert manifest["lineage_initial_git_commit"] == "1" * 40
    assert manifest["git_commit"] == "3" * 40
    assert manifest["current_git_tree"] == "4" * 40


def test_dirty_worktree_rejects_code_rollover_even_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="clean working tree"):
        _prepare_lineage(
            tmp_path,
            monkeypatch,
            allow_code_rollover=True,
            second_worktree_clean=False,
        )


def test_replay_reference_enrichment_does_not_change_parent_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    parent = SimpleNamespace(
        ref=CheckpointRef(
            "torus9",
            "parent",
            "M95",
            95,
            "checkpoints/M95.pt",
            "sha256:" + "a" * 64,
        )
    )
    identity = CodeIdentity("1" * 40, "2" * 40, True)
    monkeypatch.setattr(production, "capture_code_identity", lambda _root: identity)
    lineage = Torus9ProductionLineage(tmp_path / "runs")
    lineage.prepare(
        topology="torus9",
        lineage_id="lineage",
        parent=parent,
        effective_config=config,
        experiment_id="experiment",
        arm_id="continuous",
    )

    manifest_path = tmp_path / "runs" / "torus9" / "active" / "lineage" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    references = [
        {
            "generation": 90,
            "path": "replay/iter-90-fresh.jsonl",
            "sha256": "sha256:" + "b" * 64,
        }
    ]
    manifest["parent_checkpoint"]["replay_references"] = references
    manifest["parent_replay_references"] = references
    manifest_path.write_text(json.dumps(manifest))

    lineage.prepare(
        topology="torus9",
        lineage_id="lineage",
        parent=parent,
        effective_config=config,
        experiment_id="experiment",
        arm_id="continuous",
    )

    resumed = json.loads(manifest_path.read_text())
    assert resumed["parent_checkpoint"]["replay_references"] == references
