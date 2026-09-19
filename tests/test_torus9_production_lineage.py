from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.orchestrator_v2.torus9_production import Torus9ProductionLineage
from gocube_golden.provenance import CodeIdentity
import gocube_golden.orchestrator_v2.torus9_production as production


def _config() -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"topology": "torus9"},
        self_play={"mcts_simulations": 64, "games_per_iteration": 32},
        training={"learning_rate": 0.0003, "optimizer_steps": 40},
        replay={"generations": 1, "cap": 5000},
        arena={"reference_gap": 1},
    )


def test_resume_allows_clean_code_rollover_without_changing_lineage_origin(
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
    identities = iter(
        [
            CodeIdentity("1" * 40, "2" * 40, True),
            CodeIdentity("3" * 40, "4" * 40, True),
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
    )

    manifest = json.loads(
        (tmp_path / "runs" / "torus9" / "active" / "lineage" / "manifest.json").read_text()
    )
    assert manifest["lineage_initial_git_commit"] == "1" * 40
    assert manifest["git_commit"] == "3" * 40
    assert manifest["current_git_tree"] == "4" * 40

