from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gocube_golden.artifact_catalog import ArtifactCatalog, sha256_file
from gocube_golden.orchestrator_v2.artifact_resolver import ArtifactResolver, checkpoint_node_path
from gocube_golden.orchestrator_v2.contracts import CheckpointRef
from tools.orchestrator_v2_backfill_nodes import (
    ScopeEntry,
    TARGET_COUNT,
    TARGET_SCOPE,
    run_backfill,
)


SHA = lambda letter: "sha256:" + hashlib.sha256(letter.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _profile() -> dict[str, object]:
    return {
        "profile_id": "synthetic-torus9",
        "profile_fingerprint": SHA("p"),
        "topology": {"topology_id": "synthetic-topology", "fingerprint": SHA("t")},
        "rules": {"profile_id": "synthetic-rules", "fingerprint": SHA("r")},
        "observation": {"schema_id": "synthetic-observation", "schema_version": 1, "fingerprint": SHA("o")},
        "target": {"contract_id": "synthetic-target", "contract_version": 1, "fingerprint": SHA("a")},
        "network": {"architecture_id": "synthetic-network", "fingerprint": SHA("n")},
        "self_play": {"mcts_simulations": 8, "games_per_iteration": 2},
        "training": {"optimizer": "Adam", "learning_rate": 0.1, "batch_size": 2},
        "replay": {"window": "rolling", "generations": 2, "cap": 4},
    }


def _make_lineage(
    runs_root: Path,
    lineage: str,
    generation: int,
    *,
    parent: tuple[str, str, int] | None,
    profile_path: Path,
) -> CheckpointRef:
    root = runs_root / "torus9" / "active" / lineage
    checkpoint = root / "checkpoints" / f"M{generation}.pt"
    replay = root / "replay" / f"iter-{generation}-fresh.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    replay.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{lineage}-{generation}".encode())
    replay.write_bytes(f"replay-{lineage}-{generation}".encode())
    cp_sha = sha256_file(checkpoint)
    replay_sha = sha256_file(replay)
    metadata = {
        "checkpoint_label": f"M{generation}",
        "run_id": lineage,
        "topology_id": "synthetic-topology",
        "topology_fingerprint": SHA("t"),
        "rules_profile_id": "synthetic-rules",
        "rules_fingerprint": SHA("r"),
        "observation_schema_id": "synthetic-observation",
        "observation_schema_version": 1,
        "observation_fingerprint": SHA("o"),
        "observation_shape": [1, 1],
        "target_contract_id": "synthetic-target",
        "target_contract_version": 1,
        "target_fingerprint": SHA("a"),
        "architecture_id": "synthetic-network",
        "architecture_fingerprint": SHA("n"),
        "model_parameter_count": 1,
        "network_heads_and_shapes": {},
        "profile_id": "synthetic-torus9",
        "profile_fingerprint": SHA("p"),
        "selfplay_contract_id": "synthetic-selfplay",
        "selfplay_contract_fingerprint": SHA("s"),
        "scientific_contract": {
            "optimizer": "Adam", "learning_rate": 0.1, "weight_decay": 0.0,
            "batch_size": 2, "optimizer_steps": 1, "samples_consumed": 2,
            "model_gating": False, "ownership_loss": False, "score_loss": False,
            "replay_generations": 2, "replay_cap": 4,
        },
        "replay_policy": "rolling-recent-generations",
        "komi": 0.5,
    }
    parent_identity = None
    if parent is not None:
        parent_lineage, parent_label, parent_generation = parent
        parent_root = runs_root / "torus9" / "active" / parent_lineage
        parent_path = parent_root / "checkpoints" / f"{parent_label}.pt"
        parent_metadata = parent_root / "checkpoints" / f"{parent_label}.metadata.json"
        parent_identity = {
            "label": parent_label,
            "lineage_id": parent_lineage,
            "generation": parent_generation,
            "path": str(parent_path),
            "metadata_path": str(parent_metadata),
            "artifact_sha256": sha256_file(parent_path),
        }
    if parent_identity is not None:
        metadata["parent_checkpoint_identity"] = parent_identity
    _write_json(root / "checkpoints" / f"M{generation}.metadata.json", metadata)
    complete = {
        "schema": "training-generation-commit-v1",
        "generation": generation,
        "label": f"M{generation}",
        "checkpoint_sha256": cp_sha,
        "fresh_replay_sha256": replay_sha,
        "checkpoint_metadata_sha256": SHA("m"),
    }
    _write_json(root / f"generation-{generation}.complete.json", complete)
    spec = {
        "schema": "synthetic-run-spec",
        "topology": "torus9",
        "profile_path": str(profile_path),
        "generation": {"driver_config": {"games": 2, "workers": 1}},
        "arena": {"enabled": False},
        "supervision": {},
    }
    _write_json(root / "run-spec.json", spec)
    manifest = {
        "lineage_id": lineage,
        "topology": "torus9",
        "status": "ACTIVE",
        "config_fingerprint": SHA("c"),
        "run_spec": {"fingerprint": SHA("q"), "path": "run-spec.json"},
        "checkpoint_hashes": {f"checkpoints/M{generation}.pt": cp_sha},
    }
    _write_json(root / "manifest.json", manifest)
    catalog_path = root / "runtime" / "artifact-catalog.json"
    catalog = ArtifactCatalog.initialize(catalog_path, lineage_id=lineage, root=root)
    catalog.register_generation(
        generation,
        [
            {"path": f"checkpoints/M{generation}.pt", "sha256": cp_sha, "size_bytes": checkpoint.stat().st_size},
            {"path": f"replay/iter-{generation}-fresh.jsonl", "sha256": replay_sha, "size_bytes": replay.stat().st_size},
        ],
    )
    return CheckpointRef("torus9", lineage, f"M{generation}", generation, f"checkpoints/M{generation}.pt", cp_sha)


def _synthetic_graph(tmp_path: Path) -> tuple[Path, CheckpointRef, CheckpointRef, Path]:
    runs_root = tmp_path / "runs"
    profile_path = tmp_path / "profile.json"
    _write_json(profile_path, _profile())
    _make_lineage(runs_root, "lineage-a", 0, parent=None, profile_path=profile_path)
    parent = _make_lineage(runs_root, "lineage-a", 1, parent=("lineage-a", "M0", 0), profile_path=profile_path)
    child = _make_lineage(runs_root, "lineage-b", 2, parent=("lineage-a", "M1", 1), profile_path=profile_path)
    return runs_root, parent, child, profile_path


def test_scope_is_fixed_to_84_primary_nodes() -> None:
    assert TARGET_COUNT == 84
    assert sum(entry.count for entry in TARGET_SCOPE) == 84
    assert all(entry.first_generation >= 18 for entry in TARGET_SCOPE)
    assert all("replay80k" not in entry.lineage_id for entry in TARGET_SCOPE)


def test_backfill_materializes_cross_lineage_node_and_resolver_window(tmp_path: Path) -> None:
    runs_root, parent, child, _ = _synthetic_graph(tmp_path)
    report = run_backfill(
        runs_root,
        repo_root=tmp_path,
        targets=(ScopeEntry("lineage-a", 1, 1), ScopeEntry("lineage-b", 2, 2)),
        apply=True,
        migration_commit="test-commit",
    )
    assert report.ready == 2
    assert report.conflicts == 0
    assert report.writes == 6  # one config per lineage, plus provenance and node per checkpoint
    resolver = ArtifactResolver(runs_root)
    resolved = resolver.checkpoint(child)
    assert resolver.parent(resolved).ref == parent  # type: ignore[union-attr]
    assert [item.ref for item in resolver.ancestors(resolved, 1)] == [parent]
    assert len(resolver.replay_window(resolved, 2)) == 2


def test_missing_explicit_parent_fails_closed_without_guessing(tmp_path: Path) -> None:
    runs_root, _, _, profile_path = _synthetic_graph(tmp_path)
    metadata_path = runs_root / "torus9/active/lineage-b/checkpoints/M2.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("parent_checkpoint_identity")
    _write_json(metadata_path, metadata)
    report = run_backfill(
        runs_root,
        repo_root=tmp_path,
        targets=(ScopeEntry("lineage-b", 2, 2),),
        migration_commit="test-commit",
    )
    assert report.ready == 0
    assert any("parent_checkpoint_identity" in problem for problem in report.problems)
    assert not (runs_root / "torus9/active/lineage-b/metadata/checkpoints/M2.json").exists()


def test_apply_is_idempotent_and_conflicts_are_not_overwritten(tmp_path: Path) -> None:
    runs_root, _, _, _ = _synthetic_graph(tmp_path)
    target = (ScopeEntry("lineage-a", 1, 1),)
    first = run_backfill(runs_root, repo_root=tmp_path, targets=target, apply=True, migration_commit="test-commit")
    assert first.writes == 3
    second = run_backfill(runs_root, repo_root=tmp_path, targets=target, apply=True, migration_commit="test-commit")
    assert second.ready == 1
    assert second.writes == 0
    node_path = runs_root / "torus9/active/lineage-a/metadata/checkpoints/M1.json"
    original = node_path.read_bytes()
    payload = json.loads(node_path.read_text())
    payload["genesis"] = True
    _write_json(node_path, payload)
    conflict = run_backfill(runs_root, repo_root=tmp_path, targets=target, apply=True, migration_commit="test-commit")
    assert conflict.ready == 0
    assert conflict.conflicts == 1
    assert node_path.read_bytes() != original
    assert json.loads(node_path.read_text())["genesis"] is True
