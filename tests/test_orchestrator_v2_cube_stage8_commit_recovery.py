from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import training_engine as training_engine_module
from training_engine import CheckpointContext, sequence_fingerprint, value_fingerprint

from gocube_golden.artifact_catalog import ArtifactCatalog, sha256_file
from gocube_golden.artifact_graph import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    validate_generation_commit,
)
from gocube_golden.cube_training_contract_v2 import CubeTrainingConfig
from gocube_golden.cube_training_v2 import create_cube_m0_state
from gocube_golden.provenance import canonical_json
from gocube_golden.orchestrator_v2 import (
    ArtifactResolver,
    GenerationRunner,
    OutputLineage,
    ProductionLineage,
    ResolvedGenerationInput,
    get_topology_binding,
)


class _SimulatedProcessCrash(BaseException):
    pass


def _effective_config() -> EffectiveConfig:
    return EffectiveConfig(
        topology="cube4",
        compatibility={"topology": "cube4", "family": "cube-v2"},
        self_play={
            "games_per_iteration": 1,
            "mcts_simulations": 1,
            "cpuct": 1.0,
            "fpu": 0.0,
            "root_noise": False,
            "dirichlet_epsilon": 0.25,
            "dirichlet_alpha": 0.11,
            "temperature_plies": [1, 1],
            "temperature_after": 0.0,
            "resign": False,
            "technical_move_limit": 4,
            "komi": 0.5,
        },
        training={
            "learning_rate": 0.001,
            "batch_size": 1,
            "optimizer_steps": 1,
            "weight_decay": 0.0,
        },
        replay={"generations": 2, "cap": 16},
        execution={
            "workers": 1,
            "active_games_per_worker": 1,
            "total_active_contexts": 1,
            "inference_batch_cap": 1,
            "inference_batch_wait_ms": 0.0,
            "device": "cpu",
            "process_start_method": "fork",
        },
        arena={
            "simulations": 1,
            "cpuct": 1.0,
            "fpu": 0.0,
            "watchdog": 4,
            "reference_gap": 1,
        },
        extensions={"master_seed": 8100},
    )


def _training_config() -> CubeTrainingConfig:
    return CubeTrainingConfig(
        learning_rate=0.001,
        batch_size=1,
        optimizer_steps=1,
        replay_generations=2,
        replay_cap=16,
        weight_decay=0.0,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _publish_cube4_m0(runs_root: Path) -> CheckpointRef:
    lineage_id = "cube4-stage8-recovery-parent"
    root = runs_root / "cube4" / "active" / lineage_id
    for directory in (
        "checkpoints",
        "metadata/checkpoints",
        "metadata/effective-config-v2",
        "metadata/provenance-v2",
        "replay",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    adapter, state = create_cube_m0_state(
        size=4,
        config=_training_config(),
        seed=8104,
        device="cpu",
    )
    with torch.no_grad():
        for parameter in state.model.parameters():
            parameter.zero_()
        state.model.pass_head.bias.fill_(8.0)
    state.model.eval()

    context = CheckpointContext(
        run_id=lineage_id,
        label="M0",
        parent_label=None,
        generation=0,
        training_seed=8104,
        fresh_positions=0,
        replay_positions=0,
        replay_generations=(),
        replay_fingerprint=sequence_fingerprint(()),
        sampled_row_ids_fingerprint=value_fingerprint(()),
        completed_games=0,
        parent_checkpoint_identity=None,
        code_identity=None,
        device="cpu",
    )
    metadata = adapter.prepare_checkpoint(state, context, {})
    checkpoint_path = root / "checkpoints" / "M0.pt"
    saved = adapter.save_checkpoint(checkpoint_path, state, metadata)
    checkpoint = CheckpointRef(
        topology="cube4",
        lineage_id=lineage_id,
        checkpoint_id="M0",
        generation=0,
        path="checkpoints/M0.pt",
        sha256=str(saved["checkpoint_sha256"]),
    )

    replay_path = root / "replay" / "rolling-after-00.jsonl"
    replay_path.write_text("", encoding="utf-8")

    config = _effective_config()
    config_path = root / "metadata" / "effective-config-v2" / f"{config.fingerprint}.json"
    config_path.write_text(config.canonical_json + "\n", encoding="utf-8")
    config_ref = EffectiveConfigRef(
        ArtifactRef(config_path.relative_to(root).as_posix(), sha256_file(config_path)),
        config.fingerprint,
    )

    provenance_path = root / "metadata" / "provenance-v2" / "M0.json"
    _write_json(
        provenance_path,
        {
            "schema": "gocube-orchestrator-v2-genesis-provenance-v1",
            "checkpoint": checkpoint.to_dict(),
            "rolling_replay": {
                "path": replay_path.relative_to(root).as_posix(),
                "sha256": sha256_file(replay_path),
            },
        },
    )
    provenance = ArtifactRef(
        provenance_path.relative_to(root).as_posix(),
        sha256_file(provenance_path),
    )
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=True,
        parent=None,
        fresh_replay=None,
        effective_config=config_ref,
        provenance=provenance,
    )
    _write_json(root / "metadata" / "checkpoints" / "M0.json", node.to_dict())
    _write_json(
        root / "manifest.json",
        {
            "schema": "gocube-orchestrator-v2-test-genesis-v1",
            "lineage_id": lineage_id,
            "topology": "cube4",
            "status": "ACTIVE",
            "parent_checkpoint": None,
            "git_commit": "test-genesis",
            "config_fingerprint": config.fingerprint,
            "created_at": "test",
            "checkpoint_hashes": {checkpoint.path: checkpoint.sha256},
            "effective_config": config_ref.to_dict(),
        },
    )
    return checkpoint


def test_cube_commit_boundary_crash_is_reconciled_and_same_generation_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runs_root = tmp_path / "runs"
    parent_ref = _publish_cube4_m0(runs_root)
    resolver = ArtifactResolver(runs_root)
    parent = resolver.checkpoint(parent_ref)
    config = _effective_config()
    lineage_id = "cube4-stage8-commit-recovery"
    root, resolved_config = ProductionLineage(runs_root).prepare(
        topology="cube4",
        lineage_id=lineage_id,
        parent=parent,
        effective_config=config,
        experiment_id="stage8-commit-recovery",
        arm_id="fault-injection",
    )
    resolved = ResolvedGenerationInput(
        parent_checkpoint=parent,
        generation=1,
        effective_config=resolved_config,
        output_lineage=OutputLineage("cube4", lineage_id, root),
    )

    original_replace = training_engine_module.os.replace

    def crash_before_completion_fence(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if (
            source_path.name == ".generation-01.complete.tmp.json"
            and target_path.name == "generation-01.complete.json"
            and target_path.parent.resolve() == root.resolve()
        ):
            raise _SimulatedProcessCrash("crash after graph publication before marker rename")
        return original_replace(source, target)

    monkeypatch.setattr(training_engine_module.os, "replace", crash_before_completion_fence)
    with pytest.raises(_SimulatedProcessCrash, match="before marker rename"):
        GenerationRunner(get_topology_binding("cube4").production_path()).run(resolved)

    marker = root / "generation-01.complete.json"
    assert not marker.exists()
    assert (root / ".generation-01.complete.tmp.json").is_file()
    for path in (
        root / "checkpoints" / "M1.pt",
        root / "checkpoints" / "M1.metadata.json",
        root / "replay" / "iter-01-fresh.jsonl",
        root / "replay" / "rolling-after-01.jsonl",
        root / "training" / "iter-01.json",
        root / "iter-01-summary.json",
        root / "metadata" / "provenance-v2" / "M1.json",
        root / "metadata" / "checkpoints" / "M1.json",
    ):
        assert path.is_file(), f"fault injection did not reach graph publication: {path}"

    dirty_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "checkpoints/M1.pt" in dirty_manifest["checkpoint_hashes"]
    assert "1" in dirty_manifest["generation_commits"]
    dirty_catalog = ArtifactCatalog.load(
        root / "runtime" / "artifact-catalog.json",
        root=root,
    )
    assert "1" in dirty_catalog.payload["generations"]

    # A new child launch sees no authoritative marker, reconciles only M1's
    # pre-fence evidence, and can execute the same generation from M0 again.
    monkeypatch.setattr(training_engine_module.os, "replace", original_replace)
    result = GenerationRunner(get_topology_binding("cube4").production_path()).run(resolved)

    assert result.generation == 1
    assert marker.is_file()
    committed_node = validate_generation_commit(
        root=root,
        lineage_id=lineage_id,
        generation=1,
    )
    assert committed_node.checkpoint == result.checkpoint

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["checkpoint_hashes"]) == {"checkpoints/M1.pt"}
    assert set(manifest["generation_commits"]) == {"1"}
    catalog = ArtifactCatalog.load(root / "runtime" / "artifact-catalog.json", root=root)
    assert set(catalog.payload["generations"]) == {"1"}
    assert not (root / ".generation-01.complete.tmp.json").exists()

    # Once the marker exists it is authoritative: recovery is a no-op and the
    # generation cannot be executed/committed a second time.
    checkpoint_sha = sha256_file(root / "checkpoints" / "M1.pt")
    with pytest.raises(FileExistsError, match="already has published artifacts"):
        get_topology_binding("cube4").production_path().run_generation(resolved)
    assert sha256_file(root / "checkpoints" / "M1.pt") == checkpoint_sha
