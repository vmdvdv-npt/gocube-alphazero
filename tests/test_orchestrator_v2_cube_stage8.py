from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from training_engine import CheckpointContext, sequence_fingerprint, value_fingerprint
from tools.arena_engine import ArenaExecutionConfig

from gocube_golden.artifact_catalog import sha256_file
from gocube_golden.artifact_graph import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from gocube_golden.cube_training_contract_v2 import CubeTrainingConfig
from gocube_golden.cube_training_v2 import create_cube_m0_state
from gocube_golden.provenance import canonical_json
from gocube_golden.orchestrator_v2 import (
    ArenaRunRequest,
    ArenaRunner,
    ArtifactResolver,
    ContinuousTrainingConfig,
    ContinuousTrainingRunnerV2,
    CubeProductionGenerationPath,
    ProductionLineage,
    get_topology_binding,
)
from gocube_golden.orchestrator_v2.version import (
    ORCHESTRATOR_VERSION,
    ORCHESTRATOR_VERSION_ENV,
)
import gocube_golden.orchestrator_v2.arena_runner as arena_runner_module
import gocube_golden.orchestrator_v2.continuous_training as continuous_module


def _effective_config(size: int = 4) -> EffectiveConfig:
    topology = f"cube{size}"
    return EffectiveConfig(
        topology=topology,
        compatibility={"topology": topology, "family": "cube-v2"},
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
        extensions={"master_seed": 7000},
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


def _arena_config() -> ArenaExecutionConfig:
    return ArenaExecutionConfig(
        games=2,
        workers=1,
        games_per_worker=2,
        inference_batch_rows=2,
        inference_batch_wait_ms=0.0,
        device="cpu",
        strict_production=False,
        early_gate_enabled=False,
        min_effective_cpu_cores=0.0,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _publish_cube_m0(runs_root: Path, *, size: int = 4) -> CheckpointRef:
    topology = f"cube{size}"
    lineage_id = f"{topology}-stage8-m0"
    root = runs_root / topology / "active" / lineage_id
    for directory in (
        "checkpoints",
        "metadata/checkpoints",
        "metadata/effective-config-v2",
        "metadata/provenance-v2",
        "replay",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    adapter, state = create_cube_m0_state(
        size=size,
        config=_training_config(),
        seed=6100 + size,
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
        training_seed=6100 + size,
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
    checkpoint_sha = str(saved["checkpoint_sha256"])
    checkpoint = CheckpointRef(
        topology=topology,
        lineage_id=lineage_id,
        checkpoint_id="M0",
        generation=0,
        path="checkpoints/M0.pt",
        sha256=checkpoint_sha,
    )

    replay_path = root / "replay" / "rolling-after-00.jsonl"
    replay_path.write_text("", encoding="utf-8")

    config = _effective_config(size)
    config_path = (
        root
        / "metadata"
        / "effective-config-v2"
        / f"{config.fingerprint}.json"
    )
    config_path.write_text(config.canonical_json + "\n", encoding="utf-8")
    config_ref = EffectiveConfigRef(
        ArtifactRef(
            config_path.relative_to(root).as_posix(),
            sha256_file(config_path),
        ),
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
    provenance_ref = ArtifactRef(
        provenance_path.relative_to(root).as_posix(),
        sha256_file(provenance_path),
    )
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=True,
        parent=None,
        fresh_replay=None,
        effective_config=config_ref,
        provenance=provenance_ref,
    )
    _write_json(root / "metadata" / "checkpoints" / "M0.json", node.to_dict())
    _write_json(
        root / "manifest.json",
        {
            "schema": "gocube-orchestrator-v2-test-genesis-v1",
            "lineage_id": lineage_id,
            "topology": topology,
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


@pytest.mark.parametrize(
    ("topology", "expected_size"),
    [("cube2", 2), ("cube4", 4), ("cube7", 7)],
)
def test_topology_registry_routes_cube_family(topology: str, expected_size: int):
    binding = get_topology_binding(topology)
    assert binding.cube_size == expected_size
    path = binding.production_path()
    assert isinstance(path, CubeProductionGenerationPath)
    assert path.size == expected_size


def test_topology_registry_preserves_torus_and_fails_closed():
    assert get_topology_binding("torus9").topology == "torus9"
    with pytest.raises(ValueError, match="unsupported Orchestrator V2 topology"):
        get_topology_binding("cube8")


def test_cube_continuous_defaults_are_profile_and_startset_aware():
    config = ContinuousTrainingConfig(
        parent_checkpoint=CheckpointRef(
            "cube4",
            "parent",
            "M0",
            0,
            "checkpoints/M0.pt",
            "sha256:" + "1" * 64,
        ),
        lineage_id="cube4-stage8",
        effective_config=_effective_config(),
        generations=2,
        arena_cadence=1,
        arena_config=_arena_config(),
    )
    assert config.arena_profile.startswith("cube-v2|size=4|")
    assert config.arena_startset.id == "cube4-canonical-empty-paired-v2"
    assert config.arena_startset.fingerprint.startswith("sha256:")

    with pytest.raises(ValueError, match="not compatible"):
        ContinuousTrainingConfig(
            parent_checkpoint=config.parent_checkpoint,
            lineage_id="cube4-bad-profile",
            effective_config=_effective_config(),
            generations=1,
            arena_cadence=1,
            arena_config=_arena_config(),
            arena_profile="torus9",
        )


def test_arena_identity_comes_from_selected_cube_profile(tmp_path: Path):
    runs_root = tmp_path / "runs"
    parent_ref = _publish_cube_m0(runs_root, size=4)
    node = ArtifactResolver(runs_root).checkpoint(parent_ref)
    config = ContinuousTrainingConfig(
        parent_checkpoint=parent_ref,
        lineage_id="cube4-identity",
        effective_config=_effective_config(),
        generations=1,
        arena_cadence=1,
        arena_config=_arena_config(),
    )
    request = ArenaRunRequest(
        candidate=node,
        reference=node,
        master_seed=123,
        startset=config.arena_startset,
        config=config.arena_config,
        profile=config.arena_profile,
    )
    identity = ArenaRunner._identity(request)
    assert identity.scientific_contract["topology"] == "cube4"
    assert identity.scientific_contract["opening"] == "canonical-empty-board"
    assert identity.scientific_contract["model_gating"] is False


def test_common_lifecycle_has_no_cube_scientific_imports_or_torus_arena_default():
    lifecycle_source = inspect.getsource(continuous_module)
    for forbidden in (
        "cube_network",
        "cube_observation",
        "cube_selfplay",
        "cube_rules",
    ):
        assert forbidden not in lifecycle_source
    arena_source = inspect.getsource(arena_runner_module)
    assert "TORUS9_ARENA_PROFILE" not in arena_source
    assert "get_profile(request.profile)" in arena_source


def test_cube4_real_orchestrated_m0_to_m2_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ORCHESTRATOR_VERSION_ENV, ORCHESTRATOR_VERSION)
    runs_root = tmp_path / "runs"
    parent_ref = _publish_cube_m0(runs_root, size=4)
    resolver = ArtifactResolver(runs_root)
    parent = resolver.checkpoint(parent_ref)
    assert parent.generation == 0

    config = ContinuousTrainingConfig(
        parent_checkpoint=parent_ref,
        lineage_id="cube4-stage8-smoke",
        effective_config=_effective_config(),
        generations=2,
        arena_cadence=1,
        arena_config=_arena_config(),
        arena_master_seed=7300,
        arena_reference_gap=1,
    )
    runner = ContinuousTrainingRunnerV2(
        config,
        resolver=resolver,
        lineage_factory=ProductionLineage(runs_root),
    )
    result = runner.run()

    root = runs_root / "cube4" / "active" / "cube4-stage8-smoke"
    assert result.state == "COMPLETED"
    assert result.final_checkpoint.generation == 2
    assert [node.generation for node in result.committed_generations] == [1, 2]
    assert (root / "generation-01.complete.json").is_file()
    assert (root / "generation-02.complete.json").is_file()
    assert (root / "checkpoints" / "M1.pt").is_file()
    assert (root / "checkpoints" / "M2.pt").is_file()
    assert (root / "replay" / "rolling-after-02.jsonl").is_file()
    assert (root / "arena" / "generation-0002" / "result.json").is_file()
    assert len(result.arenas) == 2

    state = json.loads((root / "runtime" / "state.json").read_text(encoding="utf-8"))
    assert state["last_committed_generation"] == 2
    assert state["arena_generations"] == [1, 2]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lineage_id"] == "cube4-stage8-smoke"
    assert manifest["topology"] == "cube4"
    assert manifest["status"] == "ACTIVE"
    assert set(manifest["checkpoint_hashes"]) == {
        "checkpoints/M1.pt",
        "checkpoints/M2.pt",
    }

    assert not (tmp_path / "checkpoint").exists()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "arena-results").exists()
