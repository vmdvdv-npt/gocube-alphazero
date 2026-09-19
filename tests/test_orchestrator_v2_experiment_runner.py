from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from gocube_golden import run_storage
from gocube_golden.artifact_catalog import sha256_file
from gocube_golden.orchestrator_v2 import (
    ArenaRunner,
    ArmExecutionRequest,
    ArmExecutionResult,
    ArtifactRef,
    ArtifactResolver,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    ExperimentArmConfig,
    ExperimentConfig,
    ExperimentRunnerV2,
    checkpoint_node_path,
)
from tools.arena_engine import ArenaExecutionConfig


def _config(*, learning_rate: float, games: int, steps: int, sims: int) -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"topology": "torus9", "rules": "synthetic"},
        self_play={"games_per_iteration": games, "mcts_simulations": sims},
        training={"learning_rate": learning_rate, "optimizer_steps": steps},
        replay={"generations": 0, "cap": 16},
        execution={"workers": 1},
        arena={"games": 4},
    )


@dataclass(frozen=True)
class SyntheticParent:
    root: Path
    ref: CheckpointRef


def _make_parent(tmp_path: Path) -> SyntheticParent:
    runs_root = tmp_path / "runs"
    root = runs_root / "torus9" / "active" / "parent"
    config = _config(learning_rate=0.001, games=8, steps=4, sims=8)
    run_storage.ensure_lineage_layout(
        root,
        manifest={
            "lineage_id": "parent",
            "topology": "torus9",
            "status": "ACTIVE",
            "parent_checkpoint": None,
            "git_commit": "synthetic",
            "config_fingerprint": config.fingerprint,
            "created_at": "2026-09-19T00:00:00+00:00",
            "checkpoint_hashes": {},
        },
        extra_directories=("metadata",),
    )
    config_path = root / "metadata" / "config" / "effective.json"
    checkpoint_path = root / "checkpoints" / "M0.pt"
    provenance_path = root / "metadata" / "provenance" / "M0.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), sort_keys=True), encoding="utf-8")
    checkpoint_path.write_bytes(b"shared-parent-checkpoint")
    provenance_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    config_ref = EffectiveConfigRef(
        ArtifactRef("metadata/config/effective.json", sha256_file(config_path)),
        config.fingerprint,
    )
    ref = CheckpointRef(
        "torus9", "parent", "M0", 0, "checkpoints/M0.pt", sha256_file(checkpoint_path)
    )
    node = CheckpointNode(
        checkpoint=ref,
        genesis=True,
        parent=None,
        fresh_replay=None,
        effective_config=config_ref,
        provenance=ArtifactRef("metadata/provenance/M0.json", sha256_file(provenance_path)),
    )
    node_path = checkpoint_node_path(root, ref)
    node_path.parent.mkdir(parents=True, exist_ok=True)
    node_path.write_text(json.dumps(node.to_dict(), sort_keys=True), encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["checkpoint_hashes"] = {ref.path: ref.sha256}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return SyntheticParent(root, ref)


@dataclass
class FakeArmExecutionPath:
    """Synthetic replacement for the complete arm execution interface."""

    experiment_root: Path
    resolver: ArtifactResolver
    calls: list[ArmExecutionRequest] = field(default_factory=list)

    def run_arm(self, request: ArmExecutionRequest) -> ArmExecutionResult:
        self.calls.append(request)
        arm = request.arm
        lineage_id = arm.lineage_id or f"{request.experiment_id}-{arm.arm_id}"
        root = self.experiment_root / "runs" / request.topology / "active" / lineage_id
        config = arm.effective_config
        manifest = {
            "lineage_id": lineage_id,
            "topology": request.topology,
            "status": "ACTIVE",
            "parent_checkpoint": request.common_parent.ref.to_dict(),
            "git_commit": "synthetic-arm-path",
            "config_fingerprint": config.fingerprint,
            "created_at": "2026-09-19T00:00:00+00:00",
            "checkpoint_hashes": {},
        }
        run_storage.ensure_lineage_layout(
            root,
            manifest=manifest,
            extra_directories=("metadata", "replay"),
        )
        config_path = root / "metadata" / "config" / "effective.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config.to_dict(), sort_keys=True), encoding="utf-8")
        config_ref = EffectiveConfigRef(
            ArtifactRef("metadata/config/effective.json", sha256_file(config_path)),
            config.fingerprint,
        )

        current = request.common_parent
        for generation in range(
            request.common_parent.generation + 1,
            request.common_parent.generation + arm.generations + 1,
        ):
            checkpoint_path = root / "checkpoints" / f"M{generation}.pt"
            replay_path = root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
            provenance_path = root / "metadata" / "provenance" / f"M{generation}.json"
            marker_path = root / f"generation-{generation:02d}.complete.json"
            checkpoint_path.write_bytes(f"{lineage_id}:M{generation}".encode())
            replay_path.write_text(f"fresh-{lineage_id}-{generation}\n", encoding="utf-8")
            checkpoint = CheckpointRef(
                request.topology,
                lineage_id,
                f"M{generation}",
                generation,
                f"checkpoints/M{generation}.pt",
                sha256_file(checkpoint_path),
            )
            fresh_replay = ArtifactRef(
                replay_path.relative_to(root).as_posix(),
                sha256_file(replay_path),
            )
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance_path.write_text(
                json.dumps(
                    {
                        "synthetic": True,
                        "checkpoint": checkpoint.to_dict(),
                        "parent": current.ref.to_dict(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            provenance = ArtifactRef(
                provenance_path.relative_to(root).as_posix(),
                sha256_file(provenance_path),
            )
            node = CheckpointNode(
                checkpoint=checkpoint,
                genesis=False,
                parent=current.ref,
                fresh_replay=fresh_replay,
                effective_config=config_ref,
                provenance=provenance,
            )
            node_path = checkpoint_node_path(root, checkpoint)
            node_path.parent.mkdir(parents=True, exist_ok=True)
            node_path.write_text(json.dumps(node.to_dict(), sort_keys=True), encoding="utf-8")
            marker_path.write_text(
                json.dumps({"generation": generation, "lineage_id": lineage_id}),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hashes = dict(stored_manifest["checkpoint_hashes"])
            hashes[checkpoint.path] = checkpoint.sha256
            stored_manifest["checkpoint_hashes"] = hashes
            manifest_path.write_text(json.dumps(stored_manifest), encoding="utf-8")
            current = self.resolver.checkpoint(checkpoint)
        return ArmExecutionResult(final_checkpoint=current)


def test_experiment_runner_v2_synthetic_ab_resume_and_final_arena(tmp_path, monkeypatch):
    parent = _make_parent(tmp_path)
    resolver = ArtifactResolver(tmp_path / "runs")
    arm_path = FakeArmExecutionPath(tmp_path, resolver)
    arena_calls: list[dict[str, object]] = []

    def fake_arena(**kwargs: object) -> dict[str, object]:
        arena_calls.append(kwargs)
        output = kwargs["output_dir"]
        assert isinstance(output, Path)
        summary = {
            "games": 4,
            "W/L/D": [2, 2, 0],
            "telemetry": {
                "technical_games": 0,
                "performance_status": "HEALTHY",
                "performance_failures": [],
            },
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (output / "manifest.json").write_text("{}", encoding="utf-8")
        return summary

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda _topology, evaluation_id: tmp_path / "evaluations" / evaluation_id,
    )
    experiment_config = ExperimentConfig(
        experiment_id="synthetic-ab",
        topology="torus9",
        parent=parent.ref,
        arms=(
            ExperimentArmConfig("A", 2, _config(learning_rate=0.001, games=8, steps=4, sims=8)),
            ExperimentArmConfig("B", 2, _config(learning_rate=0.0005, games=12, steps=6, sims=16)),
        ),
        arena_config=ArenaExecutionConfig(
            games=4,
            workers=1,
            games_per_worker=2,
            inference_batch_rows=2,
            inference_batch_wait_ms=0.0,
            device="cpu",
            strict_production=False,
            min_mean_inference_batch_rows=0.0,
            min_effective_cpu_cores=0.0,
            early_gate_enabled=False,
        ),
        arena_master_seed=17,
    )
    assert ExperimentConfig.from_dict(experiment_config.to_dict()).fingerprint == experiment_config.fingerprint
    runner = ExperimentRunnerV2(
        experiment_config,
        resolver=resolver,
        arm_execution_path=arm_path,
        arena_runner=ArenaRunner(engine=fake_arena),
        experiment_root=tmp_path / "experiment-state",
    )

    first = runner.run()
    assert first.state == "STOPPED"
    assert len(arm_path.calls) == 2
    assert [call.arm.generations for call in arm_path.calls] == [2, 2]
    assert all(call.common_parent.ref == parent.ref for call in arm_path.calls)
    assert len(arena_calls) == 1

    a = first.final_checkpoints["A"]
    b = first.final_checkpoints["B"]
    assert a.generation == b.generation == 2
    assert a.lineage_id != b.lineage_id
    assert a.node.parent is not None and b.node.parent is not None
    assert a.node.parent != b.node.parent
    assert resolver.ancestor(a, 2).ref == resolver.ancestor(b, 2).ref == parent.ref
    assert resolver.ancestor(a, 2).lineage_id == "parent"
    assert not (tmp_path / "runs" / "torus9" / "active" / "synthetic-ab-A" / "checkpoints" / "M0.pt").exists()
    assert not (tmp_path / "runs" / "torus9" / "active" / "synthetic-ab-B" / "checkpoints" / "M0.pt").exists()
    assert arena_calls[0]["candidate_path"] == a.path
    assert arena_calls[0]["reference_path"] == b.path

    resumed = runner.run()
    assert resumed.state == "STOPPED"
    assert len(arm_path.calls) == 2
    assert len(arena_calls) == 1
    assert resumed.arena.identity.candidate == a.ref
    assert resumed.arena.identity.reference == b.ref
