from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from gocube_golden import run_storage
from gocube_golden.artifact_catalog import sha256_file
from gocube_golden.orchestrator_v2 import (
    ArenaRunner,
    ArtifactRef,
    ArtifactResolver,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    ExperimentArmConfig,
    ExperimentConfig,
    ExperimentRunnerV2,
    GenerationExecutionResult,
    GenerationRunner,
    SupervisorPolicy,
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
    provenance_path.write_text("{\"synthetic\":true}\n", encoding="utf-8")
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
class FakeGenerationPath:
    calls: list[object]

    def run_generation(self, resolved_input):
        self.calls.append(resolved_input)
        root = resolved_input.output_lineage.root
        generation = resolved_input.generation
        checkpoint = root / "checkpoints" / f"M{generation}.pt"
        replay = root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
        marker = root / f"generation-{generation:02d}.complete.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        replay.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(
            f"{resolved_input.output_lineage.lineage_id}:M{generation}".encode()
        )
        replay.write_text(f"fresh-{generation}\n", encoding="utf-8")
        marker.write_text(
            json.dumps(
                {"generation": generation, "lineage_id": resolved_input.output_lineage.lineage_id}
            ),
            encoding="utf-8",
        )
        return GenerationExecutionResult(
            generation=generation,
            committed=True,
            checkpoint=CheckpointRef(
                "torus9",
                resolved_input.output_lineage.lineage_id,
                f"M{generation}",
                generation,
                f"checkpoints/M{generation}.pt",
                sha256_file(checkpoint),
            ),
            commit_artifact=ArtifactRef(
                marker.relative_to(root).as_posix(),
                sha256_file(marker),
            ),
        )


def test_experiment_runner_v2_synthetic_ab_resume_and_final_arena(tmp_path, monkeypatch):
    parent = _make_parent(tmp_path)
    resolver = ArtifactResolver(tmp_path / "runs")
    generation_path = FakeGenerationPath([])
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
        generation_runner=GenerationRunner(generation_path),
        arena_runner=ArenaRunner(engine=fake_arena),
        supervisor_policy=SupervisorPolicy(max_retries=0, poll_interval_seconds=0.0),
        experiment_root=tmp_path / "experiment-state",
    )

    first = runner.run()
    assert first.state == "STOPPED"
    assert len(generation_path.calls) == 4
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
    assert resolver.ancestor(a, 2).ref == parent.ref
    assert resolver.ancestor(b, 2).ref == parent.ref
    assert arena_calls[0]["candidate_path"] == a.path
    assert arena_calls[0]["reference_path"] == b.path

    resumed = runner.run()
    assert resumed.state == "STOPPED"
    assert len(generation_path.calls) == 4
    assert len(arena_calls) == 1
    assert resumed.arena.identity.candidate == a.ref
    assert resumed.arena.identity.reference == b.ref
