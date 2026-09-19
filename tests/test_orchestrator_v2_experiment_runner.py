from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import pytest

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
    EXPERIMENT_STATE_SCHEMA,
    ExperimentArmConfig,
    ExperimentConfig,
    ExperimentRunnerV2,
    ExperimentRunnerError,
    ExperimentStage2Config,
    OutputLineage,
    ResolvedArtifact,
    ResolvedEffectiveConfig,
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
class SyntheticTrainOne:
    """Synthetic lineage factory plus strict one-generation callable."""

    experiment_root: Path
    resolver: ArtifactResolver
    calls: list[object] = field(default_factory=list)
    active_arm: ExperimentArmConfig | None = None
    active_parent: object | None = None

    def prepare(
        self,
        *,
        topology: str,
        lineage_id: str,
        parent,
        effective_config,
        experiment_id: str,
        arm_id: str,
    ) -> tuple[Path, ResolvedEffectiveConfig]:
        arm = ExperimentArmConfig(arm_id, 1, effective_config, lineage_id=lineage_id)
        self.active_arm = arm
        self.active_parent = parent
        root = self.resolver.runs_root / topology / "active" / lineage_id
        config = arm.effective_config
        manifest = {
            "lineage_id": lineage_id,
            "topology": topology,
            "status": "ACTIVE",
            "parent_checkpoint": parent.ref.to_dict(),
            "git_commit": "synthetic-train-one",
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
        resolved_artifact = ResolvedArtifact(
            ref=config_ref.artifact,
            path=config_path,
            owner_root=root,
            owner_topology=topology,
            owner_lineage_id=lineage_id,
            owner_status="ACTIVE",
            identity={"immutable_verified": True},
        )
        return root, ResolvedEffectiveConfig(config_ref, resolved_artifact, config)

    def __call__(self, *, parent, config, output_lineage: OutputLineage):
        generation = parent.generation + 1
        if not any(call.arm.arm_id == self.active_arm.arm_id for call in self.calls):
            self.calls.append(type("ArmCall", (), {"arm": self.active_arm, "common_parent": self.active_parent})())
        checkpoint_path = output_lineage.root / "checkpoints" / f"M{generation}.pt"
        replay_path = output_lineage.root / "replay" / f"iter-{generation:02d}-fresh.jsonl"
        provenance_path = output_lineage.root / "metadata" / "provenance" / f"M{generation}.json"
        marker_path = output_lineage.root / f"generation-{generation:02d}.complete.json"
        if marker_path.is_file():
            node = json.loads((output_lineage.root / "metadata" / "checkpoints" / f"M{generation}.json").read_text())
            return self.resolver.checkpoint(node["checkpoint"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(f"{output_lineage.lineage_id}:M{generation}".encode())
        replay_path.write_text(f"fresh-{output_lineage.lineage_id}-{generation}\n", encoding="utf-8")
        checkpoint = CheckpointRef(
            output_lineage.topology,
            output_lineage.lineage_id,
            f"M{generation}",
            generation,
            f"checkpoints/M{generation}.pt",
            sha256_file(checkpoint_path),
        )
        fresh_replay = ArtifactRef(replay_path.relative_to(output_lineage.root).as_posix(), sha256_file(replay_path))
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps({"synthetic": True, "checkpoint": checkpoint.to_dict(), "parent": parent.ref.to_dict()}, sort_keys=True),
            encoding="utf-8",
        )
        provenance = ArtifactRef(provenance_path.relative_to(output_lineage.root).as_posix(), sha256_file(provenance_path))
        node = CheckpointNode(
            checkpoint=checkpoint,
            genesis=False,
            parent=parent.ref,
            fresh_replay=fresh_replay,
            effective_config=config.ref,
            provenance=provenance,
        )
        node_path = checkpoint_node_path(output_lineage.root, checkpoint)
        node_path.parent.mkdir(parents=True, exist_ok=True)
        node_path.write_text(json.dumps(node.to_dict(), sort_keys=True), encoding="utf-8")
        marker_path.write_text(json.dumps({"generation": generation, "lineage_id": output_lineage.lineage_id}), encoding="utf-8")
        manifest_path = output_lineage.root / "manifest.json"
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = dict(stored_manifest["checkpoint_hashes"])
        hashes[checkpoint.path] = checkpoint.sha256
        stored_manifest["checkpoint_hashes"] = hashes
        manifest_path.write_text(json.dumps(stored_manifest), encoding="utf-8")
        return self.resolver.checkpoint(checkpoint)


def test_experiment_runner_v2_synthetic_ab_resume_and_final_arena(tmp_path, monkeypatch):
    parent = _make_parent(tmp_path)
    resolver = ArtifactResolver(tmp_path / "runs")
    arm_path = SyntheticTrainOne(tmp_path, resolver)
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
        (output / "manifest.json").write_text(
            json.dumps({"run_id": str(kwargs["run_id"])}), encoding="utf-8"
        )
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
        lineage_factory=arm_path,
        train_one=arm_path,
        arena_runner=ArenaRunner(engine=fake_arena),
        experiment_root=tmp_path / "experiment-state",
    )

    first = runner.run()
    assert first.state == "STOPPED"
    assert len(arm_path.calls) == 2
    assert [call.arm.arm_id for call in arm_path.calls] == ["A", "B"]
    assert all(call.common_parent.ref == parent.ref for call in arm_path.calls)
    assert len(arena_calls) == 1

    a = first.final_checkpoints["A"]
    b = first.final_checkpoints["B"]
    assert first.final_winner is not None and first.final_winner.ref == a.ref
    assert first.decisions[1].winner == a.ref  # tie remains with the reference
    assert a.generation == b.generation == 2
    assert a.lineage_id != b.lineage_id
    assert a.node.parent is not None and b.node.parent is not None
    assert a.node.parent != b.node.parent
    assert resolver.ancestor(a, 2).ref == resolver.ancestor(b, 2).ref == parent.ref
    assert resolver.ancestor(a, 2).lineage_id == "parent"
    assert not (tmp_path / "runs" / "torus9" / "active" / "synthetic-ab-A" / "checkpoints" / "M0.pt").exists()
    assert not (tmp_path / "runs" / "torus9" / "active" / "synthetic-ab-B" / "checkpoints" / "M0.pt").exists()
    assert arena_calls[0]["candidate_path"] == b.path
    assert arena_calls[0]["reference_path"] == a.path

    resumed = runner.run()
    assert resumed.state == "STOPPED"
    assert len(arm_path.calls) == 2
    assert len(arena_calls) == 1
    assert resumed.arena.identity.candidate == b.ref
    assert resumed.arena.identity.reference == a.ref


@pytest.mark.parametrize("stage1_candidate_wins", [True, False])
def test_experiment_runner_v2_synthetic_two_stage_winner_rooted_flow(
    tmp_path, monkeypatch, stage1_candidate_wins
):
    parent = _make_parent(tmp_path)
    resolver = ArtifactResolver(tmp_path / "runs")
    arm_path = SyntheticTrainOne(tmp_path, resolver)
    arena_calls: list[dict[str, object]] = []

    def fake_arena(**kwargs: object) -> dict[str, object]:
        arena_calls.append(kwargs)
        output = kwargs["output_dir"]
        assert isinstance(output, Path)
        comparison = str(kwargs["comparison"])
        summary = {
            "games": 4,
            "W/L/D": (
                [3, 1, 0]
                if not comparison.startswith("B-") or stage1_candidate_wins
                else [1, 3, 0]
            ),
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
    b_config = _config(learning_rate=0.0005, games=12, steps=6, sims=16)
    c_config = _config(learning_rate=0.0002, games=20, steps=9, sims=24)
    experiment_config = ExperimentConfig(
        experiment_id="synthetic-two-stage",
        topology="torus9",
        parent=parent.ref,
        arms=(
            ExperimentArmConfig("A", 2, _config(learning_rate=0.001, games=8, steps=4, sims=8)),
            ExperimentArmConfig("B", 1, b_config),
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
        stage2=ExperimentStage2Config(
            control_generations=1,
            c=ExperimentArmConfig("C", 2, c_config),
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
            arena_master_seed=18,
        ),
    )
    assert ExperimentConfig.from_dict(experiment_config.to_dict()).fingerprint == experiment_config.fingerprint
    runner = ExperimentRunnerV2(
        experiment_config,
        resolver=resolver,
        lineage_factory=arm_path,
        train_one=arm_path,
        arena_runner=ArenaRunner(engine=fake_arena),
        experiment_root=tmp_path / "experiment-state",
    )

    result = runner.run()

    assert result.state == "STOPPED"
    assert set(result.final_checkpoints) == {"A", "B", "control", "C"}
    a = result.final_checkpoints["A"]
    b = result.final_checkpoints["B"]
    control = result.final_checkpoints["control"]
    c = result.final_checkpoints["C"]
    assert a.generation == 2
    assert b.generation == 1
    stage1_winner = b if stage1_candidate_wins else a
    assert result.stage1_winner is not None and result.stage1_winner.ref == stage1_winner.ref
    assert result.stage2_parent is not None and result.stage2_parent.ref == stage1_winner.ref
    assert resolver.ancestor(control, 1).ref == stage1_winner.ref
    assert resolver.ancestor(c, 2).ref == stage1_winner.ref
    assert control.lineage_id != c.lineage_id
    assert {control.lineage_id, c.lineage_id}.isdisjoint({a.lineage_id, b.lineage_id})
    assert not (
        tmp_path / "runs" / "torus9" / "active" / control.lineage_id / "checkpoints" / b.checkpoint_id
    ).with_suffix(".pt").exists()
    assert not (
        tmp_path / "runs" / "torus9" / "active" / c.lineage_id / "checkpoints" / b.checkpoint_id
    ).with_suffix(".pt").exists()
    assert control.effective_config.fingerprint == stage1_winner.effective_config.fingerprint
    assert control.effective_config.config.training["learning_rate"] == (
        0.0005 if stage1_candidate_wins else 0.001
    )
    assert c.effective_config.fingerprint == c_config.fingerprint
    assert result.final_winner is not None and result.final_winner.ref == c.ref
    assert result.decisions[1].winner == stage1_winner.ref
    assert result.decisions[2].winner == c.ref
    assert result.decisions[1].to_dict()["W/L/D"] == ([3, 1, 0] if stage1_candidate_wins else [1, 3, 0])
    assert result.decisions[2].to_dict()["W/L/D"] == [3, 1, 0]
    assert [call.arm.arm_id for call in arm_path.calls] == ["A", "B", "control", "C"]
    assert arm_path.calls[2].common_parent.ref == stage1_winner.ref
    assert arm_path.calls[3].common_parent.ref == stage1_winner.ref
    assert len(arena_calls) == 2
    assert arena_calls[0]["candidate_path"] == b.path
    assert arena_calls[0]["reference_path"] == a.path
    assert arena_calls[1]["candidate_path"] == c.path
    assert arena_calls[1]["reference_path"] == control.path

    state = json.loads((tmp_path / "experiment-state" / "state.json").read_text())
    assert state["original_parent"] == parent.ref.to_dict()
    assert state["stage1"]["winner"]["winner_checkpoint"] == stage1_winner.ref.to_dict()
    assert state["stage2"]["parent"] == stage1_winner.ref.to_dict()
    assert state["stage2"]["winner"]["winner_checkpoint"] == c.ref.to_dict()
    assert state["final_winner"] == c.ref.to_dict()

    resumed = runner.run()
    assert resumed.final_winner is not None and resumed.final_winner.ref == c.ref
    assert len(arm_path.calls) == 4
    assert len(arena_calls) == 2


def test_experiment_runner_v2_resume_after_stage1_decision_reuses_all_stage1_work(tmp_path, monkeypatch):
    parent = _make_parent(tmp_path)
    resolver = ArtifactResolver(tmp_path / "runs")
    base_path = SyntheticTrainOne(tmp_path, resolver)
    failing = {"enabled": True}

    class FailCOnce:
        def prepare(self, **kwargs):
            return base_path.prepare(**kwargs)

        def __call__(self, *, parent, config, output_lineage):
            if base_path.active_arm is not None and base_path.active_arm.arm_id == "C" and failing["enabled"]:
                failing["enabled"] = False
                raise RuntimeError("synthetic interruption after Stage 1")
            return base_path(parent=parent, config=config, output_lineage=output_lineage)

    arena_calls: list[str] = []

    def fake_arena(**kwargs: object) -> dict[str, object]:
        arena_calls.append(str(kwargs["comparison"]))
        output = kwargs["output_dir"]
        assert isinstance(output, Path)
        summary = {
            "games": 4,
            "W/L/D": [3, 1, 0],
            "telemetry": {
                "technical_games": 0,
                "performance_status": "HEALTHY",
                "performance_failures": [],
            },
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (output / "manifest.json").write_text(
            json.dumps({"run_id": str(kwargs["run_id"])}), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda _topology, evaluation_id: tmp_path / "evaluations" / evaluation_id,
    )
    arena_config = ArenaExecutionConfig(
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
    )
    config = ExperimentConfig(
        experiment_id="synthetic-resume-stage1",
        topology="torus9",
        parent=parent.ref,
        arms=(
            ExperimentArmConfig("A", 1, _config(learning_rate=0.001, games=8, steps=4, sims=8)),
            ExperimentArmConfig("B", 1, _config(learning_rate=0.0005, games=12, steps=6, sims=16)),
        ),
        arena_config=arena_config,
        arena_master_seed=31,
        stage2=ExperimentStage2Config(
            control_generations=1,
            c=ExperimentArmConfig("C", 1, _config(learning_rate=0.0002, games=20, steps=9, sims=24)),
            arena_config=arena_config,
            arena_master_seed=32,
        ),
    )
    runner = ExperimentRunnerV2(
        config,
        resolver=resolver,
        lineage_factory=FailCOnce(),
        train_one=FailCOnce(),
        arena_runner=ArenaRunner(engine=fake_arena),
        experiment_root=tmp_path / "experiment-state",
    )

    try:
        runner.run()
    except RuntimeError as exc:
        assert str(exc) == "synthetic interruption after Stage 1"
    else:
        raise AssertionError("expected synthetic interruption")

    state = json.loads((tmp_path / "experiment-state" / "state.json").read_text())
    assert state["state"] == "STAGE2_RUNNING"
    assert state["stage1"]["winner"] is not None
    assert state["stage2"]["arms"]["control"]["final_checkpoint"] is not None
    assert state["stage2"]["arms"].get("C") is None
    assert arena_calls == ["B-final-vs-A-final"]

    resumed = runner.run()
    assert resumed.state == "STOPPED"
    assert len(base_path.calls) == 4
    assert [call.arm.arm_id for call in base_path.calls] == ["A", "B", "control", "C"]
    assert arena_calls == ["B-final-vs-A-final", "C-final-vs-control-final"]


def test_experiment_runner_v2_invalid_stage1_arena_never_creates_stage2_decision(tmp_path, monkeypatch):
    parent = _make_parent(tmp_path)
    resolver = ArtifactResolver(tmp_path / "runs")
    arm_path = SyntheticTrainOne(tmp_path, resolver)
    arena_calls = 0

    def invalid_arena(**kwargs: object) -> dict[str, object]:
        nonlocal arena_calls
        arena_calls += 1
        output = kwargs["output_dir"]
        assert isinstance(output, Path)
        summary = {
            "games": 4,
            "W/L/D": [4, 0, 0],
            "telemetry": {
                "technical_games": 1,
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
    arena_config = ArenaExecutionConfig(
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
    )
    config = ExperimentConfig(
        experiment_id="synthetic-invalid-stage1",
        topology="torus9",
        parent=parent.ref,
        arms=(
            ExperimentArmConfig("A", 1, _config(learning_rate=0.001, games=8, steps=4, sims=8)),
            ExperimentArmConfig("B", 1, _config(learning_rate=0.0005, games=12, steps=6, sims=16)),
        ),
        arena_config=arena_config,
        arena_master_seed=41,
        stage2=ExperimentStage2Config(
            control_generations=1,
            c=ExperimentArmConfig("C", 1, _config(learning_rate=0.0002, games=20, steps=9, sims=24)),
            arena_config=arena_config,
            arena_master_seed=42,
        ),
    )
    runner = ExperimentRunnerV2(
        config,
        resolver=resolver,
        lineage_factory=arm_path,
        train_one=arm_path,
        arena_runner=ArenaRunner(engine=invalid_arena),
        experiment_root=tmp_path / "experiment-state",
    )

    with pytest.raises(ExperimentRunnerError, match="no scientific winner"):
        runner.run()
    state = json.loads((tmp_path / "experiment-state" / "state.json").read_text())
    assert state["state"] == "ARENA_INVALID"
    assert state["stage1"]["winner"] is None
    assert state["stage2"]["arms"] == {}
    assert arena_calls == 1

    with pytest.raises(ExperimentRunnerError, match="no winner or Stage 2"):
        runner.run()
    assert len(arm_path.calls) == 2
    assert arena_calls == 1


def test_experiment_runner_v2_migrates_legacy_v2_stopped_state_without_reexecution(tmp_path, monkeypatch):
    parent = _make_parent(tmp_path)
    resolver = ArtifactResolver(tmp_path / "runs")
    arm_path = SyntheticTrainOne(tmp_path, resolver)
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
        (output / "manifest.json").write_text(
            json.dumps({"run_id": str(kwargs["run_id"])}), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda _topology, evaluation_id: tmp_path / "evaluations" / evaluation_id,
    )
    arena_config = ArenaExecutionConfig(
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
    )
    config = ExperimentConfig(
        experiment_id="synthetic-legacy-v2",
        topology="torus9",
        parent=parent.ref,
        arms=(
            ExperimentArmConfig("A", 1, _config(learning_rate=0.001, games=8, steps=4, sims=8)),
            ExperimentArmConfig("B", 1, _config(learning_rate=0.0005, games=12, steps=6, sims=16)),
        ),
        arena_config=arena_config,
        arena_master_seed=51,
    )
    runner = ExperimentRunnerV2(
        config,
        resolver=resolver,
        lineage_factory=arm_path,
        train_one=arm_path,
        arena_runner=ArenaRunner(engine=fake_arena),
        experiment_root=tmp_path / "experiment-state",
    )

    initial = runner.run()
    current_state = json.loads((tmp_path / "experiment-state" / "state.json").read_text())
    legacy_state = {
        "schema": EXPERIMENT_STATE_SCHEMA,
        "version": 2,
        "experiment_id": config.experiment_id,
        "topology": config.topology,
        "config_fingerprint": config.legacy_fingerprint,
        "parent": parent.ref.to_dict(),
        "state": "STOPPED",
        "arms": current_state["stage1"]["arms"],
        "arena_result": current_state["stage1"]["arena"],
        "stop_reason": "final A-vs-B Arena completed",
        "created_at": current_state["created_at"],
        "updated_at": current_state["updated_at"],
    }
    (tmp_path / "experiment-state" / "state.json").write_text(
        json.dumps(legacy_state), encoding="utf-8"
    )
    assert config.legacy_fingerprint != config.fingerprint
    arm_path.calls.clear()
    arena_calls.clear()

    resumed = runner.run()

    assert resumed.state == "STOPPED"
    assert resumed.final_winner is not None
    assert resumed.final_winner.ref == initial.final_checkpoints["A"].ref
    assert resumed.decisions[1].winner == initial.final_checkpoints["A"].ref
    assert arm_path.calls == []
    assert arena_calls == []
    migrated = json.loads((tmp_path / "experiment-state" / "state.json").read_text())
    assert migrated["version"] == 3
    assert migrated["config_fingerprint"] == config.fingerprint
    assert migrated["legacy_migration"]["from_version"] == 2
    assert migrated["stage1"]["winner"]["winner_checkpoint"] == initial.final_checkpoints["A"].ref.to_dict()
