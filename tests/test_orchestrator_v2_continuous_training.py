from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from gocube_golden.artifact_graph import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from gocube_golden.artifact_resolver import ResolvedArtifact, ResolvedCheckpointNode, ResolvedEffectiveConfig
from gocube_golden.orchestrator_v2 import (
    ArenaRunResult,
    ArenaRunner,
    ContinuousTrainingRunnerV2,
)
import gocube_golden.orchestrator_v2.continuous_training as continuous_training
from gocube_golden.provenance import sha256_fingerprint
from tools.arena_engine import ArenaExecutionConfig


SHA = "sha256:" + "a" * 64


def _config(*, learning_rate: float = 0.0003, sims: int = 128, games: int = 128) -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"topology": "torus9", "rules": "positional-superko"},
        self_play={"mcts_simulations": sims, "games_per_iteration": games},
        training={"learning_rate": learning_rate, "optimizer_steps": 160},
        replay={"window": 6, "cap": 40000},
        arena={"reference_gap": 1},
    )


def _arena_config() -> ArenaExecutionConfig:
    return ArenaExecutionConfig(
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


class FakeResolver:
    def __init__(self, runs_root: Path, parent: ResolvedCheckpointNode) -> None:
        self.runs_root = runs_root
        self.nodes = {parent.ref: parent}

    def add(self, node: ResolvedCheckpointNode) -> None:
        self.nodes[node.ref] = node

    def checkpoint(self, value):
        ref = value if isinstance(value, CheckpointRef) else CheckpointRef.from_dict(value)
        return self.nodes[ref]

    def ancestor(self, node: ResolvedCheckpointNode, count: int) -> ResolvedCheckpointNode:
        current = node
        for _ in range(count):
            current = self.nodes[current.node.parent]
        return current


class FakeLineageFactory:
    def __init__(self, runs_root: Path, config: EffectiveConfig) -> None:
        self.runs_root = runs_root
        self.config = config
        self.calls: list[dict[str, object]] = []

    def prepare(self, **kwargs):
        self.calls.append(kwargs)
        root = self.runs_root / "torus9" / "active" / str(kwargs["lineage_id"])
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "lineage_id": kwargs["lineage_id"],
                    "topology": "torus9",
                    "status": "ACTIVE",
                    "parent_checkpoint": kwargs["parent"].ref.to_dict(),
                    "git_commit": "test",
                    "config_fingerprint": self.config.fingerprint,
                    "created_at": "now",
                    "checkpoint_hashes": {},
                }
            ),
            encoding="utf-8",
        )
        config_ref = EffectiveConfigRef(
            ArtifactRef("metadata/effective.json", SHA),
            self.config.fingerprint,
        )
        resolved = ResolvedEffectiveConfig(
            config_ref,
            ResolvedArtifact(
                config_ref.artifact,
                root / "metadata/effective.json",
                root,
                "torus9",
                str(kwargs["lineage_id"]),
                "ACTIVE",
                {"immutable_verified": True},
            ),
            self.config,
        )
        return root, resolved


def _node(
    root: Path,
    config: EffectiveConfig,
    lineage: str,
    generation: int,
    parent: CheckpointRef | None,
) -> ResolvedCheckpointNode:
    checkpoint = CheckpointRef(
        "torus9",
        lineage,
        f"M{generation}",
        generation,
        f"checkpoints/M{generation}.pt",
        SHA,
    )
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=parent is None,
        parent=parent,
        fresh_replay=None if parent is None else ArtifactRef("replay/fresh.jsonl", SHA),
        effective_config=EffectiveConfigRef(
            ArtifactRef("metadata/effective.json", SHA), config.fingerprint
        ),
        provenance=ArtifactRef("metadata/provenance.json", SHA),
    )
    physical = SimpleNamespace(path=root / lineage / checkpoint.path)
    return ResolvedCheckpointNode(
        node=node,
        checkpoint=physical,  # type: ignore[arg-type]
        effective_config=SimpleNamespace(ref=node.effective_config),  # type: ignore[arg-type]
        provenance=SimpleNamespace(),  # type: ignore[arg-type]
        owner_root=root / lineage,
        owner_status="ACTIVE",
    )


class FakeTrainOne:
    def __init__(self, resolver: FakeResolver, root: Path, config: EffectiveConfig) -> None:
        self.resolver = resolver
        self.root = root
        self.config = config
        self.parents: list[CheckpointRef] = []
        self.calls: list[int] = []
        self.stop_runner: ContinuousTrainingRunnerV2 | None = None
        self.stop_at: int | None = None
        self.stop_once = True

    def __call__(self, *, parent, config, output_lineage):
        generation = parent.generation + 1
        self.parents.append(parent.ref)
        self.calls.append(generation)
        child = _node(self.root, self.config, output_lineage.lineage_id, generation, parent.ref)
        self.resolver.add(child)
        if self.stop_runner is not None and self.stop_at == generation and self.stop_once:
            self.stop_runner.request_soft_stop(reason="test")
            self.stop_once = False
        return child


class FakeArena:
    def __init__(self) -> None:
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        identity = ArenaRunner._identity(request)
        return ArenaRunResult(
            evaluation_id=f"fake-{request.candidate.generation}",
            evaluation_fingerprint=sha256_fingerprint(identity.to_dict()),
            output_dir=request.output_dir or Path("/tmp/canonical-evaluation"),
            identity=identity,
            summary={"games": 4, "W/L/D": [3, 1, 0]},
            validity="VALID",
        )


def _runner(
    tmp_path: Path,
    *,
    generations: int | None,
    cadence: int = 5,
    effective_config: EffectiveConfig | None = None,
):
    config = effective_config or _config()
    parent_root = tmp_path / "parent"
    parent = _node(parent_root, config, "parent-lineage", 20, None)
    resolver = FakeResolver(tmp_path / "runs", parent)
    lineage = FakeLineageFactory(resolver.runs_root, config)
    train = FakeTrainOne(resolver, resolver.runs_root, config)
    arena = FakeArena()
    runner = ContinuousTrainingRunnerV2(
        parent_checkpoint=parent.ref,
        lineage_id="continuous-lineage",
        effective_config=config,
        generations=generations,
        arena_cadence=cadence,
        arena_config=_arena_config(),
        resolver=resolver,  # type: ignore[arg-type]
        lineage_factory=lineage,  # type: ignore[arg-type]
        train_one=train,  # type: ignore[arg-type]
        arena_runner=arena,  # type: ignore[arg-type]
        reporter=lambda *_args: None,
    )
    train.stop_runner = runner
    return runner, train, arena, resolver, parent


def test_continuous_runner_trains_sequentially_and_chains_parent(tmp_path: Path) -> None:
    runner, train, _arena, _resolver, parent = _runner(tmp_path, generations=3)

    result = runner.run()

    assert train.calls == [21, 22, 23]
    assert [ref.generation for ref in train.parents] == [20, 21, 22]
    assert result.final_checkpoint.generation == 23
    assert result.state == "COMPLETED"


def test_resume_reuses_committed_generations_and_soft_stop_is_safe(tmp_path: Path) -> None:
    runner, train, _arena, _resolver, _parent = _runner(tmp_path, generations=None)
    train.stop_at = 22

    first = runner.run()
    assert first.state == "SOFT_STOPPED"
    assert train.calls == [21, 22]
    assert json.loads(runner.state_path.read_text())['soft_stop_requested'] is True

    train.stop_at = 23
    train.stop_once = True
    second = runner.run()
    assert second.state == "SOFT_STOPPED"
    assert train.calls == [21, 22, 23]
    assert second.final_checkpoint.generation == 23


def test_soft_stop_finishes_current_generation_without_starting_next_or_arena(
    tmp_path: Path,
) -> None:
    runner, train, arena, _resolver, _parent = _runner(tmp_path, generations=None, cadence=1)
    train.stop_at = 21

    result = runner.run()

    assert result.state == "SOFT_STOPPED"
    assert train.calls == [21]
    assert arena.requests == []
    assert json.loads(runner.state_path.read_text())["state"] == "SOFT_STOPPED"


def test_arena_runs_exactly_on_relative_cadence_and_is_lineage_owned(tmp_path: Path) -> None:
    runner, train, arena, _resolver, _parent = _runner(tmp_path, generations=5, cadence=2)

    result = runner.run()

    assert result.state == "COMPLETED"
    assert [request.candidate.generation for request in arena.requests] == [22, 24]
    assert all(request.candidate.lineage_id == request.reference.lineage_id for request in arena.requests)
    assert all(request.output_dir == runner.lineage_root / "arena" / f"generation-{request.candidate.generation:04d}" for request in arena.requests)
    assert (runner.lineage_root / "arena" / "generation-0022" / "result.json").is_file()
    assert not (runner.lineage_root / "arena" / "generation-0022" / "checkpoints").exists()


def test_resume_preserves_arena_cadence_without_rerunning_completed_arena(tmp_path: Path) -> None:
    runner, train, arena, _resolver, _parent = _runner(tmp_path, generations=None, cadence=2)
    train.stop_at = 21
    runner.run()
    assert [request.candidate.generation for request in arena.requests] == []

    train.stop_at = 25
    train.stop_once = True
    runner.run()
    assert [request.candidate.generation for request in arena.requests] == [22, 24]


def test_parent_checkpoint_is_referenced_but_not_copied(tmp_path: Path) -> None:
    runner, _train, _arena, _resolver, parent = _runner(tmp_path, generations=1)

    runner.run()

    assert not (runner.lineage_root / parent.ref.path).exists()
    manifest = json.loads((runner.lineage_root / "manifest.json").read_text())
    assert manifest["parent_checkpoint"] == parent.ref.to_dict()
    state = json.loads(runner.state_path.read_text())
    assert state["parent_checkpoint"] == parent.ref.to_dict()


def test_cross_lineage_arena_uses_canonical_evaluation_scope(tmp_path: Path) -> None:
    base = _config()
    config = EffectiveConfig(
        topology=base.topology,
        compatibility=base.compatibility,
        self_play=base.self_play,
        training=base.training,
        replay=base.replay,
        arena={"reference_gap": 5},
    )
    runner, _train, arena, _resolver, _parent = _runner(
        tmp_path, generations=1, cadence=1, effective_config=config
    )

    runner.run()

    assert len(arena.requests) == 1
    assert arena.requests[0].candidate.lineage_id != arena.requests[0].reference.lineage_id
    assert arena.requests[0].output_dir is None


def test_start_report_contains_resolved_operator_values(tmp_path: Path) -> None:
    runner, _train, _arena, _resolver, _parent = _runner(tmp_path, generations=0)
    messages: list[str] = []
    runner.reporter = lambda message, *_details: messages.append(str(message))

    runner.run()

    assert messages and "LR=0.0003" in messages[0]
    assert "replay=6 generations / 40000 positions" in messages[0]
    assert "self-play MCTS=128 sims" in messages[0]
    assert "games/generation=128" in messages[0]
    assert "Arena cadence=every 5 generations" in messages[0]


def test_non_golden_operator_values_are_reported_not_rejected(tmp_path: Path) -> None:
    config = _config(learning_rate=0.0002, sims=256, games=90)
    runner, _train, _arena, _resolver, _parent = _runner(
        tmp_path, generations=0, effective_config=config
    )

    result = runner.run()

    assert result.state == "COMPLETED"
    manifest = json.loads((runner.lineage_root / "manifest.json").read_text())
    assert manifest["operator_tunables"]["learning_rate"] == 0.0002
    assert manifest["operator_tunables"]["self_play_mcts_simulations"] == 256
    assert manifest["operator_tunables"]["games_per_generation"] == 90


def test_runner_is_a_coordinator_and_does_not_import_training_engine(tmp_path: Path) -> None:
    del tmp_path
    source = inspect.getsource(continuous_training)
    assert "ProductionTrainOne" in source
    assert "ArenaRunnerV2" in source
    assert "torus9_run_driver" not in source
    assert "torch" not in source
    assert "run_generation_v2" not in source
