from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from gocube_golden.artifact_graph import ArtifactRef, CheckpointNode, CheckpointRef, EffectiveConfig, EffectiveConfigRef
from gocube_golden.orchestrator_v2.contracts import EvaluationIdentity
from gocube_golden.orchestrator_v2.komi_calibration import (
    CALIBRATION_EXTENSION,
    KomiCalibrationArenaContract,
    KomiCalibrationConfig,
    KomiCalibrationError,
    KomiCalibrationRunnerV2,
    _summary_stats,
    effective_config_with_komi,
)
from gocube_golden.orchestrator_v2.arena_runner import ArenaRunRequest, ArenaRunResult
from gocube_golden.artifact_resolver import ResolvedCheckpointNode
from gocube_golden.torus9 import (
    build_torus9_observation,
    generate_torus9_evaluation_starts,
    torus9_state_from_identity,
)
from tools.arena_profiles import get_profile


SHA = "sha256:" + "a" * 64


def _config() -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"topology": "torus9", "rules": {"komi": 0.5}},
        self_play={"games_per_iteration": 384, "mcts_simulations": 200, "komi": 0.5},
        training={"learning_rate": 0.0001, "optimizer_steps": 160},
        replay={"generations": 6, "cap": None},
        execution={"workers": 16},
        arena={"games": 64, "komi": 0.5},
    )


def test_calibration_contract_is_exact_and_config_changes_only_komi() -> None:
    contract = KomiCalibrationArenaContract()
    execution = contract.execution_config(1024)
    assert execution.workers == 16
    assert execution.games_per_worker == 12
    assert execution.inference_batch_rows == 64
    assert execution.inference_batch_wait_ms == 4.0
    assert execution.strict_production is True

    selected = effective_config_with_komi(_config(), 2.5)
    before = _config().to_dict()
    after = selected.to_dict()
    assert after["self_play"]["komi"] == 2.5
    assert after["arena"]["komi"] == 2.5
    assert after["training"] == before["training"]
    assert after["replay"] == before["replay"]
    assert after["self_play"]["games_per_iteration"] == before["self_play"]["games_per_iteration"]


def test_calibration_profile_freezes_same_seed_family_with_selected_komi() -> None:
    starts = generate_torus9_evaluation_starts(
        master_seed=1234,
        accepted_per_stratum=1,
        komi=1.5,
    )
    state = torus9_state_from_identity(starts[0]["state"], expected_komi=1.5)
    observation = build_torus9_observation(state)
    assert state.komi == 1.5
    assert float(observation[5, 0]) == 1.5
    assert starts[0]["state"]["komi"] == 1.5

    profile = get_profile("torus9-komi-calibration|2.5")
    assert profile.profile_id == "torus9-komi-calibration|2.5"
    assert profile.komi == 2.5
    assert profile.scientific_contract(SimpleNamespace(games=1024)) ["komi"] == 2.5


def test_summary_stats_fails_closed_on_technical_games(tmp_path: Path) -> None:
    result = SimpleNamespace(
        summary={
            "games": 4,
            "valid_games": 3,
            "technical_games": 1,
            "black_wins": 2,
            "white_wins": 1,
            "draws": 0,
        },
        identity=SimpleNamespace(games=4),
        output_dir=tmp_path,
    )
    with pytest.raises(KomiCalibrationError, match="technical/invalid"):
        _summary_stats(result)  # type: ignore[arg-type]


def test_ambiguous_bias_extends_both_candidates_and_ties_choose_1_5() -> None:
    config = KomiCalibrationConfig(
        calibration_id="calibration-test",
        parent_checkpoint={
            "topology": "torus9",
            "lineage_id": "parent",
            "checkpoint_id": "M137",
            "generation": 137,
            "path": "checkpoints/M137.pt",
            "sha256": "sha256:" + "1" * 64,
        },
    )
    runner = object.__new__(KomiCalibrationRunnerV2)
    runner.config = config
    transitions: list[str] = []

    def transition(state, stage, **_kwargs):
        transitions.append(stage)
        state["state"] = stage

    def extend(state, _parent, komi, _stage, *, batch):
        assert batch == 2
        state["candidates"][f"{komi:g}"]["batch"] = 2
        state["candidates"][f"{komi:g}"]["stats"]["bias"] = 0.02

    runner._transition = transition
    runner._run_candidate = extend
    state = {
        "state": "CALIBRATION_KOMI_2_5",
        "candidates": {
            "1.5": {"batch": 1, "stats": {"bias": 0.005}},
            "2.5": {"batch": 1, "stats": {"bias": 0.004}},
        },
    }
    selected = runner._select_or_extend(state, object())
    assert selected == 1.5
    assert transitions == [CALIBRATION_EXTENSION]
    assert state["candidates"]["1.5"]["batch"] == 2
    assert state["candidates"]["2.5"]["batch"] == 2


def _synthetic_node(
    root: Path,
    *,
    lineage_id: str,
    generation: int,
    checkpoint_id: str,
    config: EffectiveConfig,
    parent: CheckpointRef | None,
) -> ResolvedCheckpointNode:
    checkpoint = CheckpointRef(
        "torus9",
        lineage_id,
        checkpoint_id,
        generation,
        f"checkpoints/{checkpoint_id}.pt",
        SHA,
    )
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=parent is None,
        parent=parent,
        fresh_replay=(
            None
            if parent is None
            else ArtifactRef(f"replay/fresh-M{generation}.jsonl", SHA)
        ),
        effective_config=EffectiveConfigRef(
            ArtifactRef("metadata/effective-config.json", config.fingerprint),
            config.fingerprint,
        ),
        provenance=ArtifactRef("metadata/provenance.json", SHA),
    )
    checkpoint_path = root / checkpoint.path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"synthetic-checkpoint")
    return ResolvedCheckpointNode(
        node=node,
        checkpoint=SimpleNamespace(path=checkpoint_path),  # type: ignore[arg-type]
        effective_config=SimpleNamespace(config=config),  # type: ignore[arg-type]
        provenance=SimpleNamespace(),  # type: ignore[arg-type]
        owner_root=root,
        owner_status="ACTIVE",
    )


class _SyntheticResolver:
    def __init__(self, runs_root: Path, parent: ResolvedCheckpointNode, child: ResolvedCheckpointNode):
        self.runs_root = runs_root
        self.parent = parent
        self.child = child

    def checkpoint(self, ref):
        checkpoint = ref if isinstance(ref, CheckpointRef) else CheckpointRef.from_dict(ref)
        if checkpoint.checkpoint_id == "M137":
            return self.parent
        if checkpoint.checkpoint_id == "M138":
            return self.child
        raise RuntimeError("synthetic resolver only contains M137/M138")

    def replay_window(self, _parent, count):
        return tuple(
            SimpleNamespace(
                path=self.runs_root / "replay" / f"fresh-M{generation}.jsonl",
                ref=ArtifactRef(f"replay/fresh-M{generation}.jsonl", SHA),
            )
            for generation in range(137 - count + 1, 138)
        )


class _SyntheticLineageFactory:
    def __init__(self, runs_root: Path, config: EffectiveConfig):
        self.runs_root = runs_root
        self.config = config
        self.calls = 0

    def prepare(self, *, topology, lineage_id, parent, effective_config, **_kwargs):
        self.calls += 1
        root = self.runs_root / topology / "active" / lineage_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "lineage_id": lineage_id,
                    "topology": topology,
                    "status": "ACTIVE",
                    "parent_checkpoint": parent.ref.to_dict(),
                    "checkpoint_hashes": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return root, SimpleNamespace(
            config=effective_config,
            artifact=SimpleNamespace(owner_lineage_id=lineage_id),
        )


class _SyntheticArena:
    def __init__(self, output_root: Path, *, ambiguous: bool = False):
        self.output_root = output_root
        self.ambiguous = ambiguous
        self.requests: list[ArenaRunRequest] = []

    def run(self, request: ArenaRunRequest) -> ArenaRunResult:
        self.requests.append(request)
        komi = float(request.workload["komi"])
        batch = int(request.workload["continuation_batch"])
        black_rate = 0.505 if self.ambiguous and batch == 1 else 0.502
        if komi == 2.5 and batch == 1:
            black_rate = 0.504 if self.ambiguous else 0.60
        if komi == 2.5 and batch == 2:
            black_rate = 0.498
        games = int(request.config.games)
        black_wins = int(round(games * black_rate))
        identity = EvaluationIdentity(
            candidate=request.candidate.ref,
            reference=request.reference.ref,
            games=games,
            master_seed=request.master_seed,
            startset=request.startset,
            scientific_contract=request.scientific_contract or {},
            execution_contract=request.execution_contract or {},
            workload=request.workload,
        )
        output = self.output_root / f"arena-{komi:g}-{batch}"
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "games": games,
            "valid_games": games,
            "technical_games": 0,
            "invalid_games": 0,
            "black_wins": black_wins,
            "white_wins": games - black_wins,
            "draws": 0,
            "W/L/D": [black_wins, games - black_wins, 0],
        }
        return ArenaRunResult(
            evaluation_id=f"synthetic-{komi:g}-{batch}",
            evaluation_fingerprint=identity.fingerprint,
            output_dir=output,
            identity=identity,
            summary=summary,
            validity="VALID",
        )


def _runner_fixture(tmp_path: Path, *, ambiguous: bool = False):
    runs_root = tmp_path / "runs"
    parent_root = runs_root / "torus9" / "active" / "parent"
    base = _config()
    m136 = CheckpointRef("torus9", "parent", "M136", 136, "checkpoints/M136.pt", SHA)
    parent = _synthetic_node(
        parent_root,
        lineage_id="parent",
        generation=137,
        checkpoint_id="M137",
        config=base,
        parent=m136,
    )
    (parent_root / "generation-137.complete.json").write_text(
        json.dumps({"generation": 137, "lineage_id": "parent", "checkpoint_sha256": SHA}) + "\n",
        encoding="utf-8",
    )
    (parent_root / "manifest.json").write_text(
        json.dumps({"lineage_id": "parent", "checkpoint_hashes": {parent.ref.path: SHA}}) + "\n",
        encoding="utf-8",
    )
    child_root = runs_root / "torus9" / "active" / "selected"
    child = _synthetic_node(
        child_root,
        lineage_id="selected",
        generation=138,
        checkpoint_id="M138",
        config=effective_config_with_komi(base, 1.5),
        parent=parent.ref,
    )
    resolver = _SyntheticResolver(runs_root, parent, child)
    lineage = _SyntheticLineageFactory(runs_root, base)
    arena = _SyntheticArena(tmp_path / "arena", ambiguous=ambiguous)
    stops: list[str] = []
    config = KomiCalibrationConfig(
        calibration_id="synthetic-calibration",
        parent_checkpoint=parent.ref,
        production_effective_config=base,
        child_lineage_id="selected",
        wait_poll_seconds=0,
    )
    return config, resolver, lineage, arena, parent, child, stops


def test_m137_validation_accepts_standard_marker_without_lineage_id(tmp_path: Path) -> None:
    _config_value, _resolver, _lineage, _arena, parent, _child, _stops = _runner_fixture(tmp_path)
    marker_path = parent.owner_root / "generation-137.complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop("lineage_id")
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")

    KomiCalibrationRunnerV2._validate_m137(parent)


def test_runner_pins_m137_creates_one_child_and_resume_does_not_rerun(tmp_path: Path) -> None:
    config, resolver, lineage, arena, parent, child, stops = _runner_fixture(tmp_path)
    root = tmp_path / "runs" / "torus9" / "evaluations" / config.calibration_id
    runner = KomiCalibrationRunnerV2(
        config,
        arena_runner=arena,
        resolver=resolver,
        experiment_root=root,
        lineage_factory=lineage,
        stop_parent=lambda _parent: stops.append("stop"),
        child_training=lambda **_kwargs: child,
    )
    result = runner.run()
    assert result.selected_komi == 1.5
    assert result.parent.ref == parent.ref
    assert len(arena.requests) == 2
    assert all(request.candidate.ref == parent.ref and request.reference.ref == parent.ref for request in arena.requests)
    assert lineage.calls == 1
    assert stops == ["stop"]
    assert (root / "refs.json").is_file()
    assert (root / "results.json").is_file()
    assert (root / "winner.json").is_file()
    assert (root / "full-contract.json").is_file()

    resumed = runner.run()
    assert resumed.selected_komi == 1.5
    assert len(arena.requests) == 2
    assert lineage.calls == 1


def test_runner_extends_both_candidates_from_the_same_frozen_family(tmp_path: Path) -> None:
    config, resolver, lineage, arena, _parent, child, _stops = _runner_fixture(tmp_path, ambiguous=True)
    runner = KomiCalibrationRunnerV2(
        config,
        arena_runner=arena,
        resolver=resolver,
        experiment_root=tmp_path / "runs" / "torus9" / "evaluations" / config.calibration_id,
        lineage_factory=lineage,
        child_training=lambda **_kwargs: child,
    )
    result = runner.run()
    assert result.selected_komi == 1.5
    assert len(arena.requests) == 4
    assert {request.workload["continuation_offset_pairs"] for request in arena.requests[:2]} == {0}
    assert {request.workload["continuation_offset_pairs"] for request in arena.requests[2:]} == {512}
    assert len({request.startset.fingerprint for request in arena.requests}) == 1
