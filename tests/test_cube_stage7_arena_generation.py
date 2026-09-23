from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from generation_driver import GenerationCompletion, GenerationDriver
from training_engine import CheckpointContext, value_fingerprint
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles.cube_v2 import CubeV2ArenaProfile

from gocube_golden.cube_arena_contract_v2 import (
    CUBE_ARENA_CONTRACT_ID,
    CUBE_ARENA_RESULT_SCHEMA,
    CubeArenaSearchConfig,
)
from gocube_golden.cube_arena_v2 import run_cube_arena
from gocube_golden.cube_checkpoint_v2 import file_sha256
from gocube_golden.cube_generation_adapter_v2 import (
    CubeGenerationAdapter,
    run_cube_generation,
)
from gocube_golden.cube_generation_v2 import (
    CubeArenaRequest,
    CubeSelfPlayPlan,
    build_cube_generation_request,
)
from gocube_golden.cube_selfplay_contract import CubeSelfPlaySearchContract
from gocube_golden.cube_selfplay_v2 import CubeSelfPlayExecutionConfig
from gocube_golden.cube_training_contract_v2 import CubeTrainingConfig
from gocube_golden.cube_training_v2 import create_cube_m0_state, load_cube_checkpoint


def _training_config() -> CubeTrainingConfig:
    return CubeTrainingConfig(
        learning_rate=0.001,
        batch_size=1,
        optimizer_steps=1,
        replay_generations=2,
        replay_cap=16,
    )


def _selfplay_plan(*, games: int = 1) -> CubeSelfPlayPlan:
    return CubeSelfPlayPlan(
        games=games,
        search=CubeSelfPlaySearchContract(
            simulations=1,
            cpuct=1.0,
            fpu=0.0,
            root_noise=False,
            temperature_plies=(1, 1),
            temperature_after=0.0,
            technical_move_limit=4,
            komi=0.5,
        ),
    )


def _selfplay_execution() -> CubeSelfPlayExecutionConfig:
    return CubeSelfPlayExecutionConfig(
        workers=1,
        active_games_per_worker=1,
        total_active_contexts=1,
        inference_batch_cap=1,
        inference_batch_wait_ms=0.0,
        device="cpu",
        process_start_method="fork",
    )


def _arena_search() -> CubeArenaSearchConfig:
    return CubeArenaSearchConfig(
        simulations=1,
        cpuct=1.0,
        fpu=0.0,
        watchdog=4,
    )


def _arena_execution() -> ArenaExecutionConfig:
    return ArenaExecutionConfig(
        games=2,
        workers=2,
        games_per_worker=1,
        inference_batch_rows=1,
        inference_batch_wait_ms=0.0,
        device="cpu",
        strict_production=False,
        early_gate_enabled=False,
        min_effective_cpu_cores=0.0,
    )


def _make_pass_model(state, *, salt: float) -> None:
    with torch.no_grad():
        for parameter in state.model.parameters():
            parameter.zero_()
        state.model.pass_head.bias.fill_(8.0)
        # Keep policy behavior identical/fast but checkpoint identities distinct.
        state.model.score_head[-1].bias.fill_(float(salt))
    state.model.eval()


def _publish_temp_m0(
    root: Path,
    *,
    size: int,
    lineage_id: str,
    seed: int,
    salt: float = 0.0,
):
    lineage = root / lineage_id
    config = _training_config()
    adapter, state = create_cube_m0_state(
        size=size, config=config, seed=seed, device="cpu"
    )
    _make_pass_model(state, salt=salt)
    context = CheckpointContext(
        run_id=lineage_id,
        label="M0",
        parent_label=None,
        generation=0,
        training_seed=seed,
        fresh_positions=0,
        replay_positions=0,
        replay_generations=(),
        replay_fingerprint=value_fingerprint(()),
        sampled_row_ids_fingerprint=value_fingerprint(()),
        completed_games=0,
        parent_checkpoint_identity=None,
        code_identity=None,
        device="cpu",
    )
    metadata = adapter.prepare_checkpoint(state, context, {})
    checkpoint = lineage / "checkpoints" / "M0.pt"
    saved = adapter.save_checkpoint(checkpoint, state, metadata)
    reference = {
        "lineage_id": lineage_id,
        "checkpoint_id": "M0",
        "label": "M0",
        "path": str(checkpoint.resolve()),
        "metadata_path": str(checkpoint.with_suffix(".metadata.json").resolve()),
        "sha256": saved["checkpoint_sha256"],
        "artifact_sha256": saved["checkpoint_sha256"],
        "model_hash": saved["model_hash"],
        "generation": 0,
        "size": size,
    }
    state.parent_checkpoint_identity = dict(reference)
    return adapter, state, reference, lineage


@pytest.mark.parametrize("size", range(2, 8))
def test_cube2_to_cube7_common_arena_cpu_smoke(tmp_path: Path, size: int):
    _, _, candidate, _ = _publish_temp_m0(
        tmp_path, size=size, lineage_id=f"cube{size}-a", seed=100 + size, salt=0.01
    )
    _, _, reference, _ = _publish_temp_m0(
        tmp_path, size=size, lineage_id=f"cube{size}-b", seed=200 + size, salt=0.02
    )
    result = run_cube_arena(
        size=size,
        candidate_checkpoint=candidate,
        reference_checkpoint=reference,
        output_dir=tmp_path / f"arena-cube{size}",
        search_config=_arena_search(),
        execution_config=_arena_execution(),
        seed=300 + size,
        temporary_root=tmp_path,
    )
    assert result.arena_schema == CUBE_ARENA_RESULT_SCHEMA
    assert result.games_requested == 2
    assert result.games_valid == 2
    assert result.technical == 0
    assert result.invalid == 0
    assert result.wins_a + result.wins_b + result.draws == 2
    assert result.search_config["noise"] is False
    assert result.search_config["temperature"] == 0.0
    assert result.search_config["fast_search"] is False
    assert result.search_config["resign"] is False
    assert result.search_config["contract_id"] == CUBE_ARENA_CONTRACT_ID
    assert candidate["sha256"] == result.checkpoint_a["sha256"]
    assert reference["sha256"] == result.checkpoint_b["sha256"]


def test_cube_arena_is_deterministic_and_color_paired(tmp_path: Path):
    _, _, candidate, _ = _publish_temp_m0(
        tmp_path, size=2, lineage_id="det-a", seed=401, salt=0.01
    )
    _, _, reference, _ = _publish_temp_m0(
        tmp_path, size=2, lineage_id="det-b", seed=402, salt=0.02
    )
    kwargs = dict(
        size=2,
        candidate_checkpoint=candidate,
        reference_checkpoint=reference,
        search_config=_arena_search(),
        execution_config=_arena_execution(),
        seed=403,
        temporary_root=tmp_path,
    )
    first = run_cube_arena(output_dir=tmp_path / "det-1", **kwargs)
    second = run_cube_arena(output_dir=tmp_path / "det-2", **kwargs)
    assert (
        first.wins_a,
        first.wins_b,
        first.draws,
        first.technical,
        first.invalid,
    ) == (
        second.wins_a,
        second.wins_b,
        second.draws,
        second.technical,
        second.invalid,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "det-1" / "games.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 2
    assert {row["candidate_black"] for row in rows} == {True, False}
    assert len({row["pair_id"] for row in rows}) == 1
    assert all(row["candidate_model_hash"] == candidate["model_hash"] for row in rows)
    assert all(row["reference_model_hash"] == reference["model_hash"] for row in rows)
    assert all(row["formal_result"] in {"BLACK", "WHITE", "DRAW"} for row in rows)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("observation_fingerprint", "sha256:" + "0" * 64, "observation"),
        ("architecture_id", "GoldenCubeGraphNetV1", "architecture"),
        ("game_fingerprint", "sha256:" + "1" * 64, "game/rules"),
    ],
)
def test_cube_arena_incompatible_metadata_fails_before_games(
    tmp_path: Path, field: str, bad_value: str, match: str
):
    _, _, reference, _ = _publish_temp_m0(
        tmp_path, size=4, lineage_id=f"bad-{field}", seed=501, salt=0.01
    )
    metadata_path = Path(reference["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = bad_value
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profile = CubeV2ArenaProfile(size=4, search_config=_arena_search())
    with pytest.raises(ValueError, match=match):
        profile.load_identity(Path(reference["path"]))


def test_cube_arena_cube3_vs_cube4_fails_before_games(tmp_path: Path):
    _, _, cube3, _ = _publish_temp_m0(
        tmp_path, size=3, lineage_id="cube3", seed=511, salt=0.01
    )
    profile = CubeV2ArenaProfile(size=4, search_config=_arena_search())
    with pytest.raises(ValueError, match="size"):
        profile.load_identity(Path(cube3["path"]))


def _generation_request(
    *,
    adapter,
    generation: int,
    parent: dict[str, object],
    lineage_id: str,
    lineage: Path,
    arena_reference: dict[str, object] | None,
    seed: int,
):
    arena_request = (
        None
        if arena_reference is None
        else CubeArenaRequest(
            reference_checkpoint=arena_reference,
            search_config=_arena_search(),
            execution_config=_arena_execution(),
            seed=seed + 1000,
        )
    )
    return build_cube_generation_request(
        training_adapter=adapter,
        generation=generation,
        parent_checkpoint=parent,
        selfplay_plan=_selfplay_plan(),
        selfplay_execution_config=_selfplay_execution(),
        training_config=adapter.config,
        arena_request=arena_request,
        seed=seed,
        lineage_id=lineage_id,
        lineage_dir=lineage,
    )


def test_cube4_one_generation_resume_and_duplicate_boundary(tmp_path: Path):
    adapter, state, m0, lineage = _publish_temp_m0(
        tmp_path, size=4, lineage_id="cube4-stage7", seed=601, salt=0.01
    )
    request1 = _generation_request(
        adapter=adapter,
        generation=1,
        parent=m0,
        lineage_id="cube4-stage7",
        lineage=lineage,
        arena_reference=m0,
        seed=602,
    )
    result1 = run_cube_generation(
        request1,
        training_adapter=adapter,
        training_state=state,
        temporary_root=tmp_path,
    )
    assert result1.completion_status == GenerationCompletion.COMPLETE.value
    assert result1.formal_games == 1
    assert result1.technical_games == 0
    assert result1.samples_generated == 2
    assert result1.checkpoint_reference is not None
    assert result1.arena_result is not None
    m1 = dict(result1.checkpoint_reference)
    m1_path = Path(m1["path"])
    m1_sha = m1["sha256"]

    with pytest.raises(FileExistsError, match="already"):
        run_cube_generation(
            request1,
            training_adapter=adapter,
            training_state=state,
            temporary_root=tmp_path,
        )
    assert Path(m1["path"]).is_file()
    assert file_sha256(m1["path"]) == m1_sha

    loaded_adapter, loaded_state, _ = load_cube_checkpoint(
        m1["path"],
        config=adapter.config,
        replay_path=lineage / "replay" / "rolling-after-01.jsonl",
        expected_size=4,
    )
    m1["lineage_id"] = "cube4-stage7"
    m1["checkpoint_id"] = "M1"
    m1["label"] = "M1"
    m1["generation"] = 1
    loaded_state.parent_checkpoint_identity = dict(m1)
    request2 = _generation_request(
        adapter=loaded_adapter,
        generation=2,
        parent=m1,
        lineage_id="cube4-stage7",
        lineage=lineage,
        arena_reference=None,
        seed=603,
    )
    result2 = run_cube_generation(
        request2,
        training_adapter=loaded_adapter,
        training_state=loaded_state,
        temporary_root=tmp_path,
    )
    assert result2.completion_status == GenerationCompletion.COMPLETE.value
    assert result2.parent_checkpoint["path"] == str(m1_path)
    assert Path(result2.checkpoint_reference["path"]).name == "M2.pt"
    assert m1_path.is_file()
    assert not (lineage / "checkpoints" / "M1-copy.pt").exists()


class _ArenaFailureAdapter(CubeGenerationAdapter):
    def run_arena(self, request, training_result):
        raise RuntimeError("injected Arena failure")


class _TrainingFailureAdapter(CubeGenerationAdapter):
    def run_training(self, request, selfplay_result):
        raise RuntimeError("injected training failure")


def test_arena_failure_preserves_committed_training(tmp_path: Path):
    adapter, state, m0, lineage = _publish_temp_m0(
        tmp_path, size=2, lineage_id="arena-failure", seed=701, salt=0.01
    )
    request = _generation_request(
        adapter=adapter,
        generation=1,
        parent=m0,
        lineage_id="arena-failure",
        lineage=lineage,
        arena_reference=m0,
        seed=702,
    )
    driver = GenerationDriver(
        _ArenaFailureAdapter(
            training_adapter=adapter,
            training_state=state,
            temporary_root=tmp_path,
        )
    )
    result = driver.run(request.to_common())
    assert result.completion_status == GenerationCompletion.ARENA_FAILED.value
    assert "injected Arena failure" in result.error
    assert (lineage / "checkpoints" / "M1.pt").is_file()
    assert (lineage / "replay" / "rolling-after-01.jsonl").is_file()
    assert (lineage / "generation-01.complete.json").is_file()


def test_training_failure_publishes_no_checkpoint_or_fake_arena(tmp_path: Path):
    adapter, state, m0, lineage = _publish_temp_m0(
        tmp_path, size=2, lineage_id="training-failure", seed=801, salt=0.01
    )
    request = _generation_request(
        adapter=adapter,
        generation=1,
        parent=m0,
        lineage_id="training-failure",
        lineage=lineage,
        arena_reference=m0,
        seed=802,
    )
    driver = GenerationDriver(
        _TrainingFailureAdapter(
            training_adapter=adapter,
            training_state=state,
            temporary_root=tmp_path,
        )
    )
    result = driver.run(request.to_common())
    assert result.completion_status == GenerationCompletion.FAILED.value
    assert "injected training failure" in result.error
    assert not (lineage / "checkpoints" / "M1.pt").exists()
    assert not (lineage / "generation-01.complete.json").exists()
    assert not (lineage / "arena").exists()


def test_stage7_common_layers_remain_topology_neutral():
    import ast

    root = Path(__file__).resolve().parents[1]
    for relative in (
        "gocube_golden/selfplay_engine.py",
        "training_engine.py",
        "gocube_golden/search.py",
        "tools/arena_engine.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            "cube" in name.lower() or "torus" in name.lower()
            for name in imported
        )

    orchestrator = root / "gocube_golden" / "orchestrator_v2"
    for path in orchestrator.rglob("*.py"):
        assert "CubeGenerationAdapter" not in path.read_text(encoding="utf-8")
