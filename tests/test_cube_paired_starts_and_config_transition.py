from __future__ import annotations

from pathlib import Path

import pytest

from gocube_golden.cube_arena_startset_v1 import (
    build_cube_arena_startset,
    reconstruct_cube_arena_start,
)
from gocube_golden.cube_checkpoint_v2 import file_sha256
from gocube_golden.cube_training_contract_v2 import CubeTrainingConfig
from gocube_golden.cube_training_v2 import (
    create_cube_m0_state,
    load_cube_checkpoint,
    load_cube_checkpoint_for_config_transition,
    run_cube_training_generation,
)
from gocube_golden.cube_network_v2 import cube_graphnet_v2_model_hash


def test_cube_192_game_corpus_is_paired_diverse_and_reproducible() -> None:
    first = build_cube_arena_startset(size=4, master_seed=20260926, pairs=96)
    second = build_cube_arena_startset(size=4, master_seed=20260926, pairs=96)
    other = build_cube_arena_startset(size=4, master_seed=20260927, pairs=96)

    assert len(first.starts) == 96
    assert first.empty_control_pairs == 1
    assert first.diverse_pairs == 95
    assert len({start.start_id for start in first.starts}) == 96
    assert len({start.start_fingerprint for start in first.starts}) == 96
    assert all(
        start.opening_ply == 0 or start.opening_actions
        for start in first.starts
    )
    assert len({start.opening_ply for start in first.starts}) > 1
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint != other.fingerprint
    assert [start.opening_actions for start in first.starts[1:]] != [
        start.opening_actions for start in other.starts[1:]
    ]
    for start in first.starts:
        position = reconstruct_cube_arena_start(
            size=4, opening_actions=start.opening_actions
        )
        assert not position.is_terminal


def test_cube_startset_identity_is_not_the_legacy_empty_board_identity() -> None:
    from gocube_golden.orchestrator_v2.topology_binding import get_topology_binding

    current = get_topology_binding("cube4").arena_startset(
        master_seed=20260926, games=192
    )
    assert current.id == "cube4-evaluation-starts-v1"
    assert current.fingerprint == current.artifact.sha256
    assert "canonical-empty" not in current.id


def test_cube_arena_worker_smoke_uses_nonempty_opening_history(tmp_path: Path) -> None:
    import json

    from tests.test_cube_stage7_arena_generation import (
        _arena_search,
        _publish_temp_m0,
    )
    from gocube_golden.cube_arena_v2 import run_cube_arena
    from tools.arena_engine import ArenaExecutionConfig

    _, _, candidate, _ = _publish_temp_m0(
        tmp_path, size=2, lineage_id="paired-candidate", seed=901, salt=0.01
    )
    _, _, reference, _ = _publish_temp_m0(
        tmp_path, size=2, lineage_id="paired-reference", seed=902, salt=0.02
    )
    result = run_cube_arena(
        size=2,
        candidate_checkpoint=candidate,
        reference_checkpoint=reference,
        output_dir=tmp_path / "arena",
        search_config=_arena_search().__class__(
            simulations=1, cpuct=1.0, fpu=0.0, watchdog=24
        ),
        execution_config=ArenaExecutionConfig(
            games=4,
            workers=2,
            games_per_worker=2,
            inference_batch_rows=2,
            inference_batch_wait_ms=0.0,
            device="cpu",
            strict_production=False,
            early_gate_enabled=False,
            min_effective_cpu_cores=0.0,
        ),
        seed=903,
        temporary_root=tmp_path,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "arena" / "games.jsonl").read_text().splitlines()
    ]
    assert result.unique_start_count == 2
    assert result.empty_control_pairs == 1
    assert result.diverse_pairs == 1
    diverse = [row for row in rows if row["start_kind"] == "diverse"]
    assert len(diverse) == 2
    assert diverse[0]["opening_ply"] > 0
    assert diverse[0]["start_trace"] == diverse[1]["start_trace"]
    assert diverse[0]["start_fingerprint"] == diverse[1]["start_fingerprint"]
    assert {row["candidate_black"] for row in diverse} == {True, False}


def _config(
    *,
    learning_rate: float = 0.001,
    batch_size: int = 1,
    optimizer_steps: int = 1,
    replay_generations: int = 2,
    replay_cap: int | None = 16,
) -> CubeTrainingConfig:
    return CubeTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer_steps=optimizer_steps,
        replay_generations=replay_generations,
        replay_cap=replay_cap,
    )


def test_explicit_cube_config_transition_preserves_adam_and_replay_parent(
    tmp_path: Path,
) -> None:
    # Keep the fixture intentionally small and local; this does not touch any
    # production run, checkpoint, or replay artifact.
    from tests.test_cube_stage6_training import _formal_pass_record

    source_config = _config()
    source_adapter, source_state = create_cube_m0_state(
        size=2, config=source_config, seed=801
    )
    generation = run_cube_training_generation(
        adapter=source_adapter,
        state=source_state,
        generation=1,
        lineage_dir=tmp_path / "parent",
        run_id="parent",
        training_seed=802,
        records=(_formal_pass_record(2),),
        device="cpu",
    )
    parent_path = Path(generation.checkpoint_reference["path"])
    replay_path = Path(generation.engine_result.artifacts["rolling_replay"])
    parent_sha_before = file_sha256(parent_path)
    parent_replay_before = replay_path.read_bytes()
    parent_model_hash = str(generation.engine_result.checkpoint_metadata["model_hash"])

    target_config = _config(
        learning_rate=0.002,
        batch_size=2,
        optimizer_steps=2,
        replay_generations=1,
        replay_cap=1,
    )
    with pytest.raises(ValueError, match="concrete training config"):
        load_cube_checkpoint(
            parent_path,
            config=target_config,
            replay_path=replay_path,
            expected_size=2,
        )

    adapter, state, transition = load_cube_checkpoint_for_config_transition(
        parent_path,
        config=target_config,
        replay_path=replay_path,
        expected_size=2,
        source_checkpoint_identity={
            "lineage_id": "parent",
            "checkpoint_id": "M1",
            "generation": 1,
            "sha256": parent_sha_before,
        },
    )
    assert transition.optimizer_state_preserved is True
    assert set(transition.changed_fields) == {
        "learning_rate",
        "batch_size",
        "optimizer_steps",
        "replay_generations",
        "replay_cap",
    }
    assert cube_graphnet_v2_model_hash(state.model) == parent_model_hash
    assert state.optimizer.param_groups[0]["lr"] == target_config.learning_rate
    assert state.optimizer_updates == source_state.optimizer_updates
    assert len(state.rolling_replay.rows) == 1

    child = run_cube_training_generation(
        adapter=adapter,
        state=state,
        generation=2,
        lineage_dir=tmp_path / "child",
        run_id="child",
        training_seed=803,
        records=(_formal_pass_record(2),),
        parent_checkpoint_identity={
            "lineage_id": "parent",
            "checkpoint_id": "M1",
            "generation": 1,
            "sha256": parent_sha_before,
        },
        device="cpu",
    )
    metadata = child.engine_result.checkpoint_metadata
    assert metadata["concrete_training_config_fingerprint"] == target_config.fingerprint
    assert metadata["effective_learning_rate"] == target_config.learning_rate
    assert metadata["training_config_transition"]["source_checkpoint"]["sha256"] == parent_sha_before
    assert metadata["training_config_transition"]["optimizer_state_preserved"] is True
    assert file_sha256(parent_path) == parent_sha_before
    assert replay_path.read_bytes() == parent_replay_before
    assert not (tmp_path / "child" / "checkpoints" / "M1.pt").exists()

    reloaded, reloaded_state, _ = load_cube_checkpoint(
        child.checkpoint_reference["path"],
        config=target_config,
        replay_path=child.engine_result.artifacts["rolling_replay"],
        expected_size=2,
    )
    reloaded.validate_state(reloaded_state)
