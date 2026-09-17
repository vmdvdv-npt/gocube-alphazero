from __future__ import annotations

import os

import pytest

import gocube_golden as g
from gocube_golden import torus9_monolith as _core
from gocube_golden import torus9_parallel_validation as parallel_validation
from gocube_golden.result import result_from_terminal
from gocube_golden.torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    current_torus9_profile_fingerprint,
    current_torus9_selfplay_contract_fingerprint,
    load_torus9_current_profile,
)


def _one_hot_action(action: int | str) -> tuple[float, ...]:
    values = [0.0] * TORUS9_ACTION_COUNT
    values[TORUS9_ACTION_COUNT - 1 if action == g.PASS else int(action)] = 1.0
    return tuple(values)


def _one_visit_action(action: int | str) -> tuple[int, ...]:
    values = [0] * TORUS9_ACTION_COUNT
    values[TORUS9_ACTION_COUNT - 1 if action == g.PASS else int(action)] = 1
    return tuple(values)


def _record() -> g.Torus9SelfPlayGameRecord:
    profile = load_torus9_current_profile()
    state = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    start_state = _core.torus9_state_identity(state)
    actions: tuple[int | str, ...] = (0, 1, g.PASS, g.PASS)
    positions: list[g.Torus9SelfPlayPosition] = []
    for ply, action in enumerate(actions, 1):
        positions.append(
            g.Torus9SelfPlayPosition(
                ply=ply,
                state=_core.torus9_state_identity(state),
                side_to_move=state.side_to_move.name,
                root_visits=_one_visit_action(action),
                pi=_one_hot_action(action),
                selected_action=action,
                search_seed=7000 + ply,
                model_hash="test-model-hash",
            )
        )
        state = g.apply_action(state, action).after
    assert state.is_terminal
    return g.Torus9SelfPlayGameRecord(
        run_id="parallel-validation-test",
        game_id="game-0000",
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        profile_fingerprint=current_torus9_profile_fingerprint(profile),
        selfplay_contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        selfplay_contract_fingerprint=current_torus9_selfplay_contract_fingerprint(),
        model_checkpoint_label="M0",
        model_hash="test-model-hash",
        checkpoint_artifact_hash="test-checkpoint-hash",
        git_commit="test-commit",
        git_tree="test-tree",
        git_worktree_clean=True,
        master_seed=20260917,
        game_seed=2026091701,
        start_state=start_state,
        positions=tuple(positions),
        final_action_trace=actions,
        formal_result=result_from_terminal(state).winner.value,
        technical_termination=None,
        nn_evaluations=len(actions),
    )


def _stamped_rows(adapter: g.Torus9TrainingAdapter):
    rows = adapter.build_samples_for_replay((_record(),))
    return adapter.stamp_samples(rows, 1)


def test_production_torus9_exports_parallel_validation_adapter() -> None:
    assert g.Torus9TrainingAdapter is parallel_validation.Torus9TrainingAdapter


def test_worker_count_is_bounded_by_cpu_and_useful_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert parallel_validation.Torus9TrainingAdapter._parallel_validation_worker_count(255) == 1
    assert parallel_validation.Torus9TrainingAdapter._parallel_validation_worker_count(256) == 4
    assert parallel_validation.Torus9TrainingAdapter._parallel_validation_worker_count(512) == 8
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert parallel_validation.Torus9TrainingAdapter._parallel_validation_worker_count(10_000) == 3
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert parallel_validation.Torus9TrainingAdapter._parallel_validation_worker_count(10_000) == 1


def test_parallel_validation_preserves_replay_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = _stamped_rows(adapter)
    replay = g.Torus9RollingReplay()
    timing: dict[str, object] = {}
    adapter.set_diagnostic_timing(timing)
    monkeypatch.setattr(adapter, "_parallel_validation_worker_count", lambda _count: 2)

    metrics = adapter.update_replay(replay, 1, rows)

    assert len(replay.rows) == len(rows) == 4
    assert replay.last_generation == 1
    assert metrics["fresh_positions"] == len(rows)
    assert set(adapter._validated_sample_fingerprints) == {
        str(row["replay_row_id"]) for row in rows
    }
    assert timing["replay_new_semantic_validation_mode"] == "process-parallel"
    assert timing["replay_new_semantic_validation_workers"] == 2
    adapter.validate_replay(adapter.replay_rows(replay))


def test_parallel_semantic_failure_cannot_mutate_replay_or_publish_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = [dict(row) for row in _stamped_rows(adapter)]
    rows[-1]["pi"] = [0.0] * TORUS9_ACTION_COUNT
    replay = g.Torus9RollingReplay()
    monkeypatch.setattr(adapter, "_parallel_validation_worker_count", lambda _count: 2)

    with pytest.raises(ValueError):
        adapter.update_replay(replay, 1, rows)

    assert replay.rows == ()
    assert replay.last_generation == 0
    assert adapter._validated_sample_fingerprints == {}


def test_pool_infrastructure_failure_falls_back_to_serial_without_weakening_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = _stamped_rows(adapter)
    replay = g.Torus9RollingReplay()
    timing: dict[str, object] = {}
    adapter.set_diagnostic_timing(timing)
    monkeypatch.setattr(adapter, "_parallel_validation_worker_count", lambda _count: 2)

    def fail_pool(*_args, **_kwargs):
        raise parallel_validation.BrokenProcessPool("synthetic pool failure")

    monkeypatch.setattr(adapter, "_validate_new_rows_parallel", fail_pool)
    adapter.update_replay(replay, 1, rows)

    assert len(replay.rows) == len(rows)
    assert timing["replay_new_semantic_validation_mode"] == "serial-fallback"
    assert timing["replay_new_semantic_validation_workers"] == 1
    assert timing["replay_new_semantic_validation_fallback_reason"] == "BrokenProcessPool"


def test_serial_validation_timing_reports_selected_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = _stamped_rows(adapter)
    replay = g.Torus9RollingReplay()
    timing: dict[str, object] = {}
    adapter.set_diagnostic_timing(timing)
    monkeypatch.setattr(adapter, "_parallel_validation_worker_count", lambda _count: 1)

    adapter.update_replay(replay, 1, rows)

    assert timing["replay_new_semantic_validation_mode"] == "serial"
    assert timing["replay_new_semantic_validation_workers"] == 1
    assert timing["replay_new_semantic_validation_rows"] == len(rows)
    assert float(timing["replay_new_semantic_validation_wall_time_sec"]) >= 0.0
