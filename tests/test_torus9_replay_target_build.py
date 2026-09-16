from __future__ import annotations

import pytest

import gocube_golden as g
from gocube_golden import torus9_monolith as _core
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
    index = TORUS9_ACTION_COUNT - 1 if action == g.PASS else int(action)
    values[index] = 1.0
    return tuple(values)


def _one_visit_action(action: int | str) -> tuple[int, ...]:
    values = [0] * TORUS9_ACTION_COUNT
    index = TORUS9_ACTION_COUNT - 1 if action == g.PASS else int(action)
    values[index] = 1
    return tuple(values)


def _four_ply_double_pass_record() -> g.Torus9SelfPlayGameRecord:
    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    contract_fp = current_torus9_selfplay_contract_fingerprint()
    model_hash = "test-model-hash"
    state = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    start_state = _core.torus9_state_identity(state)
    actions: tuple[int | str, ...] = (0, 1, g.PASS, g.PASS)
    positions: list[g.Torus9SelfPlayPosition] = []
    for ply, action in enumerate(actions, 1):
        assert action in g.legal_actions(state)
        positions.append(
            g.Torus9SelfPlayPosition(
                ply=ply,
                state=_core.torus9_state_identity(state),
                side_to_move=state.side_to_move.name,
                root_visits=_one_visit_action(action),
                pi=_one_hot_action(action),
                selected_action=action,
                search_seed=1000 + ply,
                model_hash=model_hash,
            )
        )
        state = g.apply_action(state, action).after
    assert state.is_terminal
    return g.Torus9SelfPlayGameRecord(
        run_id="replay-target-build-test",
        game_id="game-0000",
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        profile_fingerprint=profile_fp,
        selfplay_contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        selfplay_contract_fingerprint=contract_fp,
        model_checkpoint_label="M0",
        model_hash=model_hash,
        checkpoint_artifact_hash="test-checkpoint-hash",
        git_commit="test-commit",
        git_tree="test-tree",
        git_worktree_clean=True,
        master_seed=20260916,
        game_seed=2026091601,
        start_state=start_state,
        positions=tuple(positions),
        final_action_trace=actions,
        formal_result=result_from_terminal(state).winner.value,
        technical_termination=None,
        nn_evaluations=4,
    )


def test_adapter_fused_auxiliary_build_is_exactly_legacy_equivalent() -> None:
    record = _four_ply_double_pass_record()
    expected = _core.torus9_build_ownership_score_replay_samples(record)
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    actual = tuple(adapter.build_samples((record,)))
    assert actual == expected


def test_adapter_fused_auxiliary_build_removes_repeated_terminal_scoring(monkeypatch) -> None:
    record = _four_ply_double_pass_record()
    counts = {"apply_action": 0, "score_terminal": 0}
    original_apply_action = _core.apply_action
    original_score_terminal = _core.score_terminal

    def counted_apply_action(*args, **kwargs):
        counts["apply_action"] += 1
        return original_apply_action(*args, **kwargs)

    def counted_score_terminal(*args, **kwargs):
        counts["score_terminal"] += 1
        return original_score_terminal(*args, **kwargs)

    monkeypatch.setattr(_core, "apply_action", counted_apply_action)
    monkeypatch.setattr(_core, "score_terminal", counted_score_terminal)

    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = tuple(adapter.build_samples((record,)))

    assert len(rows) == 4
    # One trace replay remains in the proven base row builder and one computes
    # the shared terminal state for both auxiliaries. The legacy composed path
    # replayed the same four actions three times.
    assert counts["apply_action"] == 8
    # Ownership and score are computed once per side, not once per replay row.
    # For four rows the legacy composed builder performs eight terminal scores.
    assert counts["score_terminal"] == 4


def test_construction_only_api_preserves_public_validated_contract(monkeypatch) -> None:
    record = _four_ply_double_pass_record()
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    calls = {"count": 0}
    original = adapter.validate_sample

    def counted(sample):
        calls["count"] += 1
        return original(sample)

    monkeypatch.setattr(adapter, "validate_sample", counted)
    constructed = tuple(adapter.build_samples_for_replay((record,)))
    assert calls["count"] == 0

    validated = tuple(adapter.build_samples((record,)))
    assert calls["count"] == len(validated) == 4
    assert constructed == validated


def test_validation_cache_is_bound_to_replay_row_content() -> None:
    record = _four_ply_double_pass_record()
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = adapter.build_samples((record,))
    stamped = adapter.stamp_samples(rows, 1)
    replay = g.Torus9RollingReplay()
    adapter.update_replay(replay, 1, stamped)

    # The cache contains this deterministic row ID, but changing semantic
    # content must invalidate the cache hit and force full validation.
    replay.rows[0]["pi"] = [0.0] * TORUS9_ACTION_COUNT
    with pytest.raises(ValueError, match="policy target"):
        adapter.validate_replay(adapter.replay_rows(replay))


def test_update_replay_rejects_noncanonical_id_before_mutation() -> None:
    record = _four_ply_double_pass_record()
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = adapter.build_samples((record,))
    stamped = [dict(row) for row in adapter.stamp_samples(rows, 1)]
    stamped[0]["replay_row_id"] = "non-canonical-but-non-empty"
    replay = g.Torus9RollingReplay()

    with pytest.raises(ValueError, match="row ID is not deterministic"):
        adapter.update_replay(replay, 1, stamped)

    assert replay.rows == ()
    assert replay.last_generation == 0
    assert adapter._validated_sample_fingerprints == {}


def test_failed_batch_does_not_populate_validation_cache_or_mutate_replay() -> None:
    record = _four_ply_double_pass_record()
    adapter = g.Torus9TrainingAdapter(profile=load_torus9_current_profile())
    rows = adapter.build_samples((record,))
    stamped = [dict(row) for row in adapter.stamp_samples(rows, 1)]
    stamped[-1]["pi"] = [0.0] * TORUS9_ACTION_COUNT
    replay = g.Torus9RollingReplay()

    with pytest.raises(ValueError, match="policy target"):
        adapter.update_replay(replay, 1, stamped)

    assert replay.rows == ()
    assert replay.last_generation == 0
    assert adapter._validated_sample_fingerprints == {}
