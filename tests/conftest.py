from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


# These pre-boundary regression tests intentionally exercise production internals
# directly. Give only those exact tests the private unit-test authority; boundary
# tests remain unauthorized and continue proving that env markers/direct calls
# fail closed.
_AUTHORIZED_INTERNAL_TESTS: dict[tuple[str, str], tuple[str, str]] = {
    (
        "test_orchestrator_v2_cube_stage8.py",
        "test_cube4_real_orchestrated_m0_to_m2_smoke",
    ): ("cube4", "cube4-stage8-smoke"),
    (
        "test_orchestrator_v2_cube_stage8_commit_recovery.py",
        "test_cube_commit_boundary_crash_is_reconciled_and_same_generation_retries",
    ): ("cube4", "cube4-stage8-commit-recovery"),
    (
        "test_orchestrator_v2_generation_runner.py",
        "test_torus9_production_path_forwards_exact_input_and_maps_committed_result",
    ): ("torus9", "child-lineage"),
    (
        "test_orchestrator_v2_immutable_runtime.py",
        "test_unresolvable_execution_commit_fails_before_new_child",
    ): ("torus9", "lineage"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_runs_one_generation_and_returns_immediate_child",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_acknowledges_stopped_execution_before_baseline_start",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_reuses_committed_child_without_supervisor",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_reconciles_committed_execution_for_next_generation",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_committed_reuse_does_not_delete_foreign_or_malformed_supervisor_intent",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_committed_reuse_does_not_delete_foreign_or_malformed_supervisor_stop",
    ): ("torus9", "child"),
}

# Scientific/engine regression tests run under explicit test-only V2 authority.
# Rejection tests are deliberately not included.
_AUTHORIZED_INTERNAL_TESTS.update({
    ("test_torus9_adaptation.py", "test_decision_bound_to_report_and_no_duplicate"): ("torus9", "review-regression"),
    ('test_selfplay_engine.py', 'test_malformed_inference_batch_fails_entire_engine'): ("torus9", "engine-regression"),
    ('test_selfplay_engine.py', 'test_multiple_search_lanes_per_process_route_responses_without_pid_duplication'): ("torus9", "engine-regression"),
    ('test_selfplay_engine.py', 'test_process_workers_central_batching_and_ordering'): ("torus9", "engine-regression"),
    ('test_selfplay_engine.py', 'test_worker_exception_fails_closed_instead_of_returning_partial_games'): ("torus9", "engine-regression"),
    ('test_selfplay_engine.py', 'test_worker_process_death_is_detected_fail_closed'): ("torus9", "engine-regression"),
    ('test_selfplay_engine_process_cleanup.py', 'test_failure_path_leaves_no_selfplay_worker_processes'): ("torus9", "engine-regression"),
    ('test_selfplay_engine_shared_memory.py', 'test_shared_memory_cooperative_scheduler_batches_and_replenishes'): ("torus9", "engine-regression"),
    ('test_selfplay_engine_shared_memory.py', 'test_total_active_contexts_caps_contexts_without_reducing_worker_pool'): ("torus9", "engine-regression"),
    ('test_torus9_adaptation.py', 'test_cuda_full_unfreeze_bias'): ("torus9", "engine-regression"),
    ('test_torus9_adaptation.py', 'test_freeze_and_deterministic_resume'): ("torus9", "engine-regression"),
    ('test_torus9_adaptation.py', 'test_phase_clocks_and_lr_on_load'): ("torus9", "engine-regression"),
    ('test_torus9_adaptation.py', 'test_real_trained_artifact_arena_load'): ("torus9", "engine-regression"),
    ('test_torus9_adaptation.py', 'test_retry_rolls_back_entire_phase'): ("torus9", "engine-regression"),
    ('test_torus9_adaptation.py', 'test_stage_report_pause_and_notification_retry'): ("torus9", "engine-regression"),
    ('test_training_engine_stage3.py', 'test_commit_preparation_runs_before_final_marker_fence'): ("torus9", "engine-regression"),
    ('test_training_engine_stage3.py', 'test_generic_engine_commits_all_artifacts_and_state'): ("torus9", "engine-regression"),
    ('test_training_engine_stage3.py', 'test_generic_engine_failure_does_not_publish_or_commit'): ("torus9", "engine-regression"),
    ('test_training_engine_validation_contract.py', 'test_direct_samples_do_not_use_record_construction_capability'): ("torus9", "engine-regression"),
    ('test_training_engine_validation_contract.py', 'test_direct_samples_keep_pre_validation'): ("torus9", "engine-regression"),
    ('test_training_engine_validation_contract.py', 'test_record_built_samples_are_not_revalidated_by_engine'): ("torus9", "engine-regression"),
    ('test_training_engine_validation_contract.py', 'test_record_construction_capability_moves_validation_to_replay_boundary'): ("torus9", "engine-regression"),
})


_AUTHORIZED_INTERNAL_TESTS.update({
    ('test_cube_stage5_shared_engines.py', 'test_common_cooperative_runner_accepts_structural_adapter_contract'): ('cube4', "cube-regression"),
    ('test_cube_stage6_training.py', 'test_cube2_to_cube7_one_step_finite_and_changes_parameters'): ('cube-param', "cube-regression"),
    ('test_cube_stage6_training.py', 'test_stage5_cube4_record_to_common_training_engine_checkpoint'): ('cube4', "cube-regression"),
    ('test_cube_stage6_training.py', 'test_replay_window_cap_provenance_and_deterministic_sampling'): ('cube2', "cube-regression"),
    ('test_cube_stage6_training.py', 'test_checkpoint_save_load_restores_model_optimizer_and_sampling'): ('cube4', "cube-regression"),
    ('test_cube_stage6_training.py', 'test_resume_step2_matches_uninterrupted_cpu_path'): ('cube2', "cube-regression"),
    ('test_cube_stage6_training.py', 'test_checkpoint_compatibility_fails_closed'): ('cube2', "cube-regression"),
})

_AUTHORIZED_INTERNAL_TESTS.update({
    ('test_replay_write_identity.py', 'test_training_engine_does_not_rehash_rolling_tmp_after_single_pass_write'): ("torus9", "replay-regression"),
    ('test_torus9_replay_composition_identity.py', 'test_engine_hashes_fresh_artifact_during_its_single_write'): ("torus9", "replay-regression"),
    ('test_torus9_replay_composition_identity.py', 'test_engine_uses_adapter_identity_without_full_replay_fingerprint'): ("torus9", "replay-regression"),
    ('test_torus9_replay_target_build.py', 'test_torus9_records_use_one_authoritative_semantic_validation_per_row'): ("torus9", "replay-regression"),
    ('test_torus9_selfplay_engine_boundary.py', 'test_current_profile_fingerprint_drift_fails_closed'): ("torus9", "replay-regression"),
    ('test_torus9_selfplay_engine_boundary.py', 'test_empty_batch_still_crosses_the_selfplay_engine_boundary'): ("torus9", "replay-regression"),
})

# Stage-7 Cube regressions also intentionally exercise production internals.
# Keep them authorized only inside pytest; production callers still need the
# live Orchestrator V2 authority or a supervised child permit.
_AUTHORIZED_INTERNAL_TESTS.update({
    ('test_cube_paired_starts_and_config_transition.py', 'test_cube_arena_worker_smoke_uses_nonempty_opening_history'): ('cube2', "cube-stage7-regression"),
    ('test_cube_paired_starts_and_config_transition.py', 'test_explicit_cube_config_transition_preserves_adam_and_replay_parent'): ('cube2', "cube-stage7-regression"),
    ('test_cube_stage7_arena_generation.py', 'test_cube2_to_cube7_common_arena_cpu_smoke'): ('cube-param', "cube-stage7-regression"),
    ('test_cube_stage7_arena_generation.py', 'test_cube_arena_is_deterministic_and_color_paired'): ('cube2', "cube-stage7-regression"),
    ('test_cube_stage7_arena_generation.py', 'test_cube4_one_generation_resume_and_duplicate_boundary'): ('cube4', "cube-stage7-regression"),
    ('test_cube_stage7_arena_generation.py', 'test_arena_failure_preserves_committed_training'): ('cube2', "cube-stage7-regression"),
    ('test_cube_stage7_arena_generation.py', 'test_training_failure_publishes_no_checkpoint_or_fake_arena'): ('cube2', "cube-stage7-regression"),
    ('test_shared_arena_stage7_followup.py', 'test_cube_four_games_use_four_real_lanes_and_central_batching'): ('cube2', "cube-stage7-regression"),
})

# These exact legacy tests execute Arena in-process. Give them a real signed
# Arena child permit in addition to the test-only authority. The PID shim only
# models the missing process hop inside pytest; production PID binding remains
# unchanged and boundary/rejection tests are deliberately absent from this set.
_ARENA_CHILD_PERMIT_TESTS = {
    ('test_cube_paired_starts_and_config_transition.py', 'test_cube_arena_worker_smoke_uses_nonempty_opening_history'),
    ('test_cube_stage7_arena_generation.py', 'test_cube2_to_cube7_common_arena_cpu_smoke'),
    ('test_cube_stage7_arena_generation.py', 'test_cube_arena_is_deterministic_and_color_paired'),
    ('test_cube_stage7_arena_generation.py', 'test_cube4_one_generation_resume_and_duplicate_boundary'),
    ('test_shared_arena_stage7_followup.py', 'test_cube_four_games_use_four_real_lanes_and_central_batching'),
}


@pytest.fixture(autouse=True)
def _legacy_orchestrator_v2_internal_authority(request: pytest.FixtureRequest):
    filename = Path(str(request.fspath)).name
    test_name = request.node.name.split("[", 1)[0]
    test_identity = (filename, test_name)
    identity = _AUTHORIZED_INTERNAL_TESTS.get(test_identity)
    if identity is None:
        yield
        return

    # Import lazily so lightweight suites that share this conftest (notably the
    # pinned KataGo differential job) do not import gocube_golden and therefore
    # do not acquire the heavyweight torch runtime dependency.
    from gocube_golden.orchestrator_v2 import execution_permit
    from gocube_golden.orchestrator_v2.execution_permit import (
        _child_execution_permit,
        _test_authority,
    )

    topology, run_id = identity
    if topology == "cube-param":
        topology = f"cube{request.node.callspec.params['size']}"

    if filename == "test_orchestrator_v2_production_generation.py":
        # Those unit tests predate immutable-runtime enforcement and replace the
        # supervisor with a fake. Supply the minimal committed-code fixture they
        # now need without weakening the production implementation.
        tmp_path = request.getfixturevalue("tmp_path")
        monkeypatch = request.getfixturevalue("monkeypatch")
        lineage_root = tmp_path / "lineage"
        lineage_root.mkdir(parents=True, exist_ok=True)
        (lineage_root / "manifest.json").write_text(
            '{"execution_code_commit":"test-commit"}\n',
            encoding="utf-8",
        )

        from gocube_golden.orchestrator_v2 import production_generation

        monkeypatch.setattr(
            production_generation,
            "execution_commit_from_lineage",
            lambda _root: "test-commit",
        )

        def _fake_ensure(_manager, commit: str):
            assert commit == "test-commit"
            return SimpleNamespace(
                commit=commit,
                path=tmp_path,
                environment=lambda base=None: dict(
                    os.environ if base is None else base
                ),
            )

        monkeypatch.setattr(
            production_generation.ImmutableRuntimeManager,
            "ensure",
            _fake_ensure,
        )

    with _test_authority(
        topology=topology,
        run_id=run_id,
        code_identity="test-commit",
    ):
        if test_identity not in _ARENA_CHILD_PERMIT_TESTS:
            yield
            return

        monkeypatch = request.getfixturevalue("monkeypatch")
        monkeypatch.setattr(execution_permit.os, "getppid", execution_permit.os.getpid)
        with _child_execution_permit(
            action_type="arena",
            topology=topology,
            run_id=run_id,
            code_identity="test-commit",
        ):
            yield
