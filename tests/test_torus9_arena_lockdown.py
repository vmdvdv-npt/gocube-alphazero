from __future__ import annotations

from pathlib import Path

import pytest

import gocube_golden.arena_policy as policy
import gocube_golden.torus9_arena_runtime as runtime
import tools.continue_torus9_golden_m1_m100 as continuation
import tools.torus9_alpha_score_ab as alpha_score_ab
import tools.torus9_komi_calibration as komi_calibration
import tools.torus9_ownership_ab as ownership_ab


def test_torus9_arena_policy_is_fail_closed_and_keeps_komi_half_point():
    assert policy.CANONICAL_TORUS9_WORKERS == 16
    assert policy.CANONICAL_TORUS9_KOMI == 0.5
    assert policy.CANONICAL_TORUS9_ARENA_SOURCE_COMMIT == "9723bb5ac8eb28d55d21d607e3673da5bd894315"
    assert policy.CANONICAL_GOLDEN_PROCESS_PROOF_PR == 83
    assert policy.CURRENT_TORUS9_PRODUCTION_ARENA_READY is False
    assert all(engine.production_allowed is False for engine in policy.TORUS9_ARENA_ENGINES)
    with pytest.raises(policy.ArenaPolicyError, match="LOCKED"):
        policy.require_current_torus9_production_ready()


def test_only_current_production_facade_fails_closed_until_broker_is_proven():
    with pytest.raises(policy.ArenaPolicyError, match="LOCKED"):
        runtime.run_current_torus9_production_arena()


def test_process_foundation_requires_explicit_ack_and_real_16_worker_contract():
    kwargs = dict(
        run_id="policy-test",
        comparison="M1-vs-M0",
        candidate_path=Path("candidate.pt"),
        reference_path=Path("reference.pt"),
        candidate_label="M1",
        reference_label="M0",
        starts=(),
        master_seed=1,
        output_dir=Path("arena"),
    )
    with pytest.raises(policy.ArenaPolicyError, match="acknowledge_foundation_only"):
        runtime.run_torus9_process_foundation(**kwargs)
    with pytest.raises(policy.ArenaPolicyError, match="exactly 16"):
        runtime.run_torus9_process_foundation(
            **kwargs, workers=8, acknowledge_foundation_only=True
        )
    with pytest.raises(policy.ArenaPolicyError, match="CPU-only"):
        runtime.run_torus9_process_foundation(
            **kwargs, device="cuda", acknowledge_foundation_only=True
        )


def test_public_continuation_entrypoint_cannot_execute_frozen_arena_programmatically():
    source = Path(continuation.__file__).read_text(encoding="utf-8")
    assert "run_torus9_batched_arena" not in source
    assert policy.FROZEN_ARENA_OVERRIDE_FLAG in source
    assert "_frozen_continue_torus9_golden_m1_m100" in source
    with pytest.raises(policy.ArenaPolicyError, match="frozen logical-lane"):
        continuation.run()
    with pytest.raises(policy.ArenaPolicyError, match="frozen logical-lane"):
        continuation._backfill_missing_m5_arena()


def test_historical_torus9_experiment_entrypoints_are_locked_programmatically():
    with pytest.raises(policy.ArenaPolicyError, match="frozen Torus9 ownership"):
        ownership_ab.run_experiment(None)
    with pytest.raises(policy.ArenaPolicyError, match="frozen Torus9 alpha/score"):
        alpha_score_ab.run_experiment(None)
    with pytest.raises(policy.ArenaPolicyError, match="historical Torus9 komi calibration"):
        komi_calibration.main()


def test_public_historical_scripts_require_the_same_explicit_override():
    for module in (ownership_ab, alpha_score_ab, komi_calibration):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert policy.FROZEN_ARENA_OVERRIDE_FLAG in source
        assert "run_torus9_batched_arena" not in source
        assert "run_torus9_arena(" not in source


def test_komi_calibration_keeps_read_only_analysis_api_available():
    assert komi_calibration.Trajectory is not None
    assert callable(komi_calibration.bootstrap_estimates)
    assert callable(komi_calibration.crossing_interval)
    assert callable(komi_calibration.crossing_point)
    assert callable(komi_calibration.sweep_trajectories)


def test_frozen_override_is_explicit_and_not_a_production_default():
    assert policy.FROZEN_ARENA_OVERRIDE_FLAG == "--allow-frozen-arena"
    frozen = next(
        engine
        for engine in policy.TORUS9_ARENA_ENGINES
        if engine.symbol.endswith("run_torus9_batched_arena")
    )
    assert frozen.role == "frozen-degraded-logical-lane-executor"
    assert frozen.production_allowed is False
