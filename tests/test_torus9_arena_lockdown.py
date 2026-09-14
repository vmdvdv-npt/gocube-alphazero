from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import gocube_golden
import gocube_golden.arena_policy as policy
import gocube_golden.torus9 as torus9_impl
import tools.arena as arena_cli
import tools.arena_engine as arena_engine
import tools.continue_torus9_golden_m1_m100 as continuation
import tools.torus9_alpha_score_ab as alpha_score_ab
import tools.torus9_komi_calibration as komi_calibration
import tools.torus9_ownership_ab as ownership_ab
from tools.arena_profiles import available_profiles, get_profile
from tools.arena_profiles.torus9 import Torus9ArenaProfile


def test_repository_has_exactly_one_production_arena_engine_policy():
    production = [engine for engine in policy.ARENA_ENGINES if engine.production_allowed]
    assert len(production) == 1
    assert production[0].symbol == "tools.arena.run_arena"
    assert policy.CANONICAL_ARENA_ENGINE == "process-central-inference-v1"
    assert policy.CURRENT_PRODUCTION_ARENA_READY is True
    assert policy.CANONICAL_ARENA_WORKERS == 16
    policy.require_current_production_ready()


def test_engine_is_board_agnostic_and_torus9_is_only_a_profile():
    engine_source = inspect.getsource(arena_engine)
    assert "gocube_golden.torus9" not in engine_source
    assert "TORUS9_" not in engine_source
    assert "build_torus9" not in engine_source
    assert "torus9_" not in engine_source.lower()

    profile_source = inspect.getsource(Torus9ArenaProfile)
    assert "Torus9BatchedPUCT" in profile_source
    assert "torus9_load_checkpoint" in profile_source
    assert "get_context(" not in profile_source
    assert "ctx.Process" not in profile_source


def test_only_one_normal_arena_cli_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "tools" / "arena.py").is_file()
    assert not (root / "tools" / "torus9_arena.py").exists()
    parser = arena_cli.build_parser()
    profile_action = next(action for action in parser._actions if action.dest == "profile")
    assert profile_action.default == "auto"
    assert "torus9" in profile_action.choices


def test_torus9_profile_registry_has_no_second_executor():
    assert available_profiles() == ("torus9",)
    profile = get_profile("torus9")
    assert isinstance(profile, Torus9ArenaProfile)
    assert profile.profile_id == "torus9"
    assert profile.observation_shape == (6, 81)
    assert profile.policy_size == 82
    assert profile.wdl_size == 3


def test_old_torus9_arena_executors_are_not_public_package_api():
    assert not hasattr(gocube_golden, "run_torus9_arena")
    assert not hasattr(gocube_golden, "run_torus9_batched_arena")
    assert not hasattr(gocube_golden, "Torus9BatchedPUCT")


def test_direct_old_torus9_arena_calls_are_fail_closed(monkeypatch):
    monkeypatch.delenv(policy.FROZEN_ARENA_OVERRIDE_ENV, raising=False)
    with pytest.raises(policy.ArenaPolicyError, match="frozen"):
        torus9_impl.run_torus9_arena()
    with pytest.raises(policy.ArenaPolicyError, match="frozen"):
        torus9_impl.run_torus9_batched_arena()


def test_production_defaults_encode_real_parallelism_and_batching_contract():
    config = arena_engine.ArenaExecutionConfig()
    config.validate_base()
    get_profile("torus9").validate_execution_config(config)
    assert config.games == 64
    assert config.workers == 16
    assert config.games_per_worker == 4
    assert config.inference_batch_rows == 64
    assert config.inference_batch_wait_ms == 1.0
    assert config.device == "cuda"
    assert config.strict_production is True
    assert config.min_mean_inference_batch_rows == 16.0
    assert config.min_effective_cpu_cores == 8.0


def test_torus9_profile_rejects_degraded_production_shapes():
    profile = get_profile("torus9")
    with pytest.raises(ValueError, match="at least 64"):
        profile.validate_execution_config(arena_engine.ArenaExecutionConfig(games=32))
    with pytest.raises(ValueError, match="exactly 16"):
        profile.validate_execution_config(arena_engine.ArenaExecutionConfig(workers=8))
    with pytest.raises(ValueError, match="requires CUDA"):
        profile.validate_execution_config(arena_engine.ArenaExecutionConfig(device="cpu"))
    with pytest.raises(ValueError, match=">= 16"):
        profile.validate_execution_config(
            arena_engine.ArenaExecutionConfig(inference_batch_rows=8)
        )


def test_debug_config_is_explicit_and_can_be_small_cpu():
    config = arena_engine.ArenaExecutionConfig(
        games=2,
        workers=1,
        games_per_worker=1,
        inference_batch_rows=1,
        inference_batch_wait_ms=0.0,
        device="cpu",
        strict_production=False,
    )
    config.validate_base()
    get_profile("torus9").validate_execution_config(config)


def test_worker_has_no_model_loading_and_generic_parent_owns_inference():
    profile = get_profile("torus9")
    worker_source = inspect.getsource(profile.worker_main)
    engine_source = inspect.getsource(arena_engine.run_arena)
    assert "torus9_load_checkpoint" not in worker_source
    assert "torus9_model_from_metadata" not in worker_source
    assert "torch.cuda.is_initialized" in worker_source
    assert "profile.load_parent_model" in engine_source
    assert "multiprocessing" in engine_source
    assert "worker_pids" in engine_source


def test_inference_wait_collects_before_profile_forward():
    source = inspect.getsource(arena_engine.run_arena)
    assert "_ModelAwareBatchScheduler" in source
    assert "scheduler.next_deadline()" in source
    assert "scheduler.next_ready_model" in source
    assert "dispatch_model_batch" in source
    assert "profile.infer_batch" in source


def test_frozen_operational_entrypoints_require_explicit_override():
    assert policy.FROZEN_ARENA_OVERRIDE_FLAG == "--allow-frozen-arena"
    for module in (continuation, ownership_ab, alpha_score_ab, komi_calibration):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "FROZEN_ARENA_OVERRIDE_FLAG" in source
        assert "FROZEN_ARENA_OVERRIDE_ENV" in source
        assert "tools/torus9_arena.py" not in source


def test_frozen_programmatic_entrypoints_stay_blocked():
    with pytest.raises(policy.ArenaPolicyError):
        continuation.run()
    with pytest.raises(policy.ArenaPolicyError):
        ownership_ab.run_experiment(None)
    with pytest.raises(policy.ArenaPolicyError):
        alpha_score_ab.run_experiment(None)
    with pytest.raises(policy.ArenaPolicyError):
        komi_calibration.main()


def test_komi_calibration_keeps_read_only_analysis_api_available():
    assert callable(komi_calibration.bootstrap_estimates)
    assert callable(komi_calibration.crossing_interval)
    assert callable(komi_calibration.crossing_point)
    assert callable(komi_calibration.sweep_trajectories)
