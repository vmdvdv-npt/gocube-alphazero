from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import gocube_golden
import gocube_golden.arena_policy as policy
import gocube_golden.torus9 as torus9_impl
import tools.continue_torus9_golden_m1_m100 as continuation
import tools.torus9_alpha_score_ab as alpha_score_ab
import tools.torus9_arena as production_arena
import tools.torus9_komi_calibration as komi_calibration
import tools.torus9_ownership_ab as ownership_ab


def test_torus9_has_exactly_one_production_arena_policy():
    production = [engine for engine in policy.TORUS9_ARENA_ENGINES if engine.production_allowed]
    assert len(production) == 1
    assert production[0].symbol == "tools.torus9_arena.run_arena"
    assert policy.CANONICAL_TORUS9_ARENA_ENGINE == "torus9-process-central-inference-v1"
    assert policy.CURRENT_TORUS9_PRODUCTION_ARENA_READY is True
    assert policy.CANONICAL_TORUS9_WORKERS == 16
    assert policy.CANONICAL_TORUS9_KOMI == 0.5
    policy.require_current_torus9_production_ready()


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
    config = production_arena.ArenaExecutionConfig()
    config.validate()
    assert config.games == 64
    assert config.workers == 16
    assert config.games_per_worker == 4
    assert config.inference_batch_rows == 64
    assert config.inference_batch_wait_ms == 4.0
    assert config.device == "cuda"
    assert config.strict_production is True
    assert config.min_mean_inference_batch_rows == 16.0
    assert config.min_effective_cpu_cores == 8.0


def test_production_config_rejects_degraded_execution_shapes():
    with pytest.raises(ValueError, match="at least 64"):
        production_arena.ArenaExecutionConfig(games=32).validate()
    with pytest.raises(ValueError, match="exactly 16"):
        production_arena.ArenaExecutionConfig(workers=8).validate()
    with pytest.raises(ValueError, match="requires CUDA"):
        production_arena.ArenaExecutionConfig(device="cpu").validate()
    with pytest.raises(ValueError, match=">= 16"):
        production_arena.ArenaExecutionConfig(inference_batch_rows=8).validate()


def test_debug_config_is_explicit_and_can_be_small_cpu():
    production_arena.ArenaExecutionConfig(
        games=2,
        workers=1,
        games_per_worker=1,
        inference_batch_rows=1,
        inference_batch_wait_ms=0.0,
        device="cpu",
        strict_production=False,
    ).validate()


def test_worker_has_no_model_loading_and_parent_is_model_owner():
    worker_source = inspect.getsource(production_arena._worker_main)
    parent_source = inspect.getsource(production_arena.run_arena)
    assert "torus9_load_checkpoint" not in worker_source
    assert "torus9_model_from_metadata" not in worker_source
    assert "torch.cuda.is_initialized" in worker_source
    assert "_load_parent_model" in parent_source
    assert "multiprocessing" in parent_source
    assert "worker_pids" in parent_source


def test_inference_wait_collects_before_forward_not_after_chunk_creation():
    source = inspect.getsource(production_arena.run_arena)
    deadline_index = source.index("deadline = time.perf_counter()")
    queue_get_index = source.index("extra = request_queue.get", deadline_index)
    forward_index = source.index("logits, wdl_logits = model", queue_get_index)
    assert deadline_index < queue_get_index < forward_index


def test_frozen_operational_entrypoints_require_explicit_override():
    assert policy.FROZEN_ARENA_OVERRIDE_FLAG == "--allow-frozen-arena"
    for module in (continuation, ownership_ab, alpha_score_ab, komi_calibration):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert policy.FROZEN_ARENA_OVERRIDE_FLAG in source
        assert policy.FROZEN_ARENA_OVERRIDE_ENV in source


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
