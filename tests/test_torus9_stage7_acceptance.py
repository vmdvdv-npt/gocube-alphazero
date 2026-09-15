from __future__ import annotations

from pathlib import Path

import pytest

from tools.torus9_stage7_e2e_validation import (
    BASE_SHA,
    EXECUTION,
    VALIDATION_HORIZON,
    Stage7ArenaConfig,
    _assert_isolated,
    _safe_run_id,
    paired_block_bootstrap,
)


def test_stage7_uses_post_108_base_and_six_iteration_horizon() -> None:
    assert BASE_SHA == "d0b81a51f46ef8a22dca75fac04d07cfad26d9eb"
    assert VALIDATION_HORIZON == 6
    assert EXECUTION.as_dict() == {
        "workers": 16,
        "active_games_per_worker": 4,
        "active_contexts": 64,
        "inference_batch_cap": 64,
        "inference_batch_wait_ms": 1.0,
        "worker_local_wait_ms": 0.0,
        "shared_memory": True,
        "central_model_owner": "parent",
        "device": "cuda",
    }


def test_stage7_arena_contract_is_fixed() -> None:
    config = Stage7ArenaConfig(games=256)
    config.validate()
    assert config.arena_batch_size == 8
    assert config.inference_batch_wait_ms == 6.0
    with pytest.raises(ValueError):
        Stage7ArenaConfig(games=65).validate()


def test_stage7_paired_bootstrap_is_deterministic_and_non_inferiority_aware() -> None:
    result = paired_block_bootstrap((1.0, 1.0, 1.0, 1.0), seed=123, replicates=2000)
    repeat = paired_block_bootstrap((1.0, 1.0, 1.0, 1.0), seed=123, replicates=2000)
    assert result == repeat
    assert result["paired_blocks"] == 4
    assert result["non_inferiority_margin"] == 0.05
    assert result["pass_threshold_difference"] == -0.05
    assert result["status"] == "PASS"


def test_stage7_run_id_and_canonical_namespace_guards(tmp_path: Path) -> None:
    assert _safe_run_id("run-01") == "run-01"
    with pytest.raises(ValueError):
        _safe_run_id("../escape")
    canonical = tmp_path / "canonical"
    with pytest.raises(ValueError):
        _assert_isolated(canonical, canonical)
    with pytest.raises(ValueError):
        _assert_isolated(canonical / "stage7", canonical)
