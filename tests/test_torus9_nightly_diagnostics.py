from __future__ import annotations

import pytest

from tools.torus9_nightly_diagnostics import (
    ARENA_STANDARD_64,
    CADENCE_ARMS,
    CadenceTrainingAdapter,
    standard_arena_config,
)


def test_cadence_arms_use_equal_cumulative_budget() -> None:
    assert [arm.games for arm in CADENCE_ARMS] == [64, 128, 192]
    assert [arm.iterations for arm in CADENCE_ARMS] == [6, 3, 2]
    assert [arm.optimizer_steps for arm in CADENCE_ARMS] == [80, 160, 240]
    assert {arm.total_games for arm in CADENCE_ARMS} == {384}
    assert {arm.total_optimizer_steps for arm in CADENCE_ARMS} == {480}
    assert {arm.sample_draws * arm.iterations for arm in CADENCE_ARMS} == {30720}


def test_strength_arena_contract_is_standard_64() -> None:
    config = standard_arena_config()
    assert config.games == 64
    assert config.workers == 16
    assert config.games_per_worker == 4
    assert config.inference_batch_rows == 64
    assert config.inference_batch_wait_ms == 1.0
    assert config.strict_production is False
    assert ARENA_STANDARD_64["paired_starts"] == 32
    assert ARENA_STANDARD_64["total_active_contexts"] == 64
    assert ARENA_STANDARD_64["simulations"] == 64
    assert ARENA_STANDARD_64["cpuct"] == 1.25
    assert ARENA_STANDARD_64["fpu"] == 0.0
    assert ARENA_STANDARD_64["root_noise"] is False
    assert ARENA_STANDARD_64["temperature"] == 0.0
    assert ARENA_STANDARD_64["fast_search"] is False
    assert ARENA_STANDARD_64["resign"] is False
    assert ARENA_STANDARD_64["frozen_startset"] is True
    assert ARENA_STANDARD_64["paired_color_swap"] is True
    assert ARENA_STANDARD_64["deterministic_tie_break"] is True
    assert ARENA_STANDARD_64["komi"] == 0.5


def test_cadence_adapter_rejects_unplanned_optimizer_budget() -> None:
    with pytest.raises(ValueError, match="80, 160, or 240"):
        CadenceTrainingAdapter(optimizer_steps=81)
