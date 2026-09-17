from __future__ import annotations

import pytest

from tools.arena_engine import ArenaExecutionConfig, classify_arena_performance
from tools.torus9_run_driver import _apply_arena_performance_policy


@pytest.mark.parametrize(
    ("mean_batch", "status", "hard_failures", "warnings"),
    (
        (7.0, "SEVERE_WARNING", [], ["mean_inference_batch_rows"]),
        (10.0, "WARNING", [], ["mean_inference_batch_rows"]),
        (13.25, "HEALTHY", [], []),
        (14.0, "HEALTHY", [], []),
    ),
)
def test_standard_64_arena_performance_policy_boundaries(
    mean_batch: float,
    status: str,
    hard_failures: list[str],
    warnings: list[str],
):
    config = ArenaExecutionConfig(
        games=64,
        workers=16,
        games_per_worker=4,
        inference_batch_rows=64,
        inference_batch_wait_ms=1.0,
        device="cpu",
        strict_production=True,
    )

    policy = classify_arena_performance(mean_batch, config)

    assert policy["status"] == status
    assert policy["hard_failures"] == hard_failures
    assert policy["warnings"] == warnings
    assert policy["severe_warning_threshold"] == 9.0
    assert policy["healthy_minimum"] == 13.25


def test_completed_arena_summary_is_reclassified_without_rerunning_games():
    config = ArenaExecutionConfig(
        games=64,
        workers=16,
        games_per_worker=4,
        inference_batch_rows=64,
        inference_batch_wait_ms=1.0,
        device="cpu",
        strict_production=True,
    )
    summary = {
        "games": 64,
        "technical_games": 0,
        "telemetry": {
            "mean_inference_batch_rows": 14.17,
            "performance_status": "PERFORMANCE_DEGRADED",
            "performance_failures": ["mean_inference_batch_rows"],
        },
    }

    normalized, policy = _apply_arena_performance_policy(summary, config)

    assert normalized is not summary
    assert policy["status"] == "HEALTHY"
    assert normalized["telemetry"]["performance_status"] == "HEALTHY"
    assert normalized["telemetry"]["performance_failures"] == []
    assert normalized["telemetry"]["performance_warnings"] == []
    assert summary["telemetry"]["performance_status"] == "PERFORMANCE_DEGRADED"


def test_completed_low_batch_arena_is_accepted_as_severe_warning():
    config = ArenaExecutionConfig(
        games=64,
        workers=16,
        games_per_worker=4,
        inference_batch_rows=64,
        inference_batch_wait_ms=1.0,
        device="cpu",
        strict_production=True,
    )
    summary = {
        "games": 64,
        "technical_games": 0,
        "telemetry": {
            "mean_inference_batch_rows": 7.0,
            "performance_status": "CRITICAL",
            "performance_failures": ["mean_inference_batch_rows"],
        },
    }

    normalized, policy = _apply_arena_performance_policy(summary, config)

    assert policy["status"] == "SEVERE_WARNING"
    assert normalized["telemetry"]["performance_status"] == "SEVERE_WARNING"
    assert normalized["telemetry"]["performance_failures"] == []
    assert normalized["telemetry"]["performance_warnings"] == [
        "mean_inference_batch_rows"
    ]
    assert normalized["telemetry"]["performance_gate"]["mean_inference_batch_rows"] == 7.0
