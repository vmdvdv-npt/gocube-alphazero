from __future__ import annotations

from tools.torus9_wait_benchmark import (
    BASE_VARIANT_IDS,
    _correctness_gate,
    _decision,
    _read_spec,
    _shuffled_base_order,
)


def _row(wait_ms: float, moves_per_sec: float, *, digest: str = "same", numerical_status: str = "PASS") -> dict[str, object]:
    return {
        "wait_ms": wait_ms,
        "moves_per_sec": moves_per_sec,
        "wall_time_sec": 100.0,
        "games_requested": 64,
        "completed_games": 64,
        "technical_games": 0,
        "normalized_game_digests": {"game-0000": digest},
        "numerical_divergence": {"status": numerical_status},
    }


def test_run_spec_freezes_the_controlled_wait_matrix():
    spec = _read_spec()
    assert tuple(item["id"] for item in spec["variants"]) == BASE_VARIANT_IDS
    assert spec["execution"] == {
        "workers": 16,
        "active_games_per_worker": 4,
        "total_active_contexts": 64,
        "inference_batch_cap": 64,
        "coalescing": True,
        "central_model_owner": "parent",
        "shared_memory": True,
        "device": "cuda",
    }


def test_variant_order_is_deterministically_shuffled():
    order = _shuffled_base_order(_read_spec())
    assert order != list(BASE_VARIANT_IDS)
    assert sorted(order) == sorted(BASE_VARIANT_IDS)


def test_decision_keeps_golden_when_gain_is_below_material_threshold():
    result = _decision({
        "wait-0.5ms": _row(0.5, 100.0),
        "wait-1.0ms": _row(1.0, 98.5),
        "wait-2.0ms": _row(2.0, 99.0),
    })
    assert result["recommendation"] == "NO MATERIAL IMPROVEMENT — KEEP 1 ms"
    assert result["wait_0_ran"] is False


def test_decision_runs_zero_wait_only_after_large_first_gain():
    result = _decision({
        "wait-0.5ms": _row(0.5, 106.0),
        "wait-1.0ms": _row(1.0, 100.0),
        "wait-2.0ms": _row(2.0, 101.0),
        "wait-0.0ms": _row(0.0, 107.0),
    })
    assert result["wait_0_ran"] is True
    assert result["winner_id"] == "wait-0.0ms"
    assert result["recommendation"] == "RECOMMEND GOLDEN WAIT = 0 ms"


def test_correctness_gate_allows_non_bit_exact_digest_with_bounded_fp32_variance():
    baseline = _row(2.0, 100.0, digest="baseline")
    current = _row(0.5, 100.0, digest="batch-shape-variant")
    gate = _correctness_gate(
        {"wait-0.5ms": current},
        baseline_id="wait-2.0ms",
        baseline_result=baseline,
    )
    assert gate["status"] == "PASS"
    assert gate["bit_exact_equality_required"] is False
    assert gate["digest_comparison_advisory"]["status"] == "EXPECTED_VARIANCE_ALLOWED"


def test_correctness_gate_fails_unexplained_numerical_divergence():
    baseline = _row(2.0, 100.0)
    current = _row(1.0, 100.0, numerical_status="FAIL")
    gate = _correctness_gate(
        {"wait-1.0ms": current},
        baseline_id="wait-2.0ms",
        baseline_result=baseline,
    )
    assert gate["status"] == "FAIL"
    assert gate["numerical_gate"]["status"] == "FAIL"
