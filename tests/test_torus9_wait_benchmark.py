from __future__ import annotations

from tools.torus9_wait_benchmark import (
    BASE_VARIANT_IDS,
    _decision,
    _read_spec,
    _shuffled_base_order,
)


def _row(wait_ms: float, moves_per_sec: float) -> dict[str, object]:
    return {"wait_ms": wait_ms, "moves_per_sec": moves_per_sec, "wall_time_sec": 100.0, "normalized_game_digests": {}}


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
