from __future__ import annotations

from tools.arena_engine import _ModelAwareBatchScheduler


def _request(model_hash: str, rows: int, enqueued_at: float) -> dict[str, object]:
    return {
        "model_hash": model_hash,
        "rows": rows,
        "enqueued_at": enqueued_at,
    }


def test_model_aware_scheduler_never_mixes_models_and_keeps_independent_deadlines():
    scheduler = _ModelAwareBatchScheduler(("candidate", "reference"), cap=4, wait_ms=10.0)
    scheduler.enqueue(_request("candidate", 2, 0.0))
    scheduler.enqueue(_request("reference", 2, 0.0))
    assert scheduler.pending_rows() == 4
    assert scheduler.next_ready_model(0.005) is None

    scheduler.enqueue(_request("candidate", 2, 0.001))
    assert scheduler.next_ready_model(0.005) == "candidate"
    candidate_batch, _ = scheduler.pop_batch("candidate")
    assert [request["model_hash"] for request in candidate_batch] == [
        "candidate",
        "candidate",
    ]
    assert scheduler.pending_rows_by_model() == {"candidate": 0, "reference": 2}

    # The reference queue is not forced to wait for candidate traffic: its
    # own deadline makes it ready independently and prevents starvation.
    assert scheduler.next_ready_model(0.011) == "reference"
    reference_batch, _ = scheduler.pop_batch("reference")
    assert [request["model_hash"] for request in reference_batch] == ["reference"]


def test_model_aware_scheduler_round_robins_ready_models():
    scheduler = _ModelAwareBatchScheduler(("candidate", "reference"), cap=2, wait_ms=0.0)
    for model_hash in ("candidate", "reference", "candidate", "reference"):
        scheduler.enqueue(_request(model_hash, 1, 0.0))

    dispatch_order = []
    for _ in range(2):
        model_hash = scheduler.next_ready_model(0.0)
        assert model_hash is not None
        batch, _ = scheduler.pop_batch(model_hash)
        dispatch_order.append((model_hash, [request["model_hash"] for request in batch]))
    assert dispatch_order == [
        ("candidate", ["candidate", "candidate"]),
        ("reference", ["reference", "reference"]),
    ]
