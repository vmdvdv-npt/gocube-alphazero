from __future__ import annotations

from collections import deque
from queue import Queue
import threading
import time

import pytest

from tools.arena_engine import (
    _ArenaBrokerIngress,
    _ModelAwareBatchScheduler,
    _WorkerTaskQueue,
)
from tools.arena_profiles.torus9 import _WorkerInferenceAggregator


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


def test_worker_transport_is_immediate_and_rejects_a_second_timed_window():
    central = Queue()
    transport = _WorkerInferenceAggregator(
        worker_id=0,
        central_queue=central,
        local_cap=4,
        wait_ms=0.0,
    )
    request = {"kind": "inference", "worker_enqueued_at": 12.5}
    try:
        transport.put(request)
        assert central.get(timeout=1.0) == request
        assert transport.wait_ms == 0.0
    finally:
        transport.close()

    with pytest.raises(ValueError, match="wait_ms=0"):
        _WorkerInferenceAggregator(
            worker_id=0,
            central_queue=central,
            local_cap=4,
            wait_ms=4.0,
        )


def test_worker_task_queue_uses_global_replenishment_after_initial_lane_fill():
    initial = Queue()
    global_queue = Queue()
    initial.put({"task": "initial"})
    global_queue.put({"task": "replenished"})
    task_queue = _WorkerTaskQueue(initial, global_queue)

    assert task_queue.get(timeout=1.0) == {"task": "initial"}
    assert task_queue.get(timeout=1.0) == {"task": "replenished"}


def test_broker_ingress_accepts_requests_during_a_slow_forward_and_preserves_timestamp():
    request_queue = Queue()
    control_pending = deque()
    scheduler = _ModelAwareBatchScheduler(("candidate",), cap=1, wait_ms=100.0)
    ingress = _ArenaBrokerIngress(request_queue, scheduler, control_pending)
    forward_started = threading.Event()
    release_forward = threading.Event()
    dispatch_error: list[BaseException] = []

    def slow_forward() -> None:
        try:
            with scheduler.condition:
                while scheduler.next_ready_model(time.perf_counter()) is None:
                    scheduler.condition.wait(timeout=1.0)
                # The model owner removes the first batch, then blocks in its
                # fake forward. Ingress must keep filling the pending queue.
                scheduler.pop_batch("candidate")
            forward_started.set()
            if not release_forward.wait(timeout=1.0):
                raise AssertionError("fake model forward was not released")
        except BaseException as exc:
            dispatch_error.append(exc)

    ingress.start()
    try:
        request_queue.put(
            {
                "kind": "inference",
                "model_hash": "candidate",
                "rows": 1,
                "worker_enqueued_at": 12.5,
            }
        )
        dispatcher = threading.Thread(target=slow_forward)
        dispatcher.start()
        assert forward_started.wait(timeout=1.0)

        request_queue.put(
            {
                "kind": "inference",
                "model_hash": "candidate",
                "rows": 1,
                "worker_enqueued_at": 12.75,
            }
        )
        deadline = time.monotonic() + 1.0
        while scheduler.pending_rows() < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert scheduler.pending_rows_by_model() == {"candidate": 1}
        pending = scheduler.queues["candidate"].requests[0]
        assert pending["worker_enqueued_at"] == 12.75
        assert float(pending["broker_received_at"]) > 0.0
    finally:
        release_forward.set()
        if "dispatcher" in locals():
            dispatcher.join(timeout=1.0)
        ingress.stop()
    assert not dispatch_error
