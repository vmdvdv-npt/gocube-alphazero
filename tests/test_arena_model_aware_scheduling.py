from __future__ import annotations

from collections import deque
from queue import Queue
import threading
import time
from types import SimpleNamespace

import pytest

from tools.arena_engine import (
    _ArenaBrokerIngress,
    _ModelAwareBatchScheduler,
    _WorkerTaskQueue,
    _report_progress_from_activity,
    _unreported_worker_exits,
)
from tools.arena_worker import _ImmediateInferenceTransport
from tools import arena_worker
from tools.arena_profiles import torus9 as torus9_profile
from gocube_golden.search import SearchEvaluationRequest, SearchResult
from gocube_golden.state import BLACK


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


def test_worker_transport_is_immediate_and_rejects_a_second_timed_window():
    central = Queue()
    transport = _ImmediateInferenceTransport(
        worker_id=0,
        central_queue=central,
        local_cap=4,
        wait_ms=0.0,
    )
    request = {"kind": "inference", "worker_enqueued_at": 12.5}
    try:
        transport.put(request)
        assert central.get(timeout=1.0) == request
    finally:
        transport.close()

    with pytest.raises(ValueError, match="wait_ms=0"):
        _ImmediateInferenceTransport(
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


def test_move_activity_refreshes_supervision_progress_without_incrementing_games():
    progress: list[tuple[int, int]] = []

    callback = lambda completed, total: progress.append((completed, total))
    _report_progress_from_activity("move_completed", 0, 1024, callback)
    _report_progress_from_activity("game_started", 0, 1024, callback)
    _report_progress_from_activity("game_completed", 1, 1024, callback)

    assert progress == [(0, 1024), (1, 1024)]


def test_worker_exit_without_done_message_fails_after_short_grace():
    process = SimpleNamespace(name="arena-worker-00", exitcode=0, is_alive=lambda: False)
    dead_since: dict[str, float] = {}

    assert _unreported_worker_exits([], {}, dead_since, now=0.0) == []
    assert _unreported_worker_exits(
        [process], {}, dead_since, now=0.0
    ) == []
    assert _unreported_worker_exits(
        [process], {}, dead_since, now=5.0
    ) == ["arena-worker-00 (exitcode=0)"]


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


def test_arena_worker_interleaves_blocked_lanes_and_replenishes(monkeypatch):
    """A delayed response in one lane must not serialize the other lanes."""

    class FakeState:
        side_to_move = BLACK
        is_terminal = False
        stones = ()

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            self.waiting = True

        def advance(self):
            if self.waiting:
                return SearchEvaluationRequest(FakeState(), object())
            return SearchResult(
                action=0,
                legal_actions=(0,),
                root_visits=(1,),
                pi=(1.0,),
                simulations=1,
                evaluator_calls=1,
            )

        def resume(self, _evaluation):
            self.waiting = False
            return self.advance()

    def fake_apply_action(_state, _action):
        return SimpleNamespace(
            after=SimpleNamespace(
                is_terminal=True,
                stones=(),
                side_to_move=torus9_profile.WHITE,
            )
        )

    def fake_finish(game):
        return {"game_id": str(game.task["game_id"]), "action_trace": game.trace}

    monkeypatch.setattr(arena_worker, "SequentialPUCTSession", FakeSession)
    monkeypatch.setattr(torus9_profile, "apply_action", fake_apply_action)
    monkeypatch.setattr(
        torus9_profile,
        "result_from_terminal",
        lambda _state: SimpleNamespace(winner=SimpleNamespace(value="DRAW")),
    )
    monkeypatch.setattr(torus9_profile, "_finish_game", fake_finish)
    monkeypatch.setattr(
        torus9_profile,
        "build_torus9_observation",
        lambda _state, legal_context=None: torus9_profile.torch.zeros((6, 81)),
    )
    monkeypatch.setattr(
        torus9_profile,
        "_make_game",
        lambda task: torus9_profile._WorkerGame(
            task=task,
            state=FakeState(),
            trace=[],
            ply=0,
            started_at=time.perf_counter(),
        ),
    )

    tasks = Queue()
    for index in range(4):
        tasks.put({
            "game_id": f"game-{index}",
            "state": None,
            "game_seed": index,
            "candidate_black": True,
        })
    requests = Queue()
    responses = [Queue(), Queue()]
    input_slot = torus9_profile.torch.zeros((2, 6, 81))
    policy_slot = torus9_profile.torch.zeros((2, 82))
    wdl_slot = torus9_profile.torch.zeros((2, 3))
    start_event = threading.Event()
    start_event.set()
    seen: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    stop_broker = threading.Event()

    def broker() -> None:
        while not stop_broker.is_set():
            try:
                message = requests.get(timeout=0.05)
            except Exception:
                continue
            if message.get("kind") == "inference":
                seen.append(message)

                def respond(message=message) -> None:
                    policy_slot[int(message["lane_id"])].fill_(1.0)
                    wdl_slot[int(message["lane_id"])].fill_(1.0)
                    responses[int(message["lane_id"])].put(
                        {
                            "worker_id": 0,
                            "lane_id": int(message["lane_id"]),
                            "ticket": int(message["ticket"]),
                            "generation": int(message["generation"]),
                            "game_id": message["game_id"],
                            "model_role": message["model_role"],
                            "model_hash": message["model_hash"],
                            "error": None,
                        }
                    )

                if int(message["lane_id"]) == 0:
                    threading.Timer(0.05, respond).start()
                else:
                    respond()
            elif message.get("kind") == "done":
                events.append(message)
                return
            else:
                events.append(message)

    broker_thread = threading.Thread(target=broker)
    worker_thread = threading.Thread(
        target=torus9_profile.PROFILE.worker_main,
        args=(
            0,
            tasks,
            2,
            0.0,
            "candidate-hash",
            "reference-hash",
            input_slot,
            policy_slot,
            wdl_slot,
            requests,
            responses,
            start_event,
        ),
    )
    broker_thread.start()
    worker_thread.start()
    worker_thread.join(timeout=3.0)
    stop_broker.set()
    broker_thread.join(timeout=1.0)

    assert not worker_thread.is_alive()
    assert len(seen) == 4
    assert {int(message["lane_id"]) for message in seen} == {0, 1}
    assert [message["game_id"] for message in seen[:2]] == ["game-0", "game-1"]
    completed = [
        int(message["lane_id"])
        for message in events
        if message.get("kind") == "activity" and message.get("event") == "game_completed"
    ]
    assert completed[0] == 1
    done = next(message for message in events if message.get("kind") == "done")
    assert done["used_lane_ids"] == [0, 1]
    assert done["lane_replenishments"] == 2
