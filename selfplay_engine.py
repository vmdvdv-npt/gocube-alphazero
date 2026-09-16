"""Game-agnostic process self-play execution.

The scientific game adapter is deliberately kept out of this module.  The
engine owns process scheduling, shared-memory transport, central batching and
failure handling.  ``InferenceClient.request`` remains available for older
adapters, while the production path uses ``SharedMemorySpec`` and a
cooperative game factory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import multiprocessing as mp
import os
from queue import Empty
import statistics
import threading
import time
from typing import Any, Callable, Mapping, MutableMapping, Sequence


_STOP = ("__SELFPLAY_ENGINE_STOP__",)
_TASK_STOP = ("__SELFPLAY_ENGINE_TASK_STOP__",)


class SelfPlayEngineError(RuntimeError):
    """Fail-closed execution error; partial results must not be consumed."""


class InferenceTransportError(BaseException):
    """Infrastructure error that must escape game-level ``except Exception``."""


@dataclass(frozen=True)
class SelfPlayEngineConfig:
    workers: int
    inference_batch_cap: int
    inference_batch_wait_ms: float
    device: str
    process_start_method: str = "spawn"
    inference_request_timeout_s: float = 300.0
    worker_join_timeout_s: float = 15.0
    # Deprecated compatibility setting. It is used only by the legacy payload
    # callback path. Shared-memory production execution uses
    # ``active_games_per_worker`` and does not create search threads.
    lanes_per_worker: int = 1
    active_games_per_worker: int = 1
    total_active_contexts: int | None = None

    def validate(self) -> None:
        if self.workers <= 0 or self.inference_batch_cap <= 0:
            raise ValueError("self-play workers and inference batch cap must be positive")
        if self.lanes_per_worker <= 0 or self.active_games_per_worker <= 0:
            raise ValueError("self-play lanes and active games per worker must be positive")
        if self.total_active_contexts is not None and self.total_active_contexts <= 0:
            raise ValueError("self-play total active contexts must be positive")
        if self.inference_batch_wait_ms < 0 or not math.isfinite(self.inference_batch_wait_ms):
            raise ValueError("self-play inference batch wait must be finite and non-negative")
        if self.inference_request_timeout_s <= 0 or not math.isfinite(self.inference_request_timeout_s):
            raise ValueError("self-play inference timeout must be positive and finite")
        if self.worker_join_timeout_s <= 0 or not math.isfinite(self.worker_join_timeout_s):
            raise ValueError("self-play worker join timeout must be positive and finite")
        if self.process_start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method: {self.process_start_method}")


@dataclass(frozen=True)
class SharedMemorySpec:
    """Adapter-owned dimensions and codecs for shared inference slots."""

    observation_shape: tuple[int, ...]
    policy_size: int
    wdl_size: int
    write_input: Callable[[object, Any], None]
    decode_output: Callable[[Any, Any], object]

    def validate(self) -> None:
        if not self.observation_shape or any(int(value) <= 0 for value in self.observation_shape):
            raise ValueError("shared inference observation shape must be positive")
        if self.policy_size <= 0 or self.wdl_size <= 0:
            raise ValueError("shared inference output dimensions must be positive")
        if not callable(self.write_input) or not callable(self.decode_output):
            raise ValueError("shared inference codecs must be callable")


@dataclass(frozen=True)
class SharedInferenceResult:
    """Optional parent-side timing envelope for a shared model forward."""

    policy: Any
    wdl: Any
    h2d_started_at: float | None = None
    h2d_finished_at: float | None = None
    forward_started_at: float | None = None
    forward_finished_at: float | None = None


@dataclass(frozen=True)
class InferenceNeed:
    """Yielded by a cooperative game when it needs one neural evaluation."""

    payload: object


@dataclass(frozen=True)
class GameFinished:
    """Returned by a cooperative game when its record is complete."""

    record: object


@dataclass(frozen=True)
class _Request:
    worker_id: int
    lane_id: int
    request_id: int
    payload: object | None = None
    slot_ids: tuple[int, ...] = ()
    worker_sent_at: float = 0.0
    broker_received_at: float | None = None

    @property
    def rows(self) -> int:
        return len(self.slot_ids) if self.slot_ids else 1


@dataclass(frozen=True)
class _Response:
    request_id: int
    result: object | None = None
    error: str | None = None
    slot_ids: tuple[int, ...] = ()
    broker_received_at: float | None = None
    dispatch_started_at: float | None = None
    output_ready_at: float | None = None
    response_signaled_at: float | None = None
    h2d_started_at: float | None = None
    h2d_finished_at: float | None = None
    forward_started_at: float | None = None
    forward_finished_at: float | None = None


class InferenceClient:
    """Worker-side inference transport.

    ``request`` is retained for older adapters. ``request_shared_batch`` sends
    only request metadata through multiprocessing; the data rows remain in
    shared tensors owned by the worker slot and central owner.
    """

    def __init__(
        self,
        worker_id: int,
        lane_id: int,
        requests: Any,
        responses: Any,
        timeout_s: float,
        *,
        events: Any | None = None,
        input_slots: Any | None = None,
        policy_slots: Any | None = None,
        wdl_slots: Any | None = None,
        shared_spec: SharedMemorySpec | None = None,
    ) -> None:
        self.worker_id = int(worker_id)
        self.lane_id = int(lane_id)
        self._requests = requests
        self._responses = responses
        self._timeout_s = float(timeout_s)
        self._events = events
        self._input_slots = input_slots
        self._policy_slots = policy_slots
        self._wdl_slots = wdl_slots
        self._shared_spec = shared_spec
        self._next_id = 0
        self._leased_slots: set[int] = set()

    def _wait(self, request: _Request, *, expected_slots: tuple[int, ...] = ()) -> _Response:
        try:
            response = self._responses.get(timeout=self._timeout_s)
        except Empty as exc:
            raise InferenceTransportError(
                f"inference request timed out (worker={self.worker_id}, request={request.request_id})"
            ) from exc
        if not isinstance(response, _Response):
            raise InferenceTransportError("malformed inference response envelope")
        if response.request_id != request.request_id:
            raise InferenceTransportError(
                f"inference response id mismatch: expected {request.request_id}, got {response.request_id}"
            )
        if tuple(response.slot_ids) != expected_slots:
            raise InferenceTransportError("inference response slot identity mismatch")
        if response.error is not None:
            raise InferenceTransportError(response.error)
        if not expected_slots and response.result is None:
            raise InferenceTransportError("inference response contained no result")
        resumed = time.perf_counter()
        if self._events is not None:
            self._events.put(
                (
                    "inference_resumed",
                    self.worker_id,
                    os.getpid(),
                    request.request_id,
                    request.worker_sent_at,
                    response.broker_received_at,
                    response.dispatch_started_at,
                    response.output_ready_at,
                    response.response_signaled_at,
                    response.h2d_started_at,
                    response.h2d_finished_at,
                    response.forward_started_at,
                    response.forward_finished_at,
                    resumed,
                    max(0.0, resumed - request.worker_sent_at),
                )
            )
        return response

    def request(self, payload: object) -> object:
        request_id = self._next_id
        self._next_id += 1
        request = _Request(
            self.worker_id,
            self.lane_id,
            request_id,
            payload=payload,
            worker_sent_at=time.perf_counter(),
        )
        self._requests.put(request)
        response = self._wait(request)
        return response.result

    def request_shared_batch(self, rows: Sequence[tuple[int, object]]) -> tuple[object, ...]:
        if self._shared_spec is None or self._input_slots is None:
            raise InferenceTransportError("shared-memory inference was not configured for this worker")
        if not rows:
            raise InferenceTransportError("shared-memory inference batch is empty")
        slots = tuple(int(slot) for slot, _payload in rows)
        if len(set(slots)) != len(slots) or any(slot < 0 or slot >= len(self._input_slots) for slot in slots):
            raise InferenceTransportError("shared-memory inference slot identity is invalid")
        if self._leased_slots.intersection(slots):
            raise InferenceTransportError("shared-memory inference slot is still owned by an active request")
        self._leased_slots.update(slots)
        try:
            for slot, payload in rows:
                self._shared_spec.write_input(payload, self._input_slots[int(slot)])
            request_id = self._next_id
            self._next_id += 1
            request = _Request(
                self.worker_id,
                0,
                request_id,
                slot_ids=slots,
                worker_sent_at=time.perf_counter(),
            )
            self._requests.put(request)
            response = self._wait(request, expected_slots=slots)
            if self._policy_slots is None or self._wdl_slots is None:
                raise InferenceTransportError("shared-memory inference output slots are missing")
            return tuple(
                self._shared_spec.decode_output(self._policy_slots[slot], self._wdl_slots[slot])
                for slot in slots
            )
        finally:
            self._leased_slots.difference_update(slots)


def _worker_lane_main(
    worker_id: int,
    lane_id: int,
    tasks: Any,
    requests: Any,
    responses: Any,
    events: Any,
    worker_play: Callable[[object, str, InferenceClient], object],
    worker_context: object,
    request_timeout_s: float,
) -> None:
    pid = os.getpid()
    client = InferenceClient(worker_id, lane_id, requests, responses, request_timeout_s, events=events)
    while True:
        task = tasks.get()
        if task == _TASK_STOP:
            events.put(("lane_stopped", worker_id, pid, lane_id))
            return
        game_id = str(task)
        cpu_clock = time.thread_time
        cpu_started = cpu_clock()
        events.put(("game_started", worker_id, pid, lane_id, game_id))
        try:
            record = worker_play(worker_context, game_id, client)
        except BaseException as exc:
            events.put(
                (
                    "game_failed",
                    worker_id,
                    pid,
                    lane_id,
                    game_id,
                    f"{type(exc).__name__}: {exc}",
                    max(0.0, cpu_clock() - cpu_started),
                )
            )
            return
        events.put(
            (
                "game_completed",
                worker_id,
                pid,
                lane_id,
                game_id,
                record,
                max(0.0, cpu_clock() - cpu_started),
            )
        )


def _worker_main(
    worker_id: int,
    lanes_per_worker: int,
    tasks: Any,
    requests: Any,
    responses: Sequence[Any],
    events: Any,
    worker_play: Callable[[object, str, InferenceClient], object],
    worker_context: object,
    request_timeout_s: float,
) -> None:
    pid = os.getpid()
    process_cpu_started = time.process_time()
    events.put(("worker_started", worker_id, pid))
    lanes = [
        threading.Thread(
            target=_worker_lane_main,
            name=f"selfplay-compat-lane-{worker_id:02d}-{lane_id:02d}",
            args=(
                worker_id,
                lane_id,
                tasks,
                requests,
                responses[lane_id],
                events,
                worker_play,
                worker_context,
                request_timeout_s,
            ),
        )
        for lane_id in range(lanes_per_worker)
    ]
    for lane in lanes:
        lane.start()
    for lane in lanes:
        lane.join()
    events.put(("worker_stopped", worker_id, pid, max(0.0, time.process_time() - process_cpu_started)))


def _shared_worker_main(
    worker_id: int,
    max_contexts: int,
    tasks: Any,
    requests: Any,
    response_queue: Any,
    events: Any,
    worker_context: object,
    request_timeout_s: float,
    game_factory: Callable[[object, str, InferenceClient], object],
    input_slots: Any,
    policy_slots: Any,
    wdl_slots: Any,
    shared_spec: SharedMemorySpec,
) -> None:
    """Run several independent games by round-robin cooperative stepping."""
    pid = os.getpid()
    process_cpu_started = time.process_time()
    events.put(("worker_started", worker_id, pid))
    client = InferenceClient(
        worker_id,
        0,
        requests,
        response_queue,
        request_timeout_s,
        events=events,
        input_slots=input_slots,
        policy_slots=policy_slots,
        wdl_slots=wdl_slots,
        shared_spec=shared_spec,
    )
    active: dict[int, tuple[str, object, float]] = {}
    stop_seen = False

    def add_task(task: object, slot: int) -> None:
        game_id = str(task)
        game = game_factory(worker_context, game_id, client)
        active[slot] = (game_id, game, 0.0)
        events.put(("game_started", worker_id, pid, slot, game_id))

    def fill_slots() -> None:
        nonlocal stop_seen
        while len(active) < max_contexts and not stop_seen:
            try:
                task = tasks.get_nowait()
            except Empty:
                break
            if task == _TASK_STOP:
                stop_seen = True
                break
            free = next(slot for slot in range(max_contexts) if slot not in active)
            add_task(task, free)

    try:
        while True:
            fill_slots()
            if not active:
                if stop_seen:
                    break
                try:
                    task = tasks.get(timeout=0.1)
                except Empty:
                    continue
                if task == _TASK_STOP:
                    stop_seen = True
                    continue
                add_task(task, 0)

            pending: list[tuple[int, object]] = []
            for slot in tuple(sorted(active)):
                game_id, game, cpu_seconds = active[slot]
                cpu_started = time.thread_time()
                try:
                    outcome = game.advance()
                except BaseException as exc:
                    events.put(("game_failed", worker_id, pid, slot, game_id, f"{type(exc).__name__}: {exc}", cpu_seconds + max(0.0, time.thread_time() - cpu_started)))
                    return
                cpu_seconds += max(0.0, time.thread_time() - cpu_started)
                active[slot] = (game_id, game, cpu_seconds)
                if isinstance(outcome, InferenceNeed):
                    pending.append((slot, outcome.payload))
                elif isinstance(outcome, GameFinished):
                    del active[slot]
                    events.put(("game_completed", worker_id, pid, slot, game_id, outcome.record, cpu_seconds))
                else:
                    events.put(("game_failed", worker_id, pid, slot, game_id, f"RuntimeError: cooperative game returned {type(outcome).__name__}", cpu_seconds))
                    return

            if pending:
                try:
                    evaluations = client.request_shared_batch(pending)
                except BaseException as exc:
                    slot = pending[0][0]
                    game_id = str(active[slot][0]) if slot in active else "unknown"
                    cpu_seconds = active[slot][2] if slot in active else 0.0
                    events.put(("game_failed", worker_id, pid, slot, game_id, f"{type(exc).__name__}: {exc}", cpu_seconds))
                    return
                for (slot, _payload), evaluation in zip(pending, evaluations):
                    if slot not in active:
                        raise InferenceTransportError("cooperative response referenced an inactive context")
                    game_id, game, cpu_seconds = active[slot]
                    cpu_started = time.thread_time()
                    try:
                        game.resume(evaluation)
                    except BaseException as exc:
                        events.put(("game_failed", worker_id, pid, slot, game_id, f"{type(exc).__name__}: {exc}", cpu_seconds + max(0.0, time.thread_time() - cpu_started)))
                        return
                    active[slot] = (game_id, game, cpu_seconds + max(0.0, time.thread_time() - cpu_started))
    finally:
        events.put(("worker_stopped", worker_id, pid, max(0.0, time.process_time() - process_cpu_started)))


def _percentile(values: Sequence[float | int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def _summary(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))],
        "max": max(ordered),
    }


class _CentralInference:
    """Dedicated ingress plus single-threaded model dispatcher."""

    def __init__(
        self,
        requests: Any,
        responses: Sequence[Sequence[Any]],
        infer_batch: Callable[[Sequence[object]], Sequence[object]] | None,
        *,
        batch_cap: int,
        wait_ms: float,
        device: str,
        shared_spec: SharedMemorySpec | None = None,
        infer_shared_batch: Callable[[Any], object] | None = None,
        shared_inputs: Sequence[Any] | None = None,
        shared_policy: Sequence[Any] | None = None,
        shared_wdl: Sequence[Any] | None = None,
    ) -> None:
        self.requests = requests
        self.responses = tuple(tuple(row) for row in responses)
        self.infer_batch = infer_batch
        self.infer_shared_batch = infer_shared_batch
        self.shared_spec = shared_spec
        self.shared_inputs = tuple(shared_inputs or ())
        self.shared_policy = tuple(shared_policy or ())
        self.shared_wdl = tuple(shared_wdl or ())
        self.batch_cap = int(batch_cap)
        self.wait_ms = float(wait_ms)
        self.device = str(device)
        self._condition = threading.Condition()
        self._pending: deque[_Request] = deque()
        self._stop = threading.Event()
        self._dispatcher_stop = False
        self._fatal: str | None = None
        self._lock = threading.Lock()
        self._rows: list[int] = []
        self._latencies: dict[str, list[float]] = {
            "worker_to_broker": [],
            "broker_queue": [],
            "gpu_service": [],
            "response_to_worker": [],
            "total_blocked": [],
            "h2d": [],
            "model_forward": [],
        }
        self._timing_keys: set[tuple[int, int]] = set()
        self._staging_buffers: list[Any] = []
        self._staging_index = 0
        self._pinned_staging = False
        if self.shared_spec is not None:
            self._allocate_staging()
        self._ingress_thread = threading.Thread(target=self._ingress, name="selfplay-broker-ingress", daemon=True)
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="selfplay-inference-dispatch", daemon=True)

    @property
    def fatal(self) -> str | None:
        with self._lock:
            return self._fatal

    def _set_fatal(self, message: str) -> None:
        with self._lock:
            if self._fatal is None:
                self._fatal = message

    def _allocate_staging(self) -> None:
        try:
            import torch

            shape = (self.batch_cap, *self.shared_spec.observation_shape)  # type: ignore[union-attr]
            want_pinned = torch.device(self.device).type == "cuda"
            for _ in range(2):
                try:
                    buffer = torch.empty(shape, dtype=torch.float32, pin_memory=want_pinned)
                    self._pinned_staging = self._pinned_staging or bool(buffer.is_pinned())
                except RuntimeError:
                    buffer = torch.empty(shape, dtype=torch.float32)
                self._staging_buffers.append(buffer)
        except ImportError as exc:
            raise ValueError("shared-memory self-play requires torch execution support") from exc

    def start(self) -> None:
        self._ingress_thread.start()
        self._dispatch_thread.start()

    def stop(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            self.requests.put(_STOP)
        self._ingress_thread.join(timeout=30.0)
        with self._condition:
            self._dispatcher_stop = True
            self._condition.notify_all()
        self._dispatch_thread.join(timeout=30.0)
        if self._ingress_thread.is_alive() or self._dispatch_thread.is_alive():
            raise SelfPlayEngineError("central inference owner did not stop cleanly")

    def _ingress(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    item = self.requests.get(timeout=0.05)
                except Empty:
                    continue
                if item == _STOP:
                    return
                if not isinstance(item, _Request):
                    self._set_fatal("malformed inference request envelope")
                    continue
                if item.rows <= 0 or item.rows > self.batch_cap:
                    self._set_fatal(f"malformed inference row count {item.rows}")
                    continue
                received = replace(item, broker_received_at=time.perf_counter())
                with self._condition:
                    self._pending.append(received)
                    self._condition.notify()
        except BaseException as exc:
            self._set_fatal(f"broker ingress failed: {type(exc).__name__}: {exc}")
            with self._condition:
                self._condition.notify_all()

    def _fail_request(self, request: _Request, message: str) -> None:
        try:
            self.responses[request.worker_id][request.lane_id].put(_Response(request.request_id, error=message, slot_ids=request.slot_ids))
        except BaseException:
            pass

    def _take_batch(self) -> list[_Request] | None:
        with self._condition:
            while not self._pending and not self._dispatcher_stop:
                self._condition.wait(timeout=0.1)
            if self._dispatcher_stop and not self._pending:
                return None
            first = self._pending[0]
            first_received = first.broker_received_at or time.perf_counter()
            deadline = first_received + self.wait_ms / 1000.0
            while True:
                rows = sum(request.rows for request in self._pending)
                now = time.perf_counter()
                if rows >= self.batch_cap or now >= deadline:
                    break
                self._condition.wait(timeout=max(0.0, deadline - now))
                if self._dispatcher_stop and not self._pending:
                    return None
            batch: list[_Request] = []
            rows = 0
            while self._pending:
                request = self._pending[0]
                if rows and rows + request.rows > self.batch_cap:
                    break
                self._pending.popleft()
                batch.append(request)
                rows += request.rows
                if rows == self.batch_cap:
                    break
            return batch

    def _dispatch_loop(self) -> None:
        while True:
            batch = self._take_batch()
            if batch is None:
                return
            try:
                self._dispatch(batch)
            except BaseException as exc:
                message = f"central inference owner failed: {type(exc).__name__}: {exc}"
                self._set_fatal(message)
                for request in batch:
                    self._fail_request(request, message)

    def _dispatch(self, batch: Sequence[_Request]) -> None:
        dispatch_started = time.perf_counter()
        for request in batch:
            if request.broker_received_at is None:
                raise RuntimeError("inference request lost broker receive timestamp")
            if request.worker_id < 0 or request.worker_id >= len(self.responses):
                raise RuntimeError("inference request references an unknown worker")
            if request.lane_id < 0 or request.lane_id >= len(self.responses[request.worker_id]):
                raise RuntimeError("inference request references an unknown context")
        outputs: tuple[object, ...] = ()
        h2d_started = h2d_finished = forward_started = forward_finished = None
        if self.shared_spec is None:
            if self.infer_batch is None:
                raise RuntimeError("legacy inference callback is missing")
            outputs = tuple(self.infer_batch(tuple(request.payload for request in batch)))
            if len(outputs) != len(batch):
                raise RuntimeError(f"inference owner returned {len(outputs)} rows for {len(batch)} requests")
            output_ready = time.perf_counter()
        else:
            if self.infer_shared_batch is None:
                raise RuntimeError("shared inference callback is missing")
            total_rows = sum(request.rows for request in batch)
            if total_rows > self.batch_cap:
                raise RuntimeError("shared inference batch exceeds configured cap")
            staging = self._staging_buffers[self._staging_index]
            self._staging_index = (self._staging_index + 1) % len(self._staging_buffers)
            offset = 0
            for request in batch:
                if len(request.slot_ids) != request.rows:
                    raise RuntimeError("shared inference request row/slot mismatch")
                source = self.shared_inputs[request.worker_id]
                for slot in request.slot_ids:
                    if slot < 0 or slot >= int(source.shape[0]):
                        raise RuntimeError("shared inference request references an invalid slot")
                    staging[offset].copy_(source[slot])
                    offset += 1
            raw_result = self.infer_shared_batch(staging[:total_rows])
            if isinstance(raw_result, SharedInferenceResult):
                policy, wdl = raw_result.policy, raw_result.wdl
                h2d_started = raw_result.h2d_started_at
                h2d_finished = raw_result.h2d_finished_at
                forward_started = raw_result.forward_started_at
                forward_finished = raw_result.forward_finished_at
            elif isinstance(raw_result, tuple) and len(raw_result) == 2:
                policy, wdl = raw_result
            else:
                raise RuntimeError("shared inference callback must return policy and WDL tensors")
            if tuple(policy.shape) != (total_rows, self.shared_spec.policy_size):
                raise RuntimeError("shared inference policy shape drift")
            if tuple(wdl.shape) != (total_rows, self.shared_spec.wdl_size):
                raise RuntimeError("shared inference WDL shape drift")
            offset = 0
            for request in batch:
                destination_policy = self.shared_policy[request.worker_id]
                destination_wdl = self.shared_wdl[request.worker_id]
                for slot in request.slot_ids:
                    destination_policy[slot].copy_(policy[offset])
                    destination_wdl[slot].copy_(wdl[offset])
                    offset += 1
            output_ready = time.perf_counter()

        with self._lock:
            self._rows.append(sum(request.rows for request in batch))
        response_signaled = time.perf_counter()
        for index, request in enumerate(batch):
            result = None if self.shared_spec is not None else outputs[index]
            response = _Response(
                request.request_id,
                result=result,
                slot_ids=request.slot_ids,
                broker_received_at=request.broker_received_at,
                dispatch_started_at=dispatch_started,
                output_ready_at=output_ready,
                response_signaled_at=response_signaled,
                h2d_started_at=h2d_started,
                h2d_finished_at=h2d_finished,
                forward_started_at=forward_started,
                forward_finished_at=forward_finished,
            )
            self.responses[request.worker_id][request.lane_id].put(response)

    def record_worker_resumed(self, event: Sequence[object]) -> None:
        if len(event) < 15 or event[0] != "inference_resumed":
            return
        key = (int(event[1]), int(event[3]))
        with self._lock:
            if key in self._timing_keys:
                return
            self._timing_keys.add(key)
            worker_sent, broker_received, dispatch, output, _signaled = event[4:9]
            h2d_started, h2d_finished, forward_started, forward_finished, resumed = event[9:14]
            if broker_received is not None and worker_sent is not None:
                self._latencies["worker_to_broker"].append(max(0.0, float(broker_received) - float(worker_sent)) * 1000.0)
            if broker_received is not None and dispatch is not None:
                self._latencies["broker_queue"].append(max(0.0, float(dispatch) - float(broker_received)) * 1000.0)
            if dispatch is not None and output is not None:
                self._latencies["gpu_service"].append(max(0.0, float(output) - float(dispatch)) * 1000.0)
            if h2d_started is not None and h2d_finished is not None:
                self._latencies["h2d"].append(max(0.0, float(h2d_finished) - float(h2d_started)) * 1000.0)
            if forward_started is not None and forward_finished is not None:
                self._latencies["model_forward"].append(max(0.0, float(forward_finished) - float(forward_started)) * 1000.0)
            if output is not None and resumed is not None:
                self._latencies["response_to_worker"].append(max(0.0, float(resumed) - float(output)) * 1000.0)
            if worker_sent is not None and resumed is not None:
                self._latencies["total_blocked"].append(max(0.0, float(resumed) - float(worker_sent)) * 1000.0)

    def telemetry(self, wall_s: float) -> dict[str, object]:
        with self._lock:
            rows = list(self._rows)
            fatal = self._fatal
            latencies = {key: list(values) for key, values in self._latencies.items()}
        count = sum(rows)
        forwards = len(rows)
        stage_names = {
            "worker_to_broker": "worker_to_broker_latency_ms",
            "broker_queue": "broker_queue_latency_ms",
            "gpu_service": "gpu_service_latency_ms",
            "response_to_worker": "response_to_worker_latency_ms",
            "total_blocked": "worker_blocked_inference_ms",
            "h2d": "h2d_latency_ms",
            "model_forward": "model_forward_latency_ms",
        }
        data: dict[str, object] = {
            "inference_requests": count,
            "inference_forwards": forwards,
            "inference_rows": count,
            "batch_rows": rows,
            "mean_inference_batch_rows": statistics.fmean(rows) if rows else 0.0,
            "p50_inference_batch_rows": _percentile(rows, 0.50),
            "p95_inference_batch_rows": _percentile(rows, 0.95),
            "max_inference_batch_rows": max(rows, default=0),
            "inference_rows_per_sec": count / wall_s if wall_s > 0 else 0.0,
            "inference_forwards_per_sec": forwards / wall_s if wall_s > 0 else 0.0,
            "inference_batch_cap": self.batch_cap,
            "inference_batch_wait_ms": self.wait_ms,
            "inference_device": self.device,
            "central_inference_owner_pid": os.getpid(),
            "central_inference_fatal": fatal,
            "shared_memory_transport": self.shared_spec is not None,
            "shared_memory_staging_buffers": len(self._staging_buffers),
            "shared_memory_staging_pinned": self._pinned_staging,
        }
        for key, name in stage_names.items():
            summary = _summary(latencies[key])
            data[name] = summary
            data[f"{key}_mean_ms"] = summary["mean"]
            data[f"{key}_p50_ms"] = summary["p50"]
            data[f"{key}_p95_ms"] = summary["p95"]
        data["worker_blocked_inference_seconds"] = sum(latencies["total_blocked"]) / 1000.0
        data["worker_blocked_inference_calls"] = len(latencies["total_blocked"])
        return data


class SelfPlayEngine:
    """Process scheduler, shared slots, broker batching and fail-closed exit."""

    def __init__(self, config: SelfPlayEngineConfig) -> None:
        config.validate()
        self.config = config

    def run(
        self,
        game_ids: Sequence[str],
        *,
        worker_play: Callable[[object, str, InferenceClient], object] | None,
        worker_context: object,
        infer_batch: Callable[[Sequence[object]], Sequence[object]] | None,
        record_metrics: Callable[[object], Mapping[str, object]] | None = None,
        telemetry: MutableMapping[str, object] | None = None,
        shared_memory: SharedMemorySpec | None = None,
        infer_shared_batch: Callable[[Any], object] | None = None,
        worker_game_factory: Callable[[object, str, InferenceClient], object] | None = None,
        active_games_per_worker: int | None = None,
        total_active_contexts: int | None = None,
    ) -> tuple[object, ...]:
        ids = tuple(sorted(str(game_id) for game_id in game_ids))
        if len(set(ids)) != len(ids):
            raise ValueError("self-play game IDs must be unique")
        if shared_memory is not None:
            shared_memory.validate()
            if infer_shared_batch is None or worker_game_factory is None:
                raise ValueError("shared-memory execution requires a game factory and shared inference callback")
        elif worker_play is None:
            raise ValueError("legacy self-play execution requires worker_play")
        if not ids:
            if telemetry is not None:
                telemetry.update(self._empty_telemetry(shared_memory is not None))
            return ()

        ctx = mp.get_context(self.config.process_start_method)
        shared_path = shared_memory is not None
        per_worker_capacity = int(active_games_per_worker) if active_games_per_worker is not None else (self.config.active_games_per_worker if shared_path else self.config.lanes_per_worker)
        if per_worker_capacity <= 0:
            raise ValueError("active games per worker must be positive")
        configured_total = total_active_contexts if total_active_contexts is not None else self.config.total_active_contexts
        count = min(self.config.workers, len(ids))
        target = min(len(ids), count * per_worker_capacity)
        if configured_total is not None:
            if configured_total <= 0:
                raise ValueError("total active contexts must be positive")
            # ``total_active_contexts`` is a cap on cooperative game
            # contexts, not on the number of OS worker processes. Keeping
            # the configured worker pool intact preserves the separate
            # workers-vs-contexts tuning axes and leaves idle workers out of
            # the initial balanced fill when the cap is smaller.
            target = min(target, int(configured_total))
        if target <= 0:
            raise ValueError("self-play active context target must be positive")
        per_worker = [target // count + (1 if index < target % count else 0) for index in range(count)]
        slots_per_worker = max(per_worker)

        task_queues = [ctx.Queue() for _ in range(count)]
        requests, events = ctx.Queue(), ctx.Queue()
        shared_inputs: list[Any] = []
        shared_policy: list[Any] = []
        shared_wdl: list[Any] = []
        if shared_path:
            import torch

            for _ in range(count):
                shared_inputs.append(torch.empty((slots_per_worker, *shared_memory.observation_shape), dtype=torch.float32).share_memory_())  # type: ignore[union-attr]
                shared_policy.append(torch.empty((slots_per_worker, shared_memory.policy_size), dtype=torch.float32).share_memory_())  # type: ignore[union-attr]
                shared_wdl.append(torch.empty((slots_per_worker, shared_memory.wdl_size), dtype=torch.float32).share_memory_())  # type: ignore[union-attr]
            responses: list[list[Any]] = [[ctx.Queue()] for _ in range(count)]
        else:
            responses = [[ctx.Queue() for _ in range(self.config.lanes_per_worker)] for _ in range(count)]

        service = _CentralInference(
            requests,
            responses,
            infer_batch,
            batch_cap=self.config.inference_batch_cap,
            wait_ms=self.config.inference_batch_wait_ms,
            device=self.config.device,
            shared_spec=shared_memory,
            infer_shared_batch=infer_shared_batch,
            shared_inputs=shared_inputs,
            shared_policy=shared_policy,
            shared_wdl=shared_wdl,
        )
        processes: list[Any] = []
        for worker_id in range(count):
            if shared_path:
                process = ctx.Process(
                    target=_shared_worker_main,
                    name=f"selfplay-search-{worker_id:02d}",
                    args=(worker_id, per_worker[worker_id], task_queues[worker_id], requests, responses[worker_id][0], events, worker_context, self.config.inference_request_timeout_s, worker_game_factory, shared_inputs[worker_id], shared_policy[worker_id], shared_wdl[worker_id], shared_memory),
                )
            else:
                process = ctx.Process(
                    target=_worker_main,
                    name=f"selfplay-search-{worker_id:02d}",
                    args=(worker_id, self.config.lanes_per_worker, task_queues[worker_id], requests, responses[worker_id], events, worker_play, worker_context, self.config.inference_request_timeout_s),
                )
            processes.append(process)

        pending = deque(ids)
        for worker_id, capacity in enumerate(per_worker):
            for _ in range(capacity):
                task_queues[worker_id].put(pending.popleft())
        pending_started_nonempty = bool(pending)

        wall_started = time.perf_counter()
        parent_cpu_started = time.process_time()
        pids: set[int] = set()
        service_started = False
        stop_sent = False

        def send_stops() -> None:
            nonlocal stop_sent
            if stop_sent:
                return
            stop_sent = True
            for queue in task_queues:
                queue.put(_TASK_STOP)
                if not shared_path:
                    for _ in range(self.config.lanes_per_worker - 1):
                        queue.put(_TASK_STOP)

        try:
            for process in processes:
                process.start()
                if process.pid is not None:
                    pids.add(int(process.pid))
            service.start()
            service_started = True
            startup_wall_time = max(0.0, time.perf_counter() - wall_started)
        except BaseException:
            self._terminate(processes)
            if service_started:
                try:
                    service.stop()
                except BaseException:
                    pass
            raise

        completed: dict[str, object] = {}
        failures: list[dict[str, object]] = []
        active = peak = 0
        active_samples: list[int] = []
        pending_samples: list[int] = [len(pending)]
        pending_empty_at: float | None = None
        active_processes: set[int] = set()
        active_process_counts: dict[int, int] = {}
        peak_processes = 0
        game_cpu_seconds = 0.0
        worker_process_cpu: dict[int, float] = {}
        moves = technical = 0

        def assign_next(worker_id: int) -> None:
            nonlocal pending_empty_at
            if pending:
                task_queues[worker_id].put(pending.popleft())
                pending_samples.append(len(pending))
                if pending_started_nonempty and not pending and pending_empty_at is None:
                    pending_empty_at = time.perf_counter()

        def record_active_sample(value: int) -> None:
            # Exclude the synthetic startup/final-drain zeros. Positive
            # observations include the replenishment tail, which is part of
            # the measured active-context lifetime.
            if value > 0:
                active_samples.append(value)

        try:
            while len(completed) < len(ids):
                if service.fatal is not None:
                    raise SelfPlayEngineError(service.fatal)
                try:
                    event = events.get(timeout=0.25)
                except Empty:
                    dead = [p for p in processes if p.exitcode not in (None, 0)]
                    if dead:
                        detail = ", ".join(f"pid={p.pid}:exit={p.exitcode}" for p in dead)
                        raise SelfPlayEngineError(f"self-play worker process died: {detail}")
                    if all(p.exitcode == 0 for p in processes):
                        raise SelfPlayEngineError(f"self-play workers exited after {len(completed)}/{len(ids)} games")
                    continue
                if event and event[0] == "inference_resumed":
                    service.record_worker_resumed(event)
                    continue
                kind = event[0]
                if kind == "worker_stopped":
                    pids.add(int(event[2]))
                    if len(event) >= 4:
                        worker_process_cpu[int(event[2])] = float(event[3])
                elif kind in {"worker_started", "lane_stopped"}:
                    pids.add(int(event[2]))
                elif kind == "game_started":
                    pid = int(event[2])
                    pids.add(pid)
                    active += 1
                    peak = max(peak, active)
                    record_active_sample(active)
                    active_process_counts[pid] = active_process_counts.get(pid, 0) + 1
                    active_processes.add(pid)
                    peak_processes = max(peak_processes, len(active_processes))
                elif kind == "game_failed":
                    _, worker_id, pid, _slot, game_id, message, cpu = event
                    pid = int(pid)
                    pids.add(pid)
                    active = max(0, active - 1)
                    record_active_sample(active)
                    game_cpu_seconds += float(cpu)
                    remaining = active_process_counts.get(pid, 1) - 1
                    if remaining > 0:
                        active_process_counts[pid] = remaining
                    else:
                        active_process_counts.pop(pid, None)
                        active_processes.discard(pid)
                    failures.append({"worker_id": int(worker_id), "pid": pid, "game_id": str(game_id), "error": str(message)})
                    raise SelfPlayEngineError(f"self-play worker failed for {game_id}: {message}")
                elif kind == "game_completed":
                    _, worker_id, pid, _slot, game_id, record, cpu = event
                    pid = int(pid)
                    pids.add(pid)
                    active = max(0, active - 1)
                    record_active_sample(active)
                    game_cpu_seconds += float(cpu)
                    remaining = active_process_counts.get(pid, 1) - 1
                    if remaining > 0:
                        active_process_counts[pid] = remaining
                    else:
                        active_process_counts.pop(pid, None)
                        active_processes.discard(pid)
                    key = str(game_id)
                    if key in completed:
                        raise SelfPlayEngineError(f"duplicate self-play completion for {key}")
                    completed[key] = record
                    if record_metrics is not None:
                        metrics = record_metrics(record)
                        moves += int(metrics.get("moves", 0))
                        technical += int(bool(metrics.get("technical", False)))
                    assign_next(int(worker_id))
                    if len(completed) == len(ids):
                        send_stops()
                else:
                    raise SelfPlayEngineError(f"unknown self-play worker event: {kind!r}")
        except BaseException:
            self._terminate(processes)
            try:
                service.stop()
            except BaseException:
                pass
            raise

        teardown_started = time.perf_counter()
        for process in processes:
            process.join(timeout=self.config.worker_join_timeout_s)
            if process.is_alive() or process.exitcode != 0:
                self._terminate(processes)
                try:
                    service.stop()
                except BaseException:
                    pass
                raise SelfPlayEngineError(f"self-play worker {process.pid} failed shutdown: alive={process.is_alive()} exit={process.exitcode}")
        while True:
            try:
                event = events.get_nowait()
            except Empty:
                break
            if event and event[0] == "inference_resumed":
                service.record_worker_resumed(event)
            elif event and event[0] == "worker_stopped" and len(event) >= 4:
                worker_process_cpu[int(event[2])] = float(event[3])
        service.stop()
        teardown_wall_time = max(0.0, time.perf_counter() - teardown_started)

        wall_s = max(0.0, time.perf_counter() - wall_started)
        parent_cpu_seconds = max(0.0, time.process_time() - parent_cpu_started)
        worker_cpu_seconds = sum(worker_process_cpu.values())
        if len(worker_process_cpu) != count:
            raise SelfPlayEngineError(f"self-play worker CPU telemetry incomplete: {len(worker_process_cpu)}/{count}")
        process_tree_cpu_seconds = worker_cpu_seconds + parent_cpu_seconds
        result = tuple(completed[game_id] for game_id in ids)
        active_summary = _summary(active_samples)
        tail_duration = (
            max(0.0, time.perf_counter() - pending_empty_at)
            if pending_empty_at is not None
            else 0.0
        )
        data: dict[str, object] = {
            "configured_workers": self.config.workers,
            "worker_processes_started": count,
            "lanes_per_worker": self.config.lanes_per_worker,
            "configured_search_lanes": count * self.config.lanes_per_worker,
            "active_games_per_worker": per_worker_capacity,
            "target_active_contexts": target,
            "mean_active_contexts": active_summary["mean"],
            "p50_active_contexts": active_summary["p50"],
            "p95_active_contexts": active_summary["p95"],
            "minimum_active_contexts": min(active_samples, default=0),
            "pending_games_final": len(pending),
            "pending_games_samples": pending_samples,
            "pending_queue_exhausted": pending_empty_at is not None,
            "completed_games_over_time": len(result),
            "active_worker_count": len(pids),
            "worker_pids": sorted(pids),
            "real_worker_pid_count": len(pids),
            "peak_concurrent_search_workers": peak,
            "peak_concurrent_search_contexts": peak,
            "peak_concurrent_search_processes": peak_processes,
            "process_cpu_seconds": worker_cpu_seconds,
            "worker_process_cpu_seconds": worker_cpu_seconds,
            "game_thread_cpu_seconds": game_cpu_seconds,
            "parent_process_cpu_seconds": parent_cpu_seconds,
            "process_tree_cpu_seconds": process_tree_cpu_seconds,
            "effective_cpu_cores": worker_cpu_seconds / wall_s if wall_s > 0 else 0.0,
            "process_tree_effective_cpu_cores": process_tree_cpu_seconds / wall_s if wall_s > 0 else 0.0,
            "worker_failures": failures,
            "worker_restarts": 0,
            "games_requested": len(ids),
            "games_completed": len(result),
            "games_failed": len(failures),
            "technical_games": technical,
            "total_moves": moves,
            "games_per_sec": len(result) / wall_s if wall_s > 0 else 0.0,
            "moves_per_sec": moves / wall_s if wall_s > 0 else 0.0,
            "wall_time_sec": wall_s,
            "startup_wall_time_sec": startup_wall_time,
            "teardown_wall_time_sec": teardown_wall_time,
            "result_order": list(ids),
            "process_start_method": self.config.process_start_method,
            "global_task_replenishment": bool(any(value != pending_samples[0] for value in pending_samples)),
            "tail_duration_after_pending_empty_sec": tail_duration,
            **service.telemetry(wall_s),
        }
        if telemetry is not None:
            telemetry.update(data)
        return result

    def _empty_telemetry(self, shared: bool = False) -> dict[str, object]:
        return {
            "configured_workers": self.config.workers,
            "worker_processes_started": 0,
            "lanes_per_worker": self.config.lanes_per_worker,
            "configured_search_lanes": 0,
            "active_games_per_worker": self.config.active_games_per_worker,
            "target_active_contexts": 0,
            "mean_active_contexts": 0.0,
            "p50_active_contexts": 0.0,
            "p95_active_contexts": 0.0,
            "minimum_active_contexts": 0,
            "pending_games_final": 0,
            "pending_games_samples": [0],
            "pending_queue_exhausted": False,
            "completed_games_over_time": 0,
            "active_worker_count": 0,
            "worker_pids": [],
            "real_worker_pid_count": 0,
            "peak_concurrent_search_workers": 0,
            "peak_concurrent_search_contexts": 0,
            "peak_concurrent_search_processes": 0,
            "process_cpu_seconds": 0.0,
            "worker_process_cpu_seconds": 0.0,
            "game_thread_cpu_seconds": 0.0,
            "parent_process_cpu_seconds": 0.0,
            "process_tree_cpu_seconds": 0.0,
            "effective_cpu_cores": 0.0,
            "process_tree_effective_cpu_cores": 0.0,
            "worker_failures": [],
            "worker_restarts": 0,
            "games_requested": 0,
            "games_completed": 0,
            "games_failed": 0,
            "technical_games": 0,
            "total_moves": 0,
            "games_per_sec": 0.0,
            "moves_per_sec": 0.0,
            "wall_time_sec": 0.0,
            "result_order": [],
            "process_start_method": self.config.process_start_method,
            "global_task_replenishment": False,
            "tail_duration_after_pending_empty_sec": 0.0,
            "inference_requests": 0,
            "inference_forwards": 0,
            "inference_rows": 0,
            "batch_rows": [],
            "mean_inference_batch_rows": 0.0,
            "p50_inference_batch_rows": 0.0,
            "p95_inference_batch_rows": 0.0,
            "max_inference_batch_rows": 0,
            "inference_rows_per_sec": 0.0,
            "inference_forwards_per_sec": 0.0,
            "inference_batch_cap": self.config.inference_batch_cap,
            "inference_batch_wait_ms": self.config.inference_batch_wait_ms,
            "inference_device": self.config.device,
            "central_inference_owner_pid": os.getpid(),
            "central_inference_fatal": None,
            "shared_memory_transport": shared,
            "shared_memory_staging_buffers": 2 if shared else 0,
            "shared_memory_staging_pinned": False,
            "worker_blocked_inference_seconds": 0.0,
            "worker_blocked_inference_calls": 0,
            "worker_to_broker_latency_ms": {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "broker_queue_latency_ms": {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "gpu_service_latency_ms": {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "h2d_latency_ms": {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "model_forward_latency_ms": {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "response_to_worker_latency_ms": {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "worker_blocked_inference_ms": {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
        }

    @staticmethod
    def _terminate(processes: Sequence[Any]) -> None:
        started = [process for process in processes if process.pid is not None]
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=5)
