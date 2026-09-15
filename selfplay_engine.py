"""Game-agnostic process self-play execution with one inference owner."""
from __future__ import annotations

from dataclasses import dataclass
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
    lanes_per_worker: int = 1

    def validate(self) -> None:
        if self.workers <= 0 or self.inference_batch_cap <= 0 or self.lanes_per_worker <= 0:
            raise ValueError("self-play workers, lanes per worker and inference batch cap must be positive")
        if self.inference_batch_wait_ms < 0 or not math.isfinite(self.inference_batch_wait_ms):
            raise ValueError("self-play inference batch wait must be finite and non-negative")
        if self.inference_request_timeout_s <= 0 or not math.isfinite(self.inference_request_timeout_s):
            raise ValueError("self-play inference timeout must be positive and finite")
        if self.worker_join_timeout_s <= 0 or not math.isfinite(self.worker_join_timeout_s):
            raise ValueError("self-play worker join timeout must be positive and finite")
        if self.process_start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method: {self.process_start_method}")


@dataclass(frozen=True)
class _Request:
    worker_id: int
    lane_id: int
    request_id: int
    payload: object


@dataclass(frozen=True)
class _Response:
    request_id: int
    result: object | None = None
    error: str | None = None


class InferenceClient:
    """Blocking worker-side RPC client for central inference."""

    def __init__(self, worker_id: int, lane_id: int, requests: Any, responses: Any, timeout_s: float) -> None:
        self.worker_id = int(worker_id)
        self.lane_id = int(lane_id)
        self._requests = requests
        self._responses = responses
        self._timeout_s = float(timeout_s)
        self._next_id = 0

    def request(self, payload: object) -> object:
        request_id = self._next_id
        self._next_id += 1
        self._requests.put(_Request(self.worker_id, self.lane_id, request_id, payload))
        try:
            response = self._responses.get(timeout=self._timeout_s)
        except Empty as exc:
            raise InferenceTransportError(
                f"inference request timed out (worker={self.worker_id}, request={request_id})"
            ) from exc
        if not isinstance(response, _Response):
            raise InferenceTransportError("malformed inference response envelope")
        if response.request_id != request_id:
            raise InferenceTransportError(
                f"inference response id mismatch: expected {request_id}, got {response.request_id}"
            )
        if response.error is not None:
            raise InferenceTransportError(response.error)
        if response.result is None:
            raise InferenceTransportError("inference response contained no result")
        return response.result


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
    client = InferenceClient(worker_id, lane_id, requests, responses, request_timeout_s)
    while True:
        task = tasks.get()
        if task == _TASK_STOP:
            events.put(("lane_stopped", worker_id, pid, lane_id))
            return
        game_id = str(task)
        # ``process_time`` is process-wide and would be double-counted when
        # multiple lanes overlap inside one OS worker.  Per-lane CPU time is
        # the only correct additive measure for lane-enabled execution.
        cpu_clock = time.thread_time
        cpu_started = cpu_clock()
        events.put(("game_started", worker_id, pid, lane_id, game_id))
        try:
            record = worker_play(worker_context, game_id, client)
        except BaseException as exc:
            events.put((
                "game_failed", worker_id, pid, lane_id, game_id,
                f"{type(exc).__name__}: {exc}",
                max(0.0, cpu_clock() - cpu_started),
            ))
            return
        events.put((
            "game_completed", worker_id, pid, lane_id, game_id, record,
            max(0.0, cpu_clock() - cpu_started),
        ))


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
            name=f"selfplay-lane-{worker_id:02d}-{lane_id:02d}",
            args=(
                worker_id, lane_id, tasks, requests, responses[lane_id], events,
                worker_play, worker_context, request_timeout_s,
            ),
        )
        for lane_id in range(lanes_per_worker)
    ]
    for lane in lanes:
        lane.start()
    for lane in lanes:
        lane.join()
    events.put(("worker_stopped", worker_id, pid, max(0.0, time.process_time() - process_cpu_started)))


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(int(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


class _CentralInference:
    def __init__(
        self,
        requests: Any,
        responses: Sequence[Sequence[Any]],
        infer_batch: Callable[[Sequence[object]], Sequence[object]],
        *,
        batch_cap: int,
        wait_ms: float,
        device: str,
    ) -> None:
        self.requests = requests
        self.responses = tuple(responses)
        self.infer_batch = infer_batch
        self.batch_cap = int(batch_cap)
        self.wait_ms = float(wait_ms)
        self.device = str(device)
        self._lock = threading.Lock()
        self._rows: list[int] = []
        self._fatal: str | None = None
        self._thread = threading.Thread(target=self._serve, name="selfplay-central-inference", daemon=True)

    @property
    def fatal(self) -> str | None:
        with self._lock:
            return self._fatal

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.requests.put(_STOP)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise SelfPlayEngineError("central inference owner did not stop cleanly")

    def _fail(self, batch: Sequence[_Request], message: str) -> None:
        with self._lock:
            self._fatal = message
        for request in batch:
            self.responses[request.worker_id][request.lane_id].put(_Response(request.request_id, error=message))

    def _serve(self) -> None:
        while True:
            item = self.requests.get()
            if item == _STOP:
                return
            if not isinstance(item, _Request):
                with self._lock:
                    self._fatal = "malformed inference request envelope"
                continue
            batch = [item]
            deadline = time.monotonic() + self.wait_ms / 1000.0
            while len(batch) < self.batch_cap:
                timeout = max(0.0, deadline - time.monotonic()) if self.wait_ms > 0 else 0.0
                try:
                    item = self.requests.get(timeout=timeout)
                except Empty:
                    break
                if item == _STOP:
                    self.requests.put(_STOP)
                    break
                if not isinstance(item, _Request):
                    self._fail(batch, "malformed inference request envelope")
                    batch = []
                    break
                batch.append(item)
            if not batch:
                continue
            fatal = self.fatal
            if fatal is not None:
                self._fail(batch, fatal)
                continue
            try:
                outputs = tuple(self.infer_batch(tuple(request.payload for request in batch)))
                if len(outputs) != len(batch):
                    raise RuntimeError(f"inference owner returned {len(outputs)} rows for {len(batch)} requests")
            except BaseException as exc:
                self._fail(batch, f"central inference owner failed: {type(exc).__name__}: {exc}")
                continue
            with self._lock:
                self._rows.append(len(batch))
            for request, output in zip(batch, outputs):
                self.responses[request.worker_id][request.lane_id].put(
                    _Response(request.request_id, result=output)
                )

    def telemetry(self, wall_s: float) -> dict[str, object]:
        with self._lock:
            rows = list(self._rows)
            fatal = self._fatal
        count = sum(rows)
        forwards = len(rows)
        return {
            "inference_requests": count,
            "inference_forwards": forwards,
            "inference_rows": count,
            "batch_rows": rows,
            "mean_inference_batch_rows": statistics.fmean(rows) if rows else 0.0,
            "p50_inference_batch_rows": float(statistics.median(rows)) if rows else 0.0,
            "p95_inference_batch_rows": _percentile(rows, 0.95),
            "max_inference_batch_rows": max(rows, default=0),
            "inference_rows_per_sec": count / wall_s if wall_s > 0 else 0.0,
            "inference_forwards_per_sec": forwards / wall_s if wall_s > 0 else 0.0,
            "inference_batch_cap": self.batch_cap,
            "inference_batch_wait_ms": self.wait_ms,
            "inference_device": self.device,
            "central_inference_owner_pid": os.getpid(),
            "central_inference_fatal": fatal,
        }


class SelfPlayEngine:
    """Process scheduler, IPC, central batching, ordering and fail-closed telemetry."""

    def __init__(self, config: SelfPlayEngineConfig) -> None:
        config.validate()
        self.config = config

    def run(
        self,
        game_ids: Sequence[str],
        *,
        worker_play: Callable[[object, str, InferenceClient], object],
        worker_context: object,
        infer_batch: Callable[[Sequence[object]], Sequence[object]],
        record_metrics: Callable[[object], Mapping[str, object]] | None = None,
        telemetry: MutableMapping[str, object] | None = None,
    ) -> tuple[object, ...]:
        ids = tuple(sorted(str(game_id) for game_id in game_ids))
        if len(set(ids)) != len(ids):
            raise ValueError("self-play game IDs must be unique")
        if not ids:
            if telemetry is not None:
                telemetry.update(self._empty_telemetry())
            return ()

        ctx = mp.get_context(self.config.process_start_method)
        count = min(self.config.workers, len(ids))
        task_queues = [ctx.Queue() for _ in range(count)]
        requests, events = ctx.Queue(), ctx.Queue()
        responses = [
            [ctx.Queue() for _ in range(self.config.lanes_per_worker)]
            for _ in range(count)
        ]
        service = _CentralInference(
            requests, responses, infer_batch,
            batch_cap=self.config.inference_batch_cap,
            wait_ms=self.config.inference_batch_wait_ms,
            device=self.config.device,
        )
        processes = [
            ctx.Process(
                target=_worker_main,
                name=f"selfplay-search-{worker_id:02d}",
                args=(
                    worker_id, self.config.lanes_per_worker, task_queues[worker_id], requests, responses[worker_id], events,
                    worker_play, worker_context, self.config.inference_request_timeout_s,
                ),
            )
            for worker_id in range(count)
        ]
        for index, game_id in enumerate(ids):
            task_queues[index % count].put(game_id)
        for tasks in task_queues:
            for _ in range(self.config.lanes_per_worker):
                tasks.put(_TASK_STOP)

        wall_started = time.perf_counter()
        parent_cpu_started = time.process_time()
        pids: set[int] = set()
        service_started = False
        try:
            for process in processes:
                process.start()
                if process.pid is not None:
                    pids.add(int(process.pid))
            service.start()
            service_started = True
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
        active_processes: set[int] = set()
        active_process_counts: dict[int, int] = {}
        peak_processes = 0
        game_cpu_seconds = 0.0
        worker_process_cpu: dict[int, float] = {}
        moves = technical = 0
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
                kind = event[0]
                if kind == "worker_stopped":
                    pids.add(int(event[2]))
                    if len(event) >= 4:
                        worker_process_cpu[int(event[2])] = float(event[3])
                elif kind in {"worker_started", "lane_stopped"}:
                    pids.add(int(event[2]))
                elif kind == "game_started":
                    pid = int(event[2])
                    pids.add(pid); active += 1; peak = max(peak, active)
                    active_process_counts[pid] = active_process_counts.get(pid, 0) + 1
                    active_processes.add(pid); peak_processes = max(peak_processes, len(active_processes))
                elif kind == "game_failed":
                    _, worker_id, pid, _lane_id, game_id, message, cpu = event
                    pid = int(pid)
                    pids.add(pid); active = max(0, active - 1); game_cpu_seconds += float(cpu)
                    remaining = active_process_counts.get(pid, 1) - 1
                    if remaining > 0:
                        active_process_counts[pid] = remaining
                    else:
                        active_process_counts.pop(pid, None)
                        active_processes.discard(pid)
                    failures.append({"worker_id": int(worker_id), "pid": int(pid), "game_id": str(game_id), "error": str(message)})
                    raise SelfPlayEngineError(f"self-play worker failed for {game_id}: {message}")
                elif kind == "game_completed":
                    _, _worker_id, pid, _lane_id, game_id, record, cpu = event
                    pid = int(pid)
                    pids.add(pid); active = max(0, active - 1); game_cpu_seconds += float(cpu)
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
                else:
                    raise SelfPlayEngineError(f"unknown self-play worker event: {kind!r}")
        except BaseException:
            self._terminate(processes)
            try:
                service.stop()
            except BaseException:
                pass
            raise

        for process in processes:
            process.join(timeout=self.config.worker_join_timeout_s)
            if process.is_alive() or process.exitcode != 0:
                self._terminate(processes)
                try:
                    service.stop()
                except BaseException:
                    pass
                raise SelfPlayEngineError(
                    f"self-play worker {process.pid} failed shutdown: alive={process.is_alive()} exit={process.exitcode}"
                )
        while len(worker_process_cpu) < count:
            try:
                event = events.get(timeout=1.0)
            except Empty:
                break
            if event[0] == "worker_stopped" and len(event) >= 4:
                pids.add(int(event[2]))
                worker_process_cpu[int(event[2])] = float(event[3])
        service.stop()

        wall_s = max(0.0, time.perf_counter() - wall_started)
        parent_cpu_seconds = max(0.0, time.process_time() - parent_cpu_started)
        worker_cpu_seconds = sum(worker_process_cpu.values())
        if len(worker_process_cpu) != count:
            raise SelfPlayEngineError(
                f"self-play worker CPU telemetry incomplete: {len(worker_process_cpu)}/{count}"
            )
        process_tree_cpu_seconds = worker_cpu_seconds + parent_cpu_seconds
        result = tuple(completed[game_id] for game_id in ids)
        data: dict[str, object] = {
            "configured_workers": self.config.workers,
            "worker_processes_started": count,
            "lanes_per_worker": self.config.lanes_per_worker,
            "configured_search_lanes": count * self.config.lanes_per_worker,
            "active_worker_count": len(pids),
            "worker_pids": sorted(pids),
            "real_worker_pid_count": len(pids),
            "peak_concurrent_search_workers": peak,
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
            "result_order": list(ids),
            "process_start_method": self.config.process_start_method,
            **service.telemetry(wall_s),
        }
        if telemetry is not None:
            telemetry.update(data)
        return result

    def _empty_telemetry(self) -> dict[str, object]:
        return {
            "configured_workers": self.config.workers,
            "worker_processes_started": 0,
            "lanes_per_worker": self.config.lanes_per_worker,
            "configured_search_lanes": 0,
            "active_worker_count": 0,
            "worker_pids": [],
            "real_worker_pid_count": 0,
            "peak_concurrent_search_workers": 0,
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
        }

    @staticmethod
    def _terminate(processes: Sequence[Any]) -> None:
        started = [process for process in processes if process.pid is not None]
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=5)
