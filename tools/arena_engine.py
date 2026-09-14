#!/usr/bin/env python3
"""Board-agnostic production Arena execution engine.

One execution architecture:
    OS CPU search workers -> one parent inference broker -> central model owner(s).

Game/topology semantics live in ArenaProfile adapters under tools/arena_profiles/.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty
import statistics
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

import torch

CANONICAL_ARENA_ENGINE = "process-central-inference-v1"
DEFAULT_GAMES = 64
DEFAULT_WORKERS = 16
DEFAULT_GAMES_PER_WORKER = 4
DEFAULT_INFERENCE_BATCH_ROWS = 64
DEFAULT_INFERENCE_BATCH_WAIT_MS = 4.0
DEFAULT_MASTER_SEED = 20260914
MIN_MEAN_INFERENCE_BATCH_ROWS = 16.0
MIN_EFFECTIVE_CPU_CORES = 8.0


@dataclass(frozen=True)
class ArenaExecutionConfig:
    games: int = DEFAULT_GAMES
    workers: int = DEFAULT_WORKERS
    games_per_worker: int = DEFAULT_GAMES_PER_WORKER
    inference_batch_rows: int = DEFAULT_INFERENCE_BATCH_ROWS
    inference_batch_wait_ms: float = DEFAULT_INFERENCE_BATCH_WAIT_MS
    device: str = "cuda"
    strict_production: bool = True
    min_mean_inference_batch_rows: float = MIN_MEAN_INFERENCE_BATCH_ROWS
    min_effective_cpu_cores: float = MIN_EFFECTIVE_CPU_CORES

    def validate_base(self) -> None:
        if self.games <= 0 or self.games % 2:
            raise ValueError("Arena games must be a positive even number")
        if self.workers <= 0 or self.games_per_worker <= 0:
            raise ValueError("Arena workers/games_per_worker must be positive")
        if self.inference_batch_rows <= 0 or self.inference_batch_wait_ms < 0.0:
            raise ValueError("Arena inference batching settings are invalid")
        if self.inference_batch_rows < self.games_per_worker:
            raise ValueError("inference_batch_rows must cover at least one worker request")


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    model_hash: str
    artifact_sha256: str
    architecture_config: Mapping[str, object]
    metadata: Mapping[str, object]


class ArenaProfile(Protocol):
    profile_id: str
    run_id_prefix: str
    worker_process_prefix: str
    observation_shape: tuple[int, ...]
    policy_size: int
    wdl_size: int

    def validate_execution_config(self, config: ArenaExecutionConfig) -> None: ...
    def load_identity(self, path: Path) -> CheckpointIdentity: ...
    def load_parent_model(
        self, identity: CheckpointIdentity, device: torch.device
    ) -> torch.nn.Module: ...
    def build_tasks(
        self,
        *,
        run_id: str,
        comparison: str,
        candidate: CheckpointIdentity,
        reference: CheckpointIdentity,
        master_seed: int,
        games: int,
        workers: int,
    ) -> tuple[list[dict[str, object]], int]: ...
    def worker_main(
        self,
        worker_id: int,
        task_queue: Any,
        games_per_worker: int,
        worker_local_wait_ms: float,
        candidate_hash: str,
        reference_hash: str,
        input_slot: torch.Tensor,
        policy_slot: torch.Tensor,
        wdl_slot: torch.Tensor,
        request_queue: Any,
        response_queues: Any,
        start_event: Any,
    ) -> None: ...
    def infer_batch(
        self, model: torch.nn.Module, cpu_batch: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def summarize(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        candidate_label: str,
        reference_label: str,
        pairs: int,
    ) -> dict[str, object]: ...
    def scientific_contract(
        self, config: ArenaExecutionConfig
    ) -> Mapping[str, object]: ...


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _assert_expected_identity(
    identity: CheckpointIdentity,
    *,
    expected_model_hash: str | None,
    expected_artifact_sha256: str | None,
    label: str,
) -> None:
    def same_sha256(actual: str, expected: str) -> bool:
        return actual.removeprefix("sha256:") == expected.removeprefix("sha256:")

    if expected_model_hash and not same_sha256(identity.model_hash, expected_model_hash):
        raise ValueError(f"{label} model hash mismatch")
    if expected_artifact_sha256 and not same_sha256(
        identity.artifact_sha256, expected_artifact_sha256
    ):
        raise ValueError(f"{label} artifact SHA256 mismatch")


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[index]


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float | int]:
    """Return stable summary statistics for execution-only telemetry."""
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))],
        "max": max(ordered),
    }


def _batch_summary(values: Sequence[int]) -> dict[str, float | int]:
    """Return the standard batch telemetry shape for one model."""
    return {
        "forward_calls": len(values),
        "rows": sum(values),
        "mean_batch_rows": statistics.mean(values) if values else 0.0,
        "p50_batch_rows": _percentile(values, 0.50),
        "p95_batch_rows": _percentile(values, 0.95),
        "max_batch_rows": max(values, default=0),
    }


@dataclass
class _PendingModelQueue:
    """Pending inference requests for one model hash."""

    requests: deque[Mapping[str, object]] = field(default_factory=deque)
    rows: int = 0
    first_broker_received_at: float | None = None

    @staticmethod
    def _broker_received_at(request: Mapping[str, object]) -> float:
        """Return the broker timestamp without rewriting worker provenance.

        ``enqueued_at`` is accepted only as a compatibility fallback for
        scheduler unit tests and older in-process callers.  Production
        requests always carry the explicit ``broker_received_at`` field.
        """
        value = request.get("broker_received_at", request.get("enqueued_at"))
        if value is None:
            raise ValueError("Arena inference request has no broker receive timestamp")
        return float(value)

    def append(self, request: Mapping[str, object]) -> None:
        request_rows = int(request["rows"])
        if request_rows <= 0:
            raise ValueError("Arena inference request rows must be positive")
        if not self.requests:
            self.first_broker_received_at = self._broker_received_at(request)
        self.requests.append(request)
        self.rows += request_rows

    def ready(self, *, cap: int, deadline: float, now: float) -> bool:
        return bool(self.requests) and (self.rows >= cap or now >= deadline)

    def pop_batch(self, cap: int) -> tuple[list[Mapping[str, object]], float]:
        if not self.requests:
            raise RuntimeError("Cannot dispatch an empty model-aware inference queue")
        first_broker_received_at = self.first_broker_received_at
        if first_broker_received_at is None:
            raise RuntimeError(
                "Model-aware inference queue lost its first broker receive time"
            )
        batch: list[Mapping[str, object]] = []
        batch_rows = 0
        while self.requests:
            request = self.requests[0]
            request_rows = int(request["rows"])
            if batch and batch_rows + request_rows > cap:
                break
            if request_rows > cap:
                raise RuntimeError(
                    "Arena inference request exceeds the configured batch cap"
                )
            batch.append(self.requests.popleft())
            batch_rows += request_rows
            self.rows -= request_rows
            if batch_rows == cap:
                break
        if self.requests:
            self.first_broker_received_at = self._broker_received_at(self.requests[0])
        else:
            self.first_broker_received_at = None
        return batch, first_broker_received_at


class _ModelAwareBatchScheduler:
    """Fair row-capped coalescing queues keyed by model hash.

    Each model gets its own cap and deadline.  Ready models are dispatched in
    round-robin order, so a continuously busy model cannot starve another
    model whose independent deadline has expired.
    """

    def __init__(self, model_hashes: Sequence[str], *, cap: int, wait_ms: float) -> None:
        if not model_hashes or len(set(model_hashes)) != len(model_hashes):
            raise ValueError("Model-aware scheduler requires distinct model hashes")
        if cap <= 0 or wait_ms < 0.0 or not math.isfinite(float(wait_ms)):
            raise ValueError("Model-aware scheduler cap/wait is invalid")
        self.cap = int(cap)
        self.wait_seconds = float(wait_ms) / 1000.0
        self.model_hashes = tuple(str(value) for value in model_hashes)
        self.queues = {model_hash: _PendingModelQueue() for model_hash in self.model_hashes}
        self._next_ready_index = 0
        self._condition = threading.Condition()

    @property
    def condition(self) -> Any:
        """Condition shared by the broker ingress and dispatch loop."""
        return self._condition

    def enqueue(self, request: Mapping[str, object]) -> None:
        with self._condition:
            model_hash = str(request["model_hash"])
            if model_hash not in self.queues:
                raise RuntimeError(
                    f"Inference request references unknown model hash {model_hash}"
                )
            self.queues[model_hash].append(request)
            self._condition.notify_all()

    def pending_rows(self) -> int:
        with self._condition:
            return sum(queue.rows for queue in self.queues.values())

    def pending_rows_by_model(self) -> dict[str, int]:
        with self._condition:
            return {model_hash: queue.rows for model_hash, queue in self.queues.items()}

    def next_deadline(self) -> float | None:
        with self._condition:
            deadlines = [
                queue.first_broker_received_at + self.wait_seconds
                for queue in self.queues.values()
                if queue.first_broker_received_at is not None
            ]
            return min(deadlines) if deadlines else None

    def next_ready_model(self, now: float) -> str | None:
        with self._condition:
            for offset in range(len(self.model_hashes)):
                index = (self._next_ready_index + offset) % len(self.model_hashes)
                model_hash = self.model_hashes[index]
                queue = self.queues[model_hash]
                deadline = queue.first_broker_received_at + self.wait_seconds if queue.first_broker_received_at is not None else float("inf")
                if queue.ready(cap=self.cap, deadline=deadline, now=now):
                    self._next_ready_index = (index + 1) % len(self.model_hashes)
                    return model_hash
            return None

    def pop_batch(self, model_hash: str) -> tuple[list[Mapping[str, object]], float]:
        with self._condition:
            return self.queues[model_hash].pop_batch(self.cap)


class _ArenaBrokerIngress:
    """Continuously drain the process request queue independently of CUDA.

    The dispatch loop owns the model and remains single-threaded for CUDA
    execution.  This thread only stamps broker receipt time, routes inference
    requests into the model-aware scheduler, and forwards control messages.
    """

    def __init__(
        self,
        request_queue: Any,
        scheduler: _ModelAwareBatchScheduler,
        control_pending: deque[Mapping[str, object]],
    ) -> None:
        self.request_queue = request_queue
        self.scheduler = scheduler
        self.control_pending = control_pending
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="arena-broker-ingress",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    message = self.request_queue.get(timeout=0.1)
                except Empty:
                    continue
                if not isinstance(message, Mapping):
                    raise RuntimeError("Arena broker received a non-mapping message")
                broker_received_at = time.perf_counter()
                if message.get("kind") == "inference":
                    broker_message = dict(message)
                    worker_enqueued_at = broker_message.get(
                        "worker_enqueued_at",
                        broker_message.get("enqueued_at"),
                    )
                    if worker_enqueued_at is None:
                        raise RuntimeError(
                            "Arena inference request lost worker_enqueued_at"
                        )
                    # Preserve the original worker timestamp.  The scheduler
                    # deadline starts from this broker receipt timestamp.
                    broker_message["worker_enqueued_at"] = float(worker_enqueued_at)
                    broker_message["broker_received_at"] = broker_received_at
                    self.scheduler.enqueue(broker_message)
                else:
                    with self.scheduler.condition:
                        self.control_pending.append(dict(message))
                        self.scheduler.condition.notify_all()
        except BaseException as exc:
            self._error = exc
            with self.scheduler.condition:
                self.scheduler.condition.notify_all()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Arena broker ingress failed: {self._error}") from self._error

    def stop(self) -> None:
        self._stop.set()
        with self.scheduler.condition:
            self.scheduler.condition.notify_all()
        if self._thread.ident is None:
            return
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("Arena broker ingress did not exit")


class _WorkerTaskQueue:
    """Balanced initial lane fill backed by one global replenishment queue."""

    def __init__(self, initial_queue: Any, global_queue: Any) -> None:
        self.initial_queue = initial_queue
        self.global_queue = global_queue

    def get(self, *, timeout: float | None = None) -> Mapping[str, object]:
        try:
            return self.initial_queue.get_nowait()
        except Empty:
            return self.global_queue.get(timeout=timeout)

    def get_nowait(self) -> Mapping[str, object]:
        try:
            return self.initial_queue.get_nowait()
        except Empty:
            return self.global_queue.get_nowait()


def _terminate(processes: Sequence[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def _worker_bootstrap(
    profile_id: str,
    worker_id: int,
    task_queue: Any,
    games_per_worker: int,
    worker_local_wait_ms: float,
    candidate_hash: str,
    reference_hash: str,
    input_slot: torch.Tensor,
    policy_slot: torch.Tensor,
    wdl_slot: torch.Tensor,
    request_queue: Any,
    response_queues: Any,
    start_event: Any,
) -> None:
    from tools.arena_profiles import get_profile

    profile = get_profile(profile_id)
    profile.worker_main(
        worker_id,
        task_queue,
        games_per_worker,
        worker_local_wait_ms,
        candidate_hash,
        reference_hash,
        input_slot,
        policy_slot,
        wdl_slot,
        request_queue,
        response_queues,
        start_event,
    )


def run_arena(
    *,
    profile: ArenaProfile,
    candidate_path: Path,
    reference_path: Path | None = None,
    output_dir: Path,
    candidate_label: str | None = None,
    reference_label: str | None = None,
    run_id: str | None = None,
    comparison: str | None = None,
    master_seed: int = DEFAULT_MASTER_SEED,
    config: ArenaExecutionConfig = ArenaExecutionConfig(),
    expected_candidate_model_hash: str | None = None,
    expected_candidate_artifact_sha256: str | None = None,
    expected_reference_model_hash: str | None = None,
    expected_reference_artifact_sha256: str | None = None,
) -> dict[str, object]:
    """Run the single production Arena engine with one game-specific profile."""
    config.validate_base()
    profile.validate_execution_config(config)

    candidate_path = candidate_path.resolve()
    reference_path = (reference_path or candidate_path).resolve()
    candidate = profile.load_identity(candidate_path)
    reference = profile.load_identity(reference_path)
    _assert_expected_identity(
        candidate,
        expected_model_hash=expected_candidate_model_hash,
        expected_artifact_sha256=expected_candidate_artifact_sha256,
        label="candidate",
    )
    _assert_expected_identity(
        reference,
        expected_model_hash=expected_reference_model_hash,
        expected_artifact_sha256=expected_reference_artifact_sha256,
        label="reference",
    )

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Arena requested but CUDA is unavailable")

    candidate_label = candidate_label or candidate_path.stem
    reference_label = reference_label or reference_path.stem
    comparison = comparison or f"{candidate_label}-vs-{reference_label}"
    run_id = run_id or (
        f"{profile.run_id_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir = output_dir.resolve()
    if any(
        (output_dir / name).exists()
        for name in ("manifest.json", "games.jsonl", "summary.json")
    ):
        raise FileExistsError(f"Arena output directory already contains a run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks, pairs = profile.build_tasks(
        run_id=run_id,
        comparison=comparison,
        candidate=candidate,
        reference=reference,
        master_seed=master_seed,
        games=config.games,
        workers=config.workers,
    )
    if len(tasks) != config.games:
        raise RuntimeError(
            f"Arena profile {profile.profile_id!r} produced {len(tasks)} tasks "
            f"for requested games={config.games}"
        )
    candidate_model = profile.load_parent_model(candidate, device)
    reference_model = (
        candidate_model
        if reference.model_hash == candidate.model_hash
        and reference.architecture_config == candidate.architecture_config
        else profile.load_parent_model(reference, device)
    )
    models_by_hash = {
        candidate.model_hash: candidate_model,
        reference.model_hash: reference_model,
    }

    ctx = __import__("multiprocessing").get_context("spawn")
    request_queue = ctx.Queue()
    global_task_queue = ctx.Queue()
    initial_task_queues = [ctx.Queue() for _ in range(config.workers)]
    initial_task_counts = [0 for _ in range(config.workers)]
    for task in tasks:
        target_worker = int(task["worker_id"]) % config.workers
        if initial_task_counts[target_worker] < config.games_per_worker:
            initial_task_queues[target_worker].put(task)
            initial_task_counts[target_worker] += 1
        else:
            global_task_queue.put(task)
    worker_task_queues = [
        _WorkerTaskQueue(initial_task_queues[worker_id], global_task_queue)
        for worker_id in range(config.workers)
    ]
    response_queues = [
        [ctx.Queue() for _ in range(config.games_per_worker)]
        for _ in range(config.workers)
    ]
    start_event = ctx.Event()

    shared_inputs = [
        torch.empty(
            (config.games_per_worker, *profile.observation_shape),
            dtype=torch.float32,
        ).share_memory_()
        for _ in range(config.workers)
    ]
    shared_policy = [
        torch.empty(
            (config.games_per_worker, profile.policy_size),
            dtype=torch.float32,
        ).share_memory_()
        for _ in range(config.workers)
    ]
    shared_wdl = [
        torch.empty(
            (config.games_per_worker, profile.wdl_size),
            dtype=torch.float32,
        ).share_memory_()
        for _ in range(config.workers)
    ]

    processes = []
    for worker_id in range(config.workers):
        process = ctx.Process(
            target=_worker_bootstrap,
            args=(
                profile.profile_id,
                worker_id,
                worker_task_queues[worker_id],
                config.games_per_worker,
                # Batching is exclusively broker-owned.  Keep this explicit
                # in the process ABI so a local timed window cannot return.
                0.0,
                candidate.model_hash,
                reference.model_hash,
                shared_inputs[worker_id],
                shared_policy[worker_id],
                shared_wdl[worker_id],
                request_queue,
                response_queues[worker_id],
                start_event,
            ),
            name=f"{profile.worker_process_prefix}-{worker_id:02d}",
        )
        process.start()
        processes.append(process)

    ready: dict[int, Mapping[str, object]] = {}
    startup_deadline = time.monotonic() + 60.0
    ingress: _ArenaBrokerIngress | None = None
    try:
        while len(ready) < config.workers:
            remaining = startup_deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    f"Arena worker startup timeout: {len(ready)}/{config.workers} ready"
                )
            try:
                message = request_queue.get(timeout=min(1.0, remaining))
            except Empty:
                dead = [process.name for process in processes if not process.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"Arena workers died before startup barrier: {dead}"
                    )
                continue
            if message.get("kind") == "ready":
                ready[int(message["worker_id"])] = message
            elif message.get("kind") == "error":
                raise RuntimeError(f"Arena worker startup failed: {message}")
            else:
                raise RuntimeError(
                    f"Unexpected Arena message before startup barrier: {message}"
                )

        worker_pids = [int(ready[index]["pid"]) for index in sorted(ready)]
        if len(set(worker_pids)) != config.workers:
            raise RuntimeError(
                "Arena did not start the requested number of distinct OS worker processes"
            )
        if any(bool(ready[index].get("cuda_initialized")) for index in ready):
            raise RuntimeError("CUDA initialized inside an Arena search worker")

        started = time.perf_counter()
        done: dict[int, Mapping[str, object]] = {}
        records: list[dict[str, object]] = []
        batch_rows: list[int] = []
        queue_wait_ms: list[float] = []
        worker_to_broker_transport_ms: list[float] = []
        broker_queue_wait_ms: list[float] = []
        broker_collection_wait_ms: list[float] = []
        end_to_end_inference_ms: list[float] = []
        batch_worker_counts: list[int] = []
        cap_hits = 0
        inference_started = time.perf_counter()
        model_hashes = tuple(models_by_hash)
        model_hash_by_role = {
            "candidate": candidate.model_hash,
            "reference": reference.model_hash,
        }
        scheduler = _ModelAwareBatchScheduler(
            model_hashes,
            cap=config.inference_batch_rows,
            wait_ms=config.inference_batch_wait_ms,
        )
        control_pending: deque[Mapping[str, object]] = deque()
        ingress = _ArenaBrokerIngress(request_queue, scheduler, control_pending)
        pending_depth_samples: list[int] = [0]
        pending_depth_by_model: dict[str, list[int]] = {
            model_hash: [0] for model_hash in model_hashes
        }
        batch_rows_by_model: dict[str, list[int]] = {
            model_hash: [] for model_hash in model_hashes
        }
        batch_worker_counts_by_model: dict[str, list[int]] = {
            model_hash: [] for model_hash in model_hashes
        }
        queue_wait_ms_by_model: dict[str, list[float]] = {
            model_hash: [] for model_hash in model_hashes
        }
        worker_to_broker_transport_ms_by_model: dict[str, list[float]] = {
            model_hash: [] for model_hash in model_hashes
        }
        broker_queue_wait_ms_by_model: dict[str, list[float]] = {
            model_hash: [] for model_hash in model_hashes
        }
        end_to_end_inference_ms_by_model: dict[str, list[float]] = {
            model_hash: [] for model_hash in model_hashes
        }
        collection_wait_ms_by_model: dict[str, list[float]] = {
            model_hash: [] for model_hash in model_hashes
        }
        model_forward_time_sec: dict[str, list[float]] = {
            model_hash: [] for model_hash in model_hashes
        }

        def record_pending_depth() -> None:
            pending_depth_samples.append(scheduler.pending_rows())
            for model_hash, rows in scheduler.pending_rows_by_model().items():
                pending_depth_by_model[model_hash].append(rows)

        def handle_non_inference(message: Mapping[str, object]) -> None:
            kind = message.get("kind")
            if kind == "done":
                wid = int(message["worker_id"])
                done[wid] = message
                records.extend(dict(row) for row in message.get("records", ()))
            elif kind == "error":
                raise RuntimeError(
                    f"Arena worker {message.get('worker_id')} failed: "
                    f"{message.get('error')}\n{message.get('traceback', '')}"
                )
            elif kind != "ready":
                raise RuntimeError(f"Unexpected Arena control message: {message}")

        def dispatch_model_batch(model_hash: str) -> None:
            nonlocal cap_hits
            requests, first_broker_received_at = scheduler.pop_batch(model_hash)
            dispatch_started = time.perf_counter()
            collection_wait_ms_by_model[model_hash].append(
                max(0.0, dispatch_started - first_broker_received_at) * 1000.0
            )
            broker_collection_wait_ms.append(
                max(0.0, dispatch_started - first_broker_received_at) * 1000.0
            )
            model = models_by_hash.get(model_hash)
            if model is None:
                raise RuntimeError(
                    f"Inference request references unknown model hash {model_hash}"
                )
            segments: list[dict[str, object]] = []
            for request in requests:
                raw_segments = request.get("segments")
                if not raw_segments:
                    # The production zero-wait transport emits one ordinary
                    # request per lane. Avoid copying that hot-path mapping.
                    segments.append(request)  # type: ignore[arg-type]
                    continue
                for raw_segment in raw_segments:  # type: ignore[union-attr]
                    if not isinstance(raw_segment, Mapping):
                        raise RuntimeError("Arena inference request has malformed segments")
                    segment = dict(raw_segment)
                    segment.setdefault(
                        "worker_enqueued_at", request["worker_enqueued_at"]
                    )
                    segment.setdefault(
                        "broker_received_at", request["broker_received_at"]
                    )
                    segments.append(segment)
            for request in requests:
                broker_received_at = float(request["broker_received_at"])
                worker_enqueued_at = float(request["worker_enqueued_at"])
                wait_ms = max(0.0, dispatch_started - broker_received_at) * 1000.0
                queue_wait_ms.append(wait_ms)
                queue_wait_ms_by_model[model_hash].append(wait_ms)
                broker_queue_wait_ms.append(wait_ms)
                worker_to_broker_transport_ms.append(
                    max(0.0, broker_received_at - worker_enqueued_at) * 1000.0
                )
                worker_to_broker_transport_ms_by_model[model_hash].append(
                    max(0.0, broker_received_at - worker_enqueued_at) * 1000.0
                )
                broker_queue_wait_ms_by_model[model_hash].append(wait_ms)
            cpu_batch = torch.cat(
                [
                    shared_inputs[int(segment["worker_id"])][
                        int(segment["lane_id"]): int(segment["lane_id"]) + int(segment["rows"])
                    ]
                    for segment in segments
                ],
                dim=0,
            )
            forward_started = time.perf_counter()
            try:
                policy, wdl = profile.infer_batch(model, cpu_batch, device)
            finally:
                forward_finished = time.perf_counter()
            model_forward_time_sec[model_hash].append(
                forward_finished - forward_started
            )
            offset = 0
            response_sent_at = time.perf_counter()
            for segment in segments:
                wid = int(segment["worker_id"])
                rows = int(segment["rows"])
                lane_id = int(segment["lane_id"])
                shared_policy[wid][lane_id:lane_id + rows].copy_(
                    policy[offset : offset + rows]
                )
                shared_wdl[wid][lane_id:lane_id + rows].copy_(
                    wdl[offset : offset + rows]
                )
                response_queues[wid][lane_id].put(
                    {
                        "ticket": int(segment["ticket"]),
                        "error": None,
                        "worker_enqueued_at": segment["worker_enqueued_at"],
                        "broker_received_at": segment["broker_received_at"],
                        "dispatch_started_at": dispatch_started,
                        "forward_started_at": forward_started,
                        "forward_finished_at": forward_finished,
                        "response_sent_at": response_sent_at,
                    }
                )
                segment_worker_enqueued_at = float(
                    segment.get("worker_enqueued_at", requests[0]["worker_enqueued_at"])
                )
                end_to_end_inference_ms.append(
                    max(0.0, response_sent_at - segment_worker_enqueued_at) * 1000.0
                )
                end_to_end_inference_ms_by_model[model_hash].append(
                    max(0.0, response_sent_at - segment_worker_enqueued_at) * 1000.0
                )
                offset += rows
            rows_this_call = int(cpu_batch.shape[0])
            worker_count = len({int(segment["worker_id"]) for segment in segments})
            batch_rows.append(rows_this_call)
            batch_rows_by_model[model_hash].append(rows_this_call)
            batch_worker_counts.append(worker_count)
            batch_worker_counts_by_model[model_hash].append(worker_count)
            if rows_this_call >= config.inference_batch_rows:
                cap_hits += 1
            record_pending_depth()

        ingress.start()
        start_event.set()
        while len(done) < config.workers:
            ingress.raise_if_failed()
            dead = [
                process.name
                for process in processes
                if not process.is_alive() and process.exitcode not in (0, None)
            ]
            if dead:
                raise RuntimeError(f"Arena worker process died during run: {dead}")
            with scheduler.condition:
                if control_pending:
                    message = control_pending.popleft()
                    ready_model = None
                else:
                    message = None
                    now = time.perf_counter()
                    ready_model = scheduler.next_ready_model(now)
                    if ready_model is None:
                        deadline = scheduler.next_deadline()
                        timeout = 1.0
                        if deadline is not None:
                            timeout = min(timeout, max(0.0, deadline - now))
                        scheduler.condition.wait(timeout=timeout)
                        continue
            if message is not None:
                handle_non_inference(message)
                record_pending_depth()
                continue
            if ready_model is not None:
                dispatch_model_batch(ready_model)
                continue

        wall_time = time.perf_counter() - started
        inference_wall = time.perf_counter() - inference_started
    except BaseException:
        if ingress is not None:
            ingress.stop()
        _terminate(processes)
        raise
    finally:
        start_event.set()
        if ingress is not None:
            ingress.stop()

    for process in processes:
        process.join(timeout=15.0)
    hung = [process.name for process in processes if process.is_alive()]
    if hung:
        _terminate(processes)
        raise RuntimeError(f"Arena worker processes did not exit: {hung}")
    bad_exit = [
        (process.name, process.exitcode)
        for process in processes
        if process.exitcode != 0
    ]
    if bad_exit:
        raise RuntimeError(f"Arena worker process exit failures: {bad_exit}")

    records.sort(key=lambda row: str(row["game_id"]))
    _write_jsonl(output_dir / "games.jsonl", records)
    summary = profile.summarize(
        records,
        candidate_label=candidate_label,
        reference_label=reference_label,
        pairs=pairs,
    )
    worker_cpu_seconds = [
        float(done[index].get("cpu_active_seconds", done[index]["cpu_seconds"]))
        for index in sorted(done)
    ]
    worker_wall_seconds = [
        float(done[index].get("wall_time_seconds", wall_time)) for index in sorted(done)
    ]
    worker_blocked_seconds = [
        float(done[index].get("blocked_inference_seconds", 0.0))
        for index in sorted(done)
    ]
    worker_lane_wall_seconds = [
        float(
            done[index].get(
                "lane_wall_time_seconds",
                done[index].get("wall_time_seconds", wall_time),
            )
        )
        for index in sorted(done)
    ]
    effective_cpu_cores = sum(worker_cpu_seconds) / wall_time if wall_time else 0.0
    mean_batch = statistics.mean(batch_rows) if batch_rows else 0.0
    cross_worker_calls = sum(count > 1 for count in batch_worker_counts)
    inference_by_model: dict[str, dict[str, object]] = {}
    for role, model_hash in model_hash_by_role.items():
        model_batches = _batch_summary(batch_rows_by_model[model_hash])
        inference_by_model[role] = {
            "model_hash": model_hash,
            **model_batches,
            "worker_to_broker_transport_ms": _numeric_summary(
                worker_to_broker_transport_ms_by_model[model_hash]
            ),
            "queue_wait_ms": _numeric_summary(queue_wait_ms_by_model[model_hash]),
            "broker_queue_wait_ms": _numeric_summary(
                broker_queue_wait_ms_by_model[model_hash]
            ),
            "collection_wait_ms": _numeric_summary(
                collection_wait_ms_by_model[model_hash]
            ),
            "end_to_end_inference_ms": _numeric_summary(
                end_to_end_inference_ms_by_model[model_hash]
            ),
            "forward_time_ms": _numeric_summary(
                [value * 1000.0 for value in model_forward_time_sec[model_hash]]
            ),
            "mean_workers_per_forward": (
                statistics.mean(batch_worker_counts_by_model[model_hash])
                if batch_worker_counts_by_model[model_hash]
                else 0.0
            ),
        }

    def _model_metric(metric: str, role: str) -> object:
        return inference_by_model[role][metric]

    all_forward_time_ms = [
        value * 1000.0
        for values in model_forward_time_sec.values()
        for value in values
    ]
    total_moves = sum(
        len(row.get("action_trace", ()))
        for row in records
    )
    telemetry: dict[str, object] = {
        "arena_engine": CANONICAL_ARENA_ENGINE,
        "arena_profile": profile.profile_id,
        "broker_pid": os.getpid(),
        "model_owner_pid": os.getpid(),
        "worker_pids": worker_pids,
        "worker_processes_requested": config.workers,
        "worker_processes_observed": len(set(worker_pids)),
        "games_per_worker_lane_capacity": config.games_per_worker,
        "aggregate_worker_cpu_seconds": sum(worker_cpu_seconds),
        "effective_cpu_cores": effective_cpu_cores,
        "effective_cpu_pct_of_requested": (
            100.0 * effective_cpu_cores / float(config.workers)
        ),
        "worker_cpu_seconds": worker_cpu_seconds,
        "worker_cpu_active_seconds": worker_cpu_seconds,
        "worker_wall_time_seconds": worker_wall_seconds,
        "worker_blocked_on_inference_seconds": worker_blocked_seconds,
        "worker_blocked_on_inference_calls": [
            int(done[index].get("blocked_inference_calls", 0))
            for index in sorted(done)
        ],
        "worker_lane_wall_time_seconds": worker_lane_wall_seconds,
        "worker_blocked_on_inference_time": _numeric_summary(worker_blocked_seconds),
        "worker_cpu_active_time": _numeric_summary(worker_cpu_seconds),
        "worker_inference_wait_fraction_by_worker": [
            blocked / lane_wall if lane_wall else 0.0
            for blocked, lane_wall in zip(
                worker_blocked_seconds, worker_lane_wall_seconds
            )
        ],
        "fraction_wall_time_workers_spent_waiting_for_inference": (
            sum(worker_blocked_seconds) / sum(worker_lane_wall_seconds)
            if sum(worker_lane_wall_seconds)
            else 0.0
        ),
        "worker_max_rss_kb": [
            int(done[index]["max_rss_kb"]) for index in sorted(done)
        ],
        "worker_cuda_initialized_after_run": [
            bool(done[index]["cuda_initialized"]) for index in sorted(done)
        ],
        "inference_forward_calls": len(batch_rows),
        "inference_rows": sum(batch_rows),
        "inference_rows_per_sec": (
            sum(batch_rows) / inference_wall if inference_wall else 0.0
        ),
        "mean_inference_batch_rows": mean_batch,
        "p50_inference_batch_rows": _percentile(batch_rows, 0.50),
        "p95_inference_batch_rows": _percentile(batch_rows, 0.95),
        "max_inference_batch_rows": max(batch_rows, default=0),
        "inference_batch_rows_cap": config.inference_batch_rows,
        "inference_batch_wait_ms": config.inference_batch_wait_ms,
        "worker_local_inference_batch_wait_ms": 0.0,
        "batch_aggregation_scope": "cross_process",
        "batch_cap_hits": cap_hits,
        "batch_cap_hit_rate": cap_hits / len(batch_rows) if batch_rows else 0.0,
        "cross_worker_inference_calls": cross_worker_calls,
        "cross_worker_inference_call_rate": (
            cross_worker_calls / len(batch_worker_counts)
            if batch_worker_counts
            else 0.0
        ),
        "mean_workers_per_inference_call": (
            statistics.mean(batch_worker_counts) if batch_worker_counts else 0.0
        ),
        "queue_wait_ms_mean": (
            statistics.mean(queue_wait_ms) if queue_wait_ms else 0.0
        ),
        "queue_wait_ms_p95": (
            float(
                _percentile(
                    [int(round(value * 1000.0)) for value in queue_wait_ms],
                    0.95,
                )
            )
            / 1000.0
            if queue_wait_ms
            else 0.0
        ),
        "worker_to_broker_transport_ms": _numeric_summary(
            worker_to_broker_transport_ms
        ),
        "broker_queue_wait_ms": _numeric_summary(broker_queue_wait_ms),
        "broker_collection_wait_ms": _numeric_summary(
            broker_collection_wait_ms
        ),
        "end_to_end_inference_ms": _numeric_summary(end_to_end_inference_ms),
        "inference_timestamp_semantics": {
            "worker_enqueued_at": "worker before request_queue.put",
            "broker_received_at": "ingress thread receipt; central deadline origin",
            "dispatch_started_at": "model-aware batch removed from pending queue",
            "forward_started_at": "profile.infer_batch entry",
            "forward_finished_at": "profile.infer_batch return",
            "response_sent_at": "response queue put",
        },
        "model_aware_batching": True,
        "inference_by_model": inference_by_model,
        "candidate_forward_calls": int(_model_metric("forward_calls", "candidate")),
        "candidate_inference_rows": int(_model_metric("rows", "candidate")),
        "candidate_mean_inference_batch_rows": float(
            _model_metric("mean_batch_rows", "candidate")
        ),
        "candidate_p50_inference_batch_rows": int(
            _model_metric("p50_batch_rows", "candidate")
        ),
        "candidate_p95_inference_batch_rows": int(
            _model_metric("p95_batch_rows", "candidate")
        ),
        "candidate_max_inference_batch_rows": int(
            _model_metric("max_batch_rows", "candidate")
        ),
        "reference_forward_calls": int(_model_metric("forward_calls", "reference")),
        "reference_inference_rows": int(_model_metric("rows", "reference")),
        "reference_mean_inference_batch_rows": float(
            _model_metric("mean_batch_rows", "reference")
        ),
        "reference_p50_inference_batch_rows": int(
            _model_metric("p50_batch_rows", "reference")
        ),
        "reference_p95_inference_batch_rows": int(
            _model_metric("p95_batch_rows", "reference")
        ),
        "reference_max_inference_batch_rows": int(
            _model_metric("max_batch_rows", "reference")
        ),
        "global_pending_queue_depth": _numeric_summary(pending_depth_samples),
        "pending_queue_depth_by_model": {
            role: _numeric_summary(pending_depth_by_model[model_hash])
            for role, model_hash in model_hash_by_role.items()
        },
        "broker_collection_wait_ms_by_model": {
            role: _numeric_summary(collection_wait_ms_by_model[model_hash])
            for role, model_hash in model_hash_by_role.items()
        },
        "model_forward_time_ms": _numeric_summary(all_forward_time_ms),
        "model_forward_time_ms_by_model": {
            role: _numeric_summary(
                [value * 1000.0 for value in model_forward_time_sec[model_hash]]
            )
            for role, model_hash in model_hash_by_role.items()
        },
        "wall_time_sec": wall_time,
        "moves": total_moves,
        "moves_per_sec": total_moves / wall_time if wall_time else 0.0,
        "games_per_hour": (
            len(records) * 3600.0 / wall_time if wall_time else 0.0
        ),
        "technical_games": int(summary["technical_games"]),
    }

    performance_failures: list[str] = []
    if len(set(worker_pids)) != config.workers:
        performance_failures.append("worker_pid_count")
    if any(telemetry["worker_cuda_initialized_after_run"]):
        performance_failures.append("cuda_in_worker")
    if mean_batch < config.min_mean_inference_batch_rows:
        performance_failures.append("mean_inference_batch_rows")
    if effective_cpu_cores < config.min_effective_cpu_cores:
        performance_failures.append("effective_cpu_cores")
    if int(summary["technical_games"]) != 0:
        performance_failures.append("technical_games")
    telemetry["performance_status"] = (
        "PASS" if not performance_failures else "PERFORMANCE_DEGRADED"
    )
    telemetry["performance_failures"] = performance_failures

    summary.update(
        {
            "arena_engine": CANONICAL_ARENA_ENGINE,
            "arena_profile": profile.profile_id,
            "comparison": comparison,
            "run_id": run_id,
            "candidate_model_hash": candidate.model_hash,
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "reference_model_hash": reference.model_hash,
            "reference_artifact_sha256": reference.artifact_sha256,
            "scientific_contract": dict(profile.scientific_contract(config)),
            "execution": {
                "workers": config.workers,
                "games_per_worker": config.games_per_worker,
                "inference_batch_rows": config.inference_batch_rows,
                "inference_batch_wait_ms": config.inference_batch_wait_ms,
                "device": str(device),
                "strict_production": config.strict_production,
            },
            "telemetry": telemetry,
        }
    )
    _write_json(output_dir / "summary.json", summary)

    def _identity_payload(identity: CheckpointIdentity) -> dict[str, object]:
        return {
            "path": str(identity.path),
            "model_hash": identity.model_hash,
            "artifact_sha256": identity.artifact_sha256,
            "architecture_config": dict(identity.architecture_config),
        }

    manifest = {
        "schema_version": 2,
        "arena_engine": CANONICAL_ARENA_ENGINE,
        "arena_profile": profile.profile_id,
        "run_id": run_id,
        "comparison": comparison,
        "candidate": _identity_payload(candidate),
        "reference": _identity_payload(reference),
        "master_seed": master_seed,
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.json"),
        "games": str(output_dir / "games.jsonl"),
        "performance_status": telemetry["performance_status"],
    }
    _write_json(output_dir / "manifest.json", manifest)

    if config.strict_production and performance_failures:
        raise RuntimeError(
            "Arena completed but failed production performance/validity gates: "
            + ", ".join(performance_failures)
        )
    return summary
