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
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

_MODULE_IMPORT_STARTED_AT = time.perf_counter()
CANONICAL_ARENA_ENGINE = "process-central-inference-v1"
DEFAULT_GAMES = 192
DEFAULT_WORKERS = 16
DEFAULT_GAMES_PER_WORKER = 12
DEFAULT_INFERENCE_BATCH_ROWS = 64
DEFAULT_INFERENCE_BATCH_WAIT_MS = 4.0
DEFAULT_MASTER_SEED = 202609131004
# Standard-64 Arena keeps the Golden reference as an advisory performance
# band.  Mean batching never fails Arena closed; real execution/correctness
# failures are handled separately.  Keep the generic/default config at the
# legacy value for workloads other than standard-64.
MIN_MEAN_INFERENCE_BATCH_ROWS = 9.0
STANDARD_64_GAMES = 64
STANDARD_64_HEALTHY_MEAN_INFERENCE_BATCH_ROWS = 13.25
DEFAULT_NON_STANDARD_MIN_MEAN_INFERENCE_BATCH_ROWS = 16.0
MIN_EFFECTIVE_CPU_CORES = 8.0


def _report_progress_from_activity(
    event: str,
    completed_games: int,
    total_games: int,
    progress_callback: Callable[[int, int], None] | None,
) -> None:
    """Expose move activity to supervision without changing game counts."""
    if event in {"move_completed", "game_completed"} and progress_callback is not None:
        progress_callback(completed_games, total_games)


def _unreported_worker_exits(
    processes: Sequence[Any],
    done: Mapping[int, Mapping[str, object]],
    dead_since: dict[str, float],
    *,
    now: float,
    grace_seconds: float = 5.0,
) -> list[str]:
    """Return workers that exited without delivering their terminal message."""
    failures: list[str] = []
    for process in processes:
        if process.is_alive() or process.name in done:
            continue
        first_seen = dead_since.setdefault(process.name, now)
        if process.exitcode not in (0, None) or now - first_seen >= grace_seconds:
            failures.append(f"{process.name} (exitcode={process.exitcode})")
    return failures


def _expected_lane_ids_by_worker(
    initial_task_counts: Sequence[int], games_per_worker: int
) -> dict[int, tuple[int, ...]]:
    """Return the lanes a workload can actually fill on each worker.

    ``games_per_worker`` is a capacity, not a promise that every lane can be
    occupied when the requested workload is smaller than the global capacity.
    The initial round-robin task allocation is the authoritative expectation
    for the early and final lane-occupancy checks.
    """
    capacity = int(games_per_worker)
    return {
        worker_id: tuple(range(min(capacity, max(0, int(task_count)))))
        for worker_id, task_count in enumerate(initial_task_counts)
    }


@dataclass(frozen=True)
class ArenaExecutionConfig:
    games: int = DEFAULT_GAMES
    workers: int = DEFAULT_WORKERS
    games_per_worker: int = DEFAULT_GAMES_PER_WORKER
    inference_batch_rows: int = DEFAULT_INFERENCE_BATCH_ROWS
    inference_batch_wait_ms: float = DEFAULT_INFERENCE_BATCH_WAIT_MS
    device: str = "cuda"
    strict_production: bool = True
    # Scientific/correctness production checks remain active independently of
    # this switch.  Enable this only when a caller explicitly wants the
    # observed performance band to be a hard admission gate.
    strict_performance: bool = False
    # Explicit wiring/monitoring acceptance may use a smaller game count while
    # retaining the strict CUDA, worker, and technical checks. Performance
    # observations become hard gates only with strict_performance=True.
    monitoring_acceptance: bool = False
    min_mean_inference_batch_rows: float = DEFAULT_NON_STANDARD_MIN_MEAN_INFERENCE_BATCH_ROWS
    min_effective_cpu_cores: float = MIN_EFFECTIVE_CPU_CORES
    early_gate_enabled: bool = True
    early_gate_min_forwards: int = 128
    early_gate_min_wall_sec: float = 5.0

    def validate_base(self) -> None:
        if self.games <= 0 or self.games % 2:
            raise ValueError("Arena games must be a positive even number")
        if self.workers <= 0 or self.games_per_worker <= 0:
            raise ValueError("Arena workers/games_per_worker must be positive")
        if self.inference_batch_rows <= 0 or self.inference_batch_wait_ms < 0.0:
            raise ValueError("Arena inference batching settings are invalid")
        if type(self.strict_production) is not bool or type(self.strict_performance) is not bool:
            raise ValueError("Arena strict production/performance settings must be boolean")
        if self.inference_batch_rows < self.games_per_worker:
            raise ValueError("inference_batch_rows must cover at least one worker request")
        if self.early_gate_min_forwards <= 0 or self.early_gate_min_wall_sec < 0.0:
            raise ValueError("Arena early performance gate settings are invalid")


def classify_arena_performance(
    mean_inference_batch_rows: float,
    config: ArenaExecutionConfig,
) -> dict[str, object]:
    """Classify Arena batching without turning the healthy target into a hard gate.

    Standard-64 uses the Golden reference of 13.25 rows as the healthy line.
    Values below 9.0 are severe warnings, never hard failures. Other
    workloads retain their explicitly configured reference band.
    """
    mean = float(mean_inference_batch_rows)
    if config.games == STANDARD_64_GAMES:
        severe_warning_threshold = MIN_MEAN_INFERENCE_BATCH_ROWS
        healthy_min = STANDARD_64_HEALTHY_MEAN_INFERENCE_BATCH_ROWS
    else:
        severe_warning_threshold = float(config.min_mean_inference_batch_rows)
        healthy_min = severe_warning_threshold

    if mean < severe_warning_threshold:
        return {
            "status": "SEVERE_WARNING",
            "hard_failures": [],
            "warnings": ["mean_inference_batch_rows"],
            "mean_inference_batch_rows": mean,
            "severe_warning_threshold": severe_warning_threshold,
            "healthy_minimum": healthy_min,
        }
    if mean < healthy_min:
        return {
            "status": "WARNING",
            "hard_failures": [],
            "warnings": ["mean_inference_batch_rows"],
            "mean_inference_batch_rows": mean,
            "severe_warning_threshold": severe_warning_threshold,
            "healthy_minimum": healthy_min,
        }
    return {
        "status": "HEALTHY",
        "hard_failures": [],
        "warnings": [],
        "mean_inference_batch_rows": mean,
        "severe_warning_threshold": severe_warning_threshold,
        "healthy_minimum": healthy_min,
    }


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
        workload: Mapping[str, object] | None = None,
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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


class _ProcessTreeCpuSampler:
    """Low-overhead effective-core sampler for the Arena process tree.

    This is diagnostic only.  It reads cumulative user/system CPU ticks from
    ``/proc`` and never participates in scheduling or correctness decisions.
    """

    def __init__(self, pids: Sequence[int], interval_s: float = 1.0) -> None:
        self.pids = tuple(sorted({int(pid) for pid in pids if int(pid) > 0}))
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[float] = []
        self._previous: tuple[float, float] | None = None
        try:
            self._clock_ticks = float(os.sysconf("SC_CLK_TCK"))
        except (AttributeError, OSError, ValueError):
            self._clock_ticks = 100.0

    def _cpu_seconds(self) -> float | None:
        total_ticks = 0
        observed = False
        for pid in self.pids:
            try:
                raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
                command_end = raw.rfind(")")
                fields = raw[command_end + 2 :].split()
                if len(fields) < 13:
                    continue
                total_ticks += int(fields[11]) + int(fields[12])
                observed = True
            except (OSError, ValueError, IndexError):
                continue
        return total_ticks / self._clock_ticks if observed else None

    def _record(self) -> None:
        now = time.perf_counter()
        cpu_seconds = self._cpu_seconds()
        if cpu_seconds is not None and self._previous is not None:
            previous_at, previous_cpu = self._previous
            elapsed = now - previous_at
            if elapsed > 0.0:
                self._samples.append(max(0.0, cpu_seconds - previous_cpu) / elapsed)
        if cpu_seconds is not None:
            self._previous = (now, cpu_seconds)

    def _run(self) -> None:
        self._record()
        while not self._stop.wait(self.interval_s):
            self._record()
        self._record()

    def start(self) -> "_ProcessTreeCpuSampler":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run,
                name="arena-process-tree-cpu",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 3.0))
        self._thread = None

    def summary(self) -> dict[str, object]:
        if not self._samples:
            return {
                "average_effective_cores": None,
                "p50_effective_cores": None,
                "p95_effective_cores": None,
                "peak_effective_cores": None,
                "samples": 0,
                "source": "procfs aggregate user+system CPU ticks",
            }
        return {
            "average_effective_cores": statistics.fmean(self._samples),
            "p50_effective_cores": _percentile_float(self._samples, 0.50),
            "p95_effective_cores": _percentile_float(self._samples, 0.95),
            "peak_effective_cores": max(self._samples),
            "samples": len(self._samples),
            "source": "procfs aggregate user+system CPU ticks",
        }


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


def _percentile_float(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[index]


def _batch_summary(values: Sequence[int]) -> dict[str, float | int]:
    """Return the standard batch telemetry shape for one model."""
    return {
        "forward_calls": len(values),
        "rows": sum(values),
        "mean_batch_rows": statistics.mean(values) if values else 0.0,
        "p50_batch_rows": _percentile(values, 0.50),
        "p95_batch_rows": _percentile(values, 0.95),
        "p99_batch_rows": _percentile(values, 0.99),
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
        self.first_inference_received_at: float | None = None
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
                    if self.first_inference_received_at is None:
                        self.first_inference_received_at = broker_received_at
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
    progress_callback: Callable[[int, int], None] | None = None,
    workload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run the single production Arena engine with one game-specific profile."""
    process_started_at = time.perf_counter()
    startup_phases: dict[str, dict[str, float]] = {}
    diagnostic_output_dir: Path | None = None

    def measure_startup_phase(name: str, callback: Any) -> Any:
        phase_started_at = time.perf_counter()
        try:
            result = callback()
        except BaseException as exc:
            phase_finished_at = time.perf_counter()
            startup_phases[name] = {
                "start_sec": phase_started_at - process_started_at,
                "end_sec": phase_finished_at - process_started_at,
                "duration_sec": phase_finished_at - phase_started_at,
            }
            if diagnostic_output_dir is not None:
                try:
                    _write_json(
                        diagnostic_output_dir / "startup-failure.json",
                        {
                            "status": "STARTUP_FAILED",
                            "failed_phase": name,
                            "error": f"{type(exc).__name__}: {exc}",
                            "startup_timing": {"phases": startup_phases},
                        },
                    )
                except Exception:
                    pass
            raise
        phase_finished_at = time.perf_counter()
        startup_phases[name] = {
            "start_sec": phase_started_at - process_started_at,
            "end_sec": phase_finished_at - process_started_at,
            "duration_sec": phase_finished_at - phase_started_at,
        }
        return result

    measure_startup_phase(
        "config_profile_resolution",
        lambda: (config.validate_base(), profile.validate_execution_config(config)),
    )

    candidate_path, reference_path = measure_startup_phase(
        "checkpoint_discovery",
        lambda: (candidate_path.resolve(), (reference_path or candidate_path).resolve()),
    )
    output_dir = output_dir.resolve()
    if any(
        (output_dir / name).exists()
        for name in ("manifest.json", "games.jsonl", "summary.json")
    ):
        raise FileExistsError(f"Arena output directory already contains a run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_output_dir = output_dir
    candidate = measure_startup_phase(
        "candidate_metadata_read_and_hash",
        lambda: profile.load_identity(candidate_path),
    )
    reference = measure_startup_phase(
        "reference_metadata_read_and_hash",
        lambda: profile.load_identity(reference_path),
    )
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
    tasks, pairs = measure_startup_phase(
        "task_corpus_construction",
        lambda: profile.build_tasks(
            run_id=run_id,
            comparison=comparison,
            candidate=candidate,
            reference=reference,
            master_seed=master_seed,
            games=config.games,
            workers=config.workers,
            workload=workload,
        ),
    )
    if len(tasks) != config.games:
        raise RuntimeError(
            f"Arena profile {profile.profile_id!r} produced {len(tasks)} tasks "
            f"for requested games={config.games}"
        )
    if device.type == "cuda":
        measure_startup_phase("cuda_context_init", torch.cuda.init)
    candidate_model = measure_startup_phase(
        "candidate_model_construction_and_state_dict_load",
        lambda: profile.load_parent_model(candidate, device),
    )
    if reference.model_hash == candidate.model_hash and reference.architecture_config == candidate.architecture_config:
        reference_model = candidate_model
        startup_phases["reference_model_reuse"] = {
            "start_sec": time.perf_counter() - process_started_at,
            "end_sec": time.perf_counter() - process_started_at,
            "duration_sec": 0.0,
        }
    else:
        reference_model = measure_startup_phase(
            "reference_model_construction_and_state_dict_load",
            lambda: profile.load_parent_model(reference, device),
        )
    models_by_hash = {
        candidate.model_hash: candidate_model,
        reference.model_hash: reference_model,
    }

    warmup_input = torch.zeros(
        (1, *profile.observation_shape), dtype=torch.float32
    )
    measure_startup_phase(
        "candidate_model_warmup",
        lambda: profile.infer_batch(candidate_model, warmup_input, device),
    )
    if reference_model is not candidate_model:
        measure_startup_phase(
            "reference_model_warmup",
            lambda: profile.infer_batch(reference_model, warmup_input, device),
        )

    ctx = __import__("multiprocessing").get_context("spawn")
    def allocate_runtime() -> tuple[Any, ...]:
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
        return (
            request_queue,
            global_task_queue,
            initial_task_counts,
            worker_task_queues,
            response_queues,
            shared_inputs,
            shared_policy,
            shared_wdl,
            start_event,
        )

    (
        request_queue,
        global_task_queue,
        initial_task_counts,
        worker_task_queues,
        response_queues,
        shared_inputs,
        shared_policy,
        shared_wdl,
        start_event,
    ) = measure_startup_phase("shared_memory_and_queue_creation", allocate_runtime)

    processes = []
    process_spawn_started_at = time.perf_counter()
    try:
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
    except BaseException:
        _terminate(processes)
        raise
    startup_phases["process_spawn"] = {
        "start_sec": process_spawn_started_at - process_started_at,
        "end_sec": time.perf_counter() - process_started_at,
        "duration_sec": time.perf_counter() - process_spawn_started_at,
    }

    hardware_sampler = None
    hardware_summary: dict[str, object] = {"samples": 0, "phases": {}}
    process_tree_sampler: _ProcessTreeCpuSampler | None = None
    process_tree_summary: dict[str, object] = {}
    try:
        from tools.hardware_telemetry import HardwareTelemetry

        hardware_sampler = HardwareTelemetry(
            output_dir / "hardware-telemetry.jsonl", interval_s=1.0
        )
        hardware_sampler.set_phase("ARENA")
        hardware_sampler.start()
    except Exception:
        # Hardware sampling is diagnostic only and must never change Arena
        # correctness or fail-closed result handling.
        hardware_sampler = None

    ready: dict[int, Mapping[str, object]] = {}
    startup_deadline = time.monotonic() + 60.0
    ingress: _ArenaBrokerIngress | None = None
    try:
        ready_barrier_started_at = time.perf_counter()
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
                worker_id = int(message["worker_id"])
                expected_lane_ids = list(range(config.games_per_worker))
                if list(message.get("observed_lane_ids", ())) != expected_lane_ids:
                    raise RuntimeError(
                        f"Arena worker {worker_id} did not expose all configured lanes"
                    )
                ready[worker_id] = message
            elif message.get("kind") == "error":
                raise RuntimeError(f"Arena worker startup failed: {message}")
            else:
                raise RuntimeError(
                    f"Unexpected Arena message before startup barrier: {message}"
                )
        startup_phases["all_workers_ready_barrier"] = {
            "start_sec": ready_barrier_started_at - process_started_at,
            "end_sec": time.perf_counter() - process_started_at,
            "duration_sec": time.perf_counter() - ready_barrier_started_at,
        }

        worker_pids = [int(ready[index]["pid"]) for index in sorted(ready)]
        if len(set(worker_pids)) != config.workers:
            raise RuntimeError(
                "Arena did not start the requested number of distinct OS worker processes"
            )
        if any(bool(ready[index].get("cuda_initialized")) for index in ready):
            raise RuntimeError("CUDA initialized inside an Arena search worker")

        started = time.perf_counter()
        parent_cpu_started = time.process_time()
        process_tree_sampler = _ProcessTreeCpuSampler(
            [os.getpid(), *worker_pids], interval_s=1.0
        ).start()
        done: dict[int, Mapping[str, object]] = {}
        dead_since: dict[str, float] = {}
        records: list[dict[str, object]] = []
        batch_rows: list[int] = []
        queue_wait_ms: list[float] = []
        worker_to_broker_transport_ms: list[float] = []
        broker_queue_wait_ms: list[float] = []
        broker_collection_wait_ms: list[float] = []
        end_to_end_inference_ms: list[float] = []
        inference_request_count = 0
        h2d_ms: list[float] = []
        model_forward_ms: list[float] = []
        d2h_ms: list[float] = []
        batch_worker_counts: list[int] = []
        dispatch_reasons: dict[str, int] = {
            "cap_reached": 0,
            "deadline_reached": 0,
            "queue_drained": 0,
            "shutdown_tail": 0,
        }
        rows_pending_at_dispatch: list[int] = []
        cap_hits = 0
        inference_started = time.perf_counter()
        model_hash_by_role = {
            "candidate": candidate.model_hash,
            "reference": reference.model_hash,
        }
        model_hashes = tuple(dict.fromkeys(models_by_hash))
        configured_context_capacity = config.workers * config.games_per_worker
        initial_game_count = sum(initial_task_counts)
        expected_lane_ids_by_worker = _expected_lane_ids_by_worker(
            initial_task_counts,
            config.games_per_worker,
        )
        active_contexts_current = 0
        active_context_last_at = started
        active_context_samples: list[int] = []
        active_context_durations: dict[int, float] = {
            count: 0.0 for count in range(configured_context_capacity + 1)
        }
        per_worker_active: dict[int, int] = {
            worker_id: 0 for worker_id in range(config.workers)
        }
        per_worker_last_at: dict[int, float] = {
            worker_id: started for worker_id in range(config.workers)
        }
        per_worker_active_durations: dict[int, dict[int, float]] = {
            worker_id: {
                count: 0.0 for count in range(config.games_per_worker + 1)
            }
            for worker_id in range(config.workers)
        }
        activity_lane_ids_by_worker: dict[int, set[int]] = {
            worker_id: set() for worker_id in range(config.workers)
        }
        activity_events = 0
        started_games = 0
        completed_games = 0
        lane_replenishments = 0
        peak_active_contexts = 0
        steady_state_started_at: float | None = None
        pending_empty_at: float | None = None
        steady_state_samples: list[int] = []
        first_worker_request_at: float | None = None
        first_cuda_forward_at: float | None = None
        first_completed_move_at: float | None = None
        first_completed_game_at: float | None = None
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

        def _activity_timestamp(message: Mapping[str, object]) -> float:
            value = float(message.get("at", time.perf_counter()))
            return max(value, active_context_last_at)

        def record_activity(message: Mapping[str, object]) -> None:
            nonlocal active_contexts_current
            nonlocal active_context_last_at
            nonlocal activity_events
            nonlocal first_completed_move_at
            nonlocal first_completed_game_at
            nonlocal lane_replenishments
            nonlocal peak_active_contexts
            nonlocal pending_empty_at
            nonlocal started_games
            nonlocal completed_games
            nonlocal steady_state_started_at
            at = _activity_timestamp(message)
            elapsed = max(0.0, at - active_context_last_at)
            active_context_durations[active_contexts_current] += elapsed
            for worker_id, count in per_worker_active.items():
                per_worker_active_durations[worker_id][count] += max(
                    0.0, at - per_worker_last_at[worker_id]
                )
                per_worker_last_at[worker_id] = at
            active_context_last_at = at
            activity_events += 1
            event = str(message.get("event", ""))
            worker_id = int(message.get("worker_id", -1))
            lane_id = int(message.get("lane_id", -1))
            if worker_id in activity_lane_ids_by_worker and lane_id >= 0:
                activity_lane_ids_by_worker[worker_id].add(lane_id)
            if event == "game_started":
                started_games += 1
                if started_games > initial_game_count:
                    lane_replenishments += 1
                if started_games >= config.games and pending_empty_at is None:
                    pending_empty_at = at
                active_contexts_current += 1
                if worker_id in per_worker_active:
                    per_worker_active[worker_id] += 1
                peak_active_contexts = max(peak_active_contexts, active_contexts_current)
                if active_contexts_current >= min(config.games, configured_context_capacity):
                    if steady_state_started_at is None:
                        steady_state_started_at = at
            elif event == "game_completed":
                completed_games += 1
                active_contexts_current = max(0, active_contexts_current - 1)
                if worker_id in per_worker_active:
                    per_worker_active[worker_id] = max(
                        0, per_worker_active[worker_id] - 1
                    )
                if first_completed_game_at is None:
                    first_completed_game_at = at
            elif event == "move_completed" and first_completed_move_at is None:
                first_completed_move_at = at
            # A long game can legitimately take longer than the supervisor's
            # completed-game timeout at high simulation budgets. Report move
            # activity as progress too, while keeping the scientific
            # completed-game count unchanged.
            _report_progress_from_activity(
                event, completed_games, config.games, progress_callback
            )
            active_context_samples.append(active_contexts_current)
            if (
                steady_state_started_at is not None
                and (pending_empty_at is None or at <= pending_empty_at)
            ):
                steady_state_samples.append(active_contexts_current)

        def handle_non_inference(message: Mapping[str, object]) -> None:
            kind = message.get("kind")
            if kind == "done":
                wid = int(message["worker_id"])
                done[wid] = message
                activity_lane_ids_by_worker[wid].update(
                    int(lane_id) for lane_id in message.get("used_lane_ids", ())
                )
                records.extend(dict(row) for row in message.get("records", ()))
            elif kind == "activity":
                record_activity(message)
            elif kind == "error":
                raise RuntimeError(
                    f"Arena worker {message.get('worker_id')} failed: "
                    f"{message.get('error')}\n{message.get('traceback', '')}"
                )
            elif kind != "ready":
                raise RuntimeError(f"Unexpected Arena control message: {message}")

        def dispatch_model_batch(model_hash: str) -> None:
            nonlocal cap_hits
            nonlocal inference_request_count
            nonlocal first_worker_request_at
            nonlocal first_cuda_forward_at
            with scheduler.condition:
                queue_before_dispatch = scheduler.queues[model_hash].rows
                queue_first_received = scheduler.queues[model_hash].first_broker_received_at
            if queue_before_dispatch >= config.inference_batch_rows:
                dispatch_reasons["cap_reached"] += 1
            elif queue_first_received is not None:
                dispatch_reasons["deadline_reached"] += 1
            else:
                dispatch_reasons["queue_drained"] += 1
            rows_pending_at_dispatch.append(queue_before_dispatch)
            requests, first_broker_received_at = scheduler.pop_batch(model_hash)
            inference_request_count += len(requests)
            dispatch_started = time.perf_counter()
            if first_worker_request_at is None:
                first_worker_request_at = dispatch_started
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
                    segment.setdefault("model_role", request["model_role"])
                    segment.setdefault("model_hash", request["model_hash"])
                    segment.setdefault("generation", request["generation"])
                    segment.setdefault("ticket", request["ticket"])
                    segment.setdefault("worker_id", request["worker_id"])
                    segment.setdefault("lane_id", request["lane_id"])
                    segments.append(segment)
            for request in requests:
                role = str(request.get("model_role", ""))
                if role not in model_hash_by_role:
                    raise RuntimeError("Arena inference request has an unknown model role")
                if model_hash_by_role[role] != model_hash:
                    raise RuntimeError("Arena inference request model role/hash mismatch")
                if "generation" not in request or "ticket" not in request:
                    raise RuntimeError("Arena inference request lost ticket generation")
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
            if first_cuda_forward_at is None:
                first_cuda_forward_at = forward_started
            try:
                policy, wdl = profile.infer_batch(model, cpu_batch, device)
            finally:
                forward_finished = time.perf_counter()
            timing = getattr(profile, "last_infer_timing", {})
            if isinstance(timing, Mapping):
                for values, key in (
                    (h2d_ms, "h2d_ms"),
                    (model_forward_ms, "forward_ms"),
                    (d2h_ms, "d2h_ms"),
                ):
                    value = timing.get(key)
                    if isinstance(value, (int, float)):
                        values.append(float(value))
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
                        "worker_id": wid,
                        "lane_id": lane_id,
                        "ticket": int(segment["ticket"]),
                        "generation": int(segment["generation"]),
                        "game_id": str(segment.get("game_id", "")),
                        "model_role": str(segment["model_role"]),
                        "model_hash": str(segment["model_hash"]),
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

        early_gate_checked = False

        def check_early_performance_gate() -> None:
            nonlocal early_gate_checked
            if early_gate_checked or not config.strict_production or not config.early_gate_enabled:
                return
            if len(batch_rows) < config.early_gate_min_forwards:
                return
            elapsed = time.perf_counter() - started
            if elapsed < config.early_gate_min_wall_sec:
                return
            early_gate_checked = True
            expected_active = min(config.games, configured_context_capacity)
            recent = batch_rows[-config.early_gate_min_forwards:]
            observed_lanes = all(
                set(expected_lane_ids_by_worker[worker_id]).issubset(
                    activity_lane_ids_by_worker[worker_id]
                )
                for worker_id in range(config.workers)
            )
            performance_diagnostics: list[str] = []
            if peak_active_contexts < expected_active:
                performance_diagnostics.append("active_contexts")
            if not observed_lanes:
                performance_diagnostics.append("lane_occupancy")
            recent_policy = classify_arena_performance(statistics.mean(recent), config)
            performance_diagnostics.extend(str(value) for value in recent_policy["warnings"])
            if config.strict_performance and performance_diagnostics:
                _write_json(
                    output_dir / "performance-degraded.json",
                    {
                        "status": "CRITICAL",
                        "reason": performance_diagnostics,
                        "observed_peak_active_contexts": peak_active_contexts,
                        "expected_active_contexts": expected_active,
                        "observed_unique_lane_ids_per_worker": {
                            str(worker_id): sorted(lanes)
                            for worker_id, lanes in activity_lane_ids_by_worker.items()
                        },
                        "expected_unique_lane_ids_per_worker": {
                            str(worker_id): list(lanes)
                            for worker_id, lanes in expected_lane_ids_by_worker.items()
                        },
                        "recent_mean_inference_batch_rows": statistics.mean(recent),
                        "forwards_observed": len(batch_rows),
                        "wall_time_sec": elapsed,
                    },
                )
                raise RuntimeError(
                    "Arena execution gate failed: "
                    + ", ".join(performance_diagnostics)
                )
            if performance_diagnostics:
                _write_json(
                    output_dir / "performance-warning.json",
                    {
                        "status": "DIAGNOSTIC",
                        "reason": sorted(set(performance_diagnostics)),
                        "forwards_observed": len(batch_rows),
                        "wall_time_sec": elapsed,
                    },
                )

        broker_started_at = time.perf_counter()
        ingress.start()
        startup_phases["broker_start"] = {
            "start_sec": broker_started_at - process_started_at,
            "end_sec": time.perf_counter() - process_started_at,
            "duration_sec": time.perf_counter() - broker_started_at,
        }
        ingress_started_at = time.perf_counter()
        start_event.set()
        startup_phases["ingress_start"] = {
            "start_sec": ingress_started_at - process_started_at,
            "end_sec": time.perf_counter() - process_started_at,
            "duration_sec": time.perf_counter() - ingress_started_at,
        }
        while len(done) < config.workers:
            ingress.raise_if_failed()
            with scheduler.condition:
                if control_pending:
                    message = control_pending.popleft()
                    ready_model = None
                else:
                    dead = _unreported_worker_exits(
                        processes,
                        done,
                        dead_since,
                        now=time.monotonic(),
                    )
                    if dead:
                        raise RuntimeError(
                            "Arena worker process exited without a done message: "
                            + ", ".join(dead)
                        )
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
                check_early_performance_gate()
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
        if hardware_sampler is not None:
            hardware_sampler.stop()
            hardware_summary = hardware_sampler.summary()
        if process_tree_sampler is not None:
            process_tree_sampler.stop()
            process_tree_summary = process_tree_sampler.summary()

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

    while control_pending:
        handle_non_inference(control_pending.popleft())

    activity_end_at = max(time.perf_counter(), active_context_last_at)
    activity_elapsed = max(0.0, activity_end_at - active_context_last_at)
    active_context_durations[active_contexts_current] += activity_elapsed
    for worker_id, count in per_worker_active.items():
        per_worker_active_durations[worker_id][count] += max(
            0.0, activity_end_at - per_worker_last_at[worker_id]
        )

    active_context_summary = _numeric_summary(active_context_samples)
    steady_samples = steady_state_samples or active_context_samples
    steady_state_summary = _numeric_summary(steady_samples)

    def _duration_fractions(values: Mapping[int, float]) -> dict[str, float]:
        total = sum(float(value) for value in values.values())
        return {
            str(key): (float(value) / total if total > 0.0 else 0.0)
            for key, value in sorted(values.items())
        }

    worker_active_distribution = {
        str(worker_id): _duration_fractions(values)
        for worker_id, values in per_worker_active_durations.items()
    }
    global_active_distribution = _duration_fractions(active_context_durations)
    parent_cpu_seconds = max(0.0, time.process_time() - parent_cpu_started)

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
    process_tree_cpu_seconds = sum(worker_cpu_seconds) + parent_cpu_seconds
    process_tree_effective_cpu_cores = (
        process_tree_cpu_seconds / wall_time if wall_time else 0.0
    )
    workers_per_batch_summary = _numeric_summary(batch_worker_counts)
    hardware_phase = {}
    if isinstance(hardware_summary.get("phases"), Mapping):
        hardware_phase = hardware_summary["phases"].get("ARENA", {})  # type: ignore[assignment]

    def _hardware_metric(name: str) -> Mapping[str, object]:
        value = hardware_phase.get(name, {}) if isinstance(hardware_phase, Mapping) else {}
        return dict(value) if isinstance(value, Mapping) else {}

    gpu_utilization = _hardware_metric("gpu_util_percent")
    gpu_memory = _hardware_metric("gpu_memory_used_mib")
    gpu_temperature = _hardware_metric("gpu_temperature_c")
    gpu_power = _hardware_metric("gpu_power_w")

    def _instant_phase(name: str, at: float | None) -> None:
        if at is None:
            return
        startup_phases[name] = {
            "start_sec": at - process_started_at,
            "end_sec": at - process_started_at,
            "duration_sec": 0.0,
        }

    if ingress is not None and ingress.first_inference_received_at is not None:
        first_worker_request_at = (
            ingress.first_inference_received_at
            if first_worker_request_at is None
            else min(first_worker_request_at, ingress.first_inference_received_at)
        )
    _instant_phase("first_worker_request", first_worker_request_at)
    _instant_phase("first_cuda_forward", first_cuda_forward_at)
    _instant_phase("first_completed_move", first_completed_move_at)
    _instant_phase("first_completed_game", first_completed_game_at)
    startup_phases["process_exit_after_gameplay"] = {
        "start_sec": process_started_at - process_started_at,
        "end_sec": time.perf_counter() - process_started_at,
        "duration_sec": time.perf_counter() - process_started_at,
    }
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
        "configured_workers": config.workers,
        "observed_worker_pids": worker_pids,
        "configured_games_per_worker": config.games_per_worker,
        "configured_context_capacity": configured_context_capacity,
        "observed_unique_lane_ids_per_worker": {
            str(worker_id): sorted(lane_ids)
            for worker_id, lane_ids in activity_lane_ids_by_worker.items()
        },
        "expected_unique_lane_ids_per_worker": {
            str(worker_id): list(lane_ids)
            for worker_id, lane_ids in expected_lane_ids_by_worker.items()
        },
        "active_contexts": {
            "current": active_contexts_current,
            "mean": active_context_summary["mean"],
            "p50": active_context_summary["p50"],
            "p95": active_context_summary["p95"],
            "max": peak_active_contexts,
        },
        "steady_state_active_contexts": {
            "mean": steady_state_summary["mean"],
            "p50": steady_state_summary["p50"],
            "p95": steady_state_summary["p95"],
            "min": min(steady_samples, default=0),
            "max": steady_state_summary["max"],
        },
        "per_worker_active_context_distribution": worker_active_distribution,
        "lane_occupancy": {
            "global_active_contexts": global_active_distribution,
            "by_worker_active_contexts": worker_active_distribution,
        },
        "peak_contexts_global": peak_active_contexts,
        "number_of_lane_replenishments": lane_replenishments,
        "activity_event_count": activity_events,
        "global_task_replenishment": lane_replenishments > 0,
        "tail_phase": {
            "pending_empty_at_sec": (
                pending_empty_at - process_started_at
                if pending_empty_at is not None
                else None
            ),
            "steady_state_started_at_sec": (
                steady_state_started_at - process_started_at
                if steady_state_started_at is not None
                else None
            ),
        },
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
        "worker_runnable_cpu_time": _numeric_summary(worker_cpu_seconds),
        "worker_inference_blocked_time": _numeric_summary(worker_blocked_seconds),
        "worker_queue_blocked_time": _numeric_summary(
            [float(done[index].get("queue_blocked_seconds", 0.0)) for index in sorted(done)]
        ),
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
        "inference_requests": inference_request_count,
        "inference_rows": sum(batch_rows),
        "inference_rows_per_sec": (
            sum(batch_rows) / inference_wall if inference_wall else 0.0
        ),
        "mean_inference_batch_rows": mean_batch,
        "p50_inference_batch_rows": _percentile(batch_rows, 0.50),
        "p95_inference_batch_rows": _percentile(batch_rows, 0.95),
        "p99_inference_batch_rows": _percentile(batch_rows, 0.99),
        "max_inference_batch_rows": max(batch_rows, default=0),
        "inference_batch_rows_cap": config.inference_batch_rows,
        "inference_batch_wait_ms": config.inference_batch_wait_ms,
        "worker_local_inference_batch_wait_ms": 0.0,
        "batch_aggregation_scope": "cross_process",
        "batch_cap_hits": cap_hits,
        "batch_cap_hit_rate": cap_hits / len(batch_rows) if batch_rows else 0.0,
        "rows_pending_at_dispatch": _numeric_summary(rows_pending_at_dispatch),
        "dispatch_reason": dispatch_reasons,
        "dispatch_reason_counts": dispatch_reasons,
        "cross_worker_inference_calls": cross_worker_calls,
        "cross_worker_inference_call_rate": (
            cross_worker_calls / len(batch_worker_counts)
            if batch_worker_counts
            else 0.0
        ),
        "mean_workers_per_inference_call": (
            statistics.mean(batch_worker_counts) if batch_worker_counts else 0.0
        ),
        "workers_per_inference_call": workers_per_batch_summary,
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
        "inference": {
            "requests": inference_request_count,
            "rows": sum(batch_rows),
            "forward_calls": len(batch_rows),
            "batch_rows": _batch_summary(batch_rows),
            "workers_per_batch": workers_per_batch_summary,
            "rows_pending_at_dispatch": _numeric_summary(rows_pending_at_dispatch),
            "dispatch_reason_counts": dict(dispatch_reasons),
            "candidate_reference_isolation": True,
            "by_model_role": inference_by_model,
        },
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
        "candidate_p99_inference_batch_rows": int(
            _model_metric("p99_batch_rows", "candidate")
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
        "reference_p99_inference_batch_rows": int(
            _model_metric("p99_batch_rows", "reference")
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
        "h2d_ms": _numeric_summary(h2d_ms),
        "model_forward_ms": _numeric_summary(model_forward_ms),
        "d2h_ms": _numeric_summary(d2h_ms),
        "model_forward_time_ms_by_model": {
            role: _numeric_summary(
                [value * 1000.0 for value in model_forward_time_sec[model_hash]]
            )
            for role, model_hash in model_hash_by_role.items()
        },
        "wall_time_sec": wall_time,
        "startup_wall_time_sec": started - process_started_at,
        "process_wall_time_sec": time.perf_counter() - process_started_at,
        "startup_timing": {
            "module_import_to_process_start_sec": max(
                0.0, process_started_at - _MODULE_IMPORT_STARTED_AT
            ),
            "phases": startup_phases,
            "worker_ready_timestamps": {
                str(worker_id): ready[worker_id].get("ready_at")
                for worker_id in sorted(ready)
            },
        },
        "hardware_telemetry": hardware_summary,
        "process_tree_cpu": process_tree_summary or {
            "average_effective_cores": process_tree_effective_cpu_cores,
            "p50_effective_cores": None,
            "p95_effective_cores": None,
            "peak_effective_cores": None,
            "samples": 0,
            "source": "aggregate worker and parent process CPU; interval sampler unavailable",
        },
        "gpu": {
            "utilization_percent": gpu_utilization,
            "memory_used_mib": gpu_memory,
            "temperature_c": gpu_temperature,
            "power_w": gpu_power,
        },
        "gpu_utilization_average": gpu_utilization.get("mean"),
        "gpu_utilization_p50": gpu_utilization.get("p50"),
        "gpu_utilization_p95": gpu_utilization.get("p95"),
        "gpu_utilization_peak": gpu_utilization.get("max"),
        "gpu_memory_used_mib": gpu_memory.get("mean"),
        "moves": total_moves,
        "moves_per_sec": total_moves / wall_time if wall_time else 0.0,
        "games_per_hour": (
            len(records) * 3600.0 / wall_time if wall_time else 0.0
        ),
        "technical_games": int(summary["technical_games"]),
    }

    performance_diagnostics: list[str] = []
    if len(set(worker_pids)) != config.workers:
        performance_diagnostics.append("worker_pid_count")
    lane_contract_observed = all(
        set(expected_lane_ids_by_worker[worker_id]).issubset(
            activity_lane_ids_by_worker[worker_id]
        )
        for worker_id in range(config.workers)
    )
    if not lane_contract_observed:
        performance_diagnostics.append("lane_occupancy")
    if peak_active_contexts < min(config.games, configured_context_capacity):
        performance_diagnostics.append("active_contexts")
    if any(telemetry["worker_cuda_initialized_after_run"]):
        performance_diagnostics.append("cuda_in_worker")
    mean_batch_policy = classify_arena_performance(mean_batch, config)
    performance_diagnostics.extend(str(value) for value in mean_batch_policy["hard_failures"])
    performance_warnings = [str(value) for value in mean_batch_policy["warnings"]]
    if int(summary["technical_games"]) != 0:
        performance_diagnostics.append("technical_games")
    correctness_failures = [
        value for value in performance_diagnostics
        if value in {"worker_pid_count", "cuda_in_worker", "technical_games"}
    ]
    if config.strict_performance:
        performance_failures = sorted(set(performance_diagnostics))
    else:
        performance_failures = sorted(set(correctness_failures))
        performance_warnings.extend(
            value for value in performance_diagnostics
            if value not in correctness_failures
        )
    telemetry["effective_cpu_cores_target"] = config.min_effective_cpu_cores
    telemetry["effective_cpu_cores_target_met"] = (
        effective_cpu_cores >= config.min_effective_cpu_cores
    )
    telemetry["effective_cpu_cores_role"] = "diagnostic_only"
    telemetry["performance_status"] = (
        "CRITICAL"
        if performance_failures
        else (str(mean_batch_policy["status"]) if performance_warnings else "HEALTHY")
    )
    telemetry["performance_failures"] = performance_failures
    telemetry["performance_warnings"] = sorted(set(performance_warnings))
    telemetry["performance_diagnostics"] = sorted(set(performance_diagnostics))
    telemetry["performance_gate"] = {
        "mean_inference_batch_rows": mean_batch_policy["mean_inference_batch_rows"],
        "severe_warning_threshold": mean_batch_policy["severe_warning_threshold"],
        "healthy_minimum": mean_batch_policy["healthy_minimum"],
    }

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
                "strict_performance": config.strict_performance,
            },
            "telemetry": telemetry,
        }
    )
    _write_json(output_dir / "summary.json", summary)
    if performance_failures:
        _write_json(
            output_dir / "performance-degraded.json",
            {
                "status": "CRITICAL",
                "run_id": run_id,
                "reasons": performance_failures,
                "observed": {
                    "active_contexts": telemetry["active_contexts"],
                    "steady_state_active_contexts": telemetry[
                        "steady_state_active_contexts"
                    ],
                    "observed_unique_lane_ids_per_worker": telemetry[
                        "observed_unique_lane_ids_per_worker"
                    ],
                    "mean_inference_batch_rows": telemetry[
                        "mean_inference_batch_rows"
                    ],
                    "effective_cpu_cores": telemetry["effective_cpu_cores"],
                    "technical_games": telemetry["technical_games"],
                },
                "summary": str(output_dir / "summary.json"),
            },
        )
    elif performance_warnings or performance_diagnostics:
        _write_json(
            output_dir / "performance-warning.json",
            {
                "status": str(mean_batch_policy["status"]),
                "run_id": run_id,
                "reasons": sorted(set(performance_warnings + performance_diagnostics)),
                "observed": {
                    "mean_inference_batch_rows": telemetry[
                        "mean_inference_batch_rows"
                    ],
                    "severe_warning_threshold": mean_batch_policy["severe_warning_threshold"],
                    "healthy_minimum": mean_batch_policy["healthy_minimum"],
                    "technical_games": telemetry["technical_games"],
                },
                "summary": str(output_dir / "summary.json"),
            },
        )

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
