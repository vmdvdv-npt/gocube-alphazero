#!/usr/bin/env python3
"""Board-agnostic production Arena execution engine.

One execution architecture:
    OS CPU search workers -> one parent inference broker -> central model owner(s).

Game/topology semantics live in ArenaProfile adapters under tools/arena_profiles/.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty
import statistics
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
        tasks: Sequence[Mapping[str, object]],
        games_per_worker: int,
        candidate_hash: str,
        reference_hash: str,
        input_slot: torch.Tensor,
        policy_slot: torch.Tensor,
        wdl_slot: torch.Tensor,
        request_queue: Any,
        response_queue: Any,
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
    if expected_model_hash and identity.model_hash != expected_model_hash:
        raise ValueError(f"{label} model hash mismatch")
    if expected_artifact_sha256 and identity.artifact_sha256 != expected_artifact_sha256:
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


def _terminate(processes: Sequence[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def _worker_bootstrap(
    profile_id: str,
    worker_id: int,
    tasks: Sequence[Mapping[str, object]],
    games_per_worker: int,
    candidate_hash: str,
    reference_hash: str,
    input_slot: torch.Tensor,
    policy_slot: torch.Tensor,
    wdl_slot: torch.Tensor,
    request_queue: Any,
    response_queue: Any,
    start_event: Any,
) -> None:
    from tools.arena_profiles import get_profile

    profile = get_profile(profile_id)
    profile.worker_main(
        worker_id,
        tasks,
        games_per_worker,
        candidate_hash,
        reference_hash,
        input_slot,
        policy_slot,
        wdl_slot,
        request_queue,
        response_queue,
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
    buckets: list[list[dict[str, object]]] = [[] for _ in range(config.workers)]
    for task in tasks:
        buckets[int(task["worker_id"])].append(task)

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
    response_queues = [ctx.Queue() for _ in range(config.workers)]
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
                buckets[worker_id],
                config.games_per_worker,
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
        start_event.set()
        pending: deque[Mapping[str, object]] = deque()
        done: dict[int, Mapping[str, object]] = {}
        records: list[dict[str, object]] = []
        batch_rows: list[int] = []
        queue_wait_ms: list[float] = []
        batch_worker_counts: list[int] = []
        cap_hits = 0
        inference_started = time.perf_counter()

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

        while len(done) < config.workers:
            if pending:
                message = pending.popleft()
            else:
                try:
                    message = request_queue.get(timeout=1.0)
                except Empty:
                    dead = [
                        process.name
                        for process in processes
                        if not process.is_alive() and process.exitcode not in (0, None)
                    ]
                    if dead:
                        raise RuntimeError(
                            f"Arena worker process died during run: {dead}"
                        )
                    continue

            if message.get("kind") != "inference":
                handle_non_inference(message)
                continue

            requests: list[Mapping[str, object]] = [message]
            collected_rows = int(message["rows"])
            deadline = (
                time.perf_counter() + config.inference_batch_wait_ms / 1000.0
            )
            while collected_rows < config.inference_batch_rows:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    extra = request_queue.get(timeout=remaining)
                except Empty:
                    break
                if extra.get("kind") != "inference":
                    pending.append(extra)
                    continue
                extra_rows = int(extra["rows"])
                if collected_rows + extra_rows > config.inference_batch_rows:
                    pending.append(extra)
                    break
                requests.append(extra)
                collected_rows += extra_rows

            grouped: dict[str, list[Mapping[str, object]]] = {}
            for request in requests:
                grouped.setdefault(str(request["model_hash"]), []).append(request)

            for requested_hash, group in grouped.items():
                model = models_by_hash.get(requested_hash)
                if model is None:
                    raise RuntimeError(
                        f"Inference request references unknown model hash {requested_hash}"
                    )
                now = time.perf_counter()
                for request in group:
                    queue_wait_ms.append(
                        max(0.0, now - float(request["enqueued_at"])) * 1000.0
                    )
                cpu_batch = torch.cat(
                    [
                        shared_inputs[int(request["worker_id"])][
                            : int(request["rows"])
                        ]
                        for request in group
                    ],
                    dim=0,
                )
                policy, wdl = profile.infer_batch(model, cpu_batch, device)
                offset = 0
                for request in group:
                    wid = int(request["worker_id"])
                    rows = int(request["rows"])
                    shared_policy[wid][:rows].copy_(policy[offset : offset + rows])
                    shared_wdl[wid][:rows].copy_(wdl[offset : offset + rows])
                    response_queues[wid].put(
                        {"ticket": int(request["ticket"]), "error": None}
                    )
                    offset += rows
                rows_this_call = int(cpu_batch.shape[0])
                batch_rows.append(rows_this_call)
                batch_worker_counts.append(
                    len({int(request["worker_id"]) for request in group})
                )
                if rows_this_call >= config.inference_batch_rows:
                    cap_hits += 1

        wall_time = time.perf_counter() - started
        inference_wall = time.perf_counter() - inference_started
    except BaseException:
        _terminate(processes)
        raise
    finally:
        start_event.set()

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
    worker_cpu_seconds = [float(done[index]["cpu_seconds"]) for index in sorted(done)]
    effective_cpu_cores = sum(worker_cpu_seconds) / wall_time if wall_time else 0.0
    mean_batch = statistics.mean(batch_rows) if batch_rows else 0.0
    cross_worker_calls = sum(count > 1 for count in batch_worker_counts)
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
        "wall_time_sec": wall_time,
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
