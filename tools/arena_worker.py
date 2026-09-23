"""Topology-neutral cooperative Arena search worker.

Scientific profiles own game state, observations, scoring and result mapping.
This module owns the single worker execution mechanism: lanes, suspended
``SequentialPUCTSession`` searches, remote inference transport, ticket
validation, timeouts, replenishment and execution telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from queue import Empty
import resource
import time
import traceback
from typing import Any, Callable, Mapping

import torch

from gocube_golden.search import (
    Evaluation,
    SearchEvaluationRequest,
    SearchResult,
    SequentialPUCTSession,
)

ARENA_INFERENCE_TIMEOUT_SEC = 300.0


@dataclass(frozen=True)
class CooperativeArenaCallbacks:
    search_settings: Any
    search_adapter: Any
    make_game: Callable[[Mapping[str, object]], Any]
    game_id: Callable[[Any], str]
    search_state: Callable[[Any], Any]
    search_seed: Callable[[Any], int]
    model_role: Callable[[Any], str]
    build_observation: Callable[[Any, Any], torch.Tensor]
    apply_search_result: Callable[[Any, SearchResult], bool]
    mark_search_error: Callable[[Any, Exception], None]
    finish_record: Callable[[Any], Mapping[str, object]]


@dataclass
class _LaneRuntime:
    game: Any
    session: SequentialPUCTSession | None = None
    pending_evaluator: "_RemoteEvaluator | None" = None
    generation: int = 0


class _ImmediateInferenceTransport:
    """Zero-wait worker transport; all timed coalescing is central."""

    def __init__(
        self,
        *,
        worker_id: int,
        central_queue: Any,
        local_cap: int,
        wait_ms: float,
    ) -> None:
        if int(local_cap) <= 0:
            raise ValueError("Arena worker local capacity must be positive")
        if not math.isfinite(float(wait_ms)) or float(wait_ms) != 0.0:
            raise ValueError("Arena worker inference transport requires wait_ms=0")
        self.worker_id = int(worker_id)
        self.central_queue = central_queue
        self.local_cap = int(local_cap)
        self._closed = False

    def put(self, request: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("Arena worker inference transport is closed")
        self.central_queue.put(dict(request))

    def close(self) -> None:
        self._closed = True


class _RemoteEvaluator:
    """One non-blocking inference endpoint for one worker lane/model role."""

    def __init__(
        self,
        *,
        worker_id: int,
        lane_id: int,
        model_role: str,
        model_hash: str,
        input_slot: torch.Tensor,
        policy_slot: torch.Tensor,
        wdl_slot: torch.Tensor,
        request_queue: Any,
        response_queue: Any,
        build_observation: Callable[[Any, Any], torch.Tensor],
    ) -> None:
        self.worker_id = int(worker_id)
        self.lane_id = int(lane_id)
        self.model_role = str(model_role)
        self.model_hash = str(model_hash)
        self.input_slot = input_slot
        self.policy_slot = policy_slot
        self.wdl_slot = wdl_slot
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.build_observation = build_observation
        self.ticket = 0
        self.blocked_inference_seconds = 0.0
        self.blocked_inference_calls = 0
        self._pending: dict[str, object] | None = None

    def submit(
        self,
        state: Any,
        legal_context: Any,
        *,
        generation: int,
        game_id: str,
    ) -> int:
        if self._pending is not None:
            raise RuntimeError(
                f"Arena lane {self.lane_id} already has a pending inference request"
            )
        if self.lane_id < 0 or self.lane_id >= int(self.input_slot.shape[0]):
            raise RuntimeError("Arena lane exceeds its shared-memory input slot")
        observation = self.build_observation(state, legal_context)
        expected_shape = tuple(self.input_slot[self.lane_id].shape)
        if not isinstance(observation, torch.Tensor):
            raise RuntimeError("Arena observation builder must return a torch.Tensor")
        if tuple(observation.shape) != expected_shape:
            raise RuntimeError(
                "Arena observation shape mismatch: "
                f"expected {expected_shape}, got {tuple(observation.shape)}"
            )
        self.input_slot[self.lane_id].copy_(observation)

        ticket = self.ticket
        self.ticket += 1
        submitted_at = time.perf_counter()
        self._pending = {
            "ticket": int(ticket),
            "generation": int(generation),
            "submitted_at": submitted_at,
            "game_id": str(game_id),
        }
        self.request_queue.put(
            {
                "kind": "inference",
                "worker_id": self.worker_id,
                "lane_id": self.lane_id,
                "pid": os.getpid(),
                "ticket": int(ticket),
                "generation": int(generation),
                "game_id": str(game_id),
                "model_role": self.model_role,
                "model_hash": self.model_hash,
                "rows": 1,
                "worker_enqueued_at": submitted_at,
            }
        )
        return ticket

    def poll(self) -> Evaluation | None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("Arena lane has no pending inference request")
        try:
            response = self.response_queue.get_nowait()
        except Empty:
            elapsed = time.perf_counter() - float(pending["submitted_at"])
            if elapsed > ARENA_INFERENCE_TIMEOUT_SEC:
                raise RuntimeError(
                    "Arena inference response timed out "
                    f"(worker={self.worker_id}, lane={self.lane_id}, "
                    f"ticket={pending['ticket']})"
                )
            return None

        resumed_at = time.perf_counter()
        self.blocked_inference_seconds += max(
            0.0, resumed_at - float(pending["submitted_at"])
        )
        self.blocked_inference_calls += 1
        if not isinstance(response, Mapping):
            raise RuntimeError("Arena inference response is malformed")

        expected = {
            "worker_id": self.worker_id,
            "lane_id": self.lane_id,
            "ticket": int(pending["ticket"]),
            "generation": int(pending["generation"]),
            "model_role": self.model_role,
            "model_hash": self.model_hash,
        }
        for key, value in expected.items():
            if response.get(key) != value:
                raise RuntimeError(f"Arena inference response {key} mismatch")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))

        policy = tuple(float(value) for value in self.policy_slot[self.lane_id].tolist())
        wdl = tuple(float(value) for value in self.wdl_slot[self.lane_id].tolist())
        if not all(
            math.isfinite(value) and value >= 0.0 for value in policy + wdl
        ):
            raise RuntimeError("Remote Arena evaluator received invalid probabilities")
        self._pending = None
        return Evaluation(policy=policy, wdl=wdl)


def run_cooperative_arena_worker(
    *,
    callbacks: CooperativeArenaCallbacks,
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
    """Run all scientific profiles through the same cooperative lane scheduler."""

    try:
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was already initialized inside an Arena search worker")
        lane_capacity = int(games_per_worker)
        if lane_capacity <= 0:
            raise ValueError("Arena games_per_worker must be positive")

        # This ready message declares the addressable lane space for the
        # startup barrier. Final telemetry reports only lanes actually used.
        request_queue.put(
            {
                "kind": "ready",
                "worker_id": int(worker_id),
                "pid": os.getpid(),
                "cuda_initialized": False,
                "configured_games_per_worker": lane_capacity,
                "observed_lane_ids": list(range(lane_capacity)),
                "ready_at": time.perf_counter(),
            }
        )
        start_event.wait()
        run_started = time.perf_counter()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        transport = _ImmediateInferenceTransport(
            worker_id=worker_id,
            central_queue=request_queue,
            local_cap=lane_capacity,
            wait_ms=float(worker_local_wait_ms),
        )

        evaluators_by_lane = {
            lane_id: {
                "candidate": _RemoteEvaluator(
                    worker_id=worker_id,
                    lane_id=lane_id,
                    model_role="candidate",
                    model_hash=candidate_hash,
                    input_slot=input_slot,
                    policy_slot=policy_slot,
                    wdl_slot=wdl_slot,
                    request_queue=transport,
                    response_queue=response_queues[lane_id],
                    build_observation=callbacks.build_observation,
                ),
                "reference": _RemoteEvaluator(
                    worker_id=worker_id,
                    lane_id=lane_id,
                    model_role="reference",
                    model_hash=reference_hash,
                    input_slot=input_slot,
                    policy_slot=policy_slot,
                    wdl_slot=wdl_slot,
                    request_queue=transport,
                    response_queue=response_queues[lane_id],
                    build_observation=callbacks.build_observation,
                ),
            }
            for lane_id in range(lane_capacity)
        }

        active: dict[int, _LaneRuntime] = {}
        records: list[dict[str, object]] = []
        lane_wall_time_seconds = 0.0
        queue_blocked_seconds = 0.0
        first_move_published = False
        used_lane_ids: set[int] = set()
        lane_start_counts = {lane_id: 0 for lane_id in range(lane_capacity)}

        def take_task(*, block: bool) -> Mapping[str, object] | None:
            nonlocal queue_blocked_seconds
            wait_started = time.perf_counter() if block else None
            try:
                task = task_queue.get(timeout=1.0) if block else task_queue.get_nowait()
            except Empty:
                if wait_started is not None:
                    queue_blocked_seconds += max(
                        0.0, time.perf_counter() - wait_started
                    )
                return None
            if wait_started is not None:
                queue_blocked_seconds += max(
                    0.0, time.perf_counter() - wait_started
                )
            if not isinstance(task, Mapping):
                raise RuntimeError("Arena worker task is malformed")
            return task

        def publish_activity(
            event: str,
            lane_id: int,
            game: Any,
            *,
            replenished: bool = False,
        ) -> None:
            request_queue.put(
                {
                    "kind": "activity",
                    "event": event,
                    "worker_id": int(worker_id),
                    "pid": int(os.getpid()),
                    "lane_id": int(lane_id),
                    "game_id": callbacks.game_id(game),
                    "at": time.perf_counter(),
                    "replenished": bool(replenished),
                }
            )

        def start_lane(task: Mapping[str, object], lane_id: int) -> None:
            game = callbacks.make_game(dict(task, worker_id=worker_id))
            replenished = lane_start_counts[lane_id] > 0
            lane_start_counts[lane_id] += 1
            active[lane_id] = _LaneRuntime(game=game)
            used_lane_ids.add(lane_id)
            publish_activity(
                "game_started",
                lane_id,
                game,
                replenished=replenished,
            )

        def finish_lane(lane_id: int) -> None:
            nonlocal lane_wall_time_seconds
            runtime = active[lane_id]
            record = dict(callbacks.finish_record(runtime.game))
            records.append(record)
            wall = record.get("wall_time_sec")
            if isinstance(wall, (int, float)) and math.isfinite(float(wall)):
                lane_wall_time_seconds += max(0.0, float(wall))
            publish_activity("game_completed", lane_id, runtime.game)
            del active[lane_id]

        def fill_lanes() -> None:
            while len(active) < lane_capacity:
                task = take_task(block=False)
                if task is None:
                    return
                free_lane = next(
                    lane_id for lane_id in range(lane_capacity) if lane_id not in active
                )
                start_lane(task, free_lane)

        def finish_with_search_error(lane_id: int, exc: Exception) -> None:
            runtime = active[lane_id]
            callbacks.mark_search_error(runtime.game, exc)
            runtime.session = None
            runtime.pending_evaluator = None
            finish_lane(lane_id)

        def process_step(
            lane_id: int,
            step: SearchEvaluationRequest | SearchResult,
        ) -> bool:
            """Return True when this processing completed the game."""

            nonlocal first_move_published
            runtime = active[lane_id]
            if isinstance(step, SearchEvaluationRequest):
                role = callbacks.model_role(runtime.game)
                if role not in {"candidate", "reference"}:
                    raise RuntimeError(f"Arena scientific adapter returned invalid role {role!r}")
                evaluator = evaluators_by_lane[lane_id][role]
                evaluator.submit(
                    step.state,
                    step.legal_context,
                    generation=runtime.generation,
                    game_id=callbacks.game_id(runtime.game),
                )
                runtime.pending_evaluator = evaluator
                return False
            if not isinstance(step, SearchResult):
                raise RuntimeError("Cooperative Arena search returned malformed result")

            finished = bool(callbacks.apply_search_result(runtime.game, step))
            runtime.session = None
            runtime.pending_evaluator = None
            runtime.generation += 1
            if not first_move_published:
                publish_activity("move_completed", lane_id, runtime.game)
                first_move_published = True
            if finished:
                finish_lane(lane_id)
                return True
            return False

        def start_or_advance_lane(lane_id: int) -> bool:
            runtime = active[lane_id]
            try:
                if runtime.session is None:
                    runtime.session = SequentialPUCTSession(
                        callbacks.search_state(runtime.game),
                        callbacks.search_settings,
                        adapter=callbacks.search_adapter,
                        seed=callbacks.search_seed(runtime.game),
                    )
                step = runtime.session.advance()
                process_step(lane_id, step)
                return True
            except Exception as exc:
                finish_with_search_error(lane_id, exc)
                return True

        while True:
            fill_lanes()
            if not active:
                task = take_task(block=True)
                if task is None:
                    break
                start_lane(task, 0)
                continue

            progress = False
            for lane_id in tuple(sorted(active)):
                if lane_id not in active:
                    continue
                runtime = active[lane_id]
                if runtime.pending_evaluator is not None:
                    try:
                        evaluation = runtime.pending_evaluator.poll()
                    except Exception as exc:
                        finish_with_search_error(lane_id, exc)
                        progress = True
                        continue
                    if evaluation is None:
                        continue
                    runtime.pending_evaluator = None
                    try:
                        if runtime.session is None:
                            raise RuntimeError(
                                "Arena lane lost its suspended search session"
                            )
                        step = runtime.session.resume(evaluation)
                        process_step(lane_id, step)
                    except Exception as exc:
                        if lane_id in active:
                            finish_with_search_error(lane_id, exc)
                    progress = True
                    continue

                start_or_advance_lane(lane_id)
                progress = True

            if not progress:
                # Every active lane is suspended. Poll again quickly instead
                # of blocking on one lane and starving the others.
                time.sleep(0.0005)

        transport.close()
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = (cpu_end.ru_utime + cpu_end.ru_stime) - (
            cpu_start.ru_utime + cpu_start.ru_stime
        )
        request_queue.put(
            {
                "kind": "done",
                "worker_id": int(worker_id),
                "pid": os.getpid(),
                "cuda_initialized": torch.cuda.is_initialized(),
                "cpu_seconds": float(cpu_seconds),
                "cpu_active_seconds": float(cpu_seconds),
                "wall_time_seconds": time.perf_counter() - run_started,
                "blocked_inference_seconds": float(
                    sum(
                        evaluator.blocked_inference_seconds
                        for by_role in evaluators_by_lane.values()
                        for evaluator in by_role.values()
                    )
                ),
                "blocked_inference_calls": int(
                    sum(
                        evaluator.blocked_inference_calls
                        for by_role in evaluators_by_lane.values()
                        for evaluator in by_role.values()
                    )
                ),
                "lane_wall_time_seconds": float(lane_wall_time_seconds),
                "max_rss_kb": int(cpu_end.ru_maxrss),
                "observed_lane_ids": sorted(used_lane_ids),
                "used_lane_ids": sorted(used_lane_ids),
                "lane_replenishments": int(
                    sum(max(0, count - 1) for count in lane_start_counts.values())
                ),
                "queue_blocked_seconds": float(queue_blocked_seconds),
                "records": records,
            }
        )
    except BaseException as exc:
        request_queue.put(
            {
                "kind": "error",
                "worker_id": int(worker_id),
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


__all__ = [
    "ARENA_INFERENCE_TIMEOUT_SEC",
    "CooperativeArenaCallbacks",
    "run_cooperative_arena_worker",
]
