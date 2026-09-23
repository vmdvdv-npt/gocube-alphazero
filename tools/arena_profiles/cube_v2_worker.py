"""CPU/search-worker implementation for the Cube V2 Arena profile."""
from __future__ import annotations

import math
import os
from queue import Empty
import resource
import time
import traceback
from typing import Any, Mapping

import torch

from gocube_golden.cube_family import initial_cube_state
from gocube_golden.cube_observation_v2 import build_cube_observation, initial_cube_observation_context
from gocube_golden.cube_search import CubeSearchAdapter, CubeSearchPosition
from gocube_golden.provenance import derive_seed
from gocube_golden.result import result_from_terminal
from gocube_golden.search import Evaluation, SequentialPUCT
from gocube_golden.state import BLACK, WHITE

ARENA_INFERENCE_TIMEOUT_SEC = 300.0


class RemoteCubeEvaluator:
    """Synchronous worker-side proxy to the common central inference broker."""

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
        self.ticket = 0
        self.blocked_inference_seconds = 0.0
        self.blocked_inference_calls = 0

    def evaluate_prepared(self, position: CubeSearchPosition, legal_context: Any) -> Evaluation:
        observation = build_cube_observation(
            position.game_state,
            position.observation_context,
            legal_context=legal_context,
        )
        self.input_slot[self.lane_id].copy_(observation)
        ticket = self.ticket
        self.ticket += 1
        enqueued_at = time.perf_counter()
        self.request_queue.put(
            {
                "kind": "inference",
                "worker_id": self.worker_id,
                "lane_id": self.lane_id,
                "pid": os.getpid(),
                "ticket": ticket,
                "generation": ticket,
                "game_id": f"worker-{self.worker_id}-lane-{self.lane_id}",
                "model_role": self.model_role,
                "model_hash": self.model_hash,
                "rows": 1,
                "worker_enqueued_at": enqueued_at,
            }
        )
        try:
            response = self.response_queue.get(timeout=ARENA_INFERENCE_TIMEOUT_SEC)
        except Empty as exc:
            raise RuntimeError("Cube Arena inference response timed out") from exc
        resumed_at = time.perf_counter()
        self.blocked_inference_seconds += resumed_at - enqueued_at
        self.blocked_inference_calls += 1
        if not isinstance(response, Mapping):
            raise RuntimeError("Cube Arena inference response is malformed")
        expected = {
            "worker_id": self.worker_id,
            "lane_id": self.lane_id,
            "ticket": ticket,
            "generation": ticket,
            "model_role": self.model_role,
            "model_hash": self.model_hash,
        }
        for key, value in expected.items():
            if response.get(key) != value:
                raise RuntimeError(f"Cube Arena inference response {key} mismatch")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        policy = tuple(float(value) for value in self.policy_slot[self.lane_id].tolist())
        wdl = tuple(float(value) for value in self.wdl_slot[self.lane_id].tolist())
        if not all(math.isfinite(value) and value >= 0.0 for value in policy + wdl):
            raise RuntimeError("Cube Arena inference returned invalid probabilities")
        return Evaluation(policy=policy, wdl=wdl)


def _finish_record(
    task: Mapping[str, object],
    position: CubeSearchPosition,
    trace: list[dict[str, object]],
    *,
    formal_result: str | None,
    technical: str | None,
    error: str | None,
    started_at: float,
) -> dict[str, object]:
    candidate_black = bool(task["candidate_black"])
    row: dict[str, object] = {
        "run_id": str(task["run_id"]),
        "comparison": str(task["comparison"]),
        "pair_id": str(task["pair_id"]),
        "game_id": str(task["game_id"]),
        "start_id": str(task["start_id"]),
        "worker_id": int(task["worker_id"]),
        "worker_pid": os.getpid(),
        "candidate_black": candidate_black,
        "candidate_model_hash": str(task["candidate_hash"]),
        "reference_model_hash": str(task["reference_hash"]),
        "action_trace": trace,
        "formal_result": formal_result,
        "technical_termination": technical,
        "error": error,
        "mapped_result": None,
        "wall_time_sec": time.perf_counter() - started_at,
    }
    if formal_result is not None:
        if formal_result == "DRAW":
            row["mapped_result"] = "DRAW"
        elif (formal_result == "BLACK") == candidate_black:
            row["mapped_result"] = "A_WIN"
        else:
            row["mapped_result"] = "B_WIN"
        result = result_from_terminal(position.game_state)
        row.update(
            black_area=result.black_area,
            white_area=result.white_area,
            margin_black=result.margin_black,
        )
    return row


def run_cube_v2_worker(
    *,
    size: int,
    search_config: Any,
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
    """Run Cube games in one common Arena worker; execution stays broker-owned."""

    try:
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was already initialized inside a Cube Arena worker")
        if float(worker_local_wait_ms) != 0.0:
            raise ValueError("Cube Arena worker-local inference wait must be zero")
        request_queue.put(
            {
                "kind": "ready",
                "worker_id": int(worker_id),
                "pid": os.getpid(),
                "cuda_initialized": False,
                "configured_games_per_worker": int(games_per_worker),
                "observed_lane_ids": list(range(int(games_per_worker))),
                "ready_at": time.perf_counter(),
            }
        )
        start_event.wait()
        run_started = time.perf_counter()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        records: list[dict[str, object]] = []
        queue_blocked_seconds = 0.0
        lane_wall_time_seconds = 0.0
        used_lane_ids: set[int] = set()
        evaluators = {
            "candidate": RemoteCubeEvaluator(
                worker_id=worker_id,
                lane_id=0,
                model_role="candidate",
                model_hash=candidate_hash,
                input_slot=input_slot,
                policy_slot=policy_slot,
                wdl_slot=wdl_slot,
                request_queue=request_queue,
                response_queue=response_queues[0],
            ),
            "reference": RemoteCubeEvaluator(
                worker_id=worker_id,
                lane_id=0,
                model_role="reference",
                model_hash=reference_hash,
                input_slot=input_slot,
                policy_slot=policy_slot,
                wdl_slot=wdl_slot,
                request_queue=request_queue,
                response_queue=response_queues[0],
            ),
        }
        search_adapter = CubeSearchAdapter()
        search_settings = search_config.search_settings
        first_move_published = False

        while True:
            wait_started = time.perf_counter()
            try:
                task = task_queue.get(timeout=1.0)
            except Empty:
                queue_blocked_seconds += time.perf_counter() - wait_started
                break
            queue_blocked_seconds += time.perf_counter() - wait_started
            used_lane_ids.add(0)
            started_at = time.perf_counter()
            request_queue.put(
                {
                    "kind": "activity",
                    "event": "game_started",
                    "worker_id": int(worker_id),
                    "pid": os.getpid(),
                    "lane_id": 0,
                    "game_id": str(task["game_id"]),
                    "at": started_at,
                    "replenished": bool(records),
                }
            )
            state = initial_cube_state(size)
            position = CubeSearchPosition(state, initial_cube_observation_context(state))
            trace: list[dict[str, object]] = []
            formal_result: str | None = None
            technical: str | None = None
            error: str | None = None

            for ply in range(1, int(search_config.watchdog) + 1):
                side = position.game_state.side_to_move
                candidate_turn = (
                    side == BLACK and bool(task["candidate_black"])
                ) or (
                    side == WHITE and not bool(task["candidate_black"])
                )
                role = "candidate" if candidate_turn else "reference"
                try:
                    result = SequentialPUCT(
                        search_settings,
                        adapter=search_adapter,
                    ).search(
                        position,
                        evaluators[role],
                        seed=derive_seed(
                            int(task["game_seed"]),
                            ply,
                            "cube-arena-search",
                        ),
                    )
                    previous = position
                    action_index = search_adapter.action_index(previous, result.action)
                    position = search_adapter.apply_action(previous, result.action)
                    trace.append(
                        {
                            "ply": ply,
                            "side_to_move": previous.game_state.side_to_move.name,
                            "player": role,
                            "action": action_index,
                            "legal": True,
                        }
                    )
                    if not first_move_published:
                        request_queue.put(
                            {
                                "kind": "activity",
                                "event": "move_completed",
                                "worker_id": int(worker_id),
                                "pid": os.getpid(),
                                "lane_id": 0,
                                "game_id": str(task["game_id"]),
                                "at": time.perf_counter(),
                                "replenished": False,
                            }
                        )
                        first_move_published = True
                    if position.is_terminal:
                        formal_result = result_from_terminal(position.game_state).winner.value
                        break
                except Exception as exc:
                    technical = "ERROR_SEARCH"
                    error = f"{type(exc).__name__}: {exc}"
                    break
            else:
                technical = "WATCHDOG"
                error = f"Cube Arena watchdog reached {search_config.watchdog} actions"

            records.append(
                _finish_record(
                    task,
                    position,
                    trace,
                    formal_result=formal_result,
                    technical=technical,
                    error=error,
                    started_at=started_at,
                )
            )
            lane_wall_time_seconds += time.perf_counter() - started_at
            request_queue.put(
                {
                    "kind": "activity",
                    "event": "game_completed",
                    "worker_id": int(worker_id),
                    "pid": os.getpid(),
                    "lane_id": 0,
                    "game_id": str(task["game_id"]),
                    "at": time.perf_counter(),
                    "replenished": False,
                }
            )

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
                    sum(e.blocked_inference_seconds for e in evaluators.values())
                ),
                "blocked_inference_calls": int(
                    sum(e.blocked_inference_calls for e in evaluators.values())
                ),
                "lane_wall_time_seconds": float(lane_wall_time_seconds),
                "max_rss_kb": int(cpu_end.ru_maxrss),
                "observed_lane_ids": list(range(int(games_per_worker))),
                "used_lane_ids": sorted(used_lane_ids),
                "lane_replenishments": max(0, len(records) - int(games_per_worker)),
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


__all__ = ["run_cube_v2_worker"]
