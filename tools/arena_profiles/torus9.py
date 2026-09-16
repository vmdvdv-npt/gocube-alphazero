"""Torus 9x9 adapter for the universal Arena engine.

This module contains only Torus9 scientific/game semantics. Multiprocessing,
central inference batching, lifecycle and telemetry live in tools.arena_engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from queue import Empty
import resource
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.neural import model_hash
from gocube_golden.provenance import derive_seed, file_sha256
from gocube_golden.result import result_from_terminal
from gocube_golden.rules import IllegalMoveError, apply_action
from gocube_golden.scoring import score_terminal
from gocube_golden.search import (
    Evaluation,
    SearchEvaluationRequest,
    SearchResult,
    SequentialPUCTSession,
)
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, WHITE
from gocube_golden.torus9 import (
    TORUS9_TOPOLOGY_FINGERPRINT,
    build_torus9_observation,
    generate_torus9_evaluation_starts,
    summarize_torus9_arena,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    torus9_state_from_identity,
)
from gocube_golden.torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_ARENA_MOVE_LIMIT,
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_KOMI,
    TORUS9_POINT_COUNT,
)
from tools.arena_engine import ArenaExecutionConfig, CheckpointIdentity

PROFILE_ID = "torus9"
PRODUCTION_WORKERS = 16
PRODUCTION_MIN_GAMES = 64
PRODUCTION_MIN_BATCH_ROWS = 16
ARENA_INFERENCE_TIMEOUT_SEC = 300.0


class _WorkerInferenceAggregator:
    """Zero-wait transport shim for thread-safe cross-process request ingress.

    The old implementation had a second, worker-local timed coalescing window.
    That serialized every lane behind ``inference_batch_wait_ms`` before the
    central broker could see it.  The production path now forwards every
    request immediately; all timed coalescing belongs to the central broker.
    The historical name is retained as a narrow compatibility boundary.
    """

    def __init__(
        self,
        *,
        worker_id: int,
        central_queue: Any,
        local_cap: int,
        wait_ms: float,
    ) -> None:
        if local_cap <= 0 or not math.isfinite(float(wait_ms)) or float(wait_ms) != 0.0:
            raise ValueError("Worker inference transport requires wait_ms=0")
        self.worker_id = int(worker_id)
        self.central_queue = central_queue
        self.local_cap = int(local_cap)
        self.wait_ms = 0.0
        self._closed = False

    def put(self, request: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("Worker inference transport is closed")
        self.central_queue.put(dict(request))

    def close(self) -> None:
        self._closed = True


class _RemoteEvaluator:
    """Worker-side evaluator proxy; no model or CUDA owner exists in workers.

    A request is submitted independently from response consumption.  The
    cooperative Arena worker owns one instance per lane, which means a slow
    response can leave that lane suspended without blocking runnable lanes.
    """

    def __init__(
        self,
        *,
        worker_id: int,
        model_role: str,
        model_hash_value: str,
        lane_id: int,
        input_slot: torch.Tensor,
        policy_slot: torch.Tensor,
        wdl_slot: torch.Tensor,
        request_queue: Any,
        response_queue: Any,
    ) -> None:
        self.worker_id = int(worker_id)
        self.model_role = str(model_role)
        self.model_hash = str(model_hash_value)
        self.lane_id = int(lane_id)
        self.input_slot = input_slot
        self.policy_slot = policy_slot
        self.wdl_slot = wdl_slot
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.ticket = 0
        self.blocked_inference_seconds = 0.0
        self.blocked_inference_calls = 0
        self._pending: dict[str, object] | None = None

    def submit_prepared(
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
        observation = build_torus9_observation(state, legal_context=legal_context)
        if self.lane_id >= int(self.input_slot.shape[0]):
            raise RuntimeError("Arena lane exceeds its shared-memory input slot")
        self.input_slot[self.lane_id].copy_(observation)
        ticket = self.ticket
        self.ticket += 1
        worker_enqueued_at = time.perf_counter()
        self._pending = {
            "ticket": ticket,
            "generation": int(generation),
            "submitted_at": worker_enqueued_at,
        }
        self.request_queue.put(
            {
                "kind": "inference",
                "worker_id": self.worker_id,
                "lane_id": self.lane_id,
                "pid": os.getpid(),
                "ticket": ticket,
                "generation": int(generation),
                "game_id": str(game_id),
                "model_role": self.model_role,
                "model_hash": self.model_hash,
                "rows": 1,
                "worker_enqueued_at": worker_enqueued_at,
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
            if time.perf_counter() - float(pending["submitted_at"]) > ARENA_INFERENCE_TIMEOUT_SEC:
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
        if int(response.get("worker_id", -1)) != self.worker_id:
            raise RuntimeError("Arena inference response worker identity mismatch")
        if int(response.get("lane_id", -1)) != self.lane_id:
            raise RuntimeError("Arena inference response lane identity mismatch")
        if int(response.get("ticket", -1)) != int(pending["ticket"]):
            raise RuntimeError("Arena inference response ticket mismatch")
        if int(response.get("generation", -1)) != int(pending["generation"]):
            raise RuntimeError("Arena inference response generation mismatch")
        if response.get("model_role") != self.model_role:
            raise RuntimeError("Arena inference response model role mismatch")
        if response.get("model_hash") != self.model_hash:
            raise RuntimeError("Arena inference response model hash mismatch")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        policy = tuple(float(value) for value in self.policy_slot[self.lane_id].tolist())
        wdl = tuple(float(value) for value in self.wdl_slot[self.lane_id].tolist())
        self._pending = None
        if not all(math.isfinite(value) and value >= 0.0 for value in wdl):
            raise RuntimeError(
                "Remote Arena evaluator received invalid WDL "
                f"role={self.model_role} lane={self.lane_id} "
                f"ticket={pending['ticket']} values={wdl}"
            )
        return Evaluation(policy=policy, wdl=wdl)

    def evaluate_prepared(self, state: Any, legal_context: Any) -> Evaluation:
        return self.evaluate_prepared_batch((state,), (legal_context,))[0]

    def evaluate_prepared_batch(
        self,
        states: Sequence[Any],
        legal_contexts: Sequence[Any],
    ) -> tuple[Evaluation, ...]:
        if len(states) != len(legal_contexts) or not states:
            raise ValueError(
                "Remote Arena inference requires matching non-empty state/context batches"
            )
        if len(states) != 1:
            raise ValueError("Arena remote evaluator supports one row per lane")
        self.submit_prepared(
            states[0],
            legal_contexts[0],
            generation=0,
            game_id="compatibility-call",
        )
        while True:
            evaluation = self.poll()
            if evaluation is not None:
                return (evaluation,)
            time.sleep(0.0005)


@dataclass
class _WorkerGame:
    task: Mapping[str, object]
    state: Any
    trace: list[dict[str, object]]
    ply: int
    started_at: float
    formal: str | None = None
    technical: str | None = None
    error: str | None = None
    session: SequentialPUCTSession | None = None
    pending_evaluator: _RemoteEvaluator | None = None
    generation: int = 0


def _make_game(task: Mapping[str, object]) -> _WorkerGame:
    return _WorkerGame(
        task=task,
        state=torus9_state_from_identity(task["state"]),  # type: ignore[arg-type]
        trace=[],
        ply=0,
        started_at=time.perf_counter(),
    )


def _finish_game(game: _WorkerGame) -> dict[str, object]:
    task = game.task
    state = game.state
    formal = game.formal
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
        "komi": TORUS9_KOMI,
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "start_state": task["state"],
        "start_trace": task["trace"],
        "action_trace": game.trace,
        "final_board": [int(stone) for stone in state.stones],
        "formal_result": formal,
        "technical_termination": game.technical,
        "error": game.error,
        "black_area": None,
        "white_area": None,
        "margin_black": None,
        "mapped_result": None,
        "wall_time_sec": time.perf_counter() - game.started_at,
    }
    if formal is not None:
        score = score_terminal(state)
        row.update(
            {
                "black_area": score.black_area,
                "white_area": score.white_area,
                "margin_black": score.margin_black,
            }
        )
        if formal == "DRAW":
            row["mapped_result"] = "DRAW"
        elif (formal == "BLACK") == candidate_black:
            row["mapped_result"] = "A_WIN"
        else:
            row["mapped_result"] = "B_WIN"
    return row


def _select_starts(master_seed: int, pairs: int) -> tuple[dict[str, object], ...]:
    if pairs <= 0:
        raise ValueError("Arena requires at least one pair")
    per_stratum = max(1, (pairs + 7) // 8)
    generated = generate_torus9_evaluation_starts(
        master_seed=master_seed,
        accepted_per_stratum=per_stratum,
    )
    selected: list[dict[str, object]] = []
    for offset in range(per_stratum):
        for stratum in range(8):
            index = stratum * per_stratum + offset
            if index < len(generated):
                selected.append(dict(generated[index]))
                if len(selected) == pairs:
                    return tuple(selected)
    raise RuntimeError("Could not build requested Torus9 Arena startset")


class Torus9ArenaProfile:
    profile_id = PROFILE_ID
    run_id_prefix = "torus9-arena"
    worker_process_prefix = "arena-torus9-worker"
    observation_shape = (6, TORUS9_POINT_COUNT)
    policy_size = TORUS9_ACTION_COUNT
    wdl_size = 3
    last_infer_timing: Mapping[str, float] = {}

    def matches_metadata(self, metadata: Mapping[str, object]) -> bool:
        return (
            metadata.get("profile_id") == TORUS9_CURRENT_PROFILE_ID
            and metadata.get("architecture_id") == TORUS9_CURRENT_ARCHITECTURE_ID
            and metadata.get("topology_fingerprint") == TORUS9_TOPOLOGY_FINGERPRINT
        )

    def validate_execution_config(self, config: ArenaExecutionConfig) -> None:
        if TORUS9_KOMI != 0.5:
            raise RuntimeError("Active Torus9 komi drifted from 0.5")
        if config.strict_production:
            if config.games < PRODUCTION_MIN_GAMES:
                raise ValueError("Production Torus9 Arena requires at least 64 games")
            if config.workers != PRODUCTION_WORKERS:
                raise ValueError("Production Torus9 Arena requires exactly 16 OS workers")
            if torch.device(config.device).type != "cuda":
                raise ValueError("Production Torus9 Arena requires CUDA central inference")
            if config.inference_batch_rows < PRODUCTION_MIN_BATCH_ROWS:
                raise ValueError(
                    "Production Torus9 Arena inference_batch_rows must be >= 16"
                )

    def load_identity(self, path: Path) -> CheckpointIdentity:
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Torus9 checkpoint or metadata: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not self.matches_metadata(metadata):
            raise ValueError("Checkpoint does not match Torus9 Arena profile")
        if float(metadata.get("komi", -1.0)) != 0.5:
            raise ValueError("Torus9 Arena checkpoint komi must be 0.5")
        architecture = metadata.get("architecture_config")
        if not isinstance(architecture, Mapping):
            raise ValueError("Torus9 checkpoint architecture metadata is malformed")
        return CheckpointIdentity(
            path=path,
            model_hash=str(metadata["model_hash"]),
            artifact_sha256=file_sha256(path),
            architecture_config=dict(architecture),
            metadata=dict(metadata),
        )

    def load_parent_model(
        self,
        identity: CheckpointIdentity,
        device: torch.device,
    ) -> torch.nn.Module:
        model = torus9_model_from_metadata(identity.metadata).to(device)
        torus9_load_checkpoint(
            identity.path,
            model=model,
            expected={"model_hash": identity.model_hash},
            device=device,
        )
        if model_hash(model) != identity.model_hash:
            raise RuntimeError("Parent inference broker loaded the wrong Torus9 checkpoint")
        model.eval()
        return model

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
    ) -> tuple[list[dict[str, object]], int]:
        pairs = games // 2
        starts = _select_starts(master_seed, pairs)
        tasks: list[dict[str, object]] = []
        for row in starts:
            pair_id = f"{comparison}--{row['start_id']}"
            for suffix, candidate_black in (("g1", True), ("g2", False)):
                game_id = f"{pair_id}--{suffix}"
                tasks.append(
                    {
                        "run_id": run_id,
                        "comparison": comparison,
                        "pair_id": pair_id,
                        "game_id": game_id,
                        "start_id": row["start_id"],
                        "state": row["state"],
                        "trace": row["trace"],
                        "candidate_black": candidate_black,
                        "candidate_hash": candidate.model_hash,
                        "reference_hash": reference.model_hash,
                        "candidate_artifact_sha256": candidate.artifact_sha256,
                        "reference_artifact_sha256": reference.artifact_sha256,
                        "game_seed": derive_seed(master_seed, pair_id, game_id),
                        "worker_id": len(tasks) % workers,
                    }
                )
        return tasks, pairs

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
    ) -> None:
        try:
            torch.set_num_threads(1)
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "CUDA was already initialized inside an Arena search worker"
                )
            request_queue.put(
                {
                    "kind": "ready",
                    "worker_id": worker_id,
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
            inference_aggregator = _WorkerInferenceAggregator(
                worker_id=worker_id,
                central_queue=request_queue,
                local_cap=int(games_per_worker),
                wait_ms=float(worker_local_wait_ms),
            )

            def take_task(*, block: bool) -> Mapping[str, object] | None:
                nonlocal queue_blocked_seconds
                wait_started = time.perf_counter() if block else None
                try:
                    task = task_queue.get(timeout=1.0) if block else task_queue.get_nowait()
                    if wait_started is not None:
                        queue_blocked_seconds += max(
                            0.0, time.perf_counter() - wait_started
                        )
                    return task
                except Empty:
                    if wait_started is not None:
                        queue_blocked_seconds += max(
                            0.0, time.perf_counter() - wait_started
                        )
                    return None

            # The central broker owns both model instances and CUDA.  Every
            # lane gets its own request/response transport and shared-memory
            # row.  A response for lane N can never be consumed by lane M.
            evaluators_by_lane = {
                lane_id: {
                    "candidate": _RemoteEvaluator(
                        worker_id=worker_id,
                        model_role="candidate",
                        model_hash_value=candidate_hash,
                        lane_id=lane_id,
                        input_slot=input_slot,
                        policy_slot=policy_slot,
                        wdl_slot=wdl_slot,
                        request_queue=inference_aggregator,
                        response_queue=response_queues[lane_id],
                    ),
                    "reference": _RemoteEvaluator(
                        worker_id=worker_id,
                        model_role="reference",
                        model_hash_value=reference_hash,
                        lane_id=lane_id,
                        input_slot=input_slot,
                        policy_slot=policy_slot,
                        wdl_slot=wdl_slot,
                        request_queue=inference_aggregator,
                        response_queue=response_queues[lane_id],
                    ),
                }
                for lane_id in range(int(games_per_worker))
            }
            search_settings = SearchSettings(
                simulations=64,
                cpuct=1.25,
                fpu=0.0,
                deterministic_tie_break=True,
            )

            active: dict[int, _WorkerGame] = {}
            records: list[dict[str, object]] = []
            lane_wall_time_seconds = 0.0
            started_games = 0
            queue_blocked_seconds = 0.0
            first_move_published = False
            used_lane_ids: set[int] = set()

            def record_game(game: _WorkerGame) -> None:
                nonlocal lane_wall_time_seconds
                records.append(_finish_game(game))
                lane_wall_time_seconds += time.perf_counter() - game.started_at

            def publish_activity(
                event: str,
                lane_id: int,
                game: _WorkerGame,
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
                        "game_id": str(game.task["game_id"]),
                        "at": time.perf_counter(),
                        "replenished": bool(replenished),
                    }
                )

            def start_lane(task: Mapping[str, object], lane_id: int) -> None:
                nonlocal started_games
                game = _make_game(dict(task, worker_id=worker_id))
                active[lane_id] = game
                used_lane_ids.add(int(lane_id))
                replenished = started_games >= int(games_per_worker)
                started_games += 1
                publish_activity(
                    "game_started",
                    lane_id,
                    game,
                    replenished=replenished,
                )

            def finish_lane(lane_id: int, game: _WorkerGame) -> None:
                record_game(game)
                publish_activity("game_completed", lane_id, game)
                del active[lane_id]

            def fill_lanes() -> None:
                while len(active) < int(games_per_worker):
                    task = take_task(block=False)
                    if task is None:
                        return
                    free_lane = next(
                        lane_id
                        for lane_id in range(int(games_per_worker))
                        if lane_id not in active
                    )
                    start_lane(task, free_lane)

            def candidate_turn(game: _WorkerGame) -> bool:
                state = game.state
                return (
                    state.side_to_move == BLACK and bool(game.task["candidate_black"])
                ) or (
                    state.side_to_move == WHITE and not bool(game.task["candidate_black"])
                )

            def mark_search_error(game: _WorkerGame, exc: Exception) -> None:
                game.technical = "ERROR_SEARCH"
                game.error = f"{type(exc).__name__}: {exc}"
                game.session = None
                game.pending_evaluator = None

            def advance_game(game: _WorkerGame, lane_id: int) -> bool:
                """Advance one session until it needs inference or finishes."""
                nonlocal first_move_published
                try:
                    while True:
                        if game.session is None:
                            search_seed = derive_seed(
                                int(game.task["game_seed"]),
                                game.ply + 1,
                                "arena-search",
                            )
                            game.session = SequentialPUCTSession(
                                game.state,
                                search_settings,
                                adapter=GoldenSearchAdapter(),
                                seed=search_seed,
                            )
                        step = game.session.advance()
                        if isinstance(step, SearchEvaluationRequest):
                            player_is_candidate = candidate_turn(game)
                            evaluator = evaluators_by_lane[lane_id][
                                "candidate" if player_is_candidate else "reference"
                            ]
                            evaluator.submit_prepared(
                                step.state,
                                step.legal_context,
                                generation=game.generation,
                                game_id=str(game.task["game_id"]),
                            )
                            game.pending_evaluator = evaluator
                            return False
                        if not isinstance(step, SearchResult):
                            raise RuntimeError(
                                "Torus9 cooperative search returned malformed result"
                            )

                        state = game.state
                        player_is_candidate = candidate_turn(game)
                        game.ply += 1
                        action = step.action
                        try:
                            next_state = apply_action(state, action).after
                        except IllegalMoveError as exc:
                            game.technical = "ERROR_ILLEGAL_PLAYER_ACTION"
                            game.error = f"{type(exc).__name__}: {exc}"
                            game.trace.append(
                                {
                                    "ply": game.ply,
                                    "side_to_move": state.side_to_move.name,
                                    "player": "candidate" if player_is_candidate else "reference",
                                    "action": action,
                                    "legal": False,
                                    "error": game.error,
                                }
                            )
                            return True

                        game.state = next_state
                        game.trace.append(
                            {
                                "ply": game.ply,
                                "side_to_move": (
                                    BLACK if next_state.side_to_move == WHITE else WHITE
                                ).name,
                                "player": "candidate" if player_is_candidate else "reference",
                                "action": action,
                                "legal": True,
                            }
                        )
                        if not first_move_published:
                            publish_activity("move_completed", lane_id, game)
                            first_move_published = True
                        game.session = None
                        game.pending_evaluator = None
                        game.generation += 1
                        if next_state.is_terminal:
                            game.formal = result_from_terminal(next_state).winner.value
                            return True
                        if game.ply >= TORUS9_ARENA_MOVE_LIMIT:
                            game.technical = "TRUNCATED_MOVE_LIMIT"
                            game.error = (
                                "Torus 9x9 Arena watchdog reached "
                                f"{TORUS9_ARENA_MOVE_LIMIT} actions"
                            )
                            return True
                except Exception as exc:
                    mark_search_error(game, exc)
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
                    game = active[lane_id]
                    if game.pending_evaluator is not None:
                        evaluation = game.pending_evaluator.poll()
                        if evaluation is None:
                            continue
                        game.pending_evaluator = None
                        try:
                            if game.session is None:
                                raise RuntimeError("Arena lane lost its suspended search session")
                            game.session.resume(evaluation)
                        except Exception as exc:
                            mark_search_error(game, exc)
                            finish_lane(lane_id, game)
                            progress = True
                            continue
                        progress = True
                    if lane_id not in active:
                        continue
                    if active[lane_id].pending_evaluator is None:
                        if advance_game(game, lane_id):
                            finish_lane(lane_id, game)
                        progress = True
                if not progress:
                    # All lanes are suspended on inference.  Avoid a blocking
                    # wait on any single response queue; the next poll can
                    # resume whichever lane the broker completed first.
                    time.sleep(0.0005)

            inference_aggregator.close()

            cpu_end = resource.getrusage(resource.RUSAGE_SELF)
            cpu_seconds = (cpu_end.ru_utime + cpu_end.ru_stime) - (
                cpu_start.ru_utime + cpu_start.ru_stime
            )
            request_queue.put(
                {
                    "kind": "done",
                    "worker_id": worker_id,
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
                    "observed_lane_ids": list(range(int(games_per_worker))),
                    "used_lane_ids": sorted(used_lane_ids),
                    "lane_replenishments": max(0, int(started_games) - int(games_per_worker)),
                    "queue_blocked_seconds": float(queue_blocked_seconds),
                    "records": records,
                }
            )
        except BaseException as exc:
            request_queue.put(
                {
                    "kind": "error",
                    "worker_id": worker_id,
                    "pid": os.getpid(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    def infer_batch(
        self,
        model: torch.nn.Module,
        cpu_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h2d_started = time.perf_counter()
        device_batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
        h2d_finished = time.perf_counter()
        forward_started = time.perf_counter()
        with torch.inference_mode():
            logits, wdl_logits = model(device_batch)
            policy_gpu = torch.softmax(logits, dim=1)
            wdl_gpu = torch.softmax(wdl_logits, dim=1)
        forward_finished = time.perf_counter()
        d2h_started = time.perf_counter()
        policy = policy_gpu.to("cpu", non_blocking=device.type == "cuda")
        wdl = wdl_gpu.to("cpu", non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        d2h_finished = time.perf_counter()
        self.last_infer_timing = {
            "h2d_ms": (h2d_finished - h2d_started) * 1000.0,
            "forward_ms": (forward_finished - forward_started) * 1000.0,
            "d2h_ms": (d2h_finished - d2h_started) * 1000.0,
        }
        return policy, wdl

    def summarize(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        candidate_label: str,
        reference_label: str,
        pairs: int,
    ) -> dict[str, object]:
        return summarize_torus9_arena(
            records,
            candidate_label=candidate_label,
            reference_label=reference_label,
            pairs=pairs,
        )

    def scientific_contract(
        self,
        config: ArenaExecutionConfig,
    ) -> Mapping[str, object]:
        return {
            "profile": PROFILE_ID,
            "games": config.games,
            "komi": 0.5,
            "simulations": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "watchdog": TORUS9_ARENA_MOVE_LIMIT,
            "paired_starts_color_swap": True,
            "deterministic_tie_break": True,
            "technical_fail_closed": True,
        }


PROFILE = Torus9ArenaProfile()
