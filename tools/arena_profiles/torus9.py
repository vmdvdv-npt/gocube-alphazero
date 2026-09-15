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
from gocube_golden.search import Evaluation, SequentialPUCT
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
    """Worker-side evaluator proxy; no model or CUDA owner exists in workers."""

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
        rows = len(states)
        if rows > int(self.input_slot.shape[0]):
            raise RuntimeError("Worker inference request exceeds its shared-memory slot")
        observations = torch.stack(
            [
                build_torus9_observation(state, legal_context=context)
                for state, context in zip(states, legal_contexts)
            ]
        )
        self.input_slot[self.lane_id:self.lane_id + rows].copy_(observations)
        self.ticket += 1
        # Candidate and reference evaluators share one lane response queue.
        # Encode the role in the ticket so model-aware dispatch reordering can
        # never let the two evaluators consume each other's response.
        ticket = self.ticket * 2 + (1 if self.model_role == "candidate" else 2)
        worker_enqueued_at = time.perf_counter()
        self.request_queue.put(
            {
                "kind": "inference",
                "worker_id": self.worker_id,
                "lane_id": self.lane_id,
                "pid": os.getpid(),
                "ticket": ticket,
                "model_role": self.model_role,
                "model_hash": self.model_hash,
                "rows": rows,
                "worker_enqueued_at": worker_enqueued_at,
            }
        )
        wait_started = time.perf_counter()
        try:
            response = self.response_queue.get()
        finally:
            self.blocked_inference_seconds += time.perf_counter() - wait_started
            self.blocked_inference_calls += 1
        if int(response.get("ticket", -1)) != ticket:
            raise RuntimeError("Arena inference response ticket mismatch")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        evaluations: list[Evaluation] = []
        for row in range(rows):
            policy = tuple(float(value) for value in self.policy_slot[self.lane_id + row].tolist())
            wdl = tuple(float(value) for value in self.wdl_slot[self.lane_id + row].tolist())
            if not all(math.isfinite(value) and value >= 0.0 for value in wdl):
                raise RuntimeError(
                    "Remote Arena evaluator received invalid WDL "
                    f"role={self.model_role} lane={self.lane_id} ticket={ticket} values={wdl}"
                )
            evaluations.append(Evaluation(policy=policy, wdl=wdl))
        return tuple(evaluations)


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
                try:
                    return task_queue.get(timeout=1.0) if block else task_queue.get_nowait()
                except Empty:
                    return None

            # The central broker owns both model instances and CUDA.  Each
            # game uses the canonical SequentialPUCT implementation; requests
            # from the independent OS workers are still coalesced centrally.
            candidate_eval = _RemoteEvaluator(
                worker_id=worker_id,
                model_role="candidate",
                model_hash_value=candidate_hash,
                lane_id=0,
                input_slot=input_slot,
                policy_slot=policy_slot,
                wdl_slot=wdl_slot,
                request_queue=inference_aggregator,
                response_queue=response_queues[0],
            )
            reference_eval = (
                candidate_eval
                if candidate_hash == reference_hash
                else _RemoteEvaluator(
                    worker_id=worker_id,
                    model_role="reference",
                    model_hash_value=reference_hash,
                    lane_id=0,
                    input_slot=input_slot,
                    policy_slot=policy_slot,
                    wdl_slot=wdl_slot,
                    request_queue=inference_aggregator,
                    response_queue=response_queues[0],
                )
            )
            search_settings = SearchSettings(
                simulations=64,
                cpuct=1.25,
                fpu=0.0,
                deterministic_tie_break=True,
            )

            active: list[_WorkerGame] = []
            records: list[dict[str, object]] = []
            lane_wall_time_seconds = 0.0

            def record_game(game: _WorkerGame) -> None:
                nonlocal lane_wall_time_seconds
                records.append(_finish_game(game))
                lane_wall_time_seconds += time.perf_counter() - game.started_at

            while True:
                task = take_task(block=True)
                if task is None:
                    break
                task = dict(task)
                task["worker_id"] = worker_id
                game = _make_game(task)
                while game.formal is None and game.technical is None:
                    state = game.state
                    candidate_turn = (
                        state.side_to_move == BLACK
                        and bool(task["candidate_black"])
                    ) or (
                        state.side_to_move == WHITE
                        and not bool(task["candidate_black"])
                    )
                    try:
                        evaluator = candidate_eval if candidate_turn else reference_eval
                        result = SequentialPUCT(
                            search_settings,
                            adapter=GoldenSearchAdapter(),
                        ).search(
                            state,
                            evaluator,
                            seed=derive_seed(
                                int(task["game_seed"]),
                                game.ply + 1,
                                "arena-search",
                            ),
                        )
                    except Exception as exc:
                        game.technical = "ERROR_SEARCH"
                        game.error = f"{type(exc).__name__}: {exc}"
                        record_game(game)
                        break

                    game.ply += 1
                    action = result.action
                    try:
                        next_state = apply_action(state, action).after
                    except IllegalMoveError as exc:
                        game.technical = "ERROR_ILLEGAL_PLAYER_ACTION"
                        game.error = f"{type(exc).__name__}: {exc}"
                        game.trace.append(
                            {
                                "ply": game.ply,
                                "side_to_move": state.side_to_move.name,
                                "player": "candidate" if candidate_turn else "reference",
                                "action": action,
                                "legal": False,
                                "error": game.error,
                            }
                        )
                        record_game(game)
                        break

                    game.state = next_state
                    game.trace.append(
                        {
                            "ply": game.ply,
                            "side_to_move": (
                                BLACK if next_state.side_to_move == WHITE else WHITE
                            ).name,
                            "player": "candidate" if candidate_turn else "reference",
                            "action": action,
                            "legal": True,
                        }
                    )
                    if next_state.is_terminal:
                        game.formal = result_from_terminal(next_state).winner.value
                        record_game(game)
                    elif game.ply >= TORUS9_ARENA_MOVE_LIMIT:
                        game.technical = "TRUNCATED_MOVE_LIMIT"
                        game.error = (
                            "Torus 9x9 Arena watchdog reached "
                            f"{TORUS9_ARENA_MOVE_LIMIT} actions"
                        )
                        record_game(game)

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
                        candidate_eval.blocked_inference_seconds
                        + (0.0 if reference_eval is candidate_eval else reference_eval.blocked_inference_seconds)
                    ),
                    "blocked_inference_calls": int(
                        candidate_eval.blocked_inference_calls
                        + (0 if reference_eval is candidate_eval else reference_eval.blocked_inference_calls)
                    ),
                    "lane_wall_time_seconds": float(lane_wall_time_seconds),
                    "max_rss_kb": int(cpu_end.ru_maxrss),
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
        with torch.inference_mode():
            logits, wdl_logits = model(cpu_batch.to(device))
            policy = torch.softmax(logits, dim=1).to("cpu")
            wdl = torch.softmax(wdl_logits, dim=1).to("cpu")
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
