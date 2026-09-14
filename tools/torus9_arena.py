#!/usr/bin/env python3
"""Canonical Torus 9x9 production Arena.

Execution architecture:
    16 OS CPU MCTS workers -> one parent inference broker -> one CUDA model owner.

The scientific Arena contract is intentionally fixed. Execution knobs are
separate and reported explicitly so they can be benchmarked without changing
search semantics.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty
import resource
import statistics
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.arena_policy import (
    CANONICAL_TORUS9_ARENA_ENGINE,
    CANONICAL_TORUS9_KOMI,
    CANONICAL_TORUS9_WORKERS,
)
from gocube_golden.neural import model_hash
from gocube_golden.provenance import derive_seed, file_sha256
from gocube_golden.result import result_from_terminal
from gocube_golden.rules import IllegalMoveError, apply_action
from gocube_golden.scoring import score_terminal
from gocube_golden.search import Evaluation
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, WHITE
from gocube_golden.torus9 import (
    TORUS9_TOPOLOGY_FINGERPRINT,
    Torus9BatchedPUCT,
    build_torus9_observation,
    generate_torus9_evaluation_starts,
    summarize_torus9_arena,
    torus9_arena_termination_reason,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    torus9_state_from_identity,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_ARENA_MOVE_LIMIT,
    TORUS9_KOMI,
    TORUS9_POINT_COUNT,
)

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

    def validate(self) -> None:
        if self.games <= 0 or self.games % 2:
            raise ValueError("Torus9 Arena games must be a positive even number")
        if self.workers <= 0 or self.games_per_worker <= 0:
            raise ValueError("Torus9 Arena workers/games_per_worker must be positive")
        if self.inference_batch_rows <= 0 or self.inference_batch_wait_ms < 0.0:
            raise ValueError("Torus9 Arena inference batching settings are invalid")
        if self.inference_batch_rows < self.games_per_worker:
            raise ValueError("inference_batch_rows must cover at least one worker request")
        if self.strict_production:
            if self.games < DEFAULT_GAMES:
                raise ValueError("Production Torus9 Arena requires at least 64 games")
            if self.workers != CANONICAL_TORUS9_WORKERS:
                raise ValueError("Production Torus9 Arena requires exactly 16 OS workers")
            if torch.device(self.device).type != "cuda":
                raise ValueError("Production Torus9 Arena requires CUDA central inference")
            if self.inference_batch_rows < 16:
                raise ValueError("Production Torus9 Arena inference_batch_rows must be >= 16")


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    model_hash: str
    artifact_sha256: str
    architecture_config: Mapping[str, object]


class _RemoteEvaluator:
    """Worker-side evaluator proxy. No model and no CUDA state live here."""

    def __init__(
        self,
        *,
        worker_id: int,
        model_role: str,
        model_hash_value: str,
        input_slot: torch.Tensor,
        policy_slot: torch.Tensor,
        wdl_slot: torch.Tensor,
        request_queue: Any,
        response_queue: Any,
    ) -> None:
        self.worker_id = int(worker_id)
        self.model_role = str(model_role)
        self.model_hash = str(model_hash_value)
        self.input_slot = input_slot
        self.policy_slot = policy_slot
        self.wdl_slot = wdl_slot
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.ticket = 0

    def evaluate_prepared(self, state: Any, legal_context: Any) -> Evaluation:
        return self.evaluate_prepared_batch((state,), (legal_context,))[0]

    def evaluate_prepared_batch(
        self,
        states: Sequence[Any],
        legal_contexts: Sequence[Any],
    ) -> tuple[Evaluation, ...]:
        if len(states) != len(legal_contexts) or not states:
            raise ValueError("Remote Arena inference requires matching non-empty state/context batches")
        rows = len(states)
        if rows > int(self.input_slot.shape[0]):
            raise RuntimeError("Worker inference request exceeds its shared-memory slot")
        observations = torch.stack(
            [
                build_torus9_observation(state, legal_context=context)
                for state, context in zip(states, legal_contexts)
            ]
        )
        self.input_slot[:rows].copy_(observations)
        self.ticket += 1
        ticket = self.ticket
        self.request_queue.put(
            {
                "kind": "inference",
                "worker_id": self.worker_id,
                "pid": os.getpid(),
                "ticket": ticket,
                "model_role": self.model_role,
                "model_hash": self.model_hash,
                "rows": rows,
                "enqueued_at": time.perf_counter(),
            }
        )
        response = self.response_queue.get()
        if int(response.get("ticket", -1)) != ticket:
            raise RuntimeError("Arena inference response ticket mismatch")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return tuple(
            Evaluation(
                policy=tuple(float(value) for value in self.policy_slot[row].tolist()),
                wdl=tuple(float(value) for value in self.wdl_slot[row].tolist()),
            )
            for row in range(rows)
        )


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


def _worker_main(
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
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    try:
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was already initialized inside an Arena MCTS worker")
        request_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "cuda_initialized": False,
            }
        )
        start_event.wait()
        candidate_eval = _RemoteEvaluator(
            worker_id=worker_id,
            model_role="candidate",
            model_hash_value=candidate_hash,
            input_slot=input_slot,
            policy_slot=policy_slot,
            wdl_slot=wdl_slot,
            request_queue=request_queue,
            response_queue=response_queue,
        )
        reference_eval = (
            candidate_eval
            if candidate_hash == reference_hash
            else _RemoteEvaluator(
                worker_id=worker_id,
                model_role="reference",
                model_hash_value=reference_hash,
                input_slot=input_slot,
                policy_slot=policy_slot,
                wdl_slot=wdl_slot,
                request_queue=request_queue,
                response_queue=response_queue,
            )
        )
        search = Torus9BatchedPUCT(
            SearchSettings(
                simulations=64,
                cpuct=1.25,
                fpu=0.0,
                deterministic_tie_break=True,
            ),
            adapter=GoldenSearchAdapter(),
            max_batch_rows=int(games_per_worker),
            inference_batch_wait_ms=0.0,
        )
        waiting = deque(tasks)
        active: list[_WorkerGame] = []
        records: list[dict[str, object]] = []
        while waiting or active:
            while waiting and len(active) < games_per_worker:
                active.append(_make_game(waiting.popleft()))
            if not active:
                break
            states = [game.state for game in active]
            evaluators: list[_RemoteEvaluator] = []
            seeds: list[int] = []
            for game in active:
                task = game.task
                candidate_turn = (
                    game.state.side_to_move == BLACK and bool(task["candidate_black"])
                ) or (
                    game.state.side_to_move == WHITE and not bool(task["candidate_black"])
                )
                evaluators.append(candidate_eval if candidate_turn else reference_eval)
                seeds.append(derive_seed(int(task["game_seed"]), game.ply + 1, "arena-search"))
            try:
                results = search.search(states, evaluators, seeds=seeds)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                for game in active:
                    game.technical = "ERROR_SEARCH"
                    game.error = message
                    records.append(_finish_game(game))
                active.clear()
                continue
            survivors: list[_WorkerGame] = []
            for game, result in zip(active, results):
                task = game.task
                game.ply += 1
                state = game.state
                candidate_turn = (
                    state.side_to_move == BLACK and bool(task["candidate_black"])
                ) or (
                    state.side_to_move == WHITE and not bool(task["candidate_black"])
                )
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
                    records.append(_finish_game(game))
                    continue
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
                if torus9_arena_termination_reason(next_state, game.ply) == "DOUBLE_PASS":
                    game.formal = result_from_terminal(next_state).winner.value
                    records.append(_finish_game(game))
                elif game.ply >= TORUS9_ARENA_MOVE_LIMIT:
                    game.technical = "TRUNCATED_MOVE_LIMIT"
                    game.error = (
                        f"Torus 9x9 Arena watchdog reached {TORUS9_ARENA_MOVE_LIMIT} actions"
                    )
                    records.append(_finish_game(game))
                else:
                    survivors.append(game)
            active = survivors
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


def _load_identity(path: Path) -> CheckpointIdentity:
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Torus9 checkpoint or metadata: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if float(metadata.get("komi", -1.0)) != CANONICAL_TORUS9_KOMI:
        raise ValueError("Torus9 Arena checkpoint komi must be 0.5")
    architecture = metadata.get("architecture_config")
    if not isinstance(architecture, Mapping):
        raise ValueError("Torus9 checkpoint architecture metadata is malformed")
    return CheckpointIdentity(
        path=path,
        model_hash=str(metadata["model_hash"]),
        artifact_sha256=file_sha256(path),
        architecture_config=dict(architecture),
    )


def _load_parent_model(identity: CheckpointIdentity, device: torch.device) -> torch.nn.Module:
    metadata = json.loads(
        identity.path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    model = torus9_model_from_metadata(metadata).to(device)
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


def _select_starts(master_seed: int, pairs: int) -> tuple[dict[str, object], ...]:
    if pairs <= 0:
        raise ValueError("Arena requires at least one pair")
    per_stratum = max(1, math.ceil(pairs / 8))
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


def _build_tasks(
    *,
    starts: Sequence[Mapping[str, object]],
    run_id: str,
    comparison: str,
    candidate: CheckpointIdentity,
    reference: CheckpointIdentity,
    master_seed: int,
    workers: int,
) -> list[dict[str, object]]:
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
    return tasks


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


def run_arena(
    *,
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
    """Run the single canonical Torus9 Arena executor."""
    config.validate()
    if TORUS9_KOMI != 0.5:
        raise RuntimeError("Active Torus9 komi drifted from 0.5")
    candidate_path = candidate_path.resolve()
    reference_path = (reference_path or candidate_path).resolve()
    candidate = _load_identity(candidate_path)
    reference = _load_identity(reference_path)
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
    run_id = run_id or f"torus9-arena-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = output_dir.resolve()
    if any(
        (output_dir / name).exists()
        for name in ("manifest.json", "games.jsonl", "summary.json")
    ):
        raise FileExistsError(f"Arena output directory already contains a run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    starts = _select_starts(master_seed, config.games // 2)
    tasks = _build_tasks(
        starts=starts,
        run_id=run_id,
        comparison=comparison,
        candidate=candidate,
        reference=reference,
        master_seed=master_seed,
        workers=config.workers,
    )
    buckets: list[list[dict[str, object]]] = [[] for _ in range(config.workers)]
    for task in tasks:
        buckets[int(task["worker_id"])].append(task)

    candidate_model = _load_parent_model(candidate, device)
    reference_model = (
        candidate_model
        if reference.model_hash == candidate.model_hash
        and reference.architecture_config == candidate.architecture_config
        else _load_parent_model(reference, device)
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
            (config.games_per_worker, 6, TORUS9_POINT_COUNT),
            dtype=torch.float32,
        ).share_memory_()
        for _ in range(config.workers)
    ]
    shared_policy = [
        torch.empty(
            (config.games_per_worker, TORUS9_ACTION_COUNT),
            dtype=torch.float32,
        ).share_memory_()
        for _ in range(config.workers)
    ]
    shared_wdl = [
        torch.empty((config.games_per_worker, 3), dtype=torch.float32).share_memory_()
        for _ in range(config.workers)
    ]
    processes = []
    for worker_id in range(config.workers):
        process = ctx.Process(
            target=_worker_main,
            args=(
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
            name=f"torus9-arena-worker-{worker_id:02d}",
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
            raise RuntimeError("CUDA initialized inside an Arena MCTS worker")

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
                with torch.inference_mode():
                    logits, wdl_logits = model(cpu_batch.to(device))
                    policy = torch.softmax(logits, dim=1).to("cpu")
                    wdl = torch.softmax(wdl_logits, dim=1).to("cpu")
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
    write_jsonl(output_dir / "games.jsonl", records)
    summary = summarize_torus9_arena(
        records,
        candidate_label=candidate_label,
        reference_label=reference_label,
        pairs=len(starts),
    )
    worker_cpu_seconds = [float(done[index]["cpu_seconds"]) for index in sorted(done)]
    effective_cpu_cores = sum(worker_cpu_seconds) / wall_time if wall_time else 0.0
    mean_batch = statistics.mean(batch_rows) if batch_rows else 0.0
    cross_worker_calls = sum(count > 1 for count in batch_worker_counts)
    telemetry = {
        "arena_engine": CANONICAL_TORUS9_ARENA_ENGINE,
        "broker_pid": os.getpid(),
        "model_owner_pid": os.getpid(),
        "worker_pids": worker_pids,
        "worker_processes_requested": config.workers,
        "worker_processes_observed": len(set(worker_pids)),
        "games_per_worker_lane_capacity": config.games_per_worker,
        "aggregate_worker_cpu_seconds": sum(worker_cpu_seconds),
        "effective_cpu_cores": effective_cpu_cores,
        "effective_cpu_pct_of_16": 100.0 * effective_cpu_cores / 16.0,
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
            "comparison": comparison,
            "run_id": run_id,
            "candidate_model_hash": candidate.model_hash,
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "reference_model_hash": reference.model_hash,
            "reference_artifact_sha256": reference.artifact_sha256,
            "komi": TORUS9_KOMI,
            "scientific_contract": {
                "games": config.games,
                "simulations": 64,
                "cpuct": 1.25,
                "fpu": 0.0,
                "noise": False,
                "temperature": 0.0,
                "fast_search": False,
                "resign": False,
                "watchdog": TORUS9_ARENA_MOVE_LIMIT,
                "paired_starts_color_swap": True,
                "technical_fail_closed": True,
            },
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
    write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "arena_engine": CANONICAL_TORUS9_ARENA_ENGINE,
        "run_id": run_id,
        "comparison": comparison,
        "candidate": candidate.__dict__ | {"path": str(candidate.path)},
        "reference": reference.__dict__ | {"path": str(reference.path)},
        "master_seed": master_seed,
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.json"),
        "games": str(output_dir / "games.jsonl"),
        "performance_status": telemetry["performance_status"],
    }
    write_json(output_dir / "manifest.json", manifest)
    if config.strict_production and performance_failures:
        raise RuntimeError(
            "Torus9 Arena completed but failed production performance/validity gates: "
            + ", ".join(performance_failures)
        )
    return summary


def _default_output(candidate: Path, reference: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("arena-results") / f"torus9-{candidate.stem}-vs-{reference.stem}-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Defaults to --candidate for self A/B",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--reference-label", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--comparison", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--games-per-worker", type=int, default=DEFAULT_GAMES_PER_WORKER)
    parser.add_argument(
        "--inference-batch-rows",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_ROWS,
    )
    parser.add_argument(
        "--inference-batch-wait-ms",
        type=float,
        default=DEFAULT_INFERENCE_BATCH_WAIT_MS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-candidate-model-hash", default=None)
    parser.add_argument("--expected-candidate-artifact-sha256", default=None)
    parser.add_argument("--expected-reference-model-hash", default=None)
    parser.add_argument("--expected-reference-artifact-sha256", default=None)
    parser.add_argument(
        "--debug-non-production",
        action="store_true",
        help="Allow reduced games/workers or CPU; result is explicitly non-production.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = args.reference or args.candidate
    output = args.output or _default_output(args.candidate, reference)
    config = ArenaExecutionConfig(
        games=args.games,
        workers=args.workers,
        games_per_worker=args.games_per_worker,
        inference_batch_rows=args.inference_batch_rows,
        inference_batch_wait_ms=args.inference_batch_wait_ms,
        device=args.device,
        strict_production=not args.debug_non_production,
    )
    summary = run_arena(
        candidate_path=args.candidate,
        reference_path=reference,
        output_dir=output,
        candidate_label=args.candidate_label,
        reference_label=args.reference_label,
        run_id=args.run_id,
        comparison=args.comparison,
        master_seed=args.seed,
        config=config,
        expected_candidate_model_hash=args.expected_candidate_model_hash,
        expected_candidate_artifact_sha256=args.expected_candidate_artifact_sha256,
        expected_reference_model_hash=args.expected_reference_model_hash,
        expected_reference_artifact_sha256=args.expected_reference_artifact_sha256,
    )
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "comparison": summary["comparison"],
                "games": summary["games"],
                "W/L/D": summary["W/L/D"],
                "performance_status": summary["telemetry"]["performance_status"],
                "games_per_hour": summary["telemetry"]["games_per_hour"],
                "mean_inference_batch_rows": summary["telemetry"][
                    "mean_inference_batch_rows"
                ],
                "effective_cpu_cores": summary["telemetry"]["effective_cpu_cores"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
