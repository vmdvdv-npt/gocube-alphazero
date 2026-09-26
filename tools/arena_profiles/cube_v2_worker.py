"""Cube V2 scientific adapter for the common cooperative Arena worker."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Mapping

import torch

from gocube_golden.cube_arena_startset_v1 import (
    opening_fingerprint,
    reconstruct_cube_arena_start,
)
from gocube_golden.cube_observation_v2 import (
    build_cube_observation,
)
from gocube_golden.cube_search import CubeSearchAdapter, CubeSearchPosition
from gocube_golden.provenance import derive_seed
from gocube_golden.result import result_from_terminal
from gocube_golden.search import SearchResult
from gocube_golden.state import BLACK, WHITE
from tools.arena_worker import (
    CooperativeArenaCallbacks,
    run_cooperative_arena_worker,
)


@dataclass
class _CubeGame:
    task: Mapping[str, object]
    position: CubeSearchPosition
    trace: list[dict[str, object]]
    start_trace: list[dict[str, object]]
    ply: int
    started_at: float
    formal_result: str | None = None
    technical: str | None = None
    error: str | None = None


def _candidate_turn(game: _CubeGame) -> bool:
    side = game.position.game_state.side_to_move
    candidate_black = bool(game.task["candidate_black"])
    return (side == BLACK and candidate_black) or (
        side == WHITE and not candidate_black
    )


def _finish_record(game: _CubeGame) -> dict[str, object]:
    task = game.task
    candidate_black = bool(task["candidate_black"])
    row: dict[str, object] = {
        "run_id": str(task["run_id"]),
        "comparison": str(task["comparison"]),
        "pair_id": str(task["pair_id"]),
        "game_id": str(task["game_id"]),
        "start_id": str(task["start_id"]),
        "start_kind": str(task["start_kind"]),
        "start_fingerprint": str(task["start_fingerprint"]),
        "opening_ply": int(task["opening_ply"]),
        "start_trace": list(game.start_trace),
        "worker_id": int(task["worker_id"]),
        "worker_pid": os.getpid(),
        "candidate_black": candidate_black,
        "candidate_model_hash": str(task["candidate_hash"]),
        "reference_model_hash": str(task["reference_hash"]),
        "action_trace": game.trace,
        "formal_result": game.formal_result,
        "technical_termination": game.technical,
        "error": game.error,
        "mapped_result": None,
        "wall_time_sec": time.perf_counter() - game.started_at,
    }
    if game.formal_result is not None:
        if game.formal_result == "DRAW":
            row["mapped_result"] = "DRAW"
        elif (game.formal_result == "BLACK") == candidate_black:
            row["mapped_result"] = "A_WIN"
        else:
            row["mapped_result"] = "B_WIN"
        result = result_from_terminal(game.position.game_state)
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
    """Bind Cube science to the one topology-neutral cooperative runtime."""

    search_adapter = CubeSearchAdapter()

    def make_game(task: Mapping[str, object]) -> _CubeGame:
        raw_actions = task.get("opening_actions")
        if not isinstance(raw_actions, (list, tuple)):
            raise ValueError("Cube Arena task opening_actions is missing")
        opening_actions = tuple(int(action) for action in raw_actions)
        position = reconstruct_cube_arena_start(size=size, opening_actions=opening_actions)
        opening_ply = int(task.get("opening_ply", -1))
        if opening_ply != len(opening_actions):
            raise ValueError("Cube Arena task opening_ply does not match opening_actions")
        actual_fingerprint = opening_fingerprint(
            size=size, opening_actions=opening_actions, position=position
        )
        if actual_fingerprint != str(task.get("start_fingerprint")):
            raise ValueError("Cube Arena task opening fingerprint mismatch")
        start_trace = []
        replayed = reconstruct_cube_arena_start(size=size, opening_actions=())
        for ply, action in enumerate(opening_actions, start=1):
            start_trace.append(
                {
                    "ply": ply,
                    "side_to_move": replayed.game_state.side_to_move.name,
                    "action": int(action),
                    "legal": True,
                }
            )
            replayed = search_adapter.apply_action(replayed, int(action))
        if replayed.state_key != position.state_key:
            raise ValueError("Cube Arena opening replay is not reproducible")
        return _CubeGame(
            task=task,
            position=position,
            trace=[],
            start_trace=start_trace,
            ply=opening_ply,
            started_at=time.perf_counter(),
        )

    def build_observation(position: CubeSearchPosition, legal_context: Any) -> torch.Tensor:
        return build_cube_observation(
            position.game_state,
            position.observation_context,
            legal_context=legal_context,
        )

    def apply_search_result(game: _CubeGame, result: SearchResult) -> bool:
        previous = game.position
        role = "candidate" if _candidate_turn(game) else "reference"
        game.ply += 1
        action_index = search_adapter.action_index(previous, result.action)
        game.position = search_adapter.apply_action(previous, result.action)
        game.trace.append(
            {
                "ply": game.ply,
                "side_to_move": previous.game_state.side_to_move.name,
                "player": role,
                "action": action_index,
                "legal": True,
            }
        )
        if game.position.is_terminal:
            game.formal_result = result_from_terminal(
                game.position.game_state
            ).winner.value
            return True
        if game.ply >= int(search_config.watchdog):
            game.technical = "WATCHDOG"
            game.error = (
                f"Cube Arena watchdog reached {int(search_config.watchdog)} actions"
            )
            return True
        return False

    def mark_search_error(game: _CubeGame, exc: Exception) -> None:
        game.technical = "ERROR_SEARCH"
        game.error = f"{type(exc).__name__}: {exc}"

    callbacks = CooperativeArenaCallbacks(
        search_settings=search_config.search_settings,
        search_adapter=search_adapter,
        make_game=make_game,
        game_id=lambda game: str(game.task["game_id"]),
        search_state=lambda game: game.position,
        search_seed=lambda game: derive_seed(
            int(game.task["game_seed"]),
            game.ply + 1,
            "cube-arena-search",
        ),
        model_role=lambda game: (
            "candidate" if _candidate_turn(game) else "reference"
        ),
        build_observation=build_observation,
        apply_search_result=apply_search_result,
        mark_search_error=mark_search_error,
        finish_record=_finish_record,
    )
    run_cooperative_arena_worker(
        callbacks=callbacks,
        worker_id=worker_id,
        task_queue=task_queue,
        games_per_worker=games_per_worker,
        worker_local_wait_ms=worker_local_wait_ms,
        candidate_hash=candidate_hash,
        reference_hash=reference_hash,
        input_slot=input_slot,
        policy_slot=policy_slot,
        wdl_slot=wdl_slot,
        request_queue=request_queue,
        response_queues=response_queues,
        start_event=start_event,
    )


__all__ = ["run_cube_v2_worker"]
